from flask import Blueprint, request, jsonify, current_app, stream_template
from werkzeug.utils import secure_filename
import os
import asyncio
import threading
from queue import Queue, Empty
import json
import logging
from typing import Generator
from flask import send_from_directory

# 导入自定义模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools_agent.llm_manager import LLMManager
from config.constants import (
    TRANSLATION_PROMPT_TEMPLATE, 
    PAPER_TRANSLATION_PROMPT_TEMPLATE,
    SUPPORTED_LANGUAGES,
    TRANSLATION_SCENES,
    DEFAULT_MODEL,
    ALLOWED_EXTENSIONS
)
from utils.pdf_processor import PDFProcessor
from utils.session_manager import session_manager
from utils.agent_tool_pdf_translation import AsyncPDFTranslator

main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route('/api/languages', methods=['GET'])
def get_languages():
    """获取支持的语言列表"""
    return jsonify(SUPPORTED_LANGUAGES)

@main_bp.route('/api/scenes', methods=['GET'])
def get_scenes():
    """获取支持的翻译场景列表"""
    return jsonify(TRANSLATION_SCENES)

@main_bp.route('/api/translate', methods=['POST'])
def translate_text():
    """快速翻译文本"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        target_language = data.get('language', 'English')
        scene_id = data.get('scene', 'ecommerce_amazon')
        other_requirements = data.get('requirements', '').strip()
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        # 查找场景描述
        scene_description = "通用翻译场景"
        for scene in TRANSLATION_SCENES:
            if scene['id'] == scene_id:
                scene_description = scene['description']
                break
        
        # 构建额外要求部分
        requirements_text = ""
        if other_requirements:
            requirements_text = f"\n# 其他要求\n{other_requirements}\n"
        
        # 创建LLM管理器
        llm = LLMManager(DEFAULT_MODEL)
        
        # 构建提示词
        prompt = TRANSLATION_PROMPT_TEMPLATE.format(
            language=target_language,
            scene=scene_description,
            text=text,
            other_requirements=requirements_text
        )
        
        # 创建翻译会话
        session_id = session_manager.create_session()
        
        def generate():
            try:
                # 先发送会话ID
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"
                
                for chunk in llm.generate_stream(prompt):
                    # 检查会话是否被取消
                    if session_manager.is_session_cancelled(session_id):
                        yield f"data: {json.dumps({'cancelled': True, 'message': '翻译已被用户中断'})}\n\n"
                        break
                    
                    if chunk:
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                
                # 如果没有被取消，发送完成信号
                if not session_manager.is_session_cancelled(session_id):
                    yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                logger.error(f"翻译过程中出错: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                # 完成会话
                session_manager.finish_session(session_id)
        
        return current_app.response_class(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': 'http://localhost:3000',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            }
        )
        
    except Exception as e:
        logger.error(f"翻译API错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/file/<path:filename>', methods=['GET'])
def serve_uploaded_file(filename: str):
    """安全地返回上传的PDF文件用于前端预览
    仅允许从 UPLOAD_FOLDER 中读取，阻止路径穿越
    """
    try:
        safe_name = secure_filename(filename)
        if not safe_name:
            return jsonify({"error": "文件名不合法"}), 400

        upload_dir = current_app.config['UPLOAD_FOLDER']
        file_path = os.path.join(upload_dir, safe_name)
        if not os.path.isfile(file_path):
            return jsonify({"error": "文件不存在"}), 404

        logger.info(f"[file] preview filename={safe_name}")
        resp = send_from_directory(upload_dir, safe_name, as_attachment=False, mimetype='application/pdf')
        resp.headers['Cache-Control'] = 'no-cache'
        resp.headers['Accept-Ranges'] = 'bytes'
        resp.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        return resp
    except Exception as e:
        logger.error(f"[file] serve error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/upload', methods=['POST'])
def upload_file():
    """上传PDF文件"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "没有选择文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 处理PDF文件
            pdf_processor = PDFProcessor()
            pdf_info = pdf_processor.get_pdf_info(filepath)
            pages = pdf_processor.extract_text_from_pdf(filepath)
            
            return jsonify({
                "success": True,
                "filename": filename,
                "filepath": filepath,
                "info": pdf_info,
                "pages": pages
            })
        else:
            return jsonify({"error": "不支持的文件类型"}), 400
            
    except Exception as e:
        logger.error(f"文件上传错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/translate-pdf', methods=['POST'])
def translate_pdf():
    """整份PDF异步并行翻译（SSE流式返回）
    输入参数: { filepath: str }
    逻辑: 使用 AsyncPDFTranslator 对整份PDF进行OCR转Markdown、分段、并行翻译，按顺序流式推送到前端
    """
    try:
        data = request.json
        filepath = (data or {}).get('filepath', '').strip()

        if not filepath or not os.path.isfile(filepath):
            return jsonify({"error": "文件路径无效"}), 400
        if not filepath.lower().endswith('.pdf'):
            return jsonify({"error": "仅支持PDF文件"}), 400

        # 会话与日志
        session_id = session_manager.create_session()
        logger.info(f"[translate-pdf] session={session_id} file={filepath}")

        # 与 AsyncPDFTranslator 对齐的提示模板（占位符为 {paragraph}）
        PDF_TRANSLATION_PROMPT_TEMPLATE = (
"""
# 你的角色
具有数十年经验的高级翻译专家

# 你的任务
将我下面这段学术论文翻译为地道的中文，符合专业场景

# 论文内容
{paragraph}

# 要求
- 你的输出必须仅包含翻译后的段落，不要包含任何其他内容
- 在遇到公式、分点等内容的时候，你都需要用markdown样式，示例：$$f(x) = x^2$$；
- 即使原文展示的公式、表格没有用markdown，你也需要转换为markdown样式；
- 如果原文包含markdown格式的图片，你需要完整保留所有的内容，示例：![fig_1543](https://...);这里的所有内容你都不能修改
- 在遇到明显是标题的地方，你必须在标题前方增加一个换行符\n
示例："This is a paragraph.## title" 这个文本你需要翻译为“这里是一个段落。\n## 标题”
- 如果是参考文献段落，你不需要翻译为中文，直接展示原文即可

现在，请输出翻译：
"""
        ).strip()

        # 初始化异步翻译器
        translator = AsyncPDFTranslator(model_name=DEFAULT_MODEL, max_workers=8)

        q: Queue[str] = Queue(maxsize=64)

        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def run_stream():
                try:
                    # 聚合最终翻译内容
                    aggregated_contents = []
                    async for index, translation, total in translator.get_translation_ordered_stream(PDF_TRANSLATION_PROMPT_TEMPLATE, filepath):
                        # 检查是否取消
                        if session_manager.is_session_cancelled(session_id):
                            q.put(json.dumps({"cancelled": True, "message": "翻译已被用户中断"}))
                            break
                        # 推送逐段结果
                        q.put(json.dumps({
                            "index": index,
                            "content": translation,
                            "total": total
                        }))
                        try:
                            aggregated_contents.append(translation)
                        except Exception:
                            pass
                    # 正常完成
                    if not session_manager.is_session_cancelled(session_id):
                        # 保存聚合内容到 uploads 目录
                        try:
                            upload_dir = current_app.config.get('UPLOAD_FOLDER')
                            os.makedirs(upload_dir, exist_ok=True)
                            base_name = os.path.splitext(os.path.basename(filepath))[0]
                            out_path = os.path.join(upload_dir, f"{base_name}_stream_translation.md")
                            with open(out_path, 'w', encoding='utf-8') as f:
                                f.write('\n\n'.join(aggregated_contents))
                            logger.info(f"[translate-pdf] session={session_id} saved file={out_path}")
                        except Exception as save_err:
                            logger.error(f"[translate-pdf] save error session={session_id} err={str(save_err)}")
                        q.put(json.dumps({"done": True}))
                except Exception as e:
                    logger.error(f"[translate-pdf] session={session_id} error={str(e)}")
                    q.put(json.dumps({"error": str(e)}))
                finally:
                    session_manager.finish_session(session_id)
                    try:
                        q.put_nowait("__END__")
                    except Exception:
                        pass

            loop.run_until_complete(run_stream())
            loop.close()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def sse_stream():
            try:
                # 先发送会话ID
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"
                while True:
                    item = q.get()
                    if item == "__END__":
                        break
                    yield f"data: {item}\n\n"
            except GeneratorExit:
                logger.info(f"[translate-pdf] client disconnected session={session_id}")
                session_manager.cancel_session(session_id)
            except Exception as e:
                logger.error(f"[translate-pdf] stream error session={session_id} error={str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return current_app.response_class(
            sse_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': 'http://localhost:3000',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            }
        )

    except Exception as e:
        logger.error(f"[translate-pdf] API错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/translate-paper', methods=['POST'])
def translate_paper():
    """翻译片段"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        # 创建LLM管理器
        llm = LLMManager(DEFAULT_MODEL)
        
        # 使用论文翻译提示词
        prompt = PAPER_TRANSLATION_PROMPT_TEMPLATE.format(text=text)
        
        # 创建翻译会话
        session_id = session_manager.create_session()
        
        def generate():
            try:
                # 先发送会话ID
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"
                
                for chunk in llm.generate_stream(prompt):
                    # 检查会话是否被取消
                    if session_manager.is_session_cancelled(session_id):
                        yield f"data: {json.dumps({'cancelled': True, 'message': '翻译已被用户中断'})}\n\n"
                        break
                    
                    if chunk:
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                
                # 如果没有被取消，发送完成信号
                if not session_manager.is_session_cancelled(session_id):
                    yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                logger.error(f"论文翻译过程中出错: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                # 完成会话
                session_manager.finish_session(session_id)
        
        return current_app.response_class(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': 'http://localhost:3000',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            }
        )
        
    except Exception as e:
        logger.error(f"论文翻译API错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/cancel-translation', methods=['POST'])
def cancel_translation():
    """取消翻译任务"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "缺少会话ID"}), 400
        
        success = session_manager.cancel_session(session_id)
        
        if success:
            return jsonify({"success": True, "message": "翻译任务已取消"})
        else:
            return jsonify({"success": False, "message": "会话不存在或已完成"}), 404
            
    except Exception as e:
        logger.error(f"取消翻译API错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "healthy"})