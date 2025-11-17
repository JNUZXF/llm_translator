"""
改进的路由模块示例
展示如何使用新的工具类来提升代码质量

使用方法：
1. 逐步迁移现有的routes.py到这个文件
2. 或者参考这个文件来改进现有路由
"""

from flask import Blueprint, request, jsonify, current_app
import logging
from typing import Generator

# 导入新的工具模块
from utils.validators import InputValidator
from utils.file_validator import SecureFileValidator
from utils.sse_helper import SSEGenerator
from utils.exceptions import ValidationError, FileUploadError, LLMAPIError
from utils.logger import log_execution_time

# 导入原有模块
from tools_agent.llm_manager import LLMManager
from config.constants import (
    TRANSLATION_PROMPT_TEMPLATE,
    PAPER_TRANSLATION_PROMPT_TEMPLATE,
    SUPPORTED_LANGUAGES,
    TRANSLATION_SCENES,
    DEFAULT_MODEL
)
from utils.pdf_processor import PDFProcessor
from utils.session_manager import session_manager
from utils.agent_tool_pdf_translation import AsyncPDFTranslator

# 创建蓝图
improved_bp = Blueprint('improved', __name__)
logger = logging.getLogger(__name__)

# 创建SSE生成器实例
sse_helper = SSEGenerator(session_manager)


@improved_bp.route('/api/languages', methods=['GET'])
def get_languages():
    """获取支持的语言列表"""
    return jsonify(SUPPORTED_LANGUAGES)


@improved_bp.route('/api/scenes', methods=['GET'])
def get_scenes():
    """获取支持的翻译场景列表"""
    return jsonify(TRANSLATION_SCENES)


@improved_bp.route('/api/translate', methods=['POST'])
@log_execution_time()
def translate_text():
    """
    快速翻译文本
    使用新的验证器和SSE生成器
    """
    try:
        data = request.json
        if not data:
            raise ValidationError("请求数据不能为空")

        # ===== 使用验证器进行输入验证 =====
        text = InputValidator.sanitize_text(
            data.get('text', ''),
            allow_empty=False
        )

        target_language = InputValidator.validate_language(
            data.get('language', 'English')
        )

        scene_id = InputValidator.validate_scene(
            data.get('scene', 'general')
        )

        other_requirements = InputValidator.sanitize_text(
            data.get('requirements', ''),
            max_length=InputValidator.MAX_REQUIREMENTS_LENGTH,
            allow_empty=True
        )

        logger.info(f'翻译请求 - 语言: {target_language}, 场景: {scene_id}, 文本长度: {len(text)}')

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
        try:
            llm = LLMManager(DEFAULT_MODEL)
        except Exception as e:
            logger.error(f'LLM初始化失败: {str(e)}')
            raise LLMAPIError(f'翻译服务初始化失败: {str(e)}', provider=DEFAULT_MODEL)

        # 构建提示词
        prompt = TRANSLATION_PROMPT_TEMPLATE.format(
            language=target_language,
            scene=scene_description,
            text=text,
            other_requirements=requirements_text
        )

        # 创建翻译会话
        session_id = session_manager.create_session()
        logger.info(f'创建翻译会话: {session_id}')

        # ===== 使用SSE生成器 =====
        llm_stream = llm.generate_stream(prompt)
        sse_stream = sse_helper.generate(llm_stream, session_id)

        # 返回SSE响应
        return SSEGenerator.create_response(
            sse_stream,
            cors_origin=request.headers.get('Origin', '*')
        )

    except ValidationError as e:
        # 验证错误会被全局错误处理器处理
        raise
    except LLMAPIError as e:
        raise
    except Exception as e:
        logger.error(f'翻译API错误: {str(e)}', exc_info=True)
        raise


@improved_bp.route('/api/upload', methods=['POST'])
@log_execution_time()
def upload_file():
    """
    上传PDF文件
    使用新的文件验证器
    """
    try:
        # 检查文件字段
        if 'file' not in request.files:
            raise FileUploadError("没有选择文件")

        file = request.files['file']

        if not file or not file.filename:
            raise FileUploadError("文件为空或文件名无效")

        logger.info(f'收到文件上传请求: {file.filename}')

        # ===== 使用安全文件验证器 =====
        upload_dir = current_app.config['UPLOAD_FOLDER']

        # 验证并保存文件
        filepath, original_filename, file_hash = SecureFileValidator.save_validated_file(
            file,
            upload_dir
        )

        logger.info(f'文件保存成功: {filepath}, 哈希: {file_hash}')

        # 处理PDF文件
        try:
            pdf_processor = PDFProcessor()
            pdf_info = pdf_processor.get_pdf_info(filepath)
            pages = pdf_processor.extract_text_from_pdf(filepath)

            logger.info(f'PDF处理完成 - 页数: {pdf_info["page_count"]}')

            return jsonify({
                "success": True,
                "filename": original_filename,
                "filepath": filepath,
                "file_hash": file_hash,
                "info": pdf_info,
                "pages": pages
            })

        except Exception as e:
            logger.error(f'PDF处理失败: {str(e)}', exc_info=True)
            # 删除已上传的文件
            import os
            if os.path.exists(filepath):
                os.remove(filepath)
            raise FileUploadError(f'PDF处理失败: {str(e)}')

    except FileUploadError:
        raise
    except Exception as e:
        logger.error(f'文件上传错误: {str(e)}', exc_info=True)
        raise


@improved_bp.route('/api/translate-paper', methods=['POST'])
def translate_paper():
    """
    翻译论文片段
    使用新的验证器和SSE生成器
    """
    try:
        data = request.json
        if not data:
            raise ValidationError("请求数据不能为空")

        # 验证输入
        text = InputValidator.sanitize_text(
            data.get('text', ''),
            allow_empty=False
        )

        logger.info(f'论文翻译请求 - 文本长度: {len(text)}')

        # 创建LLM管理器
        llm = LLMManager(DEFAULT_MODEL)

        # 使用论文翻译提示词
        prompt = PAPER_TRANSLATION_PROMPT_TEMPLATE.format(text=text)

        # 创建翻译会话
        session_id = session_manager.create_session()

        # 使用SSE生成器
        llm_stream = llm.generate_stream(prompt)
        sse_stream = sse_helper.generate(llm_stream, session_id)

        return SSEGenerator.create_response(
            sse_stream,
            cors_origin=request.headers.get('Origin', '*')
        )

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f'论文翻译API错误: {str(e)}', exc_info=True)
        raise


@improved_bp.route('/api/cancel-translation', methods=['POST'])
def cancel_translation():
    """取消翻译任务"""
    try:
        data = request.json
        if not data:
            raise ValidationError("请求数据不能为空")

        # 验证会话ID
        session_id = InputValidator.validate_session_id(
            data.get('session_id', '')
        )

        success = session_manager.cancel_session(session_id)

        if success:
            logger.info(f'翻译任务已取消: {session_id}')
            return jsonify({
                "success": True,
                "message": "翻译任务已取消"
            })
        else:
            return jsonify({
                "success": False,
                "message": "会话不存在或已完成"
            }), 404

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f'取消翻译API错误: {str(e)}', exc_info=True)
        raise


@improved_bp.route('/api/health', methods=['GET'])
def health_check():
    """
    增强的健康检查接口
    """
    import psutil
    from datetime import datetime

    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {},
        "metrics": {}
    }

    # 检查文件系统
    try:
        upload_dir = current_app.config['UPLOAD_FOLDER']
        import os
        if os.path.exists(upload_dir) and os.access(upload_dir, os.W_OK):
            health_status["checks"]["filesystem"] = "ok"
        else:
            health_status["checks"]["filesystem"] = "error"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["filesystem"] = "error"
        health_status["status"] = "degraded"
        logger.error(f'文件系统检查失败: {str(e)}')

    # 检查会话管理器
    try:
        active_sessions = session_manager.get_active_sessions()
        total_sessions = session_manager.get_session_count()
        health_status["checks"]["session_manager"] = "ok"
        health_status["metrics"]["active_sessions"] = len(active_sessions)
        health_status["metrics"]["total_sessions"] = total_sessions
    except Exception as e:
        health_status["checks"]["session_manager"] = "error"
        health_status["status"] = "degraded"
        logger.error(f'会话管理器检查失败: {str(e)}')

    # 系统资源检查（可选）
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        health_status["metrics"]["memory_percent"] = memory.percent
        health_status["metrics"]["disk_percent"] = disk.percent

        # 警告阈值
        if memory.percent > 90 or disk.percent > 90:
            health_status["status"] = "degraded"
    except:
        # psutil可能未安装，忽略
        pass

    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code


# ===== 示例：使用装饰器简化代码 =====

from utils.validators import validate_json_request

@improved_bp.route('/api/example-with-decorator', methods=['POST'])
@validate_json_request('text', 'language', requirements=str)  # 声明必需和可选字段
@log_execution_time()
def example_endpoint():
    """
    使用装饰器的示例端点
    展示如何使用装饰器简化验证逻辑
    """
    data = request.json
    # 此时data已经过验证，可以直接使用
    text = data['text']
    language = data['language']
    requirements = data.get('requirements', '')

    # 处理逻辑...
    return jsonify({"message": "处理成功"})
