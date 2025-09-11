
"""
utils/agent_tool_pdf_translation.py
该文件实现整份PDF的OCR→Markdown提取与分段、并发翻译的能力。
路径: backend/utils/agent_tool_pdf_translation.py
"""


import os
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv, find_dotenv
import base64
from volcengine.visual.VisualService import VisualService
from utils.agent_tool_split_paras import *
from tools_agent.llm_manager import *
from typing import List, Tuple
import time
from utils.pdf_processor import PDFProcessor
import re

# 优先加载 backend/.env，保证无论工作目录如何变动都能正确读取
_logger = logging.getLogger(__name__)
try:
    _module_dir = os.path.dirname(os.path.abspath(__file__))
    _backend_dir = os.path.abspath(os.path.join(_module_dir, '..'))
    _explicit_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_explicit_env):
        load_dotenv(_explicit_env)
        _logger.info(f"[env] loaded .env from {_explicit_env}")
    else:
        _found = find_dotenv(filename='.env', usecwd=True)
        if _found:
            load_dotenv(_found)
            _logger.info(f"[env] loaded .env from {_found}")
        else:
            load_dotenv()
            _logger.warning("[env] .env not found explicitly, used default search")
except Exception as _e:
    _logger.error(f"[env] load .env failed: {_e}")

VOLC_AK = os.getenv("VOLC_AK")
VOLC_SK = os.getenv("VOLC_SK")

class AsyncPDFTranslator:
    def __init__(self, model_name: str, max_workers: int = 5):
        """
        初始化异步PDF翻译器
        :param model_name: 使用的模型名称
        :param max_workers: 最大并发线程数
        """
        self.model_name = model_name
        self.max_workers = max_workers
        
    def rag_chunking(self, md_file_path: str, target_length: int = 6000) -> List[str]:
        """
        分段处理markdown文件
        :param md_file_path: markdown文件路径
        :param target_length: 目标分段长度
        :return: 分段后的段落列表
        """
        with open(md_file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        paragraphs = smart_paragraph_split_v2(markdown_content, target_length=target_length)
        return paragraphs

    def get_pdf_markdown(self, pdf_path: str) -> str:
        """
        从PDF提取markdown内容
        :param pdf_path: PDF文件路径
        :return: markdown字符串
        """
        logger = logging.getLogger(__name__)

        # 若未配置AK/SK或OCR失败，则回退到本地文本提取
        def fallback_markdown() -> str:
            logger.warning("[get_pdf_markdown] OCR不可用，回退到本地文本提取(PDFProcessor)")
            processor = PDFProcessor()
            pages = processor.extract_text_from_pdf(pdf_path)
            parts: List[str] = []
            for page in pages:
                parts.append(f"# Page {page['page_number']}")
                parts.append(page.get('text', ''))
                parts.append("")
            md = "\n\n".join(parts).strip()
            if not md:
                return "[Empty PDF content]"
            return md

        if not VOLC_AK or not VOLC_SK:
            logger.warning("[get_pdf_markdown] 未检测到 VOLC_AK/VOLC_SK 环境变量，跳过OCR")
            return fallback_markdown()

        try:
            visual_service = VisualService()
            visual_service.set_ak(VOLC_AK)
            visual_service.set_sk(VOLC_SK)

            pdf_bytes = open(pdf_path, 'rb').read()
            pdf_b64 = base64.b64encode(pdf_bytes).decode()

            # 优先尝试使用 pdf_base64 参数（适配火山OCR PDF接口）
            form_pdf = {
                "pdf_base64": pdf_b64,
                "image_url": "",
                "version": "v3",
                "page_start": 0,
                "page_num": 50,
                "table_mode": "html",
                "filter_header": "true"
            }

            resp = visual_service.ocr_pdf(form_pdf)
            if isinstance(resp, dict) and resp.get("data") and resp["data"].get("markdown"):
                return resp["data"]["markdown"]

            # 兼容旧参数名：image_base64（若SDK/服务端要求）
            form_img = {
                "image_base64": pdf_b64,
                "image_url": "",
                "version": "v3",
                "page_start": 0,
                "page_num": 50,
                "table_mode": "html",
                "filter_header": "true"
            }

            resp2 = visual_service.ocr_pdf(form_img)
            if isinstance(resp2, dict) and resp2.get("data") and resp2["data"].get("markdown"):
                return resp2["data"]["markdown"]

            logger.error(f"[get_pdf_markdown] OCR返回无markdown字段: {resp2}")
            return fallback_markdown()

        except Exception as e:
            logger.error(f"[get_pdf_markdown] OCR异常: {str(e)}")
            return fallback_markdown()

    def translate_paragraph(self, paragraph: str, translate_prompt: str, index: int) -> Tuple[int, str]:
        """
        翻译单个段落（线程安全）
        :param paragraph: 要翻译的段落
        :param translate_prompt: 翻译提示模板
        :param index: 段落索引（用于保持顺序）
        :return: (索引, 翻译结果)
        """
        try:
            prompt = translate_prompt.format(paragraph=paragraph)
            llm = LLMManager(model=self.model_name)
            
            translation = ""
            for char in llm.generate_char_stream(prompt):
                translation += char
            
            # 保底：如果模型没有保留 Markdown 图片链接，则从原段落回填
            def _extract_image_tokens(text: str) -> List[str]:
                return re.findall(r'!\[[^\]]*\]\([^\)]+\)', text)

            def _preserve_images_in_output(source_text: str, output_text: str) -> str:
                image_tokens = _extract_image_tokens(source_text)
                if not image_tokens:
                    return output_text
                result = output_text
                for token in image_tokens:
                    m = re.search(r'\(([^\)]+)\)', token)
                    url = m.group(1) if m else None
                    already_present = (url and url in result) or (token in result)
                    if not already_present:
                        if result and not result.endswith('\n'):
                            result += '\n'
                        result += token
                return result

            translation = _preserve_images_in_output(paragraph, translation)
            
            print(f"段落 {index + 1} 翻译完成")
            return (index, translation)
            
        except Exception as e:
            print(f"段落 {index + 1} 翻译失败: {str(e)}")
            return (index, f"[翻译失败: {str(e)}]")

    async def translate_paragraphs_ordered_stream(self, paragraphs: List[str], translate_prompt: str):
        """
        按顺序流式输出翻译结果，后面的段落需等待前面的段落完成
        :param paragraphs: 段落列表
        :param translate_prompt: 翻译提示模板
        :yield: (段落索引, 翻译结果) 按顺序输出
        """
        # 使用ThreadPoolExecutor并行翻译所有段落
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有翻译任务
            future_to_index = {
                executor.submit(self.translate_paragraph, paragraph, translate_prompt, i): i 
                for i, paragraph in enumerate(paragraphs)
            }
            
            # 存储已完成的翻译结果
            completed_results = {}
            next_output_index = 0  # 下一个要输出的段落索引
            total_paragraphs = len(paragraphs)
            
            # 收集完成的结果
            for future in as_completed(future_to_index):
                try:
                    index, translation = future.result()
                    completed_results[index] = translation
                    print(f"🔄 段落 {index + 1} 翻译完成，等待输出...")
                except Exception as e:
                    index = future_to_index[future]
                    completed_results[index] = f"[翻译失败: {str(e)}]"
                    print(f"❌ 段落 {index + 1} 翻译失败: {str(e)}")
                
                # 检查是否可以按顺序输出结果
                while next_output_index in completed_results:
                    yield (next_output_index, completed_results[next_output_index])
                    next_output_index += 1
                    
                    # 如果所有段落都已输出，退出
                    if next_output_index >= total_paragraphs:
                        return

    async def translate_paragraphs_stream(self, paragraphs: List[str], translate_prompt: str):
        """
        流式异步翻译多个段落，完成一个输出一个（无序）
        :param paragraphs: 段落列表
        :param translate_prompt: 翻译提示模板
        :yield: (段落索引, 翻译结果) 按完成顺序输出
        """
        # 使用ThreadPoolExecutor来处理CPU密集型任务
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有翻译任务
            future_to_index = {
                executor.submit(self.translate_paragraph, paragraph, translate_prompt, i): i 
                for i, paragraph in enumerate(paragraphs)
            }
            
            # 按完成顺序逐个输出
            for future in as_completed(future_to_index):
                try:
                    index, translation = future.result()
                    yield (index, translation)
                except Exception as e:
                    index = future_to_index[future]
                    yield (index, f"[翻译失败: {str(e)}]")
                    print(f"段落 {index + 1} 处理异常: {str(e)}")

    async def translate_paragraphs_async(self, paragraphs: List[str], translate_prompt: str) -> List[str]:
        """
        异步翻译多个段落（保持原接口兼容性）
        :param paragraphs: 段落列表
        :param translate_prompt: 翻译提示模板
        :return: 按顺序排列的翻译结果列表
        """
        # 存储结果的字典，用于保持顺序
        results = {}
        
        # 收集流式结果
        async for index, translation in self.translate_paragraphs_stream(paragraphs, translate_prompt):
            results[index] = translation
        
        # 按索引顺序返回结果
        return [results[i] for i in sorted(results.keys())]

    async def get_translation_ordered_stream(self, translate_prompt: str, pdf_path: str):
        """
        按顺序流式获取翻译结果，后面的段落等待前面的段落输出
        :param translate_prompt: 翻译提示模板
        :param pdf_path: PDF文件路径
        :yield: (段落索引, 翻译结果, 总段落数)
        """
        print("正在提取PDF内容...")
        markdown = self.get_pdf_markdown(pdf_path)
        
        # 保存markdown
        md_file_path = pdf_path.replace(".pdf", ".md")
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Markdown已保存至: {md_file_path}")

        print("正在分段...")
        paragraphs = self.rag_chunking(md_file_path)
        total_paragraphs = len(paragraphs)
        print(f"共分为 {total_paragraphs} 个段落")

        print(f"开始并行翻译，使用 {self.max_workers} 个线程...")
        print("📝 按顺序输出翻译结果：")
        start_time = time.time()
        
        completed_count = 0
        async for index, translation in self.translate_paragraphs_ordered_stream(paragraphs, translate_prompt):
            completed_count += 1
            elapsed_time = time.time() - start_time
            print(f"✅ 段落 {index + 1}/{total_paragraphs} 已输出 (总耗时: {elapsed_time:.1f}s)")
            yield (index, translation, total_paragraphs)
            
        total_time = time.time() - start_time
        print(f"🎉 所有翻译按顺序输出完成！总耗时: {total_time:.2f} 秒")

    async def get_translation_stream(self, translate_prompt: str, pdf_path: str):
        """
        流式获取翻译结果，完成顺序输出（无序，兼容性保留）
        :param translate_prompt: 翻译提示模板
        :param pdf_path: PDF文件路径
        :yield: (段落索引, 翻译结果, 总段落数)
        """
        print("正在提取PDF内容...")
        markdown = self.get_pdf_markdown(pdf_path)
        
        # 保存markdown
        md_file_path = pdf_path.replace(".pdf", ".md")
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Markdown已保存至: {md_file_path}")

        print("正在分段...")
        paragraphs = self.rag_chunking(md_file_path)
        total_paragraphs = len(paragraphs)
        print(f"共分为 {total_paragraphs} 个段落")

        print(f"开始并行翻译，使用 {self.max_workers} 个线程...")
        start_time = time.time()
        
        completed_count = 0
        async for index, translation in self.translate_paragraphs_stream(paragraphs, translate_prompt):
            completed_count += 1
            elapsed_time = time.time() - start_time
            print(f"✅ 段落 {index + 1}/{total_paragraphs} 翻译完成 (耗时: {elapsed_time:.1f}s)")
            yield (index, translation, total_paragraphs)
            
        total_time = time.time() - start_time
        print(f"🎉 所有翻译完成！总耗时: {total_time:.2f} 秒")

    async def get_translation_async(self, translate_prompt: str, pdf_path: str) -> str:
        """
        异步获取完整翻译（保持原接口兼容性）
        :param translate_prompt: 翻译提示模板
        :param pdf_path: PDF文件路径
        :return: 完整翻译结果
        """
        results = {}
        total_paragraphs = 0
        
        async for index, translation, total in self.get_translation_stream(translate_prompt, pdf_path):
            results[index] = translation
            total_paragraphs = total
        
        # 按顺序组合结果
        return "\n\n".join([results[i] for i in range(total_paragraphs)])

    def translate_pdf(self, translate_prompt: str, pdf_path: str) -> str:
        """
        同步接口，内部使用异步实现
        :param translate_prompt: 翻译提示模板
        :param pdf_path: PDF文件路径
        :return: 完整翻译结果
        """
        # 在新的事件循环中运行异步代码
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.get_translation_async(translate_prompt, pdf_path)
            )
        finally:
            loop.close()


async def main():
    """主函数示例 - 按顺序流式输出"""
    MODEL = "doubao-seed-1-6-flash-250615"
    
    TRANSLATE_PROMPT = """
# 你的角色
具有数十年经验的高级翻译专家

# 你的任务
将我下面这段学术论文翻译为地道的中文，符合专业场景

# 论文内容
{paragraph}

# 要求
- 你的输出必须仅包含翻译后的文本和必要的 Markdown 结构
- 若原文包含图片链接（形如 ![alt](url) ），必须原样保留该图片链接（包含原始 url）
- 不要添加与内容无关的解释、前后缀或多余符号

现在，请输出翻译：
"""

    pdf_path = r"D:\AgentBuilding\LLM_translate\ai-translator\backend\files\Self-Organizing Agent Network for LLM-based Workflow Automation.pdf"
    
    # 创建异步翻译器，设置最大并发数
    translator = AsyncPDFTranslator(model_name=MODEL, max_workers=8)
    
    print("开始按顺序流式翻译...")
    
    # 按顺序流式输出（推荐）- 严格按照段落顺序显示
    final_translations = {}
    async for index, translation, total in translator.get_translation_ordered_stream(TRANSLATE_PROMPT, pdf_path):
        print(f"\n{'='*60}")
        print(f"📄 段落 {index + 1}/{total} 翻译结果：")
        print(f"{'='*60}")
        print(translation)
        print(f"{'='*60}\n")
        
        # 存储结果用于最终保存
        final_translations[index] = translation
    
    # 按顺序组合并保存最终结果
    final_translation = "\n\n".join([final_translations[i] for i in range(len(final_translations))])
    translation_path = pdf_path.replace(".pdf", "_ordered_stream_translation.md")
    with open(translation_path, "w", encoding="utf-8") as f:
        f.write(final_translation)
    
    print(f"📁 完整翻译已保存到：{translation_path}")


async def main_unordered():
    """无序流式输出示例"""
    MODEL = "doubao-seed-1-6-flash-250615"
    TRANSLATE_PROMPT = "..."  # 您的提示模板
    pdf_path = "..."  # 您的PDF路径
    
    translator = AsyncPDFTranslator(model_name=MODEL, max_workers=8)
    
    print("开始无序流式翻译...")
    
    # 无序流式输出 - 哪个先完成显示哪个
    final_translations = {}
    async for index, translation, total in translator.get_translation_stream(TRANSLATE_PROMPT, pdf_path):
        print(f"\n段落 {index + 1}/{total} 完成：")
        print(translation[:100] + "..." if len(translation) > 100 else translation)
        final_translations[index] = translation
    
    # 最终按顺序保存
    final_translation = "\n\n".join([final_translations[i] for i in sorted(final_translations.keys())])
    translation_path = pdf_path.replace(".pdf", "_unordered_stream_translation.md")
    with open(translation_path, "w", encoding="utf-8") as f:
        f.write(final_translation)
    
    print(f"翻译已保存到：{translation_path}")


async def main_simple():
    """简化版主函数 - 如果只想获得最终结果"""
    MODEL = "doubao-seed-1-6-flash-250615"
    TRANSLATE_PROMPT = "..." # 您的提示模板
    pdf_path = "..." # 您的PDF路径
    
    translator = AsyncPDFTranslator(model_name=MODEL, max_workers=8)
    
    print("开始异步翻译...")
    final_translation = await translator.get_translation_async(TRANSLATE_PROMPT, pdf_path)
    
    translation_path = pdf_path.replace(".pdf", "_async_translation.md")
    with open(translation_path, "w", encoding="utf-8") as f:
        f.write(final_translation)
    
    print(f"翻译已保存到：{translation_path}")


if __name__ == "__main__":
    # 方式1：按顺序流式输出（您需要的功能）
    asyncio.run(main())
    
    # 方式2：无序流式输出（先完成先显示）
    # asyncio.run(main_unordered())
    
    # 方式3：等待所有完成后一次性显示
    # asyncio.run(main_simple())
    
    # 方式4：使用同步接口（兼容原代码）
    """
    MODEL = "doubao-seed-1-6-flash-250615"
    TRANSLATE_PROMPT = "..." # 您的提示模板
    pdf_path = "..." # 您的PDF路径
    
    translator = AsyncPDFTranslator(model_name=MODEL, max_workers=8)
    final_translation = translator.translate_pdf(TRANSLATE_PROMPT, pdf_path)
    
    translation_path = pdf_path.replace(".pdf", "_translation.md")
    with open(translation_path, "w", encoding="utf-8") as f:
        f.write(final_translation)
    print(f"翻译保存到：{translation_path}")
    """
