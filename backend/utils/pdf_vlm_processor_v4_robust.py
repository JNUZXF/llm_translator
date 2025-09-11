# type: ignore

"""
PDF VLM处理器 - 鲁棒高速版本 V4
使用异步批量处理+强大的重试机制，确保识别完整性和高速度
路径：agent/utils/pdf_vlm_processor_v4_robust.py
"""

import os
import sys
import time
import threading
import queue
import pickle
import base64
import asyncio
import aiohttp
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from pdf2image import convert_from_path
from PIL import Image
import multiprocessing as mp
from functools import partial
import logging
from dataclasses import dataclass
import concurrent.futures

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """处理结果数据类"""
    page_num: int
    content: str
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    retry_count: int = 0

@dataclass
class VLMTask:
    """VLM处理任务"""
    page_num: int
    image_path: str
    api_key: str
    model: str
    retry_count: int = 0

@dataclass
class BatchVLMTask:
    """批量VLM处理任务"""
    tasks: List[VLMTask]
    batch_id: int

def _convert_page_range_worker(args: Tuple[str, int, int, str, str, int]) -> List[Tuple[int, str]]:
    """
    多进程工作函数：转换指定页码范围的PDF页面
    """
    pdf_path, start_page, end_page, output_dir, file_prefix, dpi = args
    
    try:
        start_time = time.time()
        
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=start_page,
            last_page=end_page,
            fmt='PNG'
        )
        
        results = []
        for i, image in enumerate(images):
            page_num = start_page + i
            output_path = os.path.join(output_dir, f"{file_prefix}_page_{page_num:04d}.png")
            
            # 优化图片保存
            image.save(output_path, 'PNG', optimize=True)
            results.append((page_num, output_path))
            
        end_time = time.time()
        logger.info(f"进程 {mp.current_process().name} 转换页面 {start_page}-{end_page} 完成，耗时 {end_time - start_time:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"转换页面 {start_page}-{end_page} 失败: {str(e)}")
        return []

async def _async_vlm_single_request_with_retry(session: aiohttp.ClientSession, task: VLMTask, max_retries: int = 3) -> ProcessingResult:
    """
    异步VLM单个请求（带重试机制）
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            
            # 获取图片的Base64编码
            with open(task.image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                image_data = f"data:image/png;base64,{image_base64}"
            
            # 构建请求数据
            question = """
请阅读我上传的pdf文件，使用markdown格式返回所有的信息。
如果有图片，需要你用一个markdown标题+文字描述，标题为图片的标题，文字描述需要详细全面地介绍这张图片的内容。

# 注意：
- 你的输出必须与原文的语种一致。如果我提供的图片是英文，你的输出必须是英文；如果我提供的图片是中文，你的输出必须是中文。即：你的输出语种要与我的图片语种一致，不需要做翻译。

开始输出：
"""
            
            payload = {
                "model": task.model,
                "messages": [
                    {"role": "system", "content": "你必须精准提取PDF图片的内容。"}, 
                    {"role": "user", "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]}
                ],
                "temperature": 1,
                "top_p": 0.7,
                "max_tokens": 16384,
                "thinking": {"type": "disabled"},
                "stream": False  # 使用非流式以提高并发性能
            }
            
            headers = {
                "Authorization": f"Bearer {task.api_key}",
                "Content-Type": "application/json"
            }
            
            # 根据重试次数增加超时时间
            base_timeout = 90  # 基础超时时间
            timeout_multiplier = 1 + (attempt * 0.5)  # 每次重试增加50%超时时间
            timeout_seconds = base_timeout * timeout_multiplier
            
            # 发送异步请求
            async with session.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds, connect=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    logger.info(f"页面 {task.page_num} VLM识别成功 (尝试{attempt+1}/{max_retries+1})，耗时 {processing_time:.2f}s")
                    
                    return ProcessingResult(
                        page_num=task.page_num,
                        content=content,
                        success=True,
                        processing_time=processing_time,
                        retry_count=attempt
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"API请求失败，状态码: {response.status}, 响应: {error_text}")
                    
        except Exception as e:
            last_error = e
            end_time = time.time()
            
            if attempt < max_retries:
                # 指数退避：等待时间随重试次数增加
                wait_time = min(2 ** attempt + random.uniform(0, 1), 30)  # 最多等待30秒
                logger.warning(f"页面 {task.page_num} 第{attempt+1}次尝试失败: {str(e)}, {wait_time:.1f}秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"页面 {task.page_num} 所有重试失败: {str(e)}")
    
    # 所有重试都失败了
    return ProcessingResult(
        page_num=task.page_num,
        content="",
        success=False,
        error=str(last_error),
        retry_count=max_retries
    )

async def _async_batch_vlm_worker_robust(batch_task: BatchVLMTask, max_retries: int = 3) -> List[ProcessingResult]:
    """
    鲁棒的异步批量VLM处理工作函数
    """
    try:
        # 优化连接池配置，减少连接问题
        connector = aiohttp.TCPConnector(
            limit=30,  # 降低总连接池大小
            limit_per_host=10,  # 降低每个主机的最大连接数
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        # 增加超时时间，给API更多响应时间
        timeout = aiohttp.ClientTimeout(total=180, connect=30, sock_read=120)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:
            # 创建所有异步任务（带重试）
            tasks = [_async_vlm_single_request_with_retry(session, task, max_retries) 
                    for task in batch_task.tasks]
            
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"批次 {batch_task.batch_id} 任务 {i} 异常: {str(result)}")
                    processed_results.append(ProcessingResult(
                        page_num=batch_task.tasks[i].page_num,
                        content="",
                        success=False,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            success_count = sum(1 for r in processed_results if r.success)
            logger.info(f"批次 {batch_task.batch_id} 完成，成功: {success_count}/{len(processed_results)}")
            return processed_results
            
    except Exception as e:
        logger.error(f"批次 {batch_task.batch_id} 处理失败: {str(e)}")
        return [ProcessingResult(
            page_num=task.page_num,
            content="",
            success=False,
            error=str(e)
        ) for task in batch_task.tasks]

def _batch_vlm_worker_robust(batch_task: BatchVLMTask) -> List[ProcessingResult]:
    """
    鲁棒的批量VLM处理工作函数（多进程入口）
    """
    # 在新进程中创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(_async_batch_vlm_worker_robust(batch_task))
    finally:
        loop.close()

class PDFVLMProcessorV4Robust:
    """PDF VLM处理器 - 鲁棒高速版本 V4"""
    
    def __init__(self, 
                 pdf_workers: Optional[int] = None,
                 vlm_batch_workers: Optional[int] = None,
                 batch_size: int = 5,  # 降低默认批次大小
                 concurrent_per_batch: int = 8,  # 降低默认并发数
                 dpi: int = 200,
                 model: str = "doubao-seed-1-6-flash-250615",
                 max_retries: int = 3,
                 enable_failed_retry: bool = True):
        """
        初始化PDF VLM处理器（鲁棒高速版本）
        
        Args:
            pdf_workers: PDF转图片的进程数
            vlm_batch_workers: VLM批量处理的进程数
            batch_size: 每个批次的图片数量
            concurrent_per_batch: 每个批次内的并发请求数
            dpi: 图片分辨率
            model: VLM模型名称
            max_retries: 最大重试次数
            enable_failed_retry: 是否启用失败任务重试
        """
        self.dpi = dpi
        self.model = model
        self.batch_size = batch_size
        self.concurrent_per_batch = concurrent_per_batch
        self.max_retries = max_retries
        self.enable_failed_retry = enable_failed_retry
        
        # 优化进程数配置（更保守的默认值）
        cpu_count = mp.cpu_count()
        self.pdf_workers = pdf_workers or min(cpu_count, 4)  # PDF转图片进程数
        self.vlm_batch_workers = vlm_batch_workers or min(cpu_count // 2, 4)  # VLM批量处理进程数
        
        # 统计信息
        self.stats = {
            'total_pages': 0,
            'images_converted': 0,
            'vlm_processed': 0,
            'vlm_success': 0,
            'vlm_failed': 0,
            'vlm_retried': 0,
            'start_time': None,
            'image_conversion_time': 0,
            'vlm_processing_time': 0,
            'total_vlm_time': 0,
            'total_api_calls': 0,
            'concurrent_calls': 0,
            'failed_pages': []
        }
        
        # 线程锁
        self.print_lock = threading.Lock()
        
        logger.info(f"初始化鲁棒处理器V4 - PDF进程数: {self.pdf_workers}, VLM批量进程数: {self.vlm_batch_workers}")
        logger.info(f"批次大小: {self.batch_size}, 每批次并发数: {self.concurrent_per_batch}, 最大重试: {self.max_retries}")
        
    def _get_pdf_page_count(self, pdf_path: str) -> int:
        """获取PDF总页数"""
        try:
            import fitz  # PyMuPDF
            try:
                doc = fitz.open(pdf_path)
                page_count = doc.page_count
                doc.close()
                return page_count
            except:
                return self._get_page_count_fallback(pdf_path)
                
        except Exception as e:
            logger.error(f"获取PDF页数失败: {str(e)}")
            return 0
    
    def _get_page_count_fallback(self, pdf_path: str) -> int:
        """备用方法获取PDF页数"""
        try:
            import subprocess
            result = subprocess.run(['pdfinfo', pdf_path], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    return int(line.split(':')[1].strip())
            return 0
        except:
            return self._binary_search_page_count(pdf_path)
    
    def _binary_search_page_count(self, pdf_path: str) -> int:
        """使用二分查找确定PDF页数"""
        left, right = 1, 1000
        while left < right:
            mid = (left + right + 1) // 2
            try:
                convert_from_path(pdf_path, dpi=72, first_page=mid, last_page=mid)
                left = mid
            except:
                right = mid - 1
        return left
    
    def convert_pdf_to_images_multiprocess(self, pdf_path: str, output_dir: str, 
                                         file_prefix: Optional[str] = None,
                                         pdf2image_batch_size: int = 5
                                         ) -> List[Tuple[int, str]]:
        """
        使用多进程转换PDF到图片
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取PDF总页数
        logger.info("正在获取PDF页数...")
        total_pages = self._get_pdf_page_count(pdf_path)
        if total_pages == 0:
            raise ValueError("无法获取PDF页数或PDF文件无效")
        
        self.stats['total_pages'] = total_pages
        logger.info(f"PDF总共有 {total_pages} 页")
        
        # 设置文件前缀
        if file_prefix is None:
            file_prefix = Path(pdf_path).stem
        
        # 创建页面批次（PDF转换用较小的批次）
        batches = []
        for start in range(1, total_pages + 1, pdf2image_batch_size):
            end = min(start + pdf2image_batch_size - 1, total_pages)
            batches.append((pdf_path, start, end, output_dir, file_prefix, self.dpi))
        
        logger.info(f"使用 {self.pdf_workers} 个进程分 {len(batches)} 个批次处理PDF转图片")
        
        start_time = time.time()
        all_results = []
        
        # 使用进程池处理PDF转图片
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.pdf_workers) as executor:
            future_to_batch = {executor.submit(_convert_page_range_worker, batch): batch 
                              for batch in batches}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    result = future.result()
                    all_results.extend(result)
                    
                    with self.print_lock:
                        self.stats['images_converted'] += len(result)
                        converted = self.stats['images_converted']
                        logger.info(f"PDF转图片进度: {converted}/{total_pages} 页")
                        
                except Exception as e:
                    batch = future_to_batch[future]
                    logger.error(f"批次 {batch[1]}-{batch[2]} 处理失败: {str(e)}")
        
        end_time = time.time()
        self.stats['image_conversion_time'] = end_time - start_time
        
        # 按页码排序
        all_results.sort(key=lambda x: x[0])
        
        logger.info(f"PDF转图片完成，共生成 {len(all_results)} 个图片，耗时 {end_time - start_time:.2f}s")
        return all_results
    
    def process_images_with_vlm_robust(self, image_paths: List[Tuple[int, str]]) -> Dict[int, str]:
        """
        使用鲁棒的异步批量处理进行VLM识别
        """
        api_key = os.environ.get("DOUBAO_API_KEY")
        if not api_key:
            raise ValueError("请设置 DOUBAO_API_KEY 环境变量")
        
        logger.info(f"开始鲁棒批量VLM处理，{self.vlm_batch_workers} 个进程，每批次 {self.batch_size} 张图片...")
        
        # 创建VLM任务列表
        vlm_tasks = []
        for page_num, image_path in image_paths:
            task = VLMTask(
                page_num=page_num,
                image_path=image_path,
                api_key=api_key,
                model=self.model
            )
            vlm_tasks.append(task)
        
        # 将任务分组为批次
        batch_tasks = []
        for i in range(0, len(vlm_tasks), self.batch_size):
            batch = vlm_tasks[i:i + self.batch_size]
            batch_task = BatchVLMTask(
                tasks=batch,
                batch_id=i // self.batch_size + 1
            )
            batch_tasks.append(batch_task)
        
        logger.info(f"创建了 {len(batch_tasks)} 个批次，每批次最多 {self.batch_size} 张图片")
        
        start_time = time.time()
        results = {}
        total_vlm_time = 0
        total_api_calls = 0
        failed_pages = []
        
        # 使用进程池处理VLM批次
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.vlm_batch_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {executor.submit(_batch_vlm_worker_robust, batch_task): batch_task 
                             for batch_task in batch_tasks}
            
            # 收集结果
            completed_batches = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    batch_task = future_to_batch[future]
                    
                    with self.print_lock:
                        for result in batch_results:
                            self.stats['vlm_processed'] += 1
                            total_api_calls += 1
                            
                            if result.success:
                                self.stats['vlm_success'] += 1
                                results[result.page_num] = result.content
                                total_vlm_time += result.processing_time
                                if result.retry_count > 0:
                                    self.stats['vlm_retried'] += 1
                            else:
                                self.stats['vlm_failed'] += 1
                                failed_pages.append(result.page_num)
                        
                        completed_batches += 1
                        processed = self.stats['vlm_processed']
                        total = len(vlm_tasks)
                        success = self.stats['vlm_success']
                        failed = self.stats['vlm_failed']
                        
                        logger.info(f"VLM批次进度: {completed_batches}/{len(batch_tasks)} "
                                  f"(总进度: {processed}/{total}, 成功: {success}, 失败: {failed})")
                        
                except Exception as e:
                    batch_task = future_to_batch[future]
                    logger.error(f"VLM批次异常 - 批次 {batch_task.batch_id}: {str(e)}")
        
        # 处理失败的页面
        if failed_pages and self.enable_failed_retry:
            logger.info(f"开始重新处理 {len(failed_pages)} 个失败的页面...")
            retry_results = self._retry_failed_pages(failed_pages, image_paths, api_key)
            results.update(retry_results)
        
        end_time = time.time()
        self.stats['vlm_processing_time'] = end_time - start_time
        self.stats['total_vlm_time'] = total_vlm_time
        self.stats['total_api_calls'] = total_api_calls
        self.stats['concurrent_calls'] = self.vlm_batch_workers * self.batch_size
        self.stats['failed_pages'] = failed_pages
        
        logger.info(f"鲁棒VLM识别完成，成功识别 {len(results)} 页，总耗时 {end_time - start_time:.2f}s")
        logger.info(f"成功率: {len(results)}/{len(vlm_tasks)} ({len(results)/len(vlm_tasks)*100:.1f}%)")
        return results
    
    def _retry_failed_pages(self, failed_pages: List[int], image_paths: List[Tuple[int, str]], api_key: str) -> Dict[int, str]:
        """重新处理失败的页面（单独处理，降低并发）"""
        retry_results = {}
        
        # 找到失败页面对应的图片路径
        failed_image_paths = [(page_num, image_path) for page_num, image_path in image_paths 
                             if page_num in failed_pages]
        
        if not failed_image_paths:
            return retry_results
        
        logger.info(f"重试失败页面: {failed_pages}")
        
        # 创建重试任务（使用更保守的配置）
        retry_vlm_tasks = []
        for page_num, image_path in failed_image_paths:
            task = VLMTask(
                page_num=page_num,
                image_path=image_path,
                api_key=api_key,
                model=self.model
            )
            retry_vlm_tasks.append(task)
        
        # 单独处理每个失败的页面（串行处理，避免再次失败）
        for task in retry_vlm_tasks:
            try:
                # 使用同步方式重试单个任务
                batch_task = BatchVLMTask(tasks=[task], batch_id=0)
                batch_results = _batch_vlm_worker_robust(batch_task)
                
                if batch_results and batch_results[0].success:
                    retry_results[task.page_num] = batch_results[0].content
                    logger.info(f"重试成功: 页面 {task.page_num}")
                else:
                    logger.warning(f"重试失败: 页面 {task.page_num}")
                    
                # 重试间隔
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"重试页面 {task.page_num} 异常: {str(e)}")
        
        logger.info(f"重试完成，额外成功 {len(retry_results)} 页")
        return retry_results
    
    def process_pdf_to_markdown(self, pdf_path: str, output_md_path: str, 
                              temp_image_dir: Optional[str] = None,
                              cleanup_images: bool = True,
                              pdf2image_batch_size: int = 5) -> str:
        """
        完整的PDF到Markdown处理流程（鲁棒高速版本）
        """
        start_time = time.time()
        self.stats['start_time'] = start_time
        
        # 设置临时图片目录
        if temp_image_dir is None:
            temp_image_dir = os.path.join(os.path.dirname(output_md_path), "temp_images")
        
        try:
            # 第一步：多进程PDF转图片
            logger.info("=" * 60)
            logger.info("第一步：多进程PDF转图片")
            logger.info("=" * 60)
            
            image_paths = self.convert_pdf_to_images_multiprocess(pdf_path, temp_image_dir, pdf2image_batch_size)
            
            if not image_paths:
                raise ValueError("PDF转图片失败")
            
            # 第二步：鲁棒异步批量VLM识别
            logger.info("=" * 60)
            logger.info("第二步：鲁棒异步批量VLM识别")
            logger.info("=" * 60)
            
            page_texts = self.process_images_with_vlm_robust(image_paths)
            
            if not page_texts:
                raise ValueError("VLM识别完全失败")
            
            # 第三步：合并文本并保存
            logger.info("=" * 60)
            logger.info("第三步：合并文本并保存")
            logger.info("=" * 60)
            
            # 按页码排序并合并文本
            sorted_pages = sorted(page_texts.keys())
            combined_text = []
            
            # 添加缺失页面的标记
            all_pages = set(range(1, self.stats['total_pages'] + 1))
            processed_pages = set(page_texts.keys())
            missing_pages = all_pages - processed_pages
            
            if missing_pages:
                logger.warning(f"缺失页面: {sorted(missing_pages)}")
            
            for page_num in sorted(all_pages):
                if page_num in page_texts:
                    text = page_texts[page_num]
                    if text.strip():
                        combined_text.append(f"# 第 {page_num} 页\n\n{text}\n\n")
                else:
                    combined_text.append(f"# 第 {page_num} 页\n\n**[识别失败 - 页面内容缺失]**\n\n")
            
            final_markdown = "\n".join(combined_text)
            
            # 保存到文件
            os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(final_markdown)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 输出详细统计信息
            self._print_performance_stats(total_time, len(page_texts), final_markdown, output_md_path)
            
            return final_markdown
            
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            raise
            
        finally:
            # 清理临时图片
            if cleanup_images and temp_image_dir and os.path.exists(temp_image_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_image_dir)
                    logger.info(f"已清理临时图片目录: {temp_image_dir}")
                except Exception as e:
                    logger.warning(f"清理临时图片失败: {str(e)}")
    
    def _print_performance_stats(self, total_time: float, processed_pages: int, 
                               final_markdown: str, output_path: str):
        """输出性能统计信息"""
        logger.info("=" * 60)
        logger.info("处理完成 - 性能统计（鲁棒高速版本 V4）")
        logger.info("=" * 60)
        logger.info(f"总页数: {self.stats['total_pages']}")
        logger.info(f"成功转换图片: {self.stats['images_converted']}")
        logger.info(f"VLM处理页数: {self.stats['vlm_processed']}")
        logger.info(f"VLM成功页数: {self.stats['vlm_success']}")
        logger.info(f"VLM失败页数: {self.stats['vlm_failed']}")
        logger.info(f"VLM重试页数: {self.stats['vlm_retried']}")
        logger.info(f"最终成功率: {processed_pages}/{self.stats['total_pages']} ({processed_pages/self.stats['total_pages']*100:.1f}%)")
        logger.info(f"总API调用数: {self.stats['total_api_calls']}")
        logger.info(f"理论并发数: {self.stats['concurrent_calls']}")
        logger.info(f"PDF转图片耗时: {self.stats['image_conversion_time']:.2f}s")
        logger.info(f"VLM识别总耗时: {self.stats['vlm_processing_time']:.2f}s")
        logger.info(f"VLM任务累计耗时: {self.stats['total_vlm_time']:.2f}s")
        logger.info(f"总耗时: {total_time:.2f}s")
        logger.info(f"平均每页耗时: {total_time/self.stats['total_pages']:.2f}s")
        
        # 计算VLM并行效率
        if self.stats['vlm_processing_time'] > 0:
            vlm_efficiency = self.stats['total_vlm_time'] / self.stats['vlm_processing_time']
            logger.info(f"VLM并行效率: {vlm_efficiency:.2f}x")
        
        # 计算API调用效率
        if total_time > 0:
            api_rate = self.stats['total_api_calls'] / total_time
            logger.info(f"API调用速率: {api_rate:.2f} 次/秒")
        
        logger.info(f"输出文件: {output_path}")
        logger.info(f"文件大小: {len(final_markdown)} 字符")
        
        if self.stats['failed_pages']:
            logger.warning(f"失败页面列表: {self.stats['failed_pages'][:20]}...")  # 只显示前20个

# 便捷函数
def convert_pdf_to_markdown_v4_robust(pdf_path: str, output_md_path: str, 
                                     dpi: int = 200, 
                                     batch_size: int = 5,
                                     concurrent_per_batch: int = 8,
                                     pdf_workers: Optional[int] = None,
                                     vlm_batch_workers: Optional[int] = None,
                                     model: str = "doubao-seed-1-6-flash-250615",
                                     max_retries: int = 3,
                                     cleanup_images: bool = True,
                                     pdf2image_batch_size: int = 5) -> str:
    """
    便捷的PDF到Markdown转换函数（鲁棒高速版本）
    
    Args:
        pdf_path: PDF文件路径
        output_md_path: 输出Markdown文件路径
        dpi: 图片分辨率
        batch_size: 每个批次的图片数量
        concurrent_per_batch: 每个批次内的并发请求数
        pdf_workers: PDF转图片的进程数
        vlm_batch_workers: VLM批量处理的进程数
        model: VLM模型名称
        max_retries: 最大重试次数
        cleanup_images: 是否清理临时图片
        
    Returns:
        处理后的Markdown文本
    """
    processor = PDFVLMProcessorV4Robust(
        pdf_workers=pdf_workers,
        vlm_batch_workers=vlm_batch_workers,
        batch_size=batch_size,
        concurrent_per_batch=concurrent_per_batch,
        dpi=dpi,
        model=model,
        max_retries=max_retries
    )
    
    return processor.process_pdf_to_markdown(
        pdf_path=pdf_path,
        output_md_path=output_md_path,
        cleanup_images=cleanup_images,
        pdf2image_batch_size=pdf2image_batch_size
    )

# 示例使用代码
if __name__ == "__main__":
    # 检查环境变量
    if not os.environ.get("DOUBAO_API_KEY"):
        print("错误：请设置 DOUBAO_API_KEY 环境变量")
        sys.exit(1)
    
    # 使用示例
    pdf_file = r"D:\AgentBuilding\FinAgent\files\arxiv_papers\2506.19676v3.pdf"
    output_md_file = r"D:\AgentBuilding\FinAgent\files\arxiv_papers\2506.19676v3_robust.md"
    
    try:
        logger.info("开始使用鲁棒高速版本处理PDF...")
        
        markdown_content = convert_pdf_to_markdown_v4_robust(
            pdf_path=pdf_file,
            output_md_path=output_md_file,
            dpi=200,
            batch_size=5,           # 较保守的批次大小
            concurrent_per_batch=8, # 较保守的并发数
            pdf_workers=4,          # PDF转图片进程数
            vlm_batch_workers=3,    # VLM批量处理进程数
            max_retries=3,          # 最大重试3次
            pdf2image_batch_size=5, # PDF转图片批次大小
            cleanup_images=True
        )
        
        logger.info("鲁棒高速版本处理完成！")
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        sys.exit(1) 