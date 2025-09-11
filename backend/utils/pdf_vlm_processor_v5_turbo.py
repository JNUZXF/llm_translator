# type: ignore

"""
PDF VLM处理器 - 极速优化版本 V5
使用混合优化策略实现极致性能：激进并发+智能负载均衡+预处理缓存
路径：agent/utils/pdf_vlm_processor_v5_turbo.py
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
import hashlib
import psutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
from pdf2image import convert_from_path
from PIL import Image
import multiprocessing as mp
from functools import partial
import logging
from dataclasses import dataclass, field
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import io

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    api_response_time: float = 0.0
    queue_wait_time: float = 0.0

@dataclass
class VLMTask:
    """VLM处理任务"""
    page_num: int
    image_path: str
    image_hash: str
    api_key: str
    model: str
    priority: int = 0
    created_time: float = field(default_factory=time.time)
    retry_count: int = 0

@dataclass
class BatchMetrics:
    """批次性能指标"""
    batch_id: int
    task_count: int
    success_count: int = 0
    total_time: float = 0.0
    avg_response_time: float = 0.0
    throughput: float = 0.0

class ImageCache:
    """图片缓存管理器"""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, str] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get_image_hash(self, image_path: str) -> str:
        """计算图片哈希值"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_base64(self, image_path: str, image_hash: str) -> str:
        """获取图片的Base64编码（带缓存）"""
        with self.lock:
            if image_hash in self.cache:
                self.access_times[image_hash] = time.time()
                return self.cache[image_hash]
            
            # 缓存未命中，读取并编码图片
            try:
                with open(image_path, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    image_data = f"data:image/png;base64,{image_base64}"
                
                # 添加到缓存
                if len(self.cache) >= self.max_size:
                    self._evict_oldest()
                
                self.cache[image_hash] = image_data
                self.access_times[image_hash] = time.time()
                return image_data
                
            except Exception as e:
                logger.error(f"读取图片失败 {image_path}: {str(e)}")
                raise
    
    def _evict_oldest(self):
        """移除最久未使用的缓存项"""
        if not self.access_times:
            return
        
        oldest_hash = min(self.access_times.keys(), 
                         key=lambda k: self.access_times[k])
        del self.cache[oldest_hash]
        del self.access_times[oldest_hash]

class AdaptiveConcurrencyManager:
    """自适应并发管理器"""
    
    def __init__(self, initial_concurrency: int = 20, max_concurrency: int = 50):
        self.current_concurrency = initial_concurrency
        self.max_concurrency = max_concurrency
        self.min_concurrency = 5
        self.success_rate_window = []
        self.response_time_window = []
        self.window_size = 20
        self.lock = threading.Lock()
        
    def update_metrics(self, success: bool, response_time: float):
        """更新性能指标"""
        with self.lock:
            self.success_rate_window.append(1 if success else 0)
            self.response_time_window.append(response_time)
            
            if len(self.success_rate_window) > self.window_size:
                self.success_rate_window.pop(0)
            if len(self.response_time_window) > self.window_size:
                self.response_time_window.pop(0)
            
            # 自适应调整并发数
            self._adjust_concurrency()
    
    def _adjust_concurrency(self):
        """自适应调整并发数"""
        if len(self.success_rate_window) < 10:
            return
        
        success_rate = sum(self.success_rate_window) / len(self.success_rate_window)
        avg_response_time = sum(self.response_time_window) / len(self.response_time_window)
        
        # 根据成功率和响应时间调整并发数
        if success_rate > 0.95 and avg_response_time < 20:
            # 高成功率，低延迟：增加并发
            new_concurrency = min(self.current_concurrency + 2, self.max_concurrency)
        elif success_rate < 0.8 or avg_response_time > 40:
            # 低成功率或高延迟：减少并发
            new_concurrency = max(self.current_concurrency - 3, self.min_concurrency)
        else:
            new_concurrency = self.current_concurrency
        
        if new_concurrency != self.current_concurrency:
            logger.info(f"调整并发数: {self.current_concurrency} -> {new_concurrency} "
                       f"(成功率: {success_rate:.3f}, 平均响应时间: {avg_response_time:.1f}s)")
            self.current_concurrency = new_concurrency
    
    def get_concurrency(self) -> int:
        """获取当前建议的并发数"""
        return self.current_concurrency

async def _ultra_fast_vlm_request(session: aiohttp.ClientSession, 
                                 task: VLMTask, 
                                 image_cache: ImageCache,
                                 concurrency_manager: AdaptiveConcurrencyManager,
                                 semaphore: asyncio.Semaphore) -> ProcessingResult:
    """
    超快速VLM单个请求（优化版）
    """
    async with semaphore:  # 限制并发数
        queue_start = time.time()
        request_start = time.time()
        
        try:
            # 从缓存获取图片数据
            image_data = image_cache.get_base64(task.image_path, task.image_hash)
            
            # 优化的请求配置
            question = "请阅读我上传的pdf文件，使用markdown格式返回所有的信息。如果有图片，需要你用一个markdown标题+文字描述，标题为图片的标题，文字描述需要详细全面地介绍这张图片的内容。注意：你的输出必须与原文的语种一致。我提供的图片是英文，你的输出也必须是英文。"
            
            payload = {
                "model": task.model,
                "messages": [
                    {"role": "system", "content": "你必须精准快速提取PDF图片的内容。"}, 
                    {"role": "user", "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]}
                ],
                "temperature": 0.7,  # 降低温度以提高速度
                "top_p": 0.8,
                "max_tokens": 8192,  # 适当减少max_tokens
                "thinking": {"type": "disabled"},
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {task.api_key}",
                "Content-Type": "application/json"
            }
            
            queue_wait_time = time.time() - queue_start
            api_start = time.time()
            
            # 动态超时时间（基于历史表现）
            base_timeout = 25  # 大幅缩短基础超时时间
            timeout_seconds = base_timeout + (task.retry_count * 10)
            
            # 发送请求
            async with session.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds, connect=10, sock_read=20)
            ) as response:
                
                api_response_time = time.time() - api_start
                
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    
                    processing_time = time.time() - request_start
                    
                    # 更新性能指标
                    concurrency_manager.update_metrics(True, api_response_time)
                    
                    logger.debug(f"页面 {task.page_num} 成功，耗时 {processing_time:.2f}s")
                    
                    return ProcessingResult(
                        page_num=task.page_num,
                        content=content,
                        success=True,
                        processing_time=processing_time,
                        retry_count=task.retry_count,
                        api_response_time=api_response_time,
                        queue_wait_time=queue_wait_time
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"API错误 {response.status}: {error_text}")
                    
        except Exception as e:
            processing_time = time.time() - request_start
            api_response_time = time.time() - api_start if 'api_start' in locals() else 0
            
            # 更新性能指标
            concurrency_manager.update_metrics(False, api_response_time or 30)
            
            logger.warning(f"页面 {task.page_num} 请求失败: {str(e)}")
            
            return ProcessingResult(
                page_num=task.page_num,
                content="",
                success=False,
                error=str(e),
                processing_time=processing_time,
                retry_count=task.retry_count,
                api_response_time=api_response_time,
                queue_wait_time=queue_wait_time if 'queue_wait_time' in locals() else 0
            )

async def _turbo_batch_processor(tasks: List[VLMTask], 
                               image_cache: ImageCache,
                               concurrency_manager: AdaptiveConcurrencyManager,
                               batch_id: int) -> List[ProcessingResult]:
    """
    极速批量处理器
    """
    start_time = time.time()
    
    # 动态获取并发数
    max_concurrent = concurrency_manager.get_concurrency()
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 优化的连接器配置
    connector = aiohttp.TCPConnector(
        limit=max_concurrent + 20,  # 增加连接池大小
        limit_per_host=max_concurrent,
        ttl_dns_cache=600,
        use_dns_cache=True,
        keepalive_timeout=120,
        enable_cleanup_closed=True,
        force_close=False,
        auto_decompress=True
    )
    
    # 短超时配置，快速失败
    timeout = aiohttp.ClientTimeout(total=35, connect=5, sock_read=25)
    
    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=True
        ) as session:
            
            # 创建所有异步任务
            async_tasks = [
                _ultra_fast_vlm_request(session, task, image_cache, concurrency_manager, semaphore)
                for task in tasks
            ]
            
            # 并发执行
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # 处理结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"批次 {batch_id} 任务 {i} 异常: {str(result)}")
                    processed_results.append(ProcessingResult(
                        page_num=tasks[i].page_num,
                        content="",
                        success=False,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            # 计算批次指标
            total_time = time.time() - start_time
            success_count = sum(1 for r in processed_results if r.success)
            
            batch_metrics = BatchMetrics(
                batch_id=batch_id,
                task_count=len(tasks),
                success_count=success_count,
                total_time=total_time,
                avg_response_time=sum(r.api_response_time for r in processed_results if r.success) / max(success_count, 1),
                throughput=success_count / total_time if total_time > 0 else 0
            )
            
            logger.info(f"批次 {batch_id} 完成: {success_count}/{len(tasks)} 成功, "
                       f"耗时 {total_time:.2f}s, 吞吐量 {batch_metrics.throughput:.2f} 任务/秒")
            
            return processed_results
            
    except Exception as e:
        logger.error(f"批次 {batch_id} 处理异常: {str(e)}")
        return [ProcessingResult(
            page_num=task.page_num,
            content="",
            success=False,
            error=str(e)
        ) for task in tasks]

def _turbo_batch_worker(args: Tuple[List[VLMTask], int]) -> List[ProcessingResult]:
    """
    极速批量工作进程入口
    """
    tasks, batch_id = args
    
    # 预先计算图片缓存
    image_cache = ImageCache(max_size=len(tasks) + 20)
    concurrency_manager = AdaptiveConcurrencyManager(
        initial_concurrency=min(25, len(tasks)),  # 激进的初始并发
        max_concurrency=40
    )
    
    # 在新进程中创建事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(
            _turbo_batch_processor(tasks, image_cache, concurrency_manager, batch_id)
        )
    finally:
        loop.close()

class PDFVLMProcessorV5Turbo:
    """PDF VLM处理器 - 极速优化版本 V5"""
    
    def __init__(self, 
                 pdf_workers: Optional[int] = None,
                 vlm_workers: Optional[int] = None,
                 batch_size: int = 15,  # 激进的批次大小
                 initial_concurrency: int = 25,  # 激进的初始并发
                 max_concurrency: int = 40,
                 dpi: int = 150,  # 适当降低DPI以减少数据传输
                 model: str = "doubao-seed-1-6-flash-250615",
                 enable_image_compression: bool = True,
                 cache_size: int = 200):
        """
        初始化极速PDF VLM处理器
        """
        self.dpi = dpi
        self.model = model
        self.batch_size = batch_size
        self.initial_concurrency = initial_concurrency
        self.max_concurrency = max_concurrency
        self.enable_image_compression = enable_image_compression
        
        # 智能进程数配置
        cpu_count = mp.cpu_count()
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # 根据系统资源动态配置
        self.pdf_workers = pdf_workers or min(cpu_count, 6)
        self.vlm_workers = vlm_workers or min(max(2, cpu_count // 2), 8)
        
        # 全局缓存和管理器
        self.image_cache = ImageCache(max_size=cache_size)
        self.concurrency_manager = AdaptiveConcurrencyManager(
            initial_concurrency=initial_concurrency,
            max_concurrency=max_concurrency
        )
        
        # 性能统计
        self.stats = {
            'total_pages': 0,
            'processed_pages': 0,
            'success_pages': 0,
            'failed_pages': 0,
            'total_api_calls': 0,
            'cache_hits': 0,
            'avg_response_time': 0,
            'peak_concurrency': 0,
            'throughput': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"初始化极速处理器V5 - PDF进程: {self.pdf_workers}, VLM进程: {self.vlm_workers}")
        logger.info(f"批次大小: {self.batch_size}, 初始并发: {self.initial_concurrency}, 最大并发: {self.max_concurrency}")
        logger.info(f"系统信息: CPU核心 {cpu_count}, 内存 {system_memory_gb:.1f}GB")
    
    def _optimize_image(self, image_path: str) -> str:
        """优化图片大小以提升传输速度"""
        if not self.enable_image_compression:
            return image_path
        
        try:
            with Image.open(image_path) as img:
                # 如果图片太大，进行压缩
                if img.size[0] > 2000 or img.size[1] > 2000:
                    # 计算压缩比例
                    max_size = 1800
                    ratio = min(max_size / img.size[0], max_size / img.size[1])
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    
                    # 创建压缩版本
                    compressed_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # 保存到临时文件
                    compressed_path = image_path.replace('.png', '_compressed.png')
                    compressed_img.save(compressed_path, 'PNG', optimize=True, compress_level=6)
                    
                    return compressed_path
            
            return image_path
            
        except Exception as e:
            logger.warning(f"图片优化失败 {image_path}: {str(e)}")
            return image_path
    
    def convert_pdf_to_images_turbo(self, pdf_path: str, output_dir: str, 
                                   file_prefix: Optional[str] = None) -> List[Tuple[int, str]]:
        """
        极速PDF转图片（优化版）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 快速获取页数
        try:
            import fitz
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
        except:
            logger.warning("无法使用PyMuPDF获取页数，使用备用方法")
            total_pages = self._get_page_count_fallback(pdf_path)
        
        if total_pages == 0:
            raise ValueError("无法获取PDF页数")
        
        self.stats['total_pages'] = total_pages
        logger.info(f"PDF总共 {total_pages} 页，开始极速转换...")
        
        if file_prefix is None:
            file_prefix = Path(pdf_path).stem
        
        # 优化的批次大小（更大的批次）
        batch_size = min(8, max(2, total_pages // self.pdf_workers))
        batches = []
        
        for start in range(1, total_pages + 1, batch_size):
            end = min(start + batch_size - 1, total_pages)
            batches.append((pdf_path, start, end, output_dir, file_prefix, self.dpi))
        
        logger.info(f"使用 {self.pdf_workers} 进程处理 {len(batches)} 个批次")
        
        start_time = time.time()
        all_results = []
        
        # 使用线程池而不是进程池（PDF2image在某些情况下线程更快）
        with ThreadPoolExecutor(max_workers=self.pdf_workers) as executor:
            future_to_batch = {executor.submit(_convert_page_range_worker, batch): batch 
                              for batch in batches}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    result = future.result()
                    all_results.extend(result)
                    logger.info(f"PDF转换进度: {len(all_results)}/{total_pages}")
                except Exception as e:
                    logger.error(f"PDF转换批次失败: {str(e)}")
        
        # 图片优化
        if self.enable_image_compression:
            logger.info("开始图片压缩优化...")
            optimized_results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self._optimize_image, img_path): (page_num, img_path) 
                          for page_num, img_path in all_results}
                
                for future in concurrent.futures.as_completed(futures):
                    page_num, original_path = futures[future]
                    optimized_path = future.result()
                    optimized_results.append((page_num, optimized_path))
            
            all_results = optimized_results
        
        all_results.sort(key=lambda x: x[0])
        conversion_time = time.time() - start_time
        
        logger.info(f"PDF转图片完成: {len(all_results)} 张，耗时 {conversion_time:.2f}s")
        return all_results
    
    def process_images_with_vlm_turbo(self, image_paths: List[Tuple[int, str]]) -> Dict[int, str]:
        """
        极速VLM批量处理
        """
        api_key = os.environ.get("DOUBAO_API_KEY")
        if not api_key:
            raise ValueError("请设置 DOUBAO_API_KEY 环境变量")
        
        logger.info(f"开始极速VLM处理: {len(image_paths)} 张图片")
        
        # 预处理：计算图片哈希并预缓存
        logger.info("预处理图片缓存...")
        vlm_tasks = []
        for page_num, image_path in image_paths:
            image_hash = self.image_cache.get_image_hash(image_path)
            task = VLMTask(
                page_num=page_num,
                image_path=image_path,
                image_hash=image_hash,
                api_key=api_key,
                model=self.model,
                priority=page_num  # 按页码优先级
            )
            vlm_tasks.append(task)
        
        # 创建批次（更大的批次大小）
        batches = []
        for i in range(0, len(vlm_tasks), self.batch_size):
            batch_tasks = vlm_tasks[i:i + self.batch_size]
            batches.append((batch_tasks, i // self.batch_size + 1))
        
        logger.info(f"创建 {len(batches)} 个批次，每批次最多 {self.batch_size} 张图片")
        
        start_time = time.time()
        all_results = []
        
        # 使用进程池进行批量处理
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.vlm_workers) as executor:
            future_to_batch = {executor.submit(_turbo_batch_worker, batch): batch 
                              for batch in batches}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    completed += 1
                    
                    # 实时统计
                    success_count = sum(1 for r in batch_results if r.success)
                    logger.info(f"批次完成: {completed}/{len(batches)} "
                               f"(本批次成功: {success_count}/{len(batch_results)})")
                    
                except Exception as e:
                    logger.error(f"批次处理异常: {str(e)}")
        
        # 整理结果
        results = {}
        success_count = 0
        failed_count = 0
        total_response_time = 0
        
        for result in all_results:
            self.stats['total_api_calls'] += 1
            if result.success:
                results[result.page_num] = result.content
                success_count += 1
                total_response_time += result.api_response_time
            else:
                failed_count += 1
        
        # 更新统计信息
        processing_time = time.time() - start_time
        self.stats['processed_pages'] = len(all_results)
        self.stats['success_pages'] = success_count
        self.stats['failed_pages'] = failed_count
        self.stats['avg_response_time'] = total_response_time / max(success_count, 1)
        self.stats['throughput'] = success_count / processing_time if processing_time > 0 else 0
        
        logger.info(f"极速VLM处理完成: 成功 {success_count}/{len(vlm_tasks)} "
                   f"({success_count/len(vlm_tasks)*100:.1f}%), 耗时 {processing_time:.2f}s")
        logger.info(f"平均响应时间: {self.stats['avg_response_time']:.2f}s, "
                   f"吞吐量: {self.stats['throughput']:.2f} 页/秒")
        
        return results
    
    def _get_page_count_fallback(self, pdf_path: str) -> int:
        """备用方法获取PDF页数"""
        try:
            # 尝试使用二分查找
            left, right = 1, 500
            while left < right:
                mid = (left + right + 1) // 2
                try:
                    convert_from_path(pdf_path, dpi=72, first_page=mid, last_page=mid)
                    left = mid
                except:
                    right = mid - 1
            return left
        except:
            return 0
    
    def process_pdf_to_markdown_turbo(self, pdf_path: str, output_md_path: str, 
                                     temp_image_dir: Optional[str] = None,
                                     cleanup_images: bool = True) -> str:
        """
        完整的极速PDF到Markdown处理流程
        """
        self.stats['start_time'] = time.time()
        
        if temp_image_dir is None:
            temp_image_dir = os.path.join(os.path.dirname(output_md_path), "temp_images_turbo")
        
        try:
            # 第一步：极速PDF转图片
            logger.info("🚀 第一步：极速PDF转图片")
            image_paths = self.convert_pdf_to_images_turbo(pdf_path, temp_image_dir)
            
            if not image_paths:
                raise ValueError("PDF转图片失败")
            
            # 第二步：极速VLM批量处理
            logger.info("⚡ 第二步：极速VLM批量处理")
            page_texts = self.process_images_with_vlm_turbo(image_paths)
            
            if not page_texts:
                raise ValueError("VLM处理完全失败")
            
            # 第三步：智能文本合并
            logger.info("📝 第三步：智能文本合并")
            combined_text = self._smart_combine_texts(page_texts)
            
            # 保存结果
            os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(combined_text)
            
            self.stats['end_time'] = time.time()
            self._print_turbo_stats(output_md_path, combined_text)
            
            return combined_text
            
        except Exception as e:
            logger.error(f"极速处理失败: {str(e)}")
            raise
            
        finally:
            # 清理临时文件
            if cleanup_images and temp_image_dir and os.path.exists(temp_image_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_image_dir)
                    logger.info(f"已清理临时目录: {temp_image_dir}")
                except Exception as e:
                    logger.warning(f"清理失败: {str(e)}")
    
    def _smart_combine_texts(self, page_texts: Dict[int, str]) -> str:
        """智能合并文本"""
        if not page_texts:
            return ""
        
        # 按页码排序
        sorted_pages = sorted(page_texts.keys())
        combined_parts = []
        
        # 添加文档头部
        combined_parts.append("# PDF文档内容\n\n")
        combined_parts.append(f"*文档处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        combined_parts.append(f"*成功处理页数: {len(page_texts)}/{self.stats['total_pages']}*\n\n")
        combined_parts.append("---\n\n")
        
        # 合并页面内容
        for page_num in sorted_pages:
            text = page_texts[page_num].strip()
            if text:
                combined_parts.append(f"## 第 {page_num} 页\n\n{text}\n\n")
        
        # 添加缺失页面报告
        all_pages = set(range(1, self.stats['total_pages'] + 1))
        missing_pages = all_pages - set(page_texts.keys())
        if missing_pages:
            combined_parts.append("---\n\n")
            combined_parts.append("## 处理报告\n\n")
            combined_parts.append(f"**缺失页面**: {sorted(missing_pages)}\n\n")
        
        return "".join(combined_parts)
    
    def _print_turbo_stats(self, output_path: str, content: str):
        """输出极速版性能统计"""
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        logger.info("=" * 70)
        logger.info("🏆 极速处理完成 - 性能统计（V5 Turbo版本）")
        logger.info("=" * 70)
        logger.info(f"📊 处理概览:")
        logger.info(f"   总页数: {self.stats['total_pages']}")
        logger.info(f"   成功页数: {self.stats['success_pages']}")
        logger.info(f"   失败页数: {self.stats['failed_pages']}")
        logger.info(f"   成功率: {self.stats['success_pages']/self.stats['total_pages']*100:.1f}%")
        logger.info(f"")
        logger.info(f"⚡ 性能指标:")
        logger.info(f"   总耗时: {total_time:.2f}s")
        logger.info(f"   平均每页: {total_time/self.stats['total_pages']:.2f}s")
        logger.info(f"   处理吞吐量: {self.stats['throughput']:.2f} 页/秒")
        logger.info(f"   平均API响应: {self.stats['avg_response_time']:.2f}s")
        logger.info(f"   总API调用: {self.stats['total_api_calls']}")
        logger.info(f"")
        logger.info(f"📁 输出信息:")
        logger.info(f"   文件路径: {output_path}")
        logger.info(f"   文件大小: {len(content):,} 字符")
        logger.info(f"   平均每页字符数: {len(content)//max(self.stats['success_pages'], 1):,}")

# 便捷函数
def convert_pdf_to_markdown_v5_turbo(pdf_path: str, 
                                    output_md_path: str,
                                    dpi: int = 150,
                                    batch_size: int = 15,
                                    initial_concurrency: int = 25,
                                    max_concurrency: int = 40,
                                    pdf_workers: Optional[int] = None,
                                    vlm_workers: Optional[int] = None,
                                    model: str = "doubao-seed-1-6-flash-250615",
                                    enable_compression: bool = True,
                                    cleanup_images: bool = True) -> str:
    """
    极速PDF到Markdown转换（V5 Turbo版本）
    
    主要优化:
    - 激进并发配置（25+并发）
    - 智能负载均衡
    - 图片缓存优化
    - 自适应超时
    - 图片压缩
    """
    processor = PDFVLMProcessorV5Turbo(
        pdf_workers=pdf_workers,
        vlm_workers=vlm_workers,
        batch_size=batch_size,
        initial_concurrency=initial_concurrency,
        max_concurrency=max_concurrency,
        dpi=dpi,
        model=model,
        enable_image_compression=enable_compression
    )
    
    return processor.process_pdf_to_markdown_turbo(
        pdf_path=pdf_path,
        output_md_path=output_md_path,
        cleanup_images=cleanup_images
    )

# 多进程工作函数需要在模块级别定义
def _convert_page_range_worker(args: Tuple[str, int, int, str, str, int]) -> List[Tuple[int, str]]:
    """多进程PDF转图片工作函数"""
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
            
            # 优化保存参数
            image.save(output_path, 'PNG', optimize=True, compress_level=3)
            results.append((page_num, output_path))
            
        end_time = time.time()
        logger.debug(f"转换页面 {start_page}-{end_page} 完成，耗时 {end_time - start_time:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"转换页面 {start_page}-{end_page} 失败: {str(e)}")
        return []

if __name__ == "__main__":
    # 使用示例
    if not os.environ.get("DOUBAO_API_KEY"):
        print("错误：请设置 DOUBAO_API_KEY 环境变量")
        sys.exit(1)
    
    # 测试文件
    pdf_file = r"D:\AgentBuilding\FinAgent\files\arxiv_papers\2506.19676v3.pdf"
    output_md_file = r"D:\AgentBuilding\FinAgent\files\arxiv_papers\2506.19676v3_turbo.md"
    
    try:
        logger.info("🚀 开始使用极速版本处理PDF...")
        
        markdown_content = convert_pdf_to_markdown_v5_turbo(
            pdf_path=pdf_file,
            output_md_path=output_md_file,
            dpi=150,                    # 适中的DPI
            batch_size=15,              # 激进的批次大小
            initial_concurrency=25,     # 激进的初始并发
            max_concurrency=40,         # 更高的最大并发
            pdf_workers=6,              # 更多PDF处理进程
            vlm_workers=8,              # 更多VLM处理进程
            enable_compression=True,    # 启用图片压缩
            cleanup_images=True
        )
        
        logger.info("🎉 极速版本处理完成！")
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        sys.exit(1) 