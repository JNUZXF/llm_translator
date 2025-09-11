"""
PDF VLM处理器 - 性能优化版本 V5
专注解决性能瓶颈：激进并发+智能缓存+快速失败
路径：agent/utils/pdf_vlm_processor_v5_optimized.py
"""

# type: ignore

import os
import sys
import time
import asyncio
import aiohttp
import json
import base64
import hashlib
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from pdf2image import convert_from_path
from PIL import Image
import multiprocessing as mp
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceCache:
    """高性能缓存系统"""
    
    def __init__(self, max_size: int = 200):
        self.cache: Dict[str, str] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_image_base64(self, image_path: str) -> str:
        """获取图片Base64编码（带LRU缓存）"""
        # 计算文件哈希作为缓存键
        cache_key = hashlib.md5(image_path.encode()).hexdigest()
        
        with self.lock:
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                self.hit_count += 1
                return self.cache[cache_key]
            
            # 缓存未命中
            self.miss_count += 1
            
            try:
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    image_url = f"data:image/png;base64,{image_data}"
                
                # 添加到缓存
                if len(self.cache) >= self.max_size:
                    self._evict_oldest()
                
                self.cache[cache_key] = image_url
                self.access_times[cache_key] = time.time()
                return image_url
                
            except Exception as e:
                logger.error(f"读取图片失败 {image_path}: {e}")
                raise
    
    def _evict_oldest(self):
        """删除最久未使用的缓存项"""
        if self.access_times:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total, 1) * 100
        return {
            'hits': self.hit_count,
            'misses': self.miss_count, 
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self.cache)
        }

async def ultra_fast_vlm_request(session: aiohttp.ClientSession, 
                                page_num: int, 
                                image_path: str,
                                api_key: str,
                                model: str,
                                cache: PerformanceCache,
                                semaphore: asyncio.Semaphore) -> Tuple[int, str, bool, float]:
    """超快速VLM请求"""
    async with semaphore:
        start_time = time.time()
        
        try:
            # 从缓存获取图片
            image_data = cache.get_image_base64(image_path)
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "你必须快速精准提取PDF内容。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "请阅读PDF图片，用markdown格式返回所有信息。图片是英文则输出英文。"},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]}
                ],
                "temperature": 0.5,
                "max_tokens": 8192,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 短超时，快速失败
            timeout = aiohttp.ClientTimeout(total=25, connect=5, sock_read=20)
            
            async with session.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    processing_time = time.time() - start_time
                    
                    logger.debug(f"页面 {page_num} 成功，耗时 {processing_time:.2f}s")
                    return page_num, content, True, processing_time
                else:
                    error_text = await response.text()
                    raise Exception(f"API错误 {response.status}: {error_text}")
                    
        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"页面 {page_num} 失败: {e}")
            return page_num, "", False, processing_time

async def process_batch_ultra_fast(image_paths: List[Tuple[int, str]], 
                                  api_key: str, 
                                  model: str,
                                  max_concurrent: int = 30) -> Dict[int, str]:
    """超快速批量处理"""
    
    # 创建缓存
    cache = PerformanceCache(max_size=len(image_paths) + 50)
    
    # 控制并发数
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 优化连接器配置
    connector = aiohttp.TCPConnector(
        limit=max_concurrent + 10,
        limit_per_host=max_concurrent,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(total=30, connect=8, sock_read=22)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 创建所有任务
        tasks = [
            ultra_fast_vlm_request(session, page_num, image_path, api_key, model, cache, semaphore)
            for page_num, image_path in image_paths
        ]
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        page_contents = {}
        success_count = 0
        total_time = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"任务异常: {result}")
                continue
                
            page_num, content, success, proc_time = result
            total_time += proc_time
            
            if success and content.strip():
                page_contents[page_num] = content
                success_count += 1
        
        # 输出缓存统计
        cache_stats = cache.get_stats()
        avg_time = total_time / len(results) if results else 0
        
        logger.info(f"批量处理完成: {success_count}/{len(image_paths)} 成功")
        logger.info(f"平均响应时间: {avg_time:.2f}s")
        logger.info(f"缓存统计: {cache_stats}")
        
        return page_contents

def process_batch_worker(args: Tuple[List[Tuple[int, str]], str, str, int]) -> Dict[int, str]:
    """批量处理工作进程"""
    image_paths, api_key, model, max_concurrent = args
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(
            process_batch_ultra_fast(image_paths, api_key, model, max_concurrent)
        )
    finally:
        loop.close()

def convert_pdf_to_images_fast(pdf_path: str, output_dir: str, dpi: int = 150) -> List[Tuple[int, str]]:
    """快速PDF转图片"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取页数
    try:
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
    except:
        total_pages = 100  # 默认值，实际运行时会调整
    
    logger.info(f"开始转换PDF，预计 {total_pages} 页")
    
    # 批量转换
    batch_size = 10
    all_results = []
    file_prefix = Path(pdf_path).stem
    
    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        
        try:
            images = convert_from_path(pdf_path, dpi=dpi, first_page=start, last_page=end)
            
            for i, image in enumerate(images):
                page_num = start + i
                image_path = os.path.join(output_dir, f"{file_prefix}_page_{page_num:04d}.png")
                
                # 优化图片保存
                if image.size[0] > 1800 or image.size[1] > 1800:
                    ratio = 1800 / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                image.save(image_path, 'PNG', optimize=True, compress_level=6)
                all_results.append((page_num, image_path))
                
        except Exception as e:
            logger.warning(f"转换页面 {start}-{end} 失败: {e}")
            break  # 到达文件末尾
    
    logger.info(f"PDF转换完成，生成 {len(all_results)} 张图片")
    return all_results

class PDFVLMProcessorV5:
    """PDF VLM处理器 V5 - 性能优化版"""
    
    def __init__(self, 
                 batch_size: int = 20,           # 更大批次
                 max_concurrent: int = 35,       # 更高并发  
                 max_workers: int = 6,           # 更多进程
                 dpi: int = 150,                 # 适中DPI
                 model: str = "doubao-seed-1-6-flash-250615"):
        
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.max_workers = max_workers
        self.dpi = dpi
        self.model = model
        
        # 根据系统资源优化
        cpu_count = mp.cpu_count()
        self.max_workers = min(max_workers, cpu_count)
        
        logger.info(f"初始化V5处理器 - 批次: {batch_size}, 并发: {max_concurrent}, 进程: {self.max_workers}")
    
    def process_pdf_to_markdown(self, pdf_path: str, output_md_path: str, 
                               temp_dir: Optional[str] = None,
                               cleanup: bool = True) -> str:
        """完整的PDF处理流程"""
        
        start_time = time.time()
        
        if temp_dir is None:
            temp_dir = os.path.join(os.path.dirname(output_md_path), "temp_images_v5")
        
        try:
            # 步骤1: 快速PDF转图片
            logger.info("🚀 步骤1: 快速PDF转图片")
            image_paths = convert_pdf_to_images_fast(pdf_path, temp_dir, self.dpi)
            
            if not image_paths:
                raise ValueError("PDF转换失败")
            
            # 步骤2: 超快速VLM批量处理
            logger.info("⚡ 步骤2: 超快速VLM处理")
            page_texts = self._process_vlm_multiprocess(image_paths)
            
            if not page_texts:
                raise ValueError("VLM处理失败")
            
            # 步骤3: 生成Markdown
            logger.info("📝 步骤3: 生成Markdown")
            markdown_content = self._generate_markdown(page_texts, len(image_paths))
            
            # 保存文件
            os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            # 性能统计
            total_time = time.time() - start_time
            success_rate = len(page_texts) / len(image_paths) * 100
            throughput = len(page_texts) / total_time
            
            logger.info("=" * 60)
            logger.info("🎉 V5处理完成 - 性能统计")
            logger.info("=" * 60)
            logger.info(f"📊 总页数: {len(image_paths)}")
            logger.info(f"✅ 成功页数: {len(page_texts)}")
            logger.info(f"📈 成功率: {success_rate:.1f}%")
            logger.info(f"⏱️ 总耗时: {total_time:.2f}s")
            logger.info(f"🚄 处理速度: {throughput:.2f} 页/秒")
            logger.info(f"📄 平均每页: {total_time/len(image_paths):.2f}s")
            logger.info(f"💾 文件大小: {len(markdown_content):,} 字符")
            logger.info(f"📁 输出路径: {output_md_path}")
            
            return markdown_content
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            raise
            
        finally:
            # 清理临时文件
            if cleanup and temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.info(f"已清理临时目录: {temp_dir}")
                except Exception as e:
                    logger.warning(f"清理失败: {e}")
    
    def _process_vlm_multiprocess(self, image_paths: List[Tuple[int, str]]) -> Dict[int, str]:
        """多进程VLM处理"""
        
        api_key = os.environ.get("DOUBAO_API_KEY")
        if not api_key:
            raise ValueError("请设置 DOUBAO_API_KEY 环境变量")
        
        # 分批处理
        batches = []
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            batches.append((batch, api_key, self.model, self.max_concurrent))
        
        logger.info(f"创建 {len(batches)} 个批次，使用 {self.max_workers} 个进程")
        
        all_results = {}
        
        # 使用进程池处理
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(process_batch_worker, batch): i 
                              for i, batch in enumerate(batches)}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.update(batch_results)
                    logger.info(f"批次 {batch_idx + 1}/{len(batches)} 完成，"
                               f"成功 {len(batch_results)} 页")
                except Exception as e:
                    logger.error(f"批次 {batch_idx + 1} 失败: {e}")
        
        return all_results
    
    def _generate_markdown(self, page_texts: Dict[int, str], total_pages: int) -> str:
        """生成Markdown内容"""
        parts = []
        
        # 添加头部信息
        parts.append("# PDF文档内容\n\n")
        parts.append(f"*处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        parts.append(f"*成功处理: {len(page_texts)}/{total_pages} 页*\n\n")
        parts.append("---\n\n")
        
        # 按页码顺序添加内容
        for page_num in sorted(page_texts.keys()):
            content = page_texts[page_num].strip()
            if content:
                parts.append(f"## 第 {page_num} 页\n\n{content}\n\n")
        
        # 添加处理报告
        missing_pages = set(range(1, total_pages + 1)) - set(page_texts.keys())
        if missing_pages:
            parts.append("---\n\n## 处理报告\n\n")
            parts.append(f"**未成功处理的页面**: {sorted(missing_pages)}\n\n")
        
        return "".join(parts)

# 便捷函数
def convert_pdf_to_markdown_v5_optimized(pdf_path: str, 
                                        output_md_path: str,
                                        batch_size: int = 20,
                                        max_concurrent: int = 35,
                                        max_workers: int = 6,
                                        dpi: int = 150,
                                        model: str = "doubao-seed-1-6-flash-250615",
                                        cleanup: bool = True) -> str:
    """
    V5优化版PDF转Markdown
    
    主要优化点：
    1. 激进并发配置 (35并发)
    2. 大批次处理 (20张/批)
    3. 智能图片缓存
    4. 快速失败机制
    5. 图片压缩优化
    """
    
    processor = PDFVLMProcessorV5(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        max_workers=max_workers,
        dpi=dpi,
        model=model
    )
    
    return processor.process_pdf_to_markdown(
        pdf_path=pdf_path,
        output_md_path=output_md_path,
        cleanup=cleanup
    )

if __name__ == "__main__":
    # 使用示例
    if not os.environ.get("DOUBAO_API_KEY"):
        print("错误：请设置 DOUBAO_API_KEY 环境变量")
        sys.exit(1)
    
    # 测试文件
    pdf_file = r"files/爱尔眼科：2024年年度报告.pdf"  # 相对路径
    output_file = r"files/爱尔眼科_2024年报_v5optimized.md"
    
    try:
        logger.info("🚀 开始V5优化版处理...")
        
        result = convert_pdf_to_markdown_v5_optimized(
            pdf_path=pdf_file,
            output_md_path=output_file,
            batch_size=20,         # 大批次
            max_concurrent=35,     # 高并发
            max_workers=6,         # 多进程
            dpi=150,              # 适中DPI
            cleanup=True
        )
        
        logger.info("🎉 V5优化版处理完成！")
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        sys.exit(1) 