import fitz  # PyMuPDF
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF文件处理器"""
    
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        从PDF文件中提取文本，按页面分组
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            包含页面信息和文本内容的列表
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # 清理文本
                text = self._clean_text(text)
                
                if text.strip():  # 只添加非空页面
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "char_count": len(text)
                    })
            
            doc.close()
            return pages
            
        except Exception as e:
            logger.error(f"PDF文本提取失败: {str(e)}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        将长文本分割成适合翻译的块
        
        Args:
            text: 需要分割的文本
            
        Returns:
            文本块列表
        """
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        sentences = text.split('.')
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果添加这个句子后不会超过最大长度
            if len(current_chunk) + len(sentence) + 1 <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
            else:
                # 保存当前块并开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        清理提取的文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白字符
        text = ' '.join(text.split())
        
        # 移除页眉页脚常见模式
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # 跳过可能的页码、页眉页脚
            if len(line) < 3 or line.isdigit():
                continue
            cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        获取PDF文件信息
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            PDF信息字典
        """
        try:
            doc = fitz.open(pdf_path)
            info = {
                "page_count": len(doc),
                "file_size": os.path.getsize(pdf_path),
                "metadata": doc.metadata
            }
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"获取PDF信息失败: {str(e)}")
            raise