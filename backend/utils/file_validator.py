"""
文件上传安全验证工具
提供全面的文件验证功能，包括类型检查、大小限制、内容验证等
"""

import os
import hashlib
import logging
from typing import Tuple, Optional
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import fitz  # PyMuPDF
from utils.exceptions import FileUploadError, ValidationError

logger = logging.getLogger(__name__)


class SecureFileValidator:
    """安全的文件验证器"""

    # 允许的MIME类型
    ALLOWED_MIME_TYPES = {'application/pdf'}

    # 允许的文件扩展名
    ALLOWED_EXTENSIONS = {'pdf'}

    # 文件大小限制（默认16MB）
    MAX_FILE_SIZE = 16 * 1024 * 1024

    # PDF文件魔数（文件头）
    PDF_MAGIC_NUMBERS = [
        b'%PDF-1.',  # PDF 1.x
        b'%PDF-2.',  # PDF 2.x
    ]

    @classmethod
    def validate_file(cls, file: FileStorage, max_size: Optional[int] = None) -> Tuple[bool, str]:
        """
        全面的文件验证

        Args:
            file: 上传的文件对象
            max_size: 最大文件大小（字节），None则使用默认值

        Returns:
            (是否有效, 消息)

        Raises:
            FileUploadError: 验证失败时
        """
        if max_size is None:
            max_size = cls.MAX_FILE_SIZE

        try:
            # 1. 检查文件对象
            if not file or not file.filename:
                raise FileUploadError("未选择文件或文件名为空")

            # 2. 验证文件名
            filename = secure_filename(file.filename)
            if not filename:
                raise FileUploadError("文件名不合法")

            # 3. 验证文件扩展名
            if not cls._check_extension(filename):
                raise FileUploadError(
                    f"不支持的文件类型。仅支持: {', '.join(cls.ALLOWED_EXTENSIONS)}"
                )

            # 4. 验证文件大小
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            if file_size == 0:
                raise FileUploadError("文件为空")

            if file_size > max_size:
                max_mb = max_size / (1024 * 1024)
                raise FileUploadError(
                    f"文件大小超过限制（最大 {max_mb:.1f}MB）"
                )

            logger.info(f"文件大小验证通过: {filename} ({file_size / 1024:.2f}KB)")

            # 5. 读取文件内容进行进一步验证
            file_content = file.read(8192)  # 读取前8KB用于验证
            file.seek(0)

            # 6. 验证文件魔数（真实文件类型）
            if not cls._check_magic_number(file_content):
                raise FileUploadError(
                    "文件内容与扩展名不匹配，这可能不是一个有效的PDF文件"
                )

            logger.info(f"文件魔数验证通过: {filename}")

            # 7. 验证PDF文件完整性
            full_content = file.read()
            file.seek(0)

            if not cls._validate_pdf_integrity(full_content, filename):
                raise FileUploadError("PDF文件已损坏或格式不正确")

            logger.info(f"PDF完整性验证通过: {filename}")

            # 8. 安全检查 - 检测可能的恶意内容
            cls._security_scan(full_content, filename)

            return True, "文件验证通过"

        except FileUploadError:
            raise
        except Exception as e:
            logger.error(f"文件验证过程出错: {str(e)}", exc_info=True)
            raise FileUploadError(f"文件验证失败: {str(e)}")

    @classmethod
    def _check_extension(cls, filename: str) -> bool:
        """
        检查文件扩展名

        Args:
            filename: 文件名

        Returns:
            是否为允许的扩展名
        """
        if '.' not in filename:
            return False

        ext = filename.rsplit('.', 1)[1].lower()
        return ext in cls.ALLOWED_EXTENSIONS

    @classmethod
    def _check_magic_number(cls, file_content: bytes) -> bool:
        """
        检查文件魔数（文件头）

        Args:
            file_content: 文件内容（前几个字节）

        Returns:
            是否匹配PDF魔数
        """
        for magic in cls.PDF_MAGIC_NUMBERS:
            if file_content.startswith(magic):
                return True
        return False

    @classmethod
    def _validate_pdf_integrity(cls, pdf_content: bytes, filename: str) -> bool:
        """
        验证PDF文件完整性

        Args:
            pdf_content: PDF文件内容
            filename: 文件名（用于日志）

        Returns:
            PDF是否完整有效
        """
        try:
            # 尝试使用PyMuPDF打开PDF
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            # 检查页数
            page_count = len(doc)
            if page_count == 0:
                logger.warning(f"PDF文件无页面: {filename}")
                doc.close()
                return False

            # 检查PDF元数据
            metadata = doc.metadata
            logger.debug(f"PDF元数据: {metadata}")

            # 尝试读取第一页（验证可读性）
            try:
                first_page = doc.load_page(0)
                # 尝试获取文本（即使为空也没关系）
                _ = first_page.get_text()
            except Exception as e:
                logger.error(f"无法读取PDF第一页: {str(e)}")
                doc.close()
                return False

            doc.close()
            logger.info(f"PDF完整性验证成功: {filename}, {page_count} 页")
            return True

        except Exception as e:
            logger.error(f"PDF完整性验证失败 ({filename}): {str(e)}")
            return False

    @classmethod
    def _security_scan(cls, file_content: bytes, filename: str) -> None:
        """
        安全扫描 - 检测潜在的恶意内容

        Args:
            file_content: 文件内容
            filename: 文件名

        Raises:
            FileUploadError: 发现安全问题时
        """
        # 1. 检查文件大小异常（可能的zip炸弹等）
        if len(file_content) > cls.MAX_FILE_SIZE:
            raise FileUploadError("文件内容大小异常")

        # 2. 检查PDF中的JavaScript（可能的XSS攻击）
        if b'/JavaScript' in file_content or b'/JS' in file_content:
            logger.warning(f"PDF包含JavaScript: {filename}")
            # 注意：不是所有带JS的PDF都是恶意的，但需要记录
            # 可以根据安全策略决定是否拒绝

        # 3. 检查PDF中的嵌入文件
        if b'/EmbeddedFile' in file_content:
            logger.warning(f"PDF包含嵌入文件: {filename}")

        # 4. 检查外部链接（可能的钓鱼）
        if b'/URI' in file_content:
            logger.info(f"PDF包含外部链接: {filename}")

        logger.info(f"安全扫描完成: {filename}")

    @staticmethod
    def calculate_file_hash(file_content: bytes, algorithm: str = 'sha256') -> str:
        """
        计算文件哈希值

        Args:
            file_content: 文件内容
            algorithm: 哈希算法 (md5, sha1, sha256)

        Returns:
            文件哈希值
        """
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        else:
            hasher = hashlib.sha256()

        hasher.update(file_content)
        return hasher.hexdigest()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        清理文件名，移除潜在的危险字符

        Args:
            filename: 原始文件名

        Returns:
            安全的文件名
        """
        # 使用werkzeug的secure_filename
        safe_name = secure_filename(filename)

        # 进一步清理
        # 移除多个点号（避免路径穿越）
        while '..' in safe_name:
            safe_name = safe_name.replace('..', '.')

        # 限制文件名长度
        max_length = 255
        if len(safe_name) > max_length:
            name, ext = os.path.splitext(safe_name)
            name = name[:max_length - len(ext) - 1]
            safe_name = f"{name}{ext}"

        return safe_name

    @classmethod
    def save_validated_file(cls, file: FileStorage, upload_dir: str) -> Tuple[str, str, str]:
        """
        验证并保存文件

        Args:
            file: 上传的文件对象
            upload_dir: 上传目录

        Returns:
            (保存的文件路径, 原始文件名, 文件哈希)

        Raises:
            FileUploadError: 验证或保存失败时
        """
        # 1. 验证文件
        cls.validate_file(file)

        # 2. 清理文件名
        original_filename = file.filename
        safe_filename = cls.sanitize_filename(original_filename)

        # 3. 读取文件内容
        file_content = file.read()
        file.seek(0)

        # 4. 计算文件哈希
        file_hash = cls.calculate_file_hash(file_content)

        # 5. 生成唯一文件名（使用哈希前缀避免冲突）
        name, ext = os.path.splitext(safe_filename)
        unique_filename = f"{file_hash[:8]}_{safe_filename}"

        # 6. 确保上传目录存在
        os.makedirs(upload_dir, exist_ok=True)

        # 7. 保存文件
        file_path = os.path.join(upload_dir, unique_filename)

        # 检查文件是否已存在
        if os.path.exists(file_path):
            logger.info(f"文件已存在，使用现有文件: {file_path}")
        else:
            with open(file_path, 'wb') as f:
                f.write(file_content)
            logger.info(f"文件保存成功: {file_path}")

        return file_path, original_filename, file_hash


# 便捷的装饰器
def validate_file_upload(field_name: str = 'file', max_size: Optional[int] = None):
    """
    装饰器：验证文件上传

    使用示例:
        @app.route('/upload', methods=['POST'])
        @validate_file_upload('file')
        def upload():
            file = request.files['file']
            # file已经过验证
    """
    from functools import wraps
    from flask import request, jsonify

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if field_name not in request.files:
                return jsonify({
                    "error": f"未找到文件字段: {field_name}",
                    "code": "FILE_FIELD_MISSING"
                }), 400

            file = request.files[field_name]

            try:
                SecureFileValidator.validate_file(file, max_size)
                return func(*args, **kwargs)

            except FileUploadError as e:
                return jsonify(e.to_dict()), e.status_code
            except Exception as e:
                logger.error(f"文件验证异常: {str(e)}", exc_info=True)
                return jsonify({
                    "error": "文件验证失败",
                    "code": "FILE_VALIDATION_ERROR"
                }), 500

        return wrapper

    return decorator


if __name__ == '__main__':
    # 测试文件验证器
    print("文件验证器测试")

    # 测试文件名清理
    test_filenames = [
        "../../../etc/passwd",
        "test..pdf",
        "normal_file.pdf",
        "文件名 with spaces.pdf"
    ]

    for filename in test_filenames:
        safe = SecureFileValidator.sanitize_filename(filename)
        print(f"{filename} -> {safe}")
