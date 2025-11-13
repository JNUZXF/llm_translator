"""
输入验证和清理工具
提供统一的数据验证和清理功能
"""

import re
from typing import Optional, Any, Dict
from utils.exceptions import ValidationError
import bleach


class InputValidator:
    """输入验证器"""

    # 允许的HTML标签（用于bleach清理）
    ALLOWED_TAGS = []
    ALLOWED_ATTRIBUTES = {}

    # 文本长度限制
    MAX_TEXT_LENGTH = 50000  # 50K字符
    MAX_FILENAME_LENGTH = 255
    MAX_REQUIREMENTS_LENGTH = 5000

    @staticmethod
    def sanitize_text(text: str, max_length: int = None, allow_empty: bool = False) -> str:
        """
        清理和验证文本输入

        Args:
            text: 输入文本
            max_length: 最大长度限制
            allow_empty: 是否允许空文本

        Returns:
            清理后的文本

        Raises:
            ValidationError: 验证失败时
        """
        if text is None:
            if allow_empty:
                return ""
            raise ValidationError("文本不能为空", field="text")

        # 去除首尾空白
        text = text.strip()

        if not text and not allow_empty:
            raise ValidationError("文本不能为空", field="text")

        # 限制长度
        if max_length is None:
            max_length = InputValidator.MAX_TEXT_LENGTH

        if len(text) > max_length:
            raise ValidationError(
                f"文本长度超过限制（最大 {max_length} 字符）",
                field="text"
            )

        # 清理HTML标签和潜在的恶意内容
        cleaned = bleach.clean(
            text,
            tags=InputValidator.ALLOWED_TAGS,
            attributes=InputValidator.ALLOWED_ATTRIBUTES,
            strip=True
        )

        return cleaned

    @staticmethod
    def validate_language(language: str) -> str:
        """
        验证语言代码

        Args:
            language: 语言代码或名称

        Returns:
            验证后的语言

        Raises:
            ValidationError: 无效的语言
        """
        if not language:
            raise ValidationError("语言不能为空", field="language")

        language = language.strip()

        # 导入语言列表（避免循环导入）
        from config.constants import SUPPORTED_LANGUAGES

        # 检查是否是有效的语言代码或名称
        valid_codes = {lang['code'] for lang in SUPPORTED_LANGUAGES}
        valid_names = {lang['name'] for lang in SUPPORTED_LANGUAGES}

        if language not in valid_codes and language not in valid_names:
            raise ValidationError(
                f"不支持的语言: {language}",
                field="language"
            )

        return language

    @staticmethod
    def validate_scene(scene_id: str) -> str:
        """
        验证翻译场景ID

        Args:
            scene_id: 场景ID

        Returns:
            验证后的场景ID

        Raises:
            ValidationError: 无效的场景
        """
        if not scene_id:
            # 返回默认场景
            return "general"

        scene_id = scene_id.strip()

        # 导入场景列表
        from config.constants import TRANSLATION_SCENES

        valid_scenes = {scene['id'] for scene in TRANSLATION_SCENES}

        if scene_id not in valid_scenes:
            raise ValidationError(
                f"不支持的翻译场景: {scene_id}",
                field="scene"
            )

        return scene_id

    @staticmethod
    def validate_filename(filename: str, allowed_extensions: set = None) -> str:
        """
        验证文件名

        Args:
            filename: 文件名
            allowed_extensions: 允许的扩展名集合

        Returns:
            验证后的文件名

        Raises:
            ValidationError: 无效的文件名
        """
        if not filename:
            raise ValidationError("文件名不能为空", field="filename")

        filename = filename.strip()

        # 检查长度
        if len(filename) > InputValidator.MAX_FILENAME_LENGTH:
            raise ValidationError(
                f"文件名过长（最大 {InputValidator.MAX_FILENAME_LENGTH} 字符）",
                field="filename"
            )

        # 检查非法字符
        illegal_chars = r'[<>:"/\\|?*\x00-\x1f]'
        if re.search(illegal_chars, filename):
            raise ValidationError(
                "文件名包含非法字符",
                field="filename"
            )

        # 检查扩展名
        if allowed_extensions:
            if '.' not in filename:
                raise ValidationError(
                    "文件名必须包含扩展名",
                    field="filename"
                )

            ext = filename.rsplit('.', 1)[1].lower()
            if ext not in allowed_extensions:
                raise ValidationError(
                    f"不支持的文件类型。允许的类型: {', '.join(allowed_extensions)}",
                    field="filename"
                )

        return filename

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        """
        验证会话ID格式

        Args:
            session_id: 会话ID

        Returns:
            验证后的会话ID

        Raises:
            ValidationError: 无效的会话ID
        """
        if not session_id:
            raise ValidationError("会话ID不能为空", field="session_id")

        session_id = session_id.strip()

        # UUID格式验证
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, session_id, re.IGNORECASE):
            raise ValidationError(
                "无效的会话ID格式",
                field="session_id"
            )

        return session_id

    @staticmethod
    def validate_page_number(page_number: Any, max_pages: int = None) -> int:
        """
        验证页码

        Args:
            page_number: 页码
            max_pages: 最大页数

        Returns:
            验证后的页码

        Raises:
            ValidationError: 无效的页码
        """
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            raise ValidationError(
                "页码必须是整数",
                field="page_number"
            )

        if page_number < 1:
            raise ValidationError(
                "页码必须大于0",
                field="page_number"
            )

        if max_pages and page_number > max_pages:
            raise ValidationError(
                f"页码超出范围（最大 {max_pages} 页）",
                field="page_number"
            )

        return page_number

    @staticmethod
    def validate_email(email: str) -> str:
        """
        验证邮箱地址

        Args:
            email: 邮箱地址

        Returns:
            验证后的邮箱

        Raises:
            ValidationError: 无效的邮箱
        """
        if not email:
            raise ValidationError("邮箱不能为空", field="email")

        email = email.strip().lower()

        # 简单的邮箱格式验证
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValidationError(
                "邮箱格式不正确",
                field="email"
            )

        return email

    @staticmethod
    def validate_request_data(data: Dict, required_fields: list, optional_fields: list = None) -> Dict:
        """
        验证请求数据

        Args:
            data: 请求数据字典
            required_fields: 必需字段列表
            optional_fields: 可选字段列表

        Returns:
            验证后的数据字典

        Raises:
            ValidationError: 验证失败
        """
        if data is None:
            raise ValidationError("请求数据不能为空")

        # 检查必需字段
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(
                f"缺少必需字段: {', '.join(missing_fields)}"
            )

        # 检查未知字段
        if optional_fields is None:
            optional_fields = []

        allowed_fields = set(required_fields) | set(optional_fields)
        unknown_fields = [field for field in data.keys() if field not in allowed_fields]

        if unknown_fields:
            raise ValidationError(
                f"包含未知字段: {', '.join(unknown_fields)}"
            )

        return data


class FileValidator:
    """文件验证器"""

    @staticmethod
    def validate_file_size(file_size: int, max_size: int = None) -> None:
        """
        验证文件大小

        Args:
            file_size: 文件大小（字节）
            max_size: 最大文件大小（字节）

        Raises:
            ValidationError: 文件过大或为空
        """
        if max_size is None:
            from config.constants import MAX_FILE_SIZE
            max_size = MAX_FILE_SIZE

        if file_size == 0:
            raise ValidationError("文件为空")

        if file_size > max_size:
            max_mb = max_size / (1024 * 1024)
            raise ValidationError(
                f"文件大小超过限制（最大 {max_mb:.1f}MB）"
            )

    @staticmethod
    def validate_mime_type(mime_type: str, allowed_types: set = None) -> None:
        """
        验证MIME类型

        Args:
            mime_type: 文件MIME类型
            allowed_types: 允许的MIME类型集合

        Raises:
            ValidationError: 不支持的文件类型
        """
        if allowed_types is None:
            allowed_types = {'application/pdf'}

        if mime_type not in allowed_types:
            raise ValidationError(
                f"不支持的文件类型: {mime_type}。允许的类型: {', '.join(allowed_types)}"
            )


# 便捷的验证装饰器
def validate_json_request(*required_fields, **optional_fields):
    """
    装饰器：验证JSON请求数据

    使用示例:
        @validate_json_request('text', 'language', requirements=str)
        def my_endpoint():
            data = request.json
            # data已经过验证
    """
    from functools import wraps
    from flask import request, jsonify

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    "error": "请求必须是JSON格式",
                    "code": "INVALID_CONTENT_TYPE"
                }), 400

            data = request.get_json()

            try:
                # 验证数据
                InputValidator.validate_request_data(
                    data,
                    list(required_fields),
                    list(optional_fields.keys())
                )

                return func(*args, **kwargs)

            except ValidationError as e:
                return jsonify(e.to_dict()), e.status_code

        return wrapper

    return decorator


if __name__ == '__main__':
    # 测试验证器
    try:
        # 测试文本验证
        text = InputValidator.sanitize_text("  Hello World  ")
        print(f"清理后的文本: '{text}'")

        # 测试语言验证
        lang = InputValidator.validate_language("en")
        print(f"验证后的语言: {lang}")

        # 测试文件名验证
        filename = InputValidator.validate_filename("document.pdf", {'pdf'})
        print(f"验证后的文件名: {filename}")

    except ValidationError as e:
        print(f"验证失败: {e.message}")
