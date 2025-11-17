"""
自定义异常类
定义应用中使用的所有自定义异常
"""


class TranslationError(Exception):
    """翻译相关错误基类"""

    def __init__(self, message: str, code: str = "TRANSLATION_ERROR", status_code: int = 400):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self):
        """转换为字典格式，便于JSON响应"""
        return {
            "error": self.message,
            "code": self.code
        }


class ValidationError(TranslationError):
    """输入验证错误"""

    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message, "VALIDATION_ERROR", 400)

    def to_dict(self):
        result = super().to_dict()
        if self.field:
            result["field"] = self.field
        return result


class LLMAPIError(TranslationError):
    """LLM API调用错误"""

    def __init__(self, message: str, provider: str = None):
        self.provider = provider
        super().__init__(message, "LLM_API_ERROR", 502)

    def to_dict(self):
        result = super().to_dict()
        if self.provider:
            result["provider"] = self.provider
        return result


class PDFProcessingError(TranslationError):
    """PDF处理错误"""

    def __init__(self, message: str, filename: str = None):
        self.filename = filename
        super().__init__(message, "PDF_PROCESSING_ERROR", 400)

    def to_dict(self):
        result = super().to_dict()
        if self.filename:
            result["filename"] = self.filename
        return result


class FileUploadError(TranslationError):
    """文件上传错误"""

    def __init__(self, message: str):
        super().__init__(message, "FILE_UPLOAD_ERROR", 400)


class SessionError(TranslationError):
    """会话管理错误"""

    def __init__(self, message: str, session_id: str = None):
        self.session_id = session_id
        super().__init__(message, "SESSION_ERROR", 400)

    def to_dict(self):
        result = super().to_dict()
        if self.session_id:
            result["session_id"] = self.session_id
        return result


class RateLimitError(TranslationError):
    """速率限制错误"""

    def __init__(self, message: str = "请求过于频繁，请稍后再试"):
        super().__init__(message, "RATE_LIMIT_ERROR", 429)


class AuthenticationError(TranslationError):
    """认证错误"""

    def __init__(self, message: str = "认证失败"):
        super().__init__(message, "AUTHENTICATION_ERROR", 401)


class AuthorizationError(TranslationError):
    """授权错误"""

    def __init__(self, message: str = "权限不足"):
        super().__init__(message, "AUTHORIZATION_ERROR", 403)


class ConfigurationError(TranslationError):
    """配置错误"""

    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(message, "CONFIGURATION_ERROR", 500)

    def to_dict(self):
        result = super().to_dict()
        if self.config_key:
            result["config_key"] = self.config_key
        return result
