from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv

# 导入新的工具模块
from utils.logger import setup_logger
from utils.exceptions import TranslationError, ValidationError, FileUploadError

# 加载环境变量
load_dotenv()


def create_app(config_name=None):
    """
    创建并配置Flask应用

    Args:
        config_name: 配置名称 ('development', 'production', 'testing')

    Returns:
        配置好的Flask应用实例
    """
    app = Flask(__name__)

    # ========== 基础配置 ==========
    _configure_app(app, config_name)

    # ========== 日志系统 ==========
    setup_logger(app)
    app.logger.info(f'应用启动 - 环境: {app.config.get("ENV", "unknown")}')

    # ========== CORS配置 ==========
    _configure_cors(app)

    # ========== 错误处理 ==========
    _register_error_handlers(app)

    # ========== 请求钩子 ==========
    _register_request_hooks(app)

    # ========== 注册蓝图 ==========
    _register_blueprints(app)

    app.logger.info('应用初始化完成')

    return app


def _configure_app(app, config_name):
    """配置应用参数"""
    # 基础配置
    app.config['ENV'] = os.getenv('FLASK_ENV', 'production')
    app.config['DEBUG'] = app.config['ENV'] == 'development'
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

    # 文件上传配置
    upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = upload_folder
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

    # JSON配置
    app.config['JSON_AS_ASCII'] = False  # 支持中文
    app.config['JSON_SORT_KEYS'] = False  # 不排序键

    # 会话配置
    app.config['SESSION_COOKIE_SECURE'] = app.config['ENV'] == 'production'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'


def _configure_cors(app):
    """配置CORS"""
    # 从环境变量读取允许的源
    allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')

    # 清理空白字符
    allowed_origins = [origin.strip() for origin in allowed_origins]

    CORS(
        app,
        origins=allowed_origins,
        methods=['GET', 'POST', 'OPTIONS'],
        allow_headers=['Content-Type', 'Authorization'],
        max_age=3600,
        supports_credentials=True
    )

    app.logger.info(f'CORS配置完成 - 允许的源: {allowed_origins}')


def _register_error_handlers(app):
    """注册错误处理器"""

    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """处理验证错误"""
        app.logger.warning(f'验证错误: {error.message}')
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(FileUploadError)
    def handle_file_upload_error(error):
        """处理文件上传错误"""
        app.logger.warning(f'文件上传错误: {error.message}')
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(TranslationError)
    def handle_translation_error(error):
        """处理翻译相关错误"""
        app.logger.error(f'翻译错误: {error.message}')
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(400)
    def handle_bad_request(error):
        """处理400错误"""
        return jsonify({
            "error": "请求参数错误",
            "code": "BAD_REQUEST",
            "details": str(error)
        }), 400

    @app.errorhandler(404)
    def handle_not_found(error):
        """处理404错误"""
        return jsonify({
            "error": "请求的资源不存在",
            "code": "NOT_FOUND"
        }), 404

    @app.errorhandler(413)
    def handle_request_entity_too_large(error):
        """处理文件过大错误"""
        max_size = app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024)
        return jsonify({
            "error": f"文件大小超过限制（最大 {max_size:.1f}MB）",
            "code": "FILE_TOO_LARGE"
        }), 413

    @app.errorhandler(429)
    def handle_rate_limit(error):
        """处理速率限制错误"""
        return jsonify({
            "error": "请求过于频繁，请稍后再试",
            "code": "RATE_LIMIT_EXCEEDED"
        }), 429

    @app.errorhandler(500)
    def handle_internal_error(error):
        """处理500错误"""
        app.logger.error(f'服务器内部错误: {str(error)}', exc_info=True)
        return jsonify({
            "error": "服务器内部错误，请稍后重试",
            "code": "INTERNAL_ERROR"
        }), 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """处理未预期的异常"""
        app.logger.error(f'未处理的异常: {str(error)}', exc_info=True)

        # 生产环境不暴露详细错误信息
        if app.config['ENV'] == 'production':
            return jsonify({
                "error": "服务器遇到未知错误",
                "code": "UNEXPECTED_ERROR"
            }), 500
        else:
            return jsonify({
                "error": "服务器遇到未知错误",
                "code": "UNEXPECTED_ERROR",
                "details": str(error),
                "type": type(error).__name__
            }), 500


def _register_request_hooks(app):
    """注册请求钩子"""

    @app.before_request
    def log_request_info():
        """记录请求信息"""
        from flask import request
        # 访问日志已在logger.py中通过中间件处理
        pass

    @app.after_request
    def add_security_headers(response):
        """添加安全响应头"""
        # 安全头
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'

        # 如果是生产环境，添加HSTS
        if app.config['ENV'] == 'production':
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        return response

    @app.teardown_appcontext
    def teardown(error=None):
        """清理资源"""
        if error:
            app.logger.error(f'请求清理时发生错误: {str(error)}')


def _register_blueprints(app):
    """注册蓝图"""
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    app.logger.info('蓝图注册完成')


# 用于开发调试的命令
def _register_cli_commands(app):
    """注册CLI命令（可选）"""

    @app.cli.command()
    def test():
        """运行测试"""
        import pytest
        pytest.main(['-v', 'tests/'])

    @app.cli.command()
    def clean_uploads():
        """清理上传目录中的旧文件"""
        import time
        upload_dir = app.config['UPLOAD_FOLDER']
        now = time.time()
        cutoff = now - (7 * 24 * 60 * 60)  # 7天前

        for filename in os.listdir(upload_dir):
            filepath = os.path.join(upload_dir, filename)
            if os.path.isfile(filepath):
                if os.path.getmtime(filepath) < cutoff:
                    os.remove(filepath)
                    app.logger.info(f'删除旧文件: {filename}')