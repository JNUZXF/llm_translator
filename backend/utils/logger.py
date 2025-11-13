"""
日志系统配置
提供统一的日志管理，支持文件日志、控制台日志和日志轮转
"""

import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime


def setup_logger(app=None, log_level=None):
    """
    配置应用日志系统

    Args:
        app: Flask应用实例（可选）
        log_level: 日志级别（可选），默认从环境变量读取

    Returns:
        配置好的logger实例
    """
    # 确定日志级别
    if log_level is None:
        log_level_str = os.getenv('LOG_LEVEL', 'INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 日志格式
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s.%(funcName)s (%(filename)s:%(lineno)d): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # 1. 主应用日志 - 按大小轮转
    app_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10,
        encoding='utf-8'
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(detailed_formatter)

    # 2. 错误日志 - 单独记录
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # 3. 访问日志 - 按天轮转
    access_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, 'access.log'),
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    access_handler.setLevel(logging.INFO)
    access_handler.setFormatter(simple_formatter)
    access_handler.suffix = '%Y%m%d'

    # 4. 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if os.getenv('FLASK_ENV') == 'development' else logging.INFO)
    console_handler.setFormatter(console_formatter)

    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除现有handlers（避免重复）
    root_logger.handlers.clear()

    # 添加handlers
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(access_handler)
    root_logger.addHandler(console_handler)

    # 如果提供了Flask app，也配置app.logger
    if app:
        app.logger.handlers.clear()
        app.logger.addHandler(app_handler)
        app.logger.addHandler(error_handler)
        app.logger.addHandler(console_handler)
        app.logger.setLevel(log_level)

        # 添加请求日志中间件
        @app.before_request
        def log_request():
            from flask import request
            access_logger = logging.getLogger('access')
            access_logger.info(f'{request.remote_addr} - {request.method} {request.path}')

        @app.after_request
        def log_response(response):
            from flask import request
            access_logger = logging.getLogger('access')
            access_logger.info(
                f'{request.remote_addr} - {request.method} {request.path} - '
                f'{response.status_code} - {response.content_length or 0} bytes'
            )
            return response

    # 禁用第三方库的过多日志
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f'日志系统初始化完成，日志目录: {log_dir}')

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger

    Args:
        name: logger名称，通常使用 __name__

    Returns:
        Logger实例
    """
    return logging.getLogger(name)


class LoggerContext:
    """日志上下文管理器，用于临时修改日志级别"""

    def __init__(self, logger_name: str, level: int):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


# 便捷的日志装饰器
def log_function_call(logger=None):
    """
    装饰器：记录函数调用

    使用示例:
        @log_function_call()
        def my_function(arg1, arg2):
            pass
    """
    import functools

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f'调用函数 {func.__name__}')
            try:
                result = func(*args, **kwargs)
                logger.debug(f'函数 {func.__name__} 执行成功')
                return result
            except Exception as e:
                logger.error(f'函数 {func.__name__} 执行失败: {str(e)}', exc_info=True)
                raise

        return wrapper

    return decorator


def log_execution_time(logger=None):
    """
    装饰器：记录函数执行时间

    使用示例:
        @log_execution_time()
        def slow_function():
            pass
    """
    import functools
    import time

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f'函数 {func.__name__} 执行时间: {execution_time:.3f}秒')
            return result

        return wrapper

    return decorator


if __name__ == '__main__':
    # 测试日志系统
    setup_logger()
    logger = get_logger(__name__)

    logger.debug('这是一条调试信息')
    logger.info('这是一条普通信息')
    logger.warning('这是一条警告信息')
    logger.error('这是一条错误信息')

    # 测试装饰器
    @log_function_call()
    @log_execution_time()
    def test_function():
        import time
        time.sleep(0.1)
        return "完成"

    result = test_function()
    print(f'结果: {result}')
