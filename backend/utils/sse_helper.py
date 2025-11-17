"""
Server-Sent Events (SSE) 辅助工具
提供统一的SSE流式响应生成和管理
"""

import json
import logging
from typing import Generator, Callable, Optional, Dict, Any
from flask import current_app, Response

logger = logging.getLogger(__name__)


class SSEGenerator:
    """SSE生成器 - 统一管理Server-Sent Events流"""

    def __init__(self, session_manager=None):
        """
        初始化SSE生成器

        Args:
            session_manager: 会话管理器实例
        """
        self.session_manager = session_manager
        self.logger = logging.getLogger(__name__)

    def generate(
            self,
            stream_generator: Generator,
            session_id: str,
            send_session_id: bool = True,
            on_chunk: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, None]:
        """
        生成SSE流

        Args:
            stream_generator: 数据流生成器
            session_id: 会话ID
            send_session_id: 是否在开始时发送会话ID
            on_chunk: 每个chunk的回调函数

        Yields:
            格式化的SSE消息
        """
        try:
            # 1. 发送会话ID
            if send_session_id:
                yield self._format_sse({'session_id': session_id})

            # 2. 流式传输数据
            for chunk in stream_generator:
                # 检查会话是否被取消
                if self.session_manager and self.session_manager.is_session_cancelled(session_id):
                    self.logger.info(f'会话 {session_id} 被用户取消')
                    yield self._format_sse({
                        'cancelled': True,
                        'message': '翻译已被用户中断'
                    })
                    break

                # 发送数据块
                if chunk:
                    yield self._format_sse({'content': chunk})

                    # 调用回调
                    if on_chunk:
                        try:
                            on_chunk(chunk)
                        except Exception as e:
                            self.logger.error(f'on_chunk回调失败: {str(e)}')

            # 3. 发送完成信号
            if not (self.session_manager and self.session_manager.is_session_cancelled(session_id)):
                yield self._format_sse({'done': True})

        except GeneratorExit:
            self.logger.warning(f'客户端断开连接 - 会话 {session_id}')
            if self.session_manager:
                self.session_manager.cancel_session(session_id)

        except Exception as e:
            self.logger.error(f'SSE生成错误 - 会话 {session_id}: {str(e)}', exc_info=True)
            yield self._format_sse({
                'error': str(e),
                'code': 'SSE_GENERATION_ERROR'
            })

        finally:
            # 4. 清理会话
            if self.session_manager:
                self.session_manager.finish_session(session_id)
                self.logger.debug(f'会话 {session_id} 已完成')

    def generate_with_progress(
            self,
            stream_generator: Generator[tuple, None, None],
            session_id: str,
            total: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        生成带进度信息的SSE流

        Args:
            stream_generator: 生成器，产出 (index, content, total) 元组
            session_id: 会话ID
            total: 总数（可选）

        Yields:
            格式化的SSE消息
        """
        try:
            # 发送会话ID
            yield self._format_sse({'session_id': session_id})

            # 流式传输带进度的数据
            for data in stream_generator:
                # 检查取消状态
                if self.session_manager and self.session_manager.is_session_cancelled(session_id):
                    yield self._format_sse({
                        'cancelled': True,
                        'message': '翻译已被用户中断'
                    })
                    break

                # 解析数据
                if isinstance(data, tuple) and len(data) >= 2:
                    index, content = data[0], data[1]
                    current_total = data[2] if len(data) > 2 else total

                    # 发送进度数据
                    yield self._format_sse({
                        'index': index,
                        'content': content,
                        'total': current_total,
                        'progress': (index + 1) / current_total * 100 if current_total else None
                    })
                else:
                    # 普通内容
                    yield self._format_sse({'content': str(data)})

            # 发送完成信号
            if not (self.session_manager and self.session_manager.is_session_cancelled(session_id)):
                yield self._format_sse({'done': True})

        except Exception as e:
            self.logger.error(f'SSE进度生成错误 - 会话 {session_id}: {str(e)}', exc_info=True)
            yield self._format_sse({
                'error': str(e),
                'code': 'SSE_PROGRESS_ERROR'
            })

        finally:
            if self.session_manager:
                self.session_manager.finish_session(session_id)

    @staticmethod
    def _format_sse(data: Dict[str, Any]) -> str:
        """
        格式化SSE消息

        Args:
            data: 数据字典

        Returns:
            格式化的SSE消息字符串
        """
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            return f"data: {json_data}\n\n"
        except Exception as e:
            logger.error(f'SSE格式化失败: {str(e)}')
            # 返回错误消息
            error_data = json.dumps({'error': 'SSE格式化失败'})
            return f"data: {error_data}\n\n"

    @staticmethod
    def create_response(
            generator: Generator[str, None, None],
            cors_origin: str = '*'
    ) -> Response:
        """
        创建SSE响应对象

        Args:
            generator: SSE生成器
            cors_origin: CORS允许的源

        Returns:
            Flask Response对象
        """
        return current_app.response_class(
            generator,
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',  # 禁用nginx缓冲
                'Access-Control-Allow-Origin': cors_origin,
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Access-Control-Allow-Methods': 'POST, OPTIONS, GET'
            }
        )

    def simple_stream(
            self,
            data_generator: Generator,
            event_type: str = 'message'
    ) -> Generator[str, None, None]:
        """
        简单的SSE流（不需要会话管理）

        Args:
            data_generator: 数据生成器
            event_type: 事件类型

        Yields:
            SSE消息
        """
        try:
            for data in data_generator:
                if isinstance(data, dict):
                    yield self._format_sse(data)
                else:
                    yield self._format_sse({'type': event_type, 'data': data})

            # 发送完成
            yield self._format_sse({'done': True})

        except Exception as e:
            self.logger.error(f'简单流生成错误: {str(e)}', exc_info=True)
            yield self._format_sse({
                'error': str(e),
                'code': 'SIMPLE_STREAM_ERROR'
            })


class SSEEvent:
    """SSE事件构建器"""

    def __init__(self, event_type: str = None):
        self.event_type = event_type
        self.data = None
        self.id = None
        self.retry = None

    def set_data(self, data: Any) -> 'SSEEvent':
        """设置事件数据"""
        self.data = data
        return self

    def set_id(self, event_id: str) -> 'SSEEvent':
        """设置事件ID"""
        self.id = event_id
        return self

    def set_retry(self, retry_ms: int) -> 'SSEEvent':
        """设置重试时间（毫秒）"""
        self.retry = retry_ms
        return self

    def format(self) -> str:
        """格式化为SSE消息"""
        lines = []

        if self.event_type:
            lines.append(f"event: {self.event_type}")

        if self.id:
            lines.append(f"id: {self.id}")

        if self.retry:
            lines.append(f"retry: {self.retry}")

        if self.data is not None:
            if isinstance(self.data, (dict, list)):
                data_str = json.dumps(self.data, ensure_ascii=False)
            else:
                data_str = str(self.data)

            # 支持多行数据
            for line in data_str.split('\n'):
                lines.append(f"data: {line}")

        lines.append('')  # 空行表示消息结束
        return '\n'.join(lines) + '\n'


# 便捷的装饰器
def sse_endpoint(session_manager=None, cors_origin='*'):
    """
    装饰器：将函数转换为SSE端点

    使用示例:
        @app.route('/stream')
        @sse_endpoint(session_manager=session_manager)
        def my_stream():
            def generator():
                for i in range(10):
                    yield f"Message {i}"
            return generator()
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 执行函数获取生成器
            result = func(*args, **kwargs)

            if not hasattr(result, '__iter__'):
                raise ValueError("SSE端点必须返回一个生成器")

            # 创建SSE生成器
            sse_gen = SSEGenerator(session_manager)

            # 如果返回的是元组 (generator, session_id)
            if isinstance(result, tuple) and len(result) == 2:
                generator, session_id = result
                sse_stream = sse_gen.generate(generator, session_id)
            else:
                # 简单流
                sse_stream = sse_gen.simple_stream(result)

            # 创建响应
            return SSEGenerator.create_response(sse_stream, cors_origin)

        return wrapper

    return decorator


if __name__ == '__main__':
    # 测试SSE生成器
    def test_generator():
        for i in range(5):
            yield f"测试消息 {i}"

    sse = SSEGenerator()

    print("测试简单流:")
    for msg in sse.simple_stream(test_generator()):
        print(msg, end='')

    print("\n测试SSE事件构建器:")
    event = SSEEvent('test') \
        .set_id('123') \
        .set_data({'message': 'Hello, SSE!'}) \
        .set_retry(3000)
    print(event.format())
