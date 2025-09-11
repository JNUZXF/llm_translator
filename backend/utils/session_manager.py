"""
翻译会话管理器
用于管理活动的翻译会话，支持中断功能
"""
import threading
import time
import uuid
from typing import Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TranslationSession:
    """翻译会话数据类"""
    session_id: str
    created_at: datetime
    is_active: bool = True
    is_cancelled: bool = False
    
class SessionManager:
    """翻译会话管理器"""
    
    def __init__(self, cleanup_interval: int = 300):  # 5分钟清理一次
        self._sessions: Dict[str, TranslationSession] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread = None
        self._running = False
        self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """启动清理线程"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._running = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
    
    def stop_cleanup_thread(self):
        """停止清理线程"""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)
    
    def create_session(self) -> str:
        """创建新的翻译会话"""
        session_id = str(uuid.uuid4())
        session = TranslationSession(
            session_id=session_id,
            created_at=datetime.now()
        )
        
        with self._lock:
            self._sessions[session_id] = session
        
        return session_id
    
    def cancel_session(self, session_id: str) -> bool:
        """取消指定的翻译会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_active:
                session.is_cancelled = True
                session.is_active = False
                return True
            return False
    
    def is_session_cancelled(self, session_id: str) -> bool:
        """检查会话是否被取消"""
        with self._lock:
            session = self._sessions.get(session_id)
            return session.is_cancelled if session else False
    
    def is_session_active(self, session_id: str) -> bool:
        """检查会话是否活跃"""
        with self._lock:
            session = self._sessions.get(session_id)
            return session.is_active if session else False
    
    def finish_session(self, session_id: str):
        """完成翻译会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.is_active = False
    
    def get_active_sessions(self) -> Set[str]:
        """获取所有活跃的会话ID"""
        with self._lock:
            return {sid for sid, session in self._sessions.items() if session.is_active}
    
    def get_session_count(self) -> int:
        """获取会话总数"""
        with self._lock:
            return len(self._sessions)
    
    def _cleanup_worker(self):
        """清理工作线程"""
        while self._running:
            try:
                self._cleanup_expired_sessions()
                time.sleep(self._cleanup_interval)
            except Exception as e:
                print(f"会话清理线程出错: {e}")
                time.sleep(60)  # 出错时等待1分钟再继续
    
    def _cleanup_expired_sessions(self):
        """清理过期的会话"""
        current_time = datetime.now()
        expired_threshold = timedelta(hours=1)  # 1小时后过期
        
        with self._lock:
            expired_sessions = []
            for session_id, session in self._sessions.items():
                if current_time - session.created_at > expired_threshold:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
            
            if expired_sessions:
                print(f"清理了 {len(expired_sessions)} 个过期会话")

# 全局会话管理器实例
session_manager = SessionManager()

# 在应用关闭时清理
import atexit
atexit.register(session_manager.stop_cleanup_thread)
