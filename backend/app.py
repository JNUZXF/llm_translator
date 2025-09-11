#!/usr/bin/env python3
"""
AI翻译应用后端服务
"""

import logging
from app import create_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = create_app()

if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )