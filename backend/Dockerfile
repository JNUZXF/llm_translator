# AI翻译应用后端 Dockerfile
# 基于Python 3.11官方镜像（满足 scipy>=1.15 要求）
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖（用于构建与运行、健康检查）
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    portaudio19-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt文件
COPY requirements.txt .

# 升级pip并安装Python依赖（尽量使用预编译wheel）
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建uploads目录
RUN mkdir -p uploads

# 暴露端口
EXPOSE 5000

# 设置启动命令
CMD ["python", "app.py"]
