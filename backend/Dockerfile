# AI翻译应用后端 Dockerfile
# 基于Python 3.9官方镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖（用于某些Python包的编译）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建uploads目录
RUN mkdir -p uploads

# 暴露端口
EXPOSE 5000

# 设置启动命令
CMD ["python", "app.py"]
