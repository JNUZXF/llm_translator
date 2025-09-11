#!/bin/bash

# AI翻译应用Docker部署脚本
# 使用方法: ./scripts/deploy.sh [环境类型]
# 环境类型: dev (开发) | prod (生产)，默认为dev

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取环境类型参数
ENV_TYPE=${1:-dev}

log_info "🚀 开始部署AI翻译应用 (环境: $ENV_TYPE)"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    log_error "Docker未安装，请先安装Docker Desktop"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 检查环境变量文件
if [ ! -f "backend/.env" ]; then
    log_warning "环境变量文件不存在，正在创建..."
    if [ -f "env.template" ]; then
        cp env.template backend/.env
        log_info "已复制环境变量模板到 backend/.env"
        log_warning "请编辑 backend/.env 文件，填入真实的API密钥后再运行此脚本"
        exit 1
    else
        log_error "env.template文件不存在，无法创建环境变量文件"
        exit 1
    fi
fi

# 选择docker-compose文件
if [ "$ENV_TYPE" = "prod" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_warning "生产环境配置文件不存在，使用默认配置"
        COMPOSE_FILE="docker-compose.yml"
    fi
else
    COMPOSE_FILE="docker-compose.yml"
fi

log_info "使用配置文件: $COMPOSE_FILE"

# 停止现有服务
log_info "🛑 停止现有服务..."
docker-compose -f $COMPOSE_FILE down || true

# 拉取最新镜像依赖
log_info "📦 拉取基础镜像..."
docker-compose -f $COMPOSE_FILE pull --ignore-pull-failures || true

# 构建应用镜像
log_info "🏗️ 构建应用镜像..."
docker-compose -f $COMPOSE_FILE build --no-cache

# 启动服务
log_info "🚀 启动服务..."
docker-compose -f $COMPOSE_FILE up -d

# 等待服务启动
log_info "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
log_info "🔍 检查服务状态..."
docker-compose -f $COMPOSE_FILE ps

# 健康检查
log_info "🩺 执行健康检查..."

# 检查后端健康状态
BACKEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/health || echo "000")
if [ "$BACKEND_HEALTH" = "200" ]; then
    log_success "后端服务健康检查通过"
else
    log_error "后端服务健康检查失败 (HTTP $BACKEND_HEALTH)"
    log_info "查看后端日志:"
    docker-compose -f $COMPOSE_FILE logs --tail=20 backend
    exit 1
fi

# 检查前端健康状态
FRONTEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/health || echo "000")
if [ "$FRONTEND_HEALTH" = "200" ]; then
    log_success "前端服务健康检查通过"
else
    log_warning "前端健康检查失败，但这可能是正常的（nginx配置可能不同）"
fi

# 显示服务信息
log_success "🎉 部署完成！"
echo ""
echo "服务访问地址:"
echo "  前端应用: http://localhost:3000"
echo "  后端API:  http://localhost:5000"
echo "  健康检查: http://localhost:5000/api/health"
echo ""
echo "常用命令:"
echo "  查看日志: docker-compose -f $COMPOSE_FILE logs -f"
echo "  停止服务: docker-compose -f $COMPOSE_FILE down"
echo "  重启服务: docker-compose -f $COMPOSE_FILE restart"
echo ""
log_info "如有问题，请查看 DOCKER_DEPLOYMENT_GUIDE.md 获取详细说明"
