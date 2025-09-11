# AI翻译应用前端 Dockerfile
# 多阶段构建：第一阶段用于构建，第二阶段用于运行

# 第一阶段：构建阶段
FROM node:18-alpine AS builder

# 设置工作目录
WORKDIR /app

# 仅复制包管理文件以充分利用缓存
COPY package.json package-lock.json ./

# 安装全部依赖（构建阶段需要构建工具和脚本）
RUN npm ci

# 复制必要的项目文件
COPY public ./public
COPY src ./src
COPY tsconfig.json ./

# 如果存在其他配置文件（不会中断构建）
COPY .env* ./ 2>/dev/null || true
COPY *.config.* ./ 2>/dev/null || true

# 构建应用
RUN npm run build

# 第二阶段：运行阶段
FROM nginx:alpine

# 复制自定义nginx配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 从构建阶段复制构建好的文件到nginx目录
COPY --from=builder /app/build /usr/share/nginx/html

# 暴露端口
EXPOSE 80

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]
