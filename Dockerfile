# ---------- 1️⃣ 前端构建 ----------
FROM node:22-trixie-slim AS frontend-builder

WORKDIR /app/one-eval-web

COPY one-eval-web/package*.json ./
RUN npm install

COPY one-eval-web/ /app/one-eval-web
RUN npm run build


# ---------- 2️⃣ 后端运行 ----------
FROM  python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    nginx \
    vim \
    wget \
    zip \
    unzip \
    procps


# ---------- 复制后端 ----------
COPY one_eval/ /app/one_eval
COPY pyproject.toml requirements.txt /app/

RUN pip install --upgrade \
    pip \
    setuptools \
    wheel

RUN pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install -e .

# ---------- 复制前端构建产物 ----------
COPY --from=frontend-builder /app/one-eval-web/dist /app/frontend_dist

# ---------- Nginx 配置 ----------
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN rm -f /etc/nginx/sites-enabled/default

# ---------- 启动脚本 ----------
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 80 8000 5173

CMD ["/app/docker-entrypoint.sh"]