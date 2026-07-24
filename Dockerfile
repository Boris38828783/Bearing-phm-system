# ---------- 基础镜像 ----------
# 使用官方 Python 3.10 精简版（slim 版比完整版小很多）
FROM python:3.10-slim

# ---------- 元信息 ----------
LABEL maintainer="Zhenqi Shi"
LABEL description="Bearing fault diagnosis inference service"

# ---------- 工作目录 ----------
# 容器内的工作目录，后续命令都在这个目录下执行
WORKDIR /app

# ---------- 安装依赖 ----------
# 先只复制依赖清单再安装，可以利用 Docker 的层缓存：
# 只要 requirements 没变，重新构建时就不用重装依赖，速度快很多
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# ---------- 复制代码和模型 ----------
COPY app/ ./app/
COPY models/cnn1d_noise_aug.onnx ./models/

# ---------- 暴露端口 ----------
EXPOSE 8000

# ---------- 健康检查 ----------
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ---------- 启动命令 ----------
# 注意 host 必须是 0.0.0.0，不能是 127.0.0.1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]