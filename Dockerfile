# 使用官方Python运行时作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=8888
ENV HOST=0.0.0.0


# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --default-timeout=60 --retries 5 -r requirements.txt

# 复制应用代码
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8888

# 健康检查（使用 Python 而非 curl，避免额外依赖）
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os,sys,urllib.request;port=os.environ.get('PORT','5000');\
from urllib.error import URLError,HTTPError;\
url=f'http://localhost:{port}/api/health';\
try:\
    resp=urllib.request.urlopen(url,timeout=5);\
    sys.exit(0 if getattr(resp,'status',200)==200 else 1)\
except (URLError,HTTPError):\
    sys.exit(1)" 

# 启动命令（Gunicorn + gevent）
CMD ["gunicorn", "-w", "2", "-k", "gevent", "-t", "60", "-b", "0.0.0.0:8888", "app:app"]