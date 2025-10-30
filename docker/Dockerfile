# OmniFoundry Dockerfile
# 多階段建構優化映像大小

# 第一階段：建構階段
FROM python:3.10-slim as builder

# 設定工作目錄
WORKDIR /build

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 複製專案檔案
COPY requirements.txt .
COPY setup.py .
COPY README.md .
COPY omnifoundry/ ./omnifoundry/

# 安裝 Python 依賴
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user -e .

# 第二階段：執行階段
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝必要的執行時依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 從建構階段複製已安裝的套件
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build /app

# 設定環境變數
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# 建立模型緩存目錄
RUN mkdir -p /app/models

# 暴露 API 埠號
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 預設命令
CMD ["omnifoundry", "--help"]

