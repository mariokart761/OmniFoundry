"""
API 伺服器模組

實作 FastAPI 伺服器。
"""

from fastapi import FastAPI


app = FastAPI(
    title="OmniFoundry API",
    description="開源模型推論環境集成程式 API",
    version="0.1.0"
)


def start_server(host="0.0.0.0", port=8000):
    """啟動 API 伺服器"""
    pass

