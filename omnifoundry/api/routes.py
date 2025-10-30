"""
API 路由模組

定義 API 端點和路由。
"""

from fastapi import APIRouter


router = APIRouter()


@router.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "ok"}


@router.post("/v1/completions")
async def completions():
    """OpenAI 相容的 completions 端點"""
    pass


@router.post("/v1/chat/completions")
async def chat_completions():
    """OpenAI 相容的 chat completions 端點"""
    pass

