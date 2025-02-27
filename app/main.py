import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from core.config import settings
from core.errors import RAGError, DartError, ImageGenerationError

# ロギング設定
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("anime-generator")

# FastAPIアプリケーション
app = FastAPI(
    title=settings.APP_NAME,
    description="日本語からDanbooruタグを生成し、アニメ調画像を生成するAPI",
    version="1.0.0",
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターをマウント
app.include_router(router, prefix="/api")

# グローバル例外ハンドラ
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"グローバル例外: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "サーバーエラーが発生しました", "detail": str(exc)},
    )

@app.exception_handler(RAGError)
async def rag_exception_handler(request: Request, exc: RAGError):
    logger.error(f"RAGエラー: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "タグ抽出エラー", "detail": str(exc)},
    )

@app.exception_handler(DartError)
async def dart_exception_handler(request: Request, exc: DartError):
    logger.error(f"Dartエラー: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "タグ補完エラー", "detail": str(exc)},
    )

@app.exception_handler(ImageGenerationError)
async def image_exception_handler(request: Request, exc: ImageGenerationError):
    logger.error(f"画像生成エラー: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "画像生成エラー", "detail": str(exc)},
    )

@app.get("/")
async def root():
    return {"message": "アニメ画像生成APIが稼働中です"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG) 