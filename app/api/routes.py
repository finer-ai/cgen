from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any

from api.schemas import (
    PromptRequest, TagsResponse, 
    ImageGenerationRequest, ImageResponse,
    ErrorResponse
)
from services.rag_service import RAGService
from services.dart_service import DartService
from services.image_service import ImageService
from core.errors import RAGError, DartError, ImageGenerationError

router = APIRouter()

# サービスのシングルトンインスタンス
rag_service = RAGService()
dart_service = DartService()
image_service = ImageService()

@router.post("/generate-tags", response_model=TagsResponse)
async def generate_tags(request: PromptRequest) -> Dict[str, Any]:
    """ユーザープロンプトからDanbooruタグを生成"""
    try:
        # RAGでタグ候補を取得
        tag_candidates = await rag_service.generate_tag_candidates(request.prompt)
        
        # Dartでタグを補完
        final_tags = await dart_service.generate_final_tags(tag_candidates)
        
        return {"tags": final_tags}
    
    except RAGError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"タグ候補生成エラー: {str(e)}"
        )
    except DartError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"タグ補完エラー: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"予期しないエラー: {str(e)}"
        )

@router.post("/generate-image", response_model=ImageResponse)
async def generate_image(request: ImageGenerationRequest) -> Dict[str, Any]:
    """タグから画像を生成"""
    try:
        # 画像生成
        result = await image_service.generate_image(
            tags=request.tags,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            width=request.width,
            height=request.height,
            negative_prompt=request.negative_prompt
        )
        
        return result
    
    except ImageGenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"画像生成エラー: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"予期しないエラー: {str(e)}"
        )

@router.post("/generate-from-prompt", response_model=ImageResponse)
async def generate_from_prompt(request: PromptRequest) -> Dict[str, Any]:
    """プロンプトから直接画像を生成（タグ生成+画像生成の統合エンドポイント）"""
    try:
        # RAGでタグ候補を取得
        tag_candidates = await rag_service.generate_tag_candidates(request.prompt)
        
        # Dartでタグを補完
        final_tags = await dart_service.generate_final_tags(tag_candidates)
        
        # 画像生成
        result = await image_service.generate_image(tags=final_tags)
        
        # タグ情報を結果に追加
        result["tags"] = final_tags
        
        return result
    
    except RAGError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"タグ候補生成エラー: {str(e)}"
        )
    except DartError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"タグ補完エラー: {str(e)}"
        )
    except ImageGenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"画像生成エラー: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"予期しないエラー: {str(e)}"
        ) 