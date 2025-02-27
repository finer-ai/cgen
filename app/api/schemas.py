from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PromptRequest(BaseModel):
    """プロンプトリクエスト"""
    prompt: str = Field(..., description="生成したい画像の説明（日本語）")

class TagsResponse(BaseModel):
    """タグ生成レスポンス"""
    tags: List[str] = Field(..., description="生成されたDanbooruタグのリスト")

class ImageGenerationRequest(BaseModel):
    """画像生成リクエスト"""
    tags: List[str] = Field(..., description="使用するDanbooruタグのリスト")
    steps: Optional[int] = Field(None, description="生成ステップ数")
    cfg_scale: Optional[float] = Field(None, description="CFGスケール")
    width: Optional[int] = Field(None, description="画像幅")
    height: Optional[int] = Field(None, description="画像高さ")
    negative_prompt: Optional[str] = Field(None, description="ネガティブプロンプト")

class ImageResponse(BaseModel):
    """画像生成レスポンス"""
    image_base64: str = Field(..., description="Base64エンコードされた画像")
    prompt: str = Field(..., description="使用されたプロンプト")
    parameters: Dict[str, Any] = Field(..., description="生成パラメータ")

class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    error: str = Field(..., description="エラーメッセージ")
    detail: Optional[str] = Field(None, description="詳細情報") 