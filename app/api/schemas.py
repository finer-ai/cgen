from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PromptRequest(BaseModel):
    """プロンプトリクエストのスキーマ"""
    prompt: str = Field(..., description="生成のためのプロンプト")
    negative_prompt: Optional[str] = Field(default="", description="ネガティブプロンプト")
    steps: Optional[int] = Field(default=30, description="生成ステップ数")
    guidance_scale: Optional[float] = Field(default=7.0, description="ガイダンススケール")
    width: Optional[int] = Field(default=512, description="生成画像の幅")
    height: Optional[int] = Field(default=768, description="生成画像の高さ")
    num_images: Optional[int] = Field(default=1, description="生成する画像の数")

class TagsResponse(BaseModel):
    """タグ生成レスポンスのスキーマ"""
    tags: List[str] = Field(..., description="生成されたタグのリスト")

class ImageGenerationRequest(BaseModel):
    """画像生成リクエストのスキーマ"""
    tags: List[str] = Field(..., description="生成に使用するタグのリスト")
    steps: Optional[int] = Field(default=30, description="生成ステップ数")
    guidance_scale: Optional[float] = Field(default=7.0, description="ガイダンススケール")
    width: Optional[int] = Field(default=512, description="生成画像の幅")
    height: Optional[int] = Field(default=768, description="生成画像の高さ")
    negative_prompt: Optional[str] = Field(default="", description="ネガティブプロンプト")
    num_images: Optional[int] = Field(default=1, description="生成する画像の数")

class ImageResponse(BaseModel):
    """画像生成レスポンスのスキーマ"""
    images: List[str] = Field(..., description="生成された画像のBase64エンコードされたリスト")
    bodylines: List[str] = Field(..., description="生成された画像のボディラインのリスト")
    generated_tags: Optional[List[str]] = Field(default=None, description="生成に使用されたタグのリスト")
    parameters: Dict[str, Any] = Field(..., description="生成パラメータ")

class ErrorResponse(BaseModel):
    """エラーレスポンスのスキーマ"""
    detail: str = Field(..., description="エラーの詳細メッセージ") 