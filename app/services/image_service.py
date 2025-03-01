from typing import Dict, Any, Optional
import os
import uuid
import base64
import io
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

from core.config import settings
from core.errors import ImageGenerationError

class ImageService:
    """画像生成サービス"""
    
    def __init__(self):
        # """初期化"""
        # try:
            # Stable Diffusion XL モデルのロード
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                settings.SD_MODEL_PATH,
                use_safetensors=True,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            # スケジューラー設定
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # GPUに移動
            self.pipe = self.pipe.to("cuda")
            
            # VAEをcache
            self.pipe.enable_vae_tiling()
            
            # メモリ効率化
            self.pipe.enable_xformers_memory_efficient_attention()
        
        # except Exception as e:
        #     raise ImageGenerationError(f"画像生成モデルの読み込みに失敗しました: {str(e)}")
    
    async def generate_image(
        self,
        tags: list[str],
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        """タグから画像を生成"""
        try:
            # パラメータのデフォルト値設定
            steps = steps or settings.DEFAULT_STEPS
            cfg_scale = cfg_scale or settings.DEFAULT_CFG_SCALE
            width = width or settings.DEFAULT_WIDTH
            height = height or settings.DEFAULT_HEIGHT
            negative_prompt = negative_prompt or "lowres, bad anatomy, bad hands, cropped, worst quality"
            
            # タグをプロンプトに変換
            prompt = ", ".join(tags)
            
            image_base64_list = []
            for i in range(num_images):
                # 画像生成
                with torch.autocast("cuda"):
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        width=width,
                        height=height,
                    )
            
                # 画像の取得
                image = result.images[0]
                
                # Base64エンコード
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_base64_list.append(image_base64)
            
            # 結果を返却
            return {
                "image_base64_list": image_base64_list,
                "prompt": prompt,
                "parameters": {
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "width": width,
                    "height": height,
                }
            }
        
        except Exception as e:
            raise ImageGenerationError(f"画像生成中にエラーが発生しました: {str(e)}") 