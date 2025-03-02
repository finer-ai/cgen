from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.models import ControlNetModel
from typing import List, Dict, Any
import io
import base64
from core.config import settings

class BodylineService:
    def __init__(self):
        # ControlNetモデルの初期化
        self.controlnet = ControlNetModel.from_single_file(
            settings.CONTROLNET_MODEL_PATH,
            torch_dtype=getattr(torch, settings.TORCH_DTYPE)
        )
        
        # SD 1.5 パイプラインの初期化
        self.pipeline = StableDiffusionControlNetPipeline.from_single_file(
            settings.SD15_MODEL_PATH,
            controlnet=self.controlnet,
            torch_dtype=getattr(torch, settings.TORCH_DTYPE)
        ).to(settings.DEVICE)

    async def resize_for_controlnet(self, image_data: str) -> Image.Image:
        """Base64画像をデコードしてControlNet用にリサイズ"""
        # Base64をデコード
        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # SD 1.5の制限に合わせてリサイズ（最大512x512）
        target_size = (512, 512)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image

    async def generate_bodyline(
        self,
        control_image: Image.Image,
        prompt: str = "1girl, simple background, white background",
        negative_prompt: str = "nsfw, nude, bad anatomy, bad proportions",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> Dict[str, Any]:
        """素体（ボディライン）を生成"""
        # 画像生成
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # 生成された画像をBase64に変換
        buffered = io.BytesIO()
        output.images[0].save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "image": image_base64,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
        } 