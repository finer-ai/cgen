from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
from diffusers.models import ControlNetModel
from typing import List, Dict, Any
import io
import base64
from core.config import settings

class BodylineService:
    def __init__(self):
        # SD 1.5 ベースモデルの初期化
        self.base_pipeline = StableDiffusionPipeline.from_single_file(
            settings.SD15_MODEL_PATH,
            torch_dtype=getattr(torch, settings.TORCH_DTYPE),
            use_safetensors=True
        ).to(settings.DEVICE)
        
        # ControlNetモデルの初期化
        self.controlnet_models = []
        self.controlnet_scales = []
        for config in settings.CONTROLNET_CONFIGS:
            controlnet = ControlNetModel.from_single_file(
                config["path"],
                torch_dtype=getattr(torch, settings.TORCH_DTYPE)
            ).to(settings.DEVICE)
            self.controlnet_models.append(controlnet)
            self.controlnet_scales.append(config["conditioning_scale"])

        # マルチControlNetパイプラインの作成
        self.pipeline = StableDiffusionControlNetPipeline(
            vae=self.base_pipeline.vae,
            text_encoder=self.base_pipeline.text_encoder,
            tokenizer=self.base_pipeline.tokenizer,
            unet=self.base_pipeline.unet,
            scheduler=self.base_pipeline.scheduler,
            safety_checker=self.base_pipeline.safety_checker,
            feature_extractor=self.base_pipeline.feature_extractor,
            controlnet=self.controlnet_models
        ).to(settings.DEVICE)

        # 不要になったベースパイプラインを解放
        del self.base_pipeline

    @staticmethod
    def calculate_resize_dimensions(image: Image.Image, max_long_side: int) -> tuple[int, int]:
        """画像の長辺を指定サイズに合わせた時の縦横サイズを計算する

        Args:
            image (Image.Image): 元画像
            max_long_side (int): リサイズ後の長辺の長さ

        Returns:
            tuple[int, int]: (width, height)のタプル
        """
        width, height = image.size
        if width >= height:
            # 横長の場合
            new_width = max_long_side
            new_height = int(height * (max_long_side / width))
        else:
            # 縦長の場合
            new_height = max_long_side
            new_width = int(width * (max_long_side / height))
        
        return (new_width, new_height)

    async def resize_for_controlnet(
        self,
        image_data: str,
        target_size: tuple[int, int] = (512, 512)
    ) -> Image.Image:
        """Base64画像をデコードしてリサイズ

        Args:
            image_data (str): Base64エンコードされた画像データ
            target_size (tuple[int, int], optional): リサイズ後のサイズ(width, height). Defaults to (512, 512).

        Returns:
            Image.Image: リサイズされた画像
        """
        # Base64をデコード
        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # 指定サイズにリサイズ
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image

    async def generate_bodyline(
        self,
        control_image: Image.Image,
        prompt: str = "1girl, simple background, white background",
        negative_prompt: str = "nsfw, nude, bad anatomy, bad proportions",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        output_size: tuple[int, int] = (512, 512)
    ) -> Dict[str, Any]:
        """素体（ボディライン）を生成

        Args:
            control_image (Image.Image): 制御用の入力画像
            prompt (str, optional): 生成時の指示プロンプト
            negative_prompt (str, optional): 生成時の否定プロンプト
            num_inference_steps (int, optional): 推論ステップ数
            guidance_scale (float, optional): ガイダンススケール
            output_size (tuple[int, int], optional): 生成する画像のサイズ(width, height). Defaults to (512, 512).

        Returns:
            Dict[str, Any]: 生成された画像とパラメータ
        """
        # 画像生成
        output = self.pipeline(
            prompt=prompt,
            image=[control_image] * len(self.controlnet_models),
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=output_size[0],
            height=output_size[1],
            denoising_strength=0.13,
            num_images_per_prompt=1,
            guess_mode=[True] * len(self.controlnet_models),
            controlnet_conditioning_scale=self.controlnet_scales,
            guidance_start=[0.0] * len(self.controlnet_models),
            guidance_end=[1.0] * len(self.controlnet_models)
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
                "guidance_scale": guidance_scale,
                "output_size": output_size
            }
        } 