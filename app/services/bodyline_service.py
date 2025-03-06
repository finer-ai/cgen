from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline, UniPCMultistepScheduler
from diffusers.models import ControlNetModel
from typing import List, Dict, Any, Tuple
import io
import base64
from core.config import settings
import cv2
import numpy as np
import random
import utils.sd_utils

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

        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)

        del self.base_pipeline

    @staticmethod        
    def binarize_image(image: Image.Image) -> np.ndarray:
        image = np.array(image.convert('L'))
        # 色反転
        image = 255 - image
        
        # ヒストグラム平坦化
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # ガウシアンブラー適用
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # 適応的二値化
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -8)

        return binary_image
    
    @staticmethod
    def create_rgba_image(binary_image: np.ndarray, color: list) -> Image.Image:
        rgba_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, 0] = color[0]
        rgba_image[:, :, 1] = color[1]
        rgba_image[:, :, 2] = color[2]
        rgba_image[:, :, 3] = binary_image
        return Image.fromarray(rgba_image, 'RGBA')

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
        control_images: List[Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        input_resolution: int = 512,
        output_size: Tuple[int, int] = (768, 768),
        seeds: List[int] = None
    ) -> List[Dict[str, Any]]:
        """複数の画像からボディラインを生成"""
        if seeds is None:
            seeds = [random.randint(0, np.iinfo(np.int32).max) for _ in range(len(control_images))]
        
        # 入力画像のリサイズ
        input_size = self.calculate_resize_dimensions(control_images[0], input_resolution)
        resized_images = []
        for image in control_images:
            resized_image = image.resize(input_size, Image.Resampling.LANCZOS)
            resized_images.append(resized_image)

        # 各画像に対して個別にパイプラインを呼び出す
        images = []
        for i, control_image in enumerate(resized_images):
            generator = utils.sd_utils.seed_everything(seeds[i])
            result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
                image=[control_image] * len(self.controlnet_models),  # 各ControlNetモデル用に同じ画像を複製
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

            image = result.images[0]
            
            # 画像を二値化
            binary_image = self.binarize_image(image)
            # 二値化した画像をRGBA画像に変換
            rgba_image = self.create_rgba_image(binary_image, [0, 0, 0])
            images.append(rgba_image)
            
        
        return {
            "images": images,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "input_resolution": input_resolution,
                "output_size": output_size,
                "seeds": seeds
            }
        } 