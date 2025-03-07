from typing import Dict, Any, Optional
import torch
from diffusers.models import AutoencoderKL
from core.config import settings
from core.errors import ImageGenerationError
import utils.sd_utils
import gc
import random
import numpy as np

# PyTorch settings for better performance and determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageService:
    """画像生成サービス"""
    # def __init__(self):
    #     # """初期化"""
    #     # Stable Diffusion XL モデルのロード
    #     self.pipe = StableDiffusionXLPipeline.from_single_file(
    #         settings.SD_MODEL_PATH,
    #         use_safetensors=True,
    #         torch_dtype=torch.float16,
    #         variant="fp16"
    #     )
        
    #     # スケジューラー設定
    #     self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    #         self.pipe.scheduler.config
    #     )
        
    #     # GPUに移動
    #     self.pipe = self.pipe.to(device)
        
    #     # VAEをcache
    #     self.pipe.enable_vae_tiling()
        
    #     # メモリ効率化
    #     self.pipe.enable_xformers_memory_efficient_attention()
        
    def __init__(self):
        # """初期化"""
        self.pipe = utils.sd_utils.load_pipeline("animagine-xl-4.0", device)
        
        # スケジューラー設定
        self.pipe.scheduler = utils.sd_utils.get_scheduler(
            self.pipe.scheduler.config,
            "Euler a"
        )
        
        # GPUに移動
        self.pipe = self.pipe.to(device)
        
        # VAEをcache
        self.pipe.enable_vae_tiling()
        
        # メモリ効率化
        self.pipe.enable_xformers_memory_efficient_attention()
    
    async def generate_image(
        self,
        tags: list[str],
        steps: Optional[int] = 20,
        guidance_scale: Optional[float] = 7.0,
        width: Optional[int] = 512,
        height: Optional[int] = 768,
        negative_prompt: Optional[str] = "lowres, bad anatomy, bad hands, cropped, worst quality",
        num_images: Optional[int] = 1,
        seeds: Optional[list[int]] = None,
    ) -> Dict[str, Any]:
        """タグから画像を生成"""
        try:
            # Memory management
            torch.cuda.empty_cache()
            gc.collect()
            
            if seeds is None:
                seeds = [random.randint(0, np.iinfo(np.int32).max) for _ in range(num_images)]

            # タグをプロンプトに変換
            prompt = ", ".join(tags)
            
            images = []
            for i in range(num_images):
                generator = utils.sd_utils.seed_everything(seeds[i])
                # 画像生成
                with torch.autocast("cuda"):
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        generator=generator,
                    )
            
                # 画像の取得
                image = result.images[0]
                images.append(image)

            # 結果を返却
            return {
                "images": images,
                "generated_tags": tags,
                "parameters": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "num_images": num_images,
                    "seeds": seeds
                }
            }
        
        except Exception as e:
            raise ImageGenerationError(f"画像生成中にエラーが発生しました: {str(e)}") 