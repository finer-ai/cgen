from typing import Dict, Any, Optional
import torch
from diffusers.models import AutoencoderKL
from core.config import settings
from core.errors import ImageGenerationError
import utils.sd_utils

# PyTorch settings for better performance and determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageService:
    """画像生成サービス"""
    
    def __init__(self):
        # """初期化"""
        # try:
            # Stable Diffusion XL モデルのロード
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16,
            )
            self.pipe = utils.sd_utils.load_pipeline(settings.SD_MODEL_PATH, device, vae=vae)
            
            # # シード値設定
            # seed = 42
            # generator = utils.sd_utils.seed_everything(seed)
            
            # スケジューラー設定
            self.pipe.scheduler = utils.sd_utils.get_scheduler(
                self.pipe.scheduler.config,
                "Euler a"
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
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        """タグから画像を生成"""
        try:
            # パラメータのデフォルト値設定
            steps = steps or 20
            guidance_scale = guidance_scale or 7.0
            width = width or 512
            height = height or 768
            negative_prompt = negative_prompt or "lowres, bad anatomy, bad hands, cropped, worst quality"
            
            # タグをプロンプトに変換
            prompt = ", ".join(tags)
            
            images = []
            for i in range(num_images):
                # 画像生成
                with torch.autocast("cuda"):
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
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
                    "num_images": num_images
                }
            }
        
        except Exception as e:
            raise ImageGenerationError(f"画像生成中にエラーが発生しました: {str(e)}") 