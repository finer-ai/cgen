import runpod
import base64
from typing import Dict, Any
from services.prompt_service import PromptService
from services.dart_service import DartService
from services.image_service import ImageService
from core.config import settings
from utils.llm_utils import load_llm

# サービスのインスタンス化
llm = load_llm()
dart_service = DartService(llm)
prompt_service = PromptService(llm)
image_service = ImageService()

async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPodのハンドラー関数"""
    try:
        # 入力データの取得
        input_data = event["input"]
        prompt = input_data.get("prompt")
        negative_prompt = input_data.get("negative_prompt", "")
        num_images = input_data.get("num_images", 1)
        
        # バリデーション
        if not prompt:
            return {
                "error": "No prompt provided"
            }

        # プロンプトからタグを生成
        try:
            # タグの生成と重み付け
            tags = await prompt_service.generate_tags(prompt)
            weighted_tags = await dart_service.filter_tags_by_context(
                tags_str=", ".join(tags),
                context_prompt=prompt
            )
            
            # 画像生成
            images = await image_service.generate_images(
                tags=weighted_tags,
                negative_prompt=negative_prompt,
                num_images=num_images
            )
            
            # 画像をBase64エンコード
            encoded_images = []
            for img in images:
                # バイト列をBase64エンコード
                img_base64 = base64.b64encode(img).decode('utf-8')
                encoded_images.append(img_base64)
            
            return {
                "generated_tags": weighted_tags,
                "images": encoded_images
            }
            
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 