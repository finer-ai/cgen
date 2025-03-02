import runpod
import base64
from typing import Dict, Any
from services.rag_service import RAGService
from services.dart_service import DartService
from services.image_service import ImageService
from core.errors import RAGError, DartError, ImageGenerationError
from utils.llm_utils import load_llm

# サービスのインスタンス化
rag_service = RAGService()
dart_service = DartService()
image_service = ImageService()

async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPodのハンドラー関数"""
    try:
        # 入力データの取得
        input_data = event["input"]
        prompt = input_data.get("prompt")
        negative_prompt = input_data.get("negative_prompt", "")
        steps = input_data.get("steps", 30)
        cfg_scale = input_data.get("cfg_scale", 7.0)
        width = input_data.get("width", 512)
        height = input_data.get("height", 768)
        num_images = input_data.get("num_images", 1)
        
        # バリデーション
        if not prompt:
            return {
                "error": "No prompt provided"
            }

        try:
            if prompt.startswith("prompt:"):
              joined_tags = [tag.strip() for tag in prompt.split("prompt:")[1].split(",")]
            else:
              # RAGでタグ候補を取得
              tag_candidates = await rag_service.generate_tag_candidates(prompt)
              print("tag_candidates", tag_candidates)
              
              # Dartでタグを補完
              final_tags = await dart_service.generate_final_tags(tag_candidates)
              print("final_tags", final_tags)

              # コンテキストに基づいてタグをフィルタリング
              filtered_tags = await dart_service.filter_tags_by_context(
                  tags_str=", ".join(final_tags),
                  context_prompt=prompt
              )
              quality_tags = ["masterpiece", "high score", "great score", "absurdres"]
              joined_tags = filtered_tags + quality_tags
            print("joined_tags", joined_tags)

            # 画像生成
            result = await image_service.generate_image(
                tags=joined_tags,
                steps=steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                negative_prompt=negative_prompt,
                num_images=num_images
            )
            

            return {
                "images": result["images"],
                "generated_tags": joined_tags,
                "parameters": {
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "width": width,
                    "height": height,
                    "negative_prompt": negative_prompt,
                    "num_images": num_images
                }
            }
            
        except (RAGError, DartError, ImageGenerationError) as e:
            return {"error": f"Generation failed: {str(e)}"}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 