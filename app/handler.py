import runpod
import base64
from typing import Dict, Any
from services.rag_service import RAGService
from services.dart_service import DartService
from services.image_service import ImageService
from services.bodyline_service import BodylineService
from core.errors import RAGError, DartError, ImageGenerationError
from PIL import Image
import io

# サービスのインスタンス化
rag_service = RAGService()
dart_service = DartService()
image_service = ImageService()
bodyline_service = BodylineService()

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
                joined_tags = [tag.strip() for tag in prompt.split("prompt:")[1].split(",")] + ["masterpiece", "high score", "great score", "absurdres"]
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
            image_result = await image_service.generate_image(
                tags=joined_tags,
                steps=steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                negative_prompt=negative_prompt,
                num_images=num_images
            )
            
            # 生成された画像を使ってボディライン生成
            output_size = bodyline_service.calculate_resize_dimensions(image_result["images"][0], 786)
            bodyline_result = await bodyline_service.generate_bodyline(
                control_images=image_result["images"],
                prompt="anime pose, girl, (white background:1.5), (monochrome:1.5), full body, sketch, eyes, breasts, (slim legs, skinny legs:1.2)",
                negative_prompt="(wings:1.6), (clothes:1.4), (garment:1.4), (lighting:1.4), (gray:1.4), (missing limb:1.4), (extra line:1.4), (extra limb:1.4), (extra arm:1.4), (extra legs:1.4), (hair:1.4), (bangs:1.4), (fringe:1.4), (forelock:1.4), (front hair:1.4), (fill:1.4), (ink pool:1.6)",
                num_inference_steps=20,
                guidance_scale=8,
                input_resolution=256,
                output_size=output_size
            )

            # Base64エンコード
            image_base64s = []
            for image in image_result["images"]:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_base64s.append(image_base64)

            bodyline_base64s = []
            for image in bodyline_result["images"]:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_base64s.append(image_base64)

            return {
                "images": image_base64s,
                "bodylines": bodyline_base64s,
                "generated_tags": joined_tags,
                "parameters": {
                    "image_parameters": image_result["parameters"],
                    "bodyline_parameters": bodyline_result["parameters"]
                }
            }
            
        except (RAGError, DartError, ImageGenerationError) as e:
            return {"error": f"Generation failed: {str(e)}"}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

import asyncio
async def main():
    test_data = {
        "input": {
            # "prompt": "prompt:original, 1girl, solo, (jumping:1.3), jacket, school uniform, pose, miniskirt, brown hair, blue eyes, pleated skirt, red footwear, red jacket, shoes, striped clothes, thighs, white thighhighs",
            "prompt": "本を読みながらジャンプをしている女の子のポーズ",
            "negative_prompt": "nsfw, sensitive, from behind, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, missing arms, extra arms, missing legs, extra legs, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry",
            "num_images": 2,
            "steps": 30,
            "cfg_scale": 10.0,
            "width": 832,
            "height": 1216
        }
    }
    result = await handler(test_data)
    print(result["parameters"])
    
    # 画像を保存
    for i, image in enumerate(result["images"]):
        image_data = base64.b64decode(image)
        with open(f"test_image_{i}.png", "wb") as f:
            f.write(image_data)
                
    # ボディラインを保存
    for i, image in enumerate(result["bodylines"]):
        image_data = base64.b64decode(image)
        with open(f"test_bodyline_{i}.png", "wb") as f:
            f.write(image_data)
            
if __name__ == "__main__":
    # runpod.serverless.start({"handler": handler})

    asyncio.run(main())