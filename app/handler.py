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
import logging
import os
from datetime import datetime
from model_downloader import download_models

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # パラメータの取得
        prompt = input_data.get("prompt")
        tag_candidate_generation_template = input_data.get("tag_candidate_generation_template", None)
        tag_normalization_template = input_data.get("tag_normalization_template", None)
        tag_filter_template = input_data.get("tag_filter_template", None)
        tag_weight_template = input_data.get("tag_weight_template", None)
        negative_prompt = input_data.get("negative_prompt", "")
        steps = input_data.get("steps", 30)
        cfg_scale = input_data.get("cfg_scale", 7.0)
        width = input_data.get("width", 512)
        height = input_data.get("height", 768)
        num_images = input_data.get("num_images", 1)
        bodyline_prompt = input_data.get("bodyline_prompt", "anime pose, girl, (white background:1.5), (monochrome:1.5), full body, sketch, eyes, breasts, (slim legs, skinny legs:1.2)")
        bodyline_negative_prompt = input_data.get("bodyline_negative_prompt", "(wings:1.6), (clothes:1.4), (garment:1.4), (lighting:1.4), (gray:1.4), (missing limb:1.4), (extra line:1.4), (extra limb:1.4), (extra arm:1.4), (extra legs:1.4), (hair:1.4), (bangs:1.4), (fringe:1.4), (forelock:1.4), (front hair:1.4), (fill:1.4), (ink pool:1.6)")

        try:
            if tag_candidate_generation_template:
                rag_service.set_tag_candidate_generation_template(tag_candidate_generation_template)
                print("tag_candidate_generation_template registered")
            if tag_normalization_template:
                rag_service.set_tag_normalization_template(tag_normalization_template)
                print("tag_normalization_template registered")
            if tag_filter_template:
                dart_service.set_tag_filter_template(tag_filter_template)
                print("tag_filter_template registered")
            if tag_weight_template:
                dart_service.set_tag_weight_template(tag_weight_template)
                print("tag_weight_template registered")

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
                prompt=bodyline_prompt,
                negative_prompt=bodyline_negative_prompt,
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
                bodyline_base64s.append(image_base64)

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

async def test():
    # タイムスタンプ付きのディレクトリ名を作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{timestamp}"
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

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
        with open(f"{output_dir}/image_{i}.png", "wb") as f:
            f.write(image_data)
                
    # ボディラインを保存
    for i, image in enumerate(result["bodylines"]):
        image_data = base64.b64decode(image)
        with open(f"{output_dir}/bodyline_{i}.png", "wb") as f:
            f.write(image_data)
            
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
    # asyncio.run(test())