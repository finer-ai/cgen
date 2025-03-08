import runpod
import base64
from typing import Dict, Any, List, Tuple
from services.rag_service import RAGService
from services.dart_service import DartService
from services.image_service import ImageService
from services.bodyline_service import BodylineService
from services.remove_bg_service import RemoveBGService
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

# モデルダウンロード 開発時
download_models()

# サービスのインスタンス化
rag_service = RAGService()
dart_service = DartService()
image_service = ImageService()
bodyline_service = BodylineService()
remove_bg_service = RemoveBGService(cpu_only=False)

async def generate_prompt_tags(
    prompt: str,
    tag_candidate_generation_template: str = None,
    tag_normalization_template: str = None,
    tag_filter_template: str = None,
    tag_weight_template: str = None
) -> List[str]:
    """プロンプトからタグを生成する関数"""
    # テンプレートの設定
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

    # プロンプト処理
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
    return joined_tags

async def generate_images(
    tags: List[str],
    negative_prompt: str = "",
    steps: int = 30,
    guidance_scale: float = 7.0,
    width: int = 512,
    height: int = 768,
    num_images: int = 1,
    seeds: List[int] = None,
    bodyline_prompt: str = "anime pose, girl, (white background:1.5), (monochrome:1.5), full body, sketch, eyes, breasts, (slim legs, skinny legs:1.2)",
    bodyline_negative_prompt: str = "(wings:1.6), (clothes:1.4), (garment:1.4), (lighting:1.4), (gray:1.4), (missing limb:1.4), (extra line:1.4), (extra limb:1.4), (extra arm:1.4), (extra legs:1.4), (hair:1.4), (bangs:1.4), (fringe:1.4), (forelock:1.4), (front hair:1.4), (fill:1.4), (ink pool:1.6)",
    bodyline_steps: int = 20,
    bodyline_guidance_scale: float = 8.0,
    bodyline_input_resolution: int = 256,
    bodyline_output_size: int = 786,
    bodyline_seeds: List[int] = None
) -> Dict[str, Any]:
    """タグから画像とボディラインを生成する関数"""
    # 画像生成
    image_result = await image_service.generate_image(
        tags=tags,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        num_images=num_images,
        seeds=seeds
    )
    
    # 生成された画像を使ってボディライン生成
    output_size = bodyline_service.calculate_resize_dimensions(image_result["images"][0], bodyline_output_size)
    bodyline_result = await bodyline_service.generate_bodyline(
        control_images=image_result["images"],
        prompt=bodyline_prompt,
        negative_prompt=bodyline_negative_prompt,
        num_inference_steps=bodyline_steps,
        guidance_scale=bodyline_guidance_scale,
        input_resolution=bodyline_input_resolution,
        output_size=output_size,
        seeds=bodyline_seeds
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
        "parameters": {
            "image_parameters": image_result["parameters"],
            "bodyline_parameters": bodyline_result["parameters"]
        }
    }

async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPodのハンドラー関数
    
    mode:
        - "prompt_only": プロンプト生成のみ実行
        - "image_only": 画像生成のみ実行（tagsパラメータが必要）
        - "remove_bg_diff": 背景除去と差分計算を実行
        - "full" または未指定: プロンプト生成と画像生成の両方を実行
    """
    # 入力の検証
    if not event or not event.get("input"):
        return {"error": "入力が不正です"}

    input_data = event["input"]
    mode = input_data.get("mode", "full").lower()
    
    # 結果を格納する辞書
    result = {}
    
    # プロンプト生成モードまたは両方実行モードの場合
    if mode in ["prompt_only", "full"]:
        # プロンプト関連パラメータの取得
        prompt = input_data.get("prompt")
        tag_candidate_generation_template = input_data.get("tag_candidate_generation_template", None)
        tag_normalization_template = input_data.get("tag_normalization_template", None)
        tag_filter_template = input_data.get("tag_filter_template", None)
        tag_weight_template = input_data.get("tag_weight_template", None)
        
        # プロンプトからタグを生成
        joined_tags = await generate_prompt_tags(
            prompt=prompt,
            tag_candidate_generation_template=tag_candidate_generation_template,
            tag_normalization_template=tag_normalization_template,
            tag_filter_template=tag_filter_template,
            tag_weight_template=tag_weight_template
        )
        
        # 結果に生成されたタグを追加
        result["generated_tags"] = joined_tags
        
        # プロンプト生成のみの場合はここで終了
        if mode == "prompt_only":
            return result
    
    # 画像生成モードまたは両方実行モードの場合
    if mode in ["image_only", "full"]:
        # 画像生成関連パラメータの取得
        negative_prompt = input_data.get("negative_prompt", "")
        steps = input_data.get("steps", 30)
        guidance_scale = input_data.get("guidance_scale", 7.0)
        width = input_data.get("width", 512)
        height = input_data.get("height", 768)
        num_images = input_data.get("num_images", 1)
        seeds = input_data.get("seeds", None)

        # ボディライン生成関連パラメータの取得
        bodyline_prompt = input_data.get("bodyline_prompt", "anime pose, girl, (white background:1.5), (monochrome:1.5), full body, sketch, eyes, breasts, (slim legs, skinny legs:1.2)")
        bodyline_negative_prompt = input_data.get("bodyline_negative_prompt", "(wings:1.6), (clothes:1.4), (garment:1.4), (lighting:1.4), (gray:1.4), (missing limb:1.4), (extra line:1.4), (extra limb:1.4), (extra arm:1.4), (extra legs:1.4), (hair:1.4), (bangs:1.4), (fringe:1.4), (forelock:1.4), (front hair:1.4), (fill:1.4), (ink pool:1.6)")
        bodyline_steps = input_data.get("bodyline_steps", 20)
        bodyline_guidance_scale = input_data.get("bodyline_guidance_scale", 8.0)
        bodyline_input_resolution = input_data.get("bodyline_input_resolution", 256)
        bodyline_output_size = input_data.get("bodyline_output_size", 786)
        bodyline_seeds = input_data.get("bodyline_seeds", None)
        
        # 画像生成のみの場合は、入力からタグを取得
        if mode == "image_only":
            tags = input_data.get("tags", [])
            if not tags:
                return {"error": "画像生成のみモードの場合、'tags'パラメータが必要です"}
        else:
            # 両方実行モードの場合は、プロンプト生成で得られたタグを使用
            tags = joined_tags
        
        # タグから画像とボディラインを生成
        image_result = await generate_images(
            tags=tags,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images=num_images,
            seeds=seeds,
            bodyline_prompt=bodyline_prompt,
            bodyline_negative_prompt=bodyline_negative_prompt,
            bodyline_steps=bodyline_steps,
            bodyline_guidance_scale=bodyline_guidance_scale,
            bodyline_input_resolution=bodyline_input_resolution,
            bodyline_output_size=bodyline_output_size,
            bodyline_seeds=bodyline_seeds
        )
        
        # 結果に画像生成結果を追加
        result["images"] = image_result["images"]
        result["bodylines"] = image_result["bodylines"]
        result["parameters"] = image_result["parameters"]
        
        # 画像生成のみの場合は使用したタグも追加
        if mode == "image_only":
            result["used_tags"] = tags
    
    
    # 背景除去と差分計算モード
    if mode in ["remove_bg_diff", "image_only", "full"]:
        if mode == "remove_bg_diff":
            if not input_data.get("image1") or not input_data.get("image2"):
                return {"error": "image1とimage2の両方が必要です"}
            image1_data = input_data["images1"]
            image2_data = input_data["images2"]
        else:
            image1_data = result["images"]
            image2_data = result["bodylines"]
            
        # 背景除去と差分計算
        diff_images = []
        white_percentage = []
        white_pixels_mask2 = []
        white_pixels_diff = []
        for i in range(len(image1_data)):
            res = remove_bg_service.process_images(image1_data[i], image2_data[i])
            diff_images.append(res["diff"])
            white_percentage.append(res["white_percentage"])
            white_pixels_mask2.append(res["white_pixels_mask2"])
            white_pixels_diff.append(res["white_pixels_diff"])
        print(white_percentage)
        print(white_pixels_mask2)
        print(white_pixels_diff)
        result["remove_bg_diff"] = {
            "diff": diff_images,
            "white_percentage": white_percentage,
            "white_pixels_mask2": white_pixels_mask2,
            "white_pixels_diff": white_pixels_diff
        }

    # モードが不正な場合
    else:
        return {"error": "不正なモードです。'prompt_only', 'image_only', 'remove_bg_diff', または 'full'を指定してください。"}
    
    # 最終結果を返却
    return result

async def test():
    # タイムスタンプ付きのディレクトリ名を作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{timestamp}"
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    test_data = {
        "input": {
            "prompt": "本を読みながらジャンプをしている女の子のポーズ",
            "negative_prompt": "nsfw, sensitive, from behind, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, missing arms, extra arms, missing legs, extra legs, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry",
            "num_images": 2,
            "steps": 30,
            "guidance_scale": 10.0,
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