import os
import json
import base64
import requests
import gradio as gr
from PIL import Image
import io
from datetime import datetime
import time
import dotenv
from pathlib import Path
from templates import (
    TAG_CANDIDATE_GENERATION_TEMPLATE,
    TAG_NORMALIZATION_TEMPLATE,
    TAG_FILTER_TEMPLATE,
    TAG_WEIGHT_TEMPLATE
)

dotenv.load_dotenv()

def save_images(images: list, output_dir: Path, prefix: str = "generated", subfolder: str = None):
    """Base64エンコードされた画像リストを保存する"""
    # タイムスタンプ付きのディレクトリ名を作成
    output_dir = output_dir / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    for i, img_base64 in enumerate(images):
        img_data = base64.b64decode(img_base64)
        output_path = output_dir / f"{prefix}_{i}.png"
        
        with open(output_path, "wb") as f:
            f.write(img_data)
        saved_paths.append(output_path)
    
    return saved_paths

# テキストからプロンプトを生成するAPI呼び出し
def generate_prompt_from_text(text_prompt, tag_candidate_generation_template=None, 
                             tag_normalization_template=None, tag_filter_template=None, 
                             tag_weight_template=None):
    """
    テキスト説明からプロンプトを生成するRunPod API呼び出し関数
    
    Args:
        text_prompt: テキストプロンプト
        tag_candidate_generation_template: タグ候補生成テンプレート
        tag_normalization_template: タグ正規化テンプレート
        tag_filter_template: タグフィルターテンプレート
        tag_weight_template: タグ重み付けテンプレート
    
    Returns:
        生成されたプロンプト
    """
    if not text_prompt.strip():
        return "", "エラー: テキストプロンプトが空です。プロンプトを生成するためのテキストを入力してください。"
    
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    
    if not endpoint_id:
        return "", "エラー: Endpoint IDが設定されていません"
    
    if not api_key:
        return "", "エラー: API KeyまたはEndpoint IDが設定されていません"
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "mode": "prompt_only",  # プロンプト生成のみモード
            "prompt": text_prompt,
            "tag_candidate_generation_template": tag_candidate_generation_template,
            "tag_normalization_template": tag_normalization_template,
            "tag_filter_template": tag_filter_template,
            "tag_weight_template": tag_weight_template
        }
    }
    
    # Noneの値を持つキーを削除
    payload["input"] = {k: v for k, v in payload["input"].items() if v is not None}
    
    print(f"プロンプト生成APIリクエスト送信: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        # タイムアウトを2分に設定
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        status = result.get("status")
        
        if status == "IN_QUEUE" or status == "IN_PROGRESS":
            task_id = result.get("id")
            status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{task_id}"
            
            # セッション作成
            session = requests.Session()
            session.headers.update(headers)
            
            # 最大2分間ポーリング
            for _ in range(24):
                time.sleep(5)
                try:
                    status_response = session.get(status_url, timeout=(3.05, 10))
                    
                    if status_response.status_code != 200:
                        print(f"Unexpected status code: {status_response.status_code}")
                        time.sleep(2)
                        continue
                        
                    status_data = status_response.json()
                    current_status = status_data.get("status")
                    print(f"Current status: {current_status}")
                    
                    if current_status == "COMPLETED":
                        result = status_data
                        break
                    elif current_status in ["FAILED", "CANCELLED"]:
                        raise Exception(f"Task failed with status: {current_status}")
                        
                except Exception as e:
                    print(f"Error during status check: {e}")
                    time.sleep(2)
                    continue
            
            # セッションのクリーンアップ
            session.close()
        
        if result.get("status") != "COMPLETED":
            raise Exception(f"API処理エラー: {result}")
        
        output = result.get("output", {})
        generated_tags = output.get("generated_tags", [])
        
        if not generated_tags:
            return "", "エラー: タグを生成できませんでした。別のテキストで試してください。"
        
        # タグをカンマ区切りの文字列に変換
        prompt_string = ", ".join(generated_tags)
        
        return prompt_string, None
        
    except Exception as e:
        return "", f"エラーが発生しました: {str(e)}"

# RunPod APIを呼び出す関数を変更
def generate_images_from_prompt(generated_prompt, negative_prompt="", 
                guidance_scale=7.0, num_inference_steps=30, 
                width=512, height=768, num_images=1, is_random_seeds=True, seeds=None, 
                bodyline_prompt=None, bodyline_negative_prompt=None,
                bodyline_steps=20, bodyline_guidance_scale=8.0,
                bodyline_input_resolution=256, bodyline_output_size=786,
                is_random_bodyline_seeds=True, bodyline_seeds=None):
    """
    生成されたプロンプトから画像を生成する関数
    
    Args:
        generated_prompt: 生成済みのプロンプト（カンマ区切りのタグ）
        negative_prompt: ネガティブプロンプト
        guidance_scale: ガイダンススケール
        num_inference_steps: 推論ステップ数
        width: 画像の幅
        height: 画像の高さ
        num_images: 生成する画像の枚数
        seeds: 乱数シード
        bodyline_prompt: ボディライン生成用プロンプト
        bodyline_negative_prompt: ボディライン生成用ネガティブプロンプト
        bodyline_steps: ボディライン生成の推論ステップ数
        bodyline_guidance_scale: ボディライン生成のガイダンススケール
        bodyline_input_resolution: ボディライン生成の入力解像度
        bodyline_output_size: ボディライン生成の出力サイズ
        is_random_bodyline_seeds: ボディライン生成用シードをランダムにするかどうか
        bodyline_seeds: ボディライン生成用シード
        api_key: RunPod API Key
        endpoint_id: RunPod Endpoint ID
    
    Returns:
        生成された画像のリスト
    """
    if not generated_prompt.strip():
        return *[None]*4, "エラー: 生成プロンプトが空です。先にプロンプトを生成してください。", seeds, bodyline_seeds
    
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    
    if not endpoint_id:
        return *[None]*4, "エラー: Endpoint IDが設定されていません", seeds, bodyline_seeds
    
    if not api_key:
        return *[None]*4, "エラー: API KeyまたはEndpoint IDが設定されていません", seeds, bodyline_seeds
    
    if is_random_seeds:
        seeds = None
    else:
        seeds = seeds.split(",")
        seeds = [int(seed) for seed in seeds]
        if len(seeds) < num_images:
            # 最後のseedを繰り返し使用
            seeds = seeds + [seeds[-1]] * (num_images - len(seeds))

    if is_random_bodyline_seeds:
        bodyline_seeds = None
    else:
        bodyline_seeds = bodyline_seeds.split(",")
        bodyline_seeds = [int(seed) for seed in bodyline_seeds]
        if len(bodyline_seeds) < num_images:
            # 最後のseedを繰り返し使用
            bodyline_seeds = bodyline_seeds + [bodyline_seeds[-1]] * (num_images - len(bodyline_seeds))
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # タグリストの作成
    tags = [tag.strip() for tag in generated_prompt.split(",")]
    
    payload = {
        "input": {
            "mode": "image_only",  # 画像生成のみモード
            "tags": tags,
            "negative_prompt": negative_prompt,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "num_images": num_images,
            "seeds": seeds,
            "bodyline_prompt": bodyline_prompt,
            "bodyline_negative_prompt": bodyline_negative_prompt,
            "bodyline_steps": bodyline_steps,
            "bodyline_guidance_scale": bodyline_guidance_scale,
            "bodyline_input_resolution": bodyline_input_resolution,
            "bodyline_output_size": bodyline_output_size,
            "bodyline_seeds": bodyline_seeds
        }
    }
    
    # Noneの値を持つキーを削除
    payload["input"] = {k: v for k, v in payload["input"].items() if v is not None}
    
    print(f"画像生成APIリクエスト送信: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        # タイムアウトを10分に設定
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        
        result = response.json()
        status = result.get("status")
        
        if status == "IN_QUEUE" or status == "IN_PROGRESS":
            task_id = result.get("id")
            status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{task_id}"
            
            # セッション作成
            session = requests.Session()
            session.headers.update(headers)
            
            # 最大5分間ポーリング
            for _ in range(60):
                time.sleep(5)
                try:
                    # タイムアウトを細かく設定
                    status_response = session.get(
                        status_url,
                        timeout=(3.05, 10)  # (接続タイムアウト, 読み込みタイムアウト)
                    )
                    
                    if status_response.status_code != 200:
                        print(f"Unexpected status code: {status_response.status_code}")
                        time.sleep(2)
                        continue
                        
                    status_data = status_response.json()
                    current_status = status_data.get("status")
                    print(f"Current status: {current_status}")
                    
                    if current_status == "COMPLETED":
                        result = status_data
                        break
                    elif current_status in ["FAILED", "CANCELLED"]:
                        raise Exception(f"Task failed with status: {current_status}")
                        
                except requests.exceptions.Timeout:
                    print("Request timed out, retrying...")
                    time.sleep(2)
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"Request error: {e}")
                    time.sleep(2)
                    continue
                except json.JSONDecodeError:
                    print("Invalid JSON response, retrying...")
                    time.sleep(2)
                    continue
            
            # セッションのクリーンアップ
            session.close()
        
        if result.get("status") != "COMPLETED":
            raise Exception(f"API処理エラー: {result}")
        
        output = result.get("output", {})
        images = [None]*num_images

        # 画像の保存処理
        if "images" in output:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output")
            
            # 生成画像の保存
            save_images(
                output["images"],
                output_dir,
                prefix="generated",
                subfolder=timestamp
            )
            
            # ボディラインの保存
            save_images(
                output["bodylines"],
                output_dir,
                prefix="bodyline",
                subfolder=timestamp
            )
            
            # seedsの保存
            seeds = output["parameters"]["image_parameters"]["seeds"]
            
            # メタデータの保存
            metadata = {
                "used_tags": output.get("used_tags", []),
                "parameters": output.get("parameters", {})
            }
            
            metadata_path = output_dir / timestamp / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"Images and metadata saved to: {output_dir / timestamp}")
            
            # Gradio用の画像変換
            for i, (img_b64, bodyline_b64, diff_b64) in enumerate(zip(output["images"], output["bodylines"], output["remove_bg_diff"]["diff"])):
                img_data = base64.b64decode(img_b64.split(",")[1] if "," in img_b64 else img_b64)
                img = Image.open(io.BytesIO(img_data))
                bodyline_data = base64.b64decode(bodyline_b64.split(",")[1] if "," in bodyline_b64 else bodyline_b64)
                bodyline = Image.open(io.BytesIO(bodyline_data))
                diff_data = base64.b64decode(diff_b64.split(",")[1] if "," in diff_b64 else diff_b64)
                diff = Image.open(io.BytesIO(diff_data))
                images[i] = [bodyline, img, diff]
                
            print(output["remove_bg_diff"]["white_percentage"])
            print(output["remove_bg_diff"]["white_pixels_mask2"])
            print(output["remove_bg_diff"]["white_pixels_diff"])

        return (
            *images, 
            json.dumps(output["parameters"], indent=2, ensure_ascii=False), 
            ', '.join(str(seed) for seed in output["parameters"]["image_parameters"]["seeds"]),
            ', '.join(str(seed) for seed in output["parameters"]["bodyline_parameters"]["seeds"])
        )

    except Exception as e:
        return *[None]*4, f"エラーが発生しました: {str(e)}", seeds, bodyline_seeds

# テンプレートを再読み込みする関数を追加
def reload_templates():
    """テンプレートモジュールを再読み込みする"""
    import importlib
    import templates
    importlib.reload(templates)
    from templates import (
        TAG_CANDIDATE_GENERATION_TEMPLATE,
        TAG_NORMALIZATION_TEMPLATE,
        TAG_FILTER_TEMPLATE,
        TAG_WEIGHT_TEMPLATE
    )
    return (
        TAG_CANDIDATE_GENERATION_TEMPLATE,
        TAG_NORMALIZATION_TEMPLATE,
        TAG_FILTER_TEMPLATE,
        TAG_WEIGHT_TEMPLATE
    )

# Gradio UIの構築
def create_ui():
    with gr.Blocks() as app:
        gr.Markdown("# Image Generation Interface")
        
        with gr.Row():
            text_prompt = gr.Textbox(label="テキスト説明", placeholder="画像にしたい内容を自然な文章で説明してください...", lines=3)
            generate_prompt_btn = gr.Button("プロンプト生成", variant="secondary")
        
        with gr.Row():
            generated_prompt = gr.Textbox(label="生成されたプロンプト", placeholder="生成されたプロンプトがここに表示されます...", lines=3)
            generate_image_btn = gr.Button("画像生成", variant="primary")
        
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="nsfw, sensitive, from behind, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, missing arms, extra arms, missing legs, extra legs, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry",
                    lines=2
                )
            
            with gr.Row():
                guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=10.0, step=0.1, label="Guidance Scale")
                num_inference_steps = gr.Slider(minimum=10, maximum=100, value=30, step=1, label="Steps")
            
            with gr.Row():
                width = gr.Slider(minimum=512, maximum=2048, value=832, step=64, label="Width")
                height = gr.Slider(minimum=512, maximum=2048, value=1216, step=64, label="Height")
                num_images = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="Number of Images")
                
            with gr.Row():
                is_random_seeds = gr.Checkbox(label="Generate Random Seeds", value=True)
                seeds = gr.Textbox(label="Seeds (Random if empty)", placeholder="Enter seed values. For multiple seeds, separate with commas.")
            
            with gr.Accordion("Bodyline Settings", open=False):
                bodyline_prompt = gr.Textbox(
                    label="Bodyline Prompt",
                    value="anime pose, girl, (white background:1.5), (monochrome:1.5), full body, sketch, eyes, breasts, (slim legs, skinny legs:1.2)",
                    lines=2
                )
                bodyline_negative_prompt = gr.Textbox(
                    label="Bodyline Negative Prompt",
                    value="(wings:1.6), (clothes:1.4), (garment:1.4), (lighting:1.4), (gray:1.4), (missing limb:1.4), (extra line:1.4), (extra limb:1.4), (extra arm:1.4), (extra legs:1.4), (hair:1.4), (bangs:1.4), (fringe:1.4), (forelock:1.4), (front hair:1.4), (fill:1.4), (ink pool:1.6)",
                    lines=2
                )
                with gr.Row():
                    bodyline_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=8.0, step=0.1, label="Bodyline Guidance Scale")
                    bodyline_steps = gr.Slider(minimum=10, maximum=100, value=20, step=1, label="Bodyline Steps")
                with gr.Row():
                    bodyline_input_resolution = gr.Slider(minimum=128, maximum=1024, value=256, step=64, label="Bodyline Input Resolution")
                    bodyline_output_size = gr.Slider(minimum=512, maximum=2048, value=786, step=64, label="Bodyline Output Size")
                with gr.Row():
                    is_random_bodyline_seeds = gr.Checkbox(label="Generate Random Bodyline Seeds", value=True)
                    bodyline_seeds = gr.Textbox(label="Bodyline Seeds (Random if empty)", placeholder="Enter seed values. For multiple seeds, separate with commas.")
            
            with gr.Accordion("Tag Template Settings", open=False, visible=False):
                with gr.Row():
                    reload_templates_btn = gr.Button("テンプレートを再読み込み", variant="secondary")

                tag_candidate_generation_template = gr.Textbox(
                    label="Tag Candidate Generation Template", 
                    value=TAG_CANDIDATE_GENERATION_TEMPLATE,
                    lines=6
                )
                tag_normalization_template = gr.Textbox(
                    label="Tag Normalization Template", 
                    value=TAG_NORMALIZATION_TEMPLATE,
                    lines=6
                )
                tag_filter_template = gr.Textbox(
                    label="Tag Filter Template", 
                    value=TAG_FILTER_TEMPLATE,
                    lines=6
                )
                tag_weight_template = gr.Textbox(
                    label="Tag Weighting Template", 
                    value=TAG_WEIGHT_TEMPLATE,
                    lines=6
                )

        with gr.Row():
            output_gallery1 = gr.Gallery(label="Generated Results", columns=1, height=400, object_fit="contain")
            output_gallery2 = gr.Gallery(label="Generated Results", columns=1, height=400, object_fit="contain")
            output_gallery3 = gr.Gallery(label="Generated Results", columns=1, height=400, object_fit="contain")
            output_gallery4 = gr.Gallery(label="Generated Results", columns=1, height=400, object_fit="contain")
        status_text = gr.Textbox(label="Status", interactive=False)

        output_gallery = [output_gallery1, output_gallery2, output_gallery3, output_gallery4]
        
        # プロンプト生成ボタンのクリックイベント
        generate_prompt_btn.click(
            fn=generate_prompt_from_text,
            inputs=[
                text_prompt, tag_candidate_generation_template,
                tag_normalization_template, tag_filter_template,
                tag_weight_template
            ],
            outputs=[
                generated_prompt,
                status_text
            ]
        )
        
        # 画像生成ボタンのクリックイベント
        generate_image_btn.click(
            fn=generate_images_from_prompt,
            inputs=[
                generated_prompt, negative_prompt,
                guidance_scale, num_inference_steps,
                width, height, num_images, is_random_seeds, seeds,
                bodyline_prompt, bodyline_negative_prompt,
                bodyline_steps, bodyline_guidance_scale,
                bodyline_input_resolution, bodyline_output_size,
                is_random_bodyline_seeds, bodyline_seeds,
            ],
            outputs=[
                output_gallery[0],
                output_gallery[1],
                output_gallery[2],
                output_gallery[3],
                status_text,
                seeds,
                bodyline_seeds
            ]
        )

        # テンプレート再読み込みボタンのクリックイベント
        reload_templates_btn.click(
            fn=reload_templates,
            inputs=[],
            outputs=[
                tag_candidate_generation_template,
                tag_normalization_template,
                tag_filter_template,
                tag_weight_template
            ]
        )
    
    return app

# メイン実行部分
if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True) 