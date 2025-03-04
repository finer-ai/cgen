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

# RunPod APIを呼び出す関数
def call_runpod(prompt, negative_prompt="", tag_candidate_generation_template=None, 
                tag_normalization_template=None, tag_filter_template=None, 
                tag_weight_template=None, guidance_scale=7.0, num_inference_steps=30, 
                width=512, height=768, num_images=1, is_random_seeds=True, seeds=None, api_key="", endpoint_id=""):
    """
    RunPod APIを呼び出して画像を生成する関数
    
    Args:
        prompt: 生成プロンプト
        negative_prompt: ネガティブプロンプト
        tag_candidate_generation_template: タグ候補生成テンプレート
        tag_normalization_template: タグ正規化テンプレート
        tag_filter_template: タグフィルターテンプレート
        tag_weight_template: タグ重み付けテンプレート
        guidance_scale: ガイダンススケール（CFGスケール）
        num_inference_steps: 推論ステップ数
        width: 画像の幅
        height: 画像の高さ
        num_images: 生成する画像の枚数
        seeds: 乱数シード
        api_key: RunPod API Key
        endpoint_id: RunPod Endpoint ID
    
    Returns:
        生成された画像のリスト
    """
    if not api_key:
        api_key = os.environ.get("RUNPOD_API_KEY", "")
    
    if not endpoint_id:
        endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    
    if not api_key or not endpoint_id:
        raise ValueError("API KeyまたはEndpoint IDが設定されていません")
    
    if is_random_seeds:
        seeds = None
    else:
        seeds = seeds.split(",")
        seeds = [int(seed) for seed in seeds]
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompt": prompt,
            "tag_candidate_generation_template": tag_candidate_generation_template,
            "tag_normalization_template": tag_normalization_template,
            "tag_filter_template": tag_filter_template,
            "tag_weight_template": tag_weight_template,
            "negative_prompt": negative_prompt,
            "num_images": num_images,
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
            "width": width,
            "height": height,
            "seeds": seeds
        }
    }
    
    # Noneの値を持つキーを削除
    payload["input"] = {k: v for k, v in payload["input"].items() if v is not None}
    
    print(f"APIリクエスト送信: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        # タイムアウトを10分に設定
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        
        result = response.json()
        status = result.get("status")
        
        if status == "IN_QUEUE" or status == "IN_PROGRESS":
            task_id = result.get("id")
            status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{task_id}"
            
            # 最大10分間ポーリング
            for _ in range(60):
                time.sleep(10)
                status_response = requests.get(status_url, headers=headers)
                status_data = status_response.json()
                print(status_data.get("status"))
                if status_data.get("status") == "COMPLETED":
                    result = status_data
                    break
        
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
            
            # メタデータの保存
            metadata = {
                "generated_tags": output.get("generated_tags", []),
                "parameters": output.get("parameters", {})
            }
            
            metadata_path = output_dir / timestamp / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"Images and metadata saved to: {output_dir / timestamp}")
            
            # Gradio用の画像変換
            for i, (img_b64, bodyline_b64) in enumerate(zip(output["images"], output["bodylines"])):
                img_data = base64.b64decode(img_b64.split(",")[1] if "," in img_b64 else img_b64)
                img = Image.open(io.BytesIO(img_data))
                bodyline_data = base64.b64decode(bodyline_b64.split(",")[1] if "," in bodyline_b64 else bodyline_b64)
                bodyline = Image.open(io.BytesIO(bodyline_data))
                images[i] = [img, bodyline]

        return *images, json.dumps(output["parameters"], indent=2, ensure_ascii=False), None #', '.join(output["parameters"]["seeds"])

    except Exception as e:
        return *[None]*4, f"エラーが発生しました: {str(e)}", None

# Gradio UIの構築
def create_ui():
    with gr.Blocks() as app:
        gr.Markdown("# Image Generation Interface")
        
        with gr.Row():
            prompt = gr.Textbox(label="生成プロンプト", placeholder="画像生成のためのプロンプトを入力してください...", lines=3)
            submit_btn = gr.Button("生成開始", variant="primary")
        
        with gr.Accordion("詳細設定", open=False):
            with gr.Row():
                negative_prompt = gr.Textbox(
                    label="ネガティブプロンプト",
                    value="nsfw, sensitive, from behind, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, missing arms, extra arms, missing legs, extra legs, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry",
                    lines=2
                )
            
            with gr.Row():
                guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=10.0, step=0.1, label="CFGスケール")
                num_inference_steps = gr.Slider(minimum=10, maximum=100, value=30, step=1, label="ステップ数")
            
            with gr.Row():
                width = gr.Slider(minimum=512, maximum=2048, value=832, step=64, label="幅")
                height = gr.Slider(minimum=512, maximum=2048, value=1216, step=64, label="高さ")
                num_images = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="生成枚数")
                
            with gr.Row():
                is_random_seeds = gr.Checkbox(label="シード値をランダムに生成", value=True)
                seeds = gr.Textbox(label="シード値 (未入力の場合はランダム)", placeholder="シード値を入力してください。複数の場合はカンマ区切りで入力してください。")
            
            with gr.Accordion("タグテンプレート設定", open=False):
                tag_candidate_generation_template = gr.Textbox(
                    label="タグ候補生成テンプレート", 
                    value=TAG_CANDIDATE_GENERATION_TEMPLATE,
                    lines=6
                )
                tag_normalization_template = gr.Textbox(
                    label="タグ正規化テンプレート", 
                    value=TAG_NORMALIZATION_TEMPLATE,
                    lines=6
                )
                tag_filter_template = gr.Textbox(
                    label="タグフィルターテンプレート", 
                    value=TAG_FILTER_TEMPLATE,
                    lines=6
                )
                tag_weight_template = gr.Textbox(
                    label="タグ重み付けテンプレート", 
                    value=TAG_WEIGHT_TEMPLATE,
                    lines=6
                )
            
            with gr.Row():
                api_key = gr.Textbox(label="RunPod API Key (未入力の場合は環境変数から取得)", type="password")
                endpoint_id = gr.Textbox(label="RunPod Endpoint ID (未入力の場合は環境変数から取得)")
        
        with gr.Row():
            output_gallery1 = gr.Gallery(label="生成結果", columns=1, height=500, object_fit="contain")
            output_gallery2 = gr.Gallery(label="生成結果", columns=1, height=500, object_fit="contain")
            output_gallery3 = gr.Gallery(label="生成結果", columns=1, height=500, object_fit="contain")
            output_gallery4 = gr.Gallery(label="生成結果", columns=1, height=500, object_fit="contain")
        status_text = gr.Textbox(label="ステータス", interactive=False)

        output_gallery = [output_gallery1, output_gallery2, output_gallery3, output_gallery4]
        
        # 送信ボタンのクリックイベント
        submit_btn.click(
            fn=call_runpod,
            inputs=[
                prompt, negative_prompt, tag_candidate_generation_template,
                tag_normalization_template, tag_filter_template,
                tag_weight_template, guidance_scale, num_inference_steps,
                width, height, num_images, is_random_seeds, seeds, api_key, endpoint_id
            ],
            outputs=[output_gallery[0], output_gallery[1], output_gallery[2], output_gallery[3], status_text, seeds]
        )
    
    return app

# メイン実行部分
if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True) 