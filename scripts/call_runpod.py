import requests
import base64
import os
from pathlib import Path
import argparse
import time
import json

def save_images(images: list, output_dir: Path, prefix: str = "generated"):
    """Base64エンコードされた画像リストを保存する"""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    for i, img_base64 in enumerate(images):
        img_data = base64.b64decode(img_base64)
        output_path = output_dir / f"{prefix}_{i}.png"
        
        with open(output_path, "wb") as f:
            f.write(img_data)
        saved_paths.append(output_path)
    
    return saved_paths

def call_runpod_endpoint(prompt: str, endpoint_id: str, api_key: str, **kwargs):
    """RunPodエンドポイントを呼び出す"""
    start_time = time.time()
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    
    # リクエストの構築
    request = {
        'input': {
            'prompt': prompt,
            'negative_prompt': kwargs.get('negative_prompt', ''),
            'num_images': kwargs.get('num_images', 1),
            'steps': kwargs.get('steps', 30),
            'cfg_scale': kwargs.get('cfg_scale', 7.0),
            'width': kwargs.get('width', 512),
            'height': kwargs.get('height', 768)
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(url, json=request, headers=headers)
        response.raise_for_status()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"API call took: {elapsed_time:.2f} seconds")

        data = response.json()
        
        if "error" in data:
            print(f"Error: {data['error']}")
            return None
            
        # 生成された画像を保存
        output_dir = Path("output")
        saved_paths = save_images(
            data["output"]["images"],
            output_dir,
            prefix="generated"
        )
        # ボディラインを保存
        saved_paths = save_images(
            data["output"]["bodylines"],
            output_dir,
            prefix="bodyline"
        )
        
        # 生成されたタグとパラメータを保存
        metadata = {
            "generated_tags": data["output"]["generated_tags"],
            "parameters": data["output"]["parameters"]
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Generated images saved to: {[str(p) for p in saved_paths]}")
        print(f"Metadata saved to: {metadata_path}")
        
        return saved_paths, metadata
        
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"API call took: {elapsed_time:.2f} seconds")
        print(f"Error calling RunPod endpoint: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Call RunPod endpoint to generate images from prompt')
    parser.add_argument('--prompt', help='Text prompt for image generation')
    parser.add_argument('--negative-prompt', help='Negative prompt for image generation')
    parser.add_argument('--num-images', type=int, help='Number of images to generate')
    parser.add_argument('--steps', type=int, help='Number of generation steps')
    parser.add_argument('--cfg-scale', type=float, help='CFG scale')
    parser.add_argument('--width', type=int, help='Image width')
    parser.add_argument('--height', type=int, help='Image height')
    parser.add_argument('--endpoint-id', help='RunPod endpoint ID')
    parser.add_argument('--api-key', help='RunPod API key')
    
    args = parser.parse_args()
    
    import dotenv
    dotenv.load_dotenv()
      
    prompt = args.prompt or "壁に寄りかかりながらピースをしている女の子のポーズ"
    # prompt = args.prompt or "prompt:original, 1girl, solo, (jumping:1.3), jacket, school uniform, pose, miniskirt, brown hair, blue eyes, pleated skirt, red footwear, red jacket, shoes, striped clothes, thighs, white thighhighs, expression"
    endpoint_id = args.endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = args.api_key or os.getenv("RUNPOD_API_KEY")
    negative_prompt = args.negative_prompt or "nsfw, sensitive, from behind, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, missing arms, extra arms, missing legs, extra legs, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
    num_images = args.num_images or 4
    steps = args.steps or 30
    cfg_scale = args.cfg_scale or 10.0
    width = args.width or 832
    height = args.height or 1216
    
    result = call_runpod_endpoint(
        prompt,
        endpoint_id,
        api_key,
        negative_prompt=negative_prompt,
        num_images=num_images,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height
    )
    
    if result:
        print("Processing completed successfully")
        
# python call.py "図書館で本を読んでいる女の子。眼鏡をかけていて、真面目そうな見た目なんですが、なんとなく大人っぽい魅力があふれています。" --endpoint-id qidc047mo1cano --api-key rpa_XWRZGKA8089VRA02YEZU574GH9MA5QTTPLY4LPKW1sjqle

if __name__ == "__main__":
    main() 
