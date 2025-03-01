import requests
import json
import os
from dotenv import load_dotenv
import base64
import time

# .envファイルから環境変数を読み込む
load_dotenv()

def call_api(endpoint, method='GET', data=None, headers=None):
    """
    APIを呼び出すための汎用関数
    
    Args:
        endpoint (str): APIのエンドポイント
        method (str): HTTPメソッド（GET, POST, PUT, DELETE等）
        data (dict): リクエストボディ（POSTリクエスト等で使用）
        headers (dict): リクエストヘッダー
    
    Returns:
        dict: APIレスポンス
    """
    base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    url = f"{base_url}{endpoint}"
    
    if headers is None:
        headers = {
            'Content-Type': 'application/json'
        }
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, json=data, headers=headers)
        elif method == 'PUT':
            response = requests.put(url, json=data, headers=headers)
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"APIコールエラー: {e}")
        if hasattr(e.response, 'text'):
            try:
                error_detail = json.loads(e.response.text)
                print(f"エラー詳細: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
            except json.JSONDecodeError:
                print(f"エラーレスポンス: {e.response.text}")
        return None

def test_individual_api():
    start_time = time.time()
    # サーバーの状態確認
    print("1. サーバーの状態確認")
    response = call_api('/')
    print("GET response:", json.dumps(response, indent=2, ensure_ascii=False))
    print(f"所要時間: {time.time() - start_time}秒")
    print()
    
    start_time = time.time()
    # タグ生成のテスト
    print("2. タグ生成のテスト")
    data = {
        # "prompt": "図書館で本を読んでいる女の子。眼鏡をかけていて、真面目そうな見た目なんですが、なんとなく大人っぽい魅力があふれています。"
        "prompt": "「制服の女の子がジャンプしているポーズ」を描いてください。セーラー服ではなくブレザーを着ている。ちょっとエッチな感じで。"
        # "prompt": "ベッドの上で横向きに寝そべっている金髪の女の子"
    }
    response = call_api('/api/generate-tags', method='POST', data=data)
    print("POST response:", json.dumps(response, indent=2, ensure_ascii=False))
    tags = response.get("tags", [])
    print(", ".join(tags))
    print(f"所要時間: {time.time() - start_time}秒")
    print()
    
    start_time = time.time()
    if response:
        # タグから画像生成のテスト
        print("3. 画像生成のテスト")
        image_data = {
            "tags": response.get("tags", []),
            "steps": 20,
            "cfg_scale": 7.5,
            "width": 512,
            "height": 512,
            "negative_prompt": "lowres, bad anatomy, bad hands, cropped, worst quality"
        }
        response = call_api('/api/generate-image', method='POST', data=image_data)
        image_base64 = response.get("image_base64", "")
        if image_base64:
            with open("output.png", "wb") as f:
                f.write(base64.b64decode(image_base64))
                print("画像をoutput.pngに保存しました")
        else:
            print("画像が生成されませんでした")
    print(f"所要時間: {time.time() - start_time}秒")

def test_all_apis():
    start_time = time.time()
    print("1. サーバーの状態確認")
    response = call_api('/')
    print("GET response:", json.dumps(response, indent=2, ensure_ascii=False))
    print(f"所要時間: {time.time() - start_time}秒")
    
    start_time = time.time()
    
    # generate-from-promptのテスト
    print("2. generate-from-promptのテスト")
    prompt = "「制服の女の子がジャンプしているポーズ」を描いてください。セーラー服ではなくブレザーを着ている。ちょっとエッチな感じで。"
    data = {
        "prompt": prompt,
        "steps": 20,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512,
        "negative_prompt": "lowres, bad anatomy, bad hands, cropped, worst quality",
        "num_images": 2
    }
    response = call_api('/api/generate-from-prompt', method='POST', data=data)
    for i in range(response["num_images"]):
        image_base64 = response["image_base64_list"][i]
        with open(f"output_{i}.png", "wb") as f:
            f.write(base64.b64decode(image_base64))
            print(f"画像をoutput_{i}.pngに保存しました")
    print("POST response:", json.dumps(response, indent=2, ensure_ascii=False))
    print(f"所要時間: {time.time() - start_time}秒")


if __name__ == "__main__":
    test_all_apis() 