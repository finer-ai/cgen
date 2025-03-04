import requests
import json
import base64
from PIL import Image
import io
import time

# APIのベースURL
BASE_URL = "http://localhost:8000/api"

def test_generate_tags():
    """タグ生成APIのテスト"""
    prompt = "ベッドの上で横向きに寝そべっている金髪の女の子、水色のパジャマ、白い枕、赤い毛布、窓から月明かりが差し込んでいる"
    
    print(f"タグ生成テスト開始: \"{prompt}\"")
    response = requests.post(
        f"{BASE_URL}/generate-tags",
        json={"prompt": prompt}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("生成されたタグ:")
        for tag in result["tags"]:
            print(f"- {tag}")
        return result["tags"]
    else:
        print(f"エラー: {response.status_code}")
        print(response.text)
        return None

def test_generate_image(tags):
    """画像生成APIのテスト"""
    if not tags:
        print("タグが提供されていないため、画像生成をスキップします")
        return
    
    print("画像生成テスト開始...")
    response = requests.post(
        f"{BASE_URL}/generate-image",
        json={
            "tags": tags,
            "steps": 20,
            "guidance_scale": 7.0,
            "width": 512,
            "height": 768
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        image_data = base64.b64decode(result["image_base64"])
        
        # 画像を保存
        image = Image.open(io.BytesIO(image_data))
        output_path = "test_output.png"
        image.save(output_path)
        
        print(f"画像を保存しました: {output_path}")
        print(f"使用されたプロンプト: {result['prompt']}")
    else:
        print(f"エラー: {response.status_code}")
        print(response.text)

def test_generate_from_prompt():
    """一括生成APIのテスト"""
    prompt = "窓際に立っている黒髪の男性、スーツを着ている、夕日の光が差し込んでいる"
    
    print(f"一括生成テスト開始: \"{prompt}\"")
    response = requests.post(
        f"{BASE_URL}/generate-from-prompt",
        json={"prompt": prompt}
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # 結果のタグを表示
        print("生成されたタグ:")
        for tag in result["tags"]:
            print(f"- {tag}")
            
        # 画像を保存
        image_data = base64.b64decode(result["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        output_path = "test_output_direct.png"
        image.save(output_path)
        
        print(f"画像を保存しました: {output_path}")
    else:
        print(f"エラー: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # タグ生成テスト
    tags = test_generate_tags()
    
    # しばらく待機
    if tags:
        print("5秒後に画像生成テストを開始します...")
        time.sleep(5)
    
    # 画像生成テスト
    test_generate_image(tags)
    
    # しばらく待機
    print("5秒後に一括生成テストを開始します...")
    time.sleep(5)
    
    # 一括生成テスト
    test_generate_from_prompt() 