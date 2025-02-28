import requests
import json
import os
from dotenv import load_dotenv

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

def main():
    # サーバーの状態確認
    print("1. サーバーの状態確認")
    response = call_api('/')
    print("GET response:", json.dumps(response, indent=2, ensure_ascii=False))
    print()
    
    # タグ生成のテスト
    print("2. タグ生成のテスト")
    data = {
        "prompt": "ベッドの上で横向きに寝そべっている金髪の女の子"
    }
    response = call_api('/api/generate-tags', method='POST', data=data)
    print("POST response:", json.dumps(response, indent=2, ensure_ascii=False))
    print()
    
    # if response:
    #     # タグから画像生成のテスト
    #     print("3. 画像生成のテスト")
    #     image_data = {
    #         "tags": response.get("tags", [])
    #     }
    #     response = call_api('/api/generate-image', method='POST', data=image_data)
    #     print("Image generation response:", json.dumps(response, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 