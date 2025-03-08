import argparse
import huggingface_hub
import onnxruntime as rt
import numpy as np
import cv2
import os
import io
import base64
from PIL import Image


class RemoveBGService:
    def __init__(self, cpu_only=False):
        """
        背景除去サービスの初期化
        
        Args:
            cpu_only (bool): CPUのみを使用するかどうか
        """
        self.providers = ['CPUExecutionProvider'] if cpu_only else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = None
        self.load_model()
    
    def load_model(self):
        """モデルをロードする"""
        try:
            model_path = os.path.join("models", "anime-seg", "isnetis.onnx")
            self.model = rt.InferenceSession(model_path, providers=self.providers)
            print("背景除去モデルが正常にロードされました！")
        except Exception as e:
            print(f"モデルのロード中にエラーが発生しました: {e}")
            raise
    
    def get_mask(self, img, s=1024):
        """
        画像からマスクを取得する
        
        Args:
            img: 入力画像（RGB or RGBA）
            s: サイズ
            
        Returns:
            マスク画像
        """
        # RGBAの場合は白背景に合成
        if img.shape[-1] == 4:
            # 白背景の画像を作成
            white_bg = np.ones((*img.shape[:2], 3), dtype=np.uint8) * 255
            # アルファチャンネルを使用して合成
            alpha = img[..., 3:] / 255.0
            img_rgb = img[..., :3]
            img = (img_rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)

        img = (img / 255).astype(np.float32)
        h, w = h0, w0 = img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        mask = self.model.run(None, {'img': img_input})[0][0]
        mask = np.transpose(mask, (1, 2, 0))
        mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
        return mask

    def calculate_mask_difference(self, mask1, mask2):
        """
        2つのマスク間の差分を計算する
        
        Args:
            mask1: 1つ目のマスク
            mask2: 2つ目のマスク
            
        Returns:
            差分画像
        """
        # 計算用に正しいフォーマットに変換
        mask1_flat = mask1.squeeze().astype(np.float32)
        mask2_flat = mask2.squeeze().astype(np.float32)
        
        # 次元が異なる場合はリサイズ
        if mask1_flat.shape != mask2_flat.shape:
            print(f"マスクのサイズが異なるので小さい方のサイズにリサイズします: {mask1_flat.shape} vs {mask2_flat.shape}")
            
            # 小さい方のサイズを取得
            min_height = min(mask1_flat.shape[0], mask2_flat.shape[0])
            min_width = min(mask1_flat.shape[1], mask2_flat.shape[1])
            
            # 両方のマスクを小さい方のサイズにリサイズ
            mask1_flat = cv2.resize(mask1_flat, (min_width, min_height))
            mask2_flat = cv2.resize(mask2_flat, (min_width, min_height))
        
        # 各ピクセルの差分を計算（負の値は0にクリップ）
        diff = np.maximum(0, mask1_flat - mask2_flat)
        
        # 8ビットグレースケール画像に変換（単一チャンネル）
        diff_image = (diff * 255).astype(np.uint8)
        
        return diff_image

    def process_images(self, image1_data, image2_data):
        """
        2つの画像を処理する:
        1. 両方の画像のマスクを取得
        2. マスク間の差分を計算
        3. マスクと差分を返す
        
        Args:
            image1_data: 1つ目の画像データ（バイト列またはbase64エンコード文字列）
            image2_data: 2つ目の画像データ（バイト列またはbase64エンコード文字列）
            
        Returns:
            差分画像、白ピクセルの割合を含む辞書
        """
        def decode_image(img_data):
            if isinstance(img_data, str):
                # 文字列の場合の詳細なデバッグ情報
                if img_data.startswith('data:image'):
                    # Base64文字列の場合
                    img_data = img_data.split(',')[1]
                    img_bytes = base64.b64decode(img_data)
                    return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                else:
                    # Base64エンコードされた画像データの可能性をチェック
                    try:
                        img_bytes = base64.b64decode(img_data)
                        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            return img
                    except:
                        pass
                    # エラーメッセージに詳細情報を含める
                    preview = img_data[:100] + '...' if len(img_data) > 100 else img_data
                    raise ValueError(f"Invalid image data format: string must be base64 encoded or data URL. Received: {preview}")
            elif isinstance(img_data, bytes):
                # バイト列の場合
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("Failed to decode image bytes")
                return img
            elif isinstance(img_data, np.ndarray):
                # NumPy配列の場合
                return img_data
            elif hasattr(img_data, 'convert'):
                # PILイメージの場合
                return np.array(img_data.convert('RGBA' if img_data.mode == 'RGBA' else 'RGB'))
            else:
                raise ValueError(f"Unsupported image data type: {type(img_data)}")

        # 画像データをデコード
        img1 = decode_image(image1_data)
        img2 = decode_image(image2_data)
        
        # マスクを取得
        mask1 = self.get_mask(img1)
        mask2 = self.get_mask(img2)

        # マスク間の差分を計算
        mask_diff = self.calculate_mask_difference(mask2, mask1)
        
        # mask2における白いピクセルの割合を計算
        mask2_flat = mask2.squeeze().astype(np.float32)
        white_pixels_mask2 = np.sum(mask2_flat > 0.5)
        white_pixels_diff = np.sum(mask_diff > 127)
        
        if white_pixels_mask2 > 0:
            white_percentage = (white_pixels_diff / white_pixels_mask2) * 100
        else:
            white_percentage = 0
        
        # 各画像をbase64エンコード
        _, diff_buffer = cv2.imencode('.png', mask_diff)
        diff_base64 = base64.b64encode(diff_buffer).decode('utf-8')
        
        return {
            "diff": diff_base64,
            "white_percentage": float(white_percentage),
            "white_pixels_mask2": int(white_pixels_mask2),
            "white_pixels_diff": int(white_pixels_diff)
        } 