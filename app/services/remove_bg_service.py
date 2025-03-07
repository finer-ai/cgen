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
            model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
            self.model = rt.InferenceSession(model_path, providers=self.providers)
            print("背景除去モデルが正常にロードされました！")
        except Exception as e:
            print(f"モデルのロード中にエラーが発生しました: {e}")
            raise
    
    def get_mask(self, img, s=1024):
        """
        画像からマスクを取得する
        
        Args:
            img: 入力画像
            s: サイズ
            
        Returns:
            マスク画像
        """
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
            print(f"警告: マスクのサイズが異なります: {mask1_flat.shape} vs {mask2_flat.shape}")
            
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
        # 画像データをデコード
        if isinstance(image1_data, str) and image1_data.startswith('data:image'):
            # Base64文字列の場合
            image1_data = image1_data.split(',')[1]
            image1_bytes = base64.b64decode(image1_data)
            img1 = cv2.imdecode(np.frombuffer(image1_bytes, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image1_data, bytes):
            # バイト列の場合
            img1 = cv2.imdecode(np.frombuffer(image1_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            # PILイメージの場合
            img1 = cv2.cvtColor(np.array(image1_data), cv2.COLOR_RGB2BGR)
        
        if isinstance(image2_data, str) and image2_data.startswith('data:image'):
            # Base64文字列の場合
            image2_data = image2_data.split(',')[1]
            image2_bytes = base64.b64decode(image2_data)
            img2 = cv2.imdecode(np.frombuffer(image2_bytes, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image2_data, bytes):
            # バイト列の場合
            img2 = cv2.imdecode(np.frombuffer(image2_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            # PILイメージの場合
            img2 = cv2.cvtColor(np.array(image2_data), cv2.COLOR_RGB2BGR)
        
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