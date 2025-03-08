#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
画像とbodylineのペアを生成するスクリプト
Dartを使用してランダムにプロンプトを生成し、image_serviceとbodyline_serviceを使用して
画像とbodylineのペアを生成します。
"""

import os
import sys
import asyncio
import random
from PIL import Image
import uuid
from datetime import datetime
import json
import argparse

# プロジェクトルートへのパスを設定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# appディレクトリへのパスを設定
APP_ROOT = os.path.join(PROJECT_ROOT, 'app')
sys.path.insert(0, APP_ROOT)

from core.config import settings
from services.dart_service import DartService
from services.image_service import ImageService
from services.bodyline_service import BodylineService
from model_downloader import download_models
# モデルダウンロード 開発時
download_models()

def setup_output_dirs(base_output_dir):
    """出力ディレクトリの設定と作成"""
    dirs = {
        'base': base_output_dir,
        'images': os.path.join(base_output_dir, 'image'),
        'bodylines': os.path.join(base_output_dir, 'bodyline'),
        'metadata': os.path.join(base_output_dir, 'metadata')
    }
    
    # ディレクトリが存在しない場合は作成
    for directory in dirs.values():
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    return dirs

# プロンプトのテーマリスト（ランダムな生成のベースとして使用）
PROMPT_THEMES = [
    "女の子が学校の制服を着ている",
    "女の子がカフェでコーヒーを飲んでいる",
    "女の子が公園でリラックスしている",
    "女の子が本を読んでいる",
    "女の子がスポーツをしている",
    "女の子が楽器を演奏している",
    "女の子が料理をしている",
    "女の子が踊っている",
    "女の子が買い物をしている",
    "女の子が海辺で遊んでいる",
    "女の子が花畑にいる",
    "女の子が雨の中で傘をさしている",
    "女の子が雪の中で遊んでいる",
    "女の子が夜空を見上げている",
    "女の子が友達と笑っている",
    "女の子がペットと遊んでいる",
    "女の子がスマートフォンを使っている",
    "女の子がゲームをしている",
    "女の子が寝転がっている",
    "女の子が走っている"
]

# ポーズのバリエーション
POSE_VARIATIONS = [
    "立っている",
    "座っている",
    "ジャンプしている",
    "歩いている",
    "走っている",
    "寝転んでいる",
    "しゃがんでいる",
    "振り向いている",
    "手を振っている",
    "手を広げている",
    "両手を上げている",
    "片手を上げている",
    "ポーズをとっている",
    "バランスを取っている",
    "伸びをしている",
    "腕を組んでいる",
    "手をポケットに入れている",
    "髪をなびかせている",
    "スカートを押さえている",
    "笑っている"
]

# 服装のバリエーション
CLOTHING_VARIATIONS = [
    "制服を着ている",
    "セーラー服を着ている",
    "ブレザーを着ている",
    "カジュアルな服装をしている",
    "ドレスを着ている",
    "スポーツウェアを着ている",
    "水着を着ている",
    "パジャマを着ている",
    "コートを着ている",
    "ニットを着ている",
    "Tシャツとジーンズを着ている",
    "ワンピースを着ている",
    "スーツを着ている",
    "ゴスロリ服を着ている",
    "和服を着ている"
]

async def generate_random_prompt():
    """ランダムなプロンプトを生成する"""
    theme = random.choice(PROMPT_THEMES)
    pose = random.choice(POSE_VARIATIONS)
    clothing = random.choice(CLOTHING_VARIATIONS)
    
    # ランダムに組み合わせる（すべての要素を使うわけではない）
    elements = [theme]
    if random.random() > 0.3:  # 70%の確率でポーズを追加
        elements.append(pose)
    if random.random() > 0.5:  # 50%の確率で服装を追加
        elements.append(clothing)
    
    # ランダムにシャッフル
    random.shuffle(elements)
    
    return "、".join(elements)

async def generate_tags_from_prompt(dart_service, prompt):
    """プロンプトからタグを生成する"""
    # ダミータグリスト（実際のタグはDartが生成）
    initial_tags = [
        "1girl", "solo", "looking_at_viewer", "smile", 
        "long_hair", "short_hair", "school_uniform", "casual", 
        "standing", "sitting", "outdoors", "indoors"
    ]
    
    # Dartを使って最終的なタグを生成
    final_tags = await dart_service.generate_final_tags(initial_tags)
    
    # # コンテキストに基づいてタグをフィルタリングして重み付け
    # weighted_tags = await dart_service.filter_tags_by_context(", ".join(final_tags), prompt)
    
    return ", ".join(final_tags)

async def generate_and_save_pair(dart_service, image_service, bodyline_service, output_dirs, index, prompt=None):
    """画像とbodylineのペアを生成して保存する"""
    try:
        # # プロンプトがない場合はランダムに生成
        # if prompt is None:
        #     prompt = await generate_random_prompt()
        
        # print(f"[{iteration}] プロンプト: {prompt}")
        
        # タグ生成
        tags = await generate_tags_from_prompt(dart_service, prompt)
        print(f"[{index}] 生成されたタグ: {tags}")
        
        # 画像生成
        image_size_list = [
            (1024, 1024),  # 1:1
            (1152, 896),   # 1.29:1
            (896, 1152),   # 1:1.29
            (832, 1216),   # 1:1.46
            (1216, 832),   # 1.46:1
            (1344, 768),   # 1.75:1
            (768, 1344),   # 1:1.75
            (1536, 640),   # 2.4:1
            (640, 1536)    # 1:2.4
        ]
        # ランダムにサイズを選択
        width, height = random.choice(image_size_list)
        
        image_result = await image_service.generate_image(
            tags=tags,
            steps=30,
            guidance_scale=8.0,
            width=width,
            height=height,
            negative_prompt="lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry",
            num_images=1
        )
        
        # 生成された画像を取得
        generated_image = image_result["images"][0]
        
        output_size = bodyline_service.calculate_resize_dimensions(generated_image, 768)
        
        # bodyline生成
        bodyline_result = await bodyline_service.generate_bodyline(
            control_images=[generated_image],
            prompt="anime pose, girl, (white background:1.5), (monochrome:1.5), full body, sketch, eyes, breasts, (slim legs, skinny legs:1.2)",
            negative_prompt="(wings:1.6), (clothes:1.4), (garment:1.4), (lighting:1.4), (gray:1.4), (missing limb:1.4), (extra line:1.4), (extra limb:1.4), (extra arm:1.4), (extra legs:1.4), (hair:1.4), (bangs:1.4), (fringe:1.4), (forelock:1.4), (front hair:1.4), (fill:1.4), (ink pool:1.6)",
            num_inference_steps=20,
            guidance_scale=8.0,
            input_resolution=512,
            output_size=output_size
        )
        
        # 生成されたbodylineを取得
        generated_bodyline = bodyline_result["images"][0]
        
        # 8桁の0埋め連番でファイル名を生成
        filename = f"{index:08d}.png"
        
        # 画像の保存
        image_path = os.path.join(output_dirs['images'], filename)
        generated_image.save(image_path)
        
        # bodylineの保存
        bodyline_path = os.path.join(output_dirs['bodylines'], filename)
        generated_bodyline.save(bodyline_path)
        
        # メタデータの保存
        metadata = {
            "id": index,
            "prompt": prompt,
            "tags": tags,
            "image_path": image_path,
            "bodyline_path": bodyline_path,
            "image_params": image_result["parameters"],
            "bodyline_params": bodyline_result["parameters"]
        }
        
        metadata_path = os.path.join(output_dirs['metadata'], f"{index:08d}.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"[{index}] 生成完了: {image_path}, {bodyline_path}")
        print("-" * 50)
        
        return image_path, bodyline_path, metadata_path
    
    except Exception as e:
        print(f"[{index}] エラーが発生しました: {str(e)}")
        return None, None, None

async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='画像とbodylineのペアを生成するスクリプト')
    parser.add_argument('--output', '-o', type=str, required=True,
                      help='出力先ディレクトリのパス')
    parser.add_argument('--count', '-c', type=int, default=1,
                      help='生成する画像の数 (デフォルト: 1)')
    parser.add_argument('--start', '-s', type=int, default=0,
                      help='開始番号 (デフォルト: 0)')
    
    args = parser.parse_args()
    
    print("画像とbodylineのペア生成プログラムを開始します...")
    print(f"出力先: {args.output}")
    print(f"生成数: {args.count}")
    print(f"開始番号: {args.start}")
    
    # 出力ディレクトリの設定
    output_dirs = setup_output_dirs(args.output)
    
    # サービスの初期化
    dart_service = DartService()
    # # LLMの初期化
    # dart_service.initialize_llm(use_local_llm=False)
    image_service = ImageService()
    bodyline_service = BodylineService()
    
    print("すべてのサービスの初期化が完了しました。生成を開始します...")
    
    success_count = 0
    try:
        for i in range(args.count):
            current_index = args.start + i
            await generate_and_save_pair(
                dart_service, 
                image_service, 
                bodyline_service, 
                output_dirs,
                current_index
            )
            success_count += 1
    
    except KeyboardInterrupt:
        print("\n生成を停止します...")
    
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {str(e)}")
    
    finally:
        print(f"合計 {success_count} 枚のペアを生成しました。")
        print(f"生成範囲: {args.start:08d} ~ {args.start + success_count - 1:08d}")
        print("プログラムを終了します。")

if __name__ == "__main__":
    # 非同期メイン関数の実行
    asyncio.run(main()) 