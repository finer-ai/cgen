#!/bin/bash

# 必要なディレクトリを作成
mkdir -p ./app/models/dart
mkdir -p ./app/models/animagine
mkdir -p ./app/data

# Dartモデルのダウンロード（例）
echo "Dartモデルをダウンロードしています..."
# 実際のダウンロードURLに置き換えてください
wget https://huggingface.co/p1atdev/dart-v2-moe-sft/resolve/main/model.safetensors -O ./app/models/dart/model.safetensors
echo "Dartモデルは手動でダウンロードして ./app/models/dart に配置してください"

# # Animagineモデルのダウンロード（例）
# echo "Animagineモデルをダウンロードしています..."
# # 実際のダウンロードURLに置き換えてください
# # wget https://huggingface.co/path/to/animagine-model -O ./app/models/animagine/model.safetensors
# echo "Animagineモデルは手動でダウンロードして ./app/models/animagine に配置してください"

# echo "モデルのダウンロードと配置手順が完了しました" 