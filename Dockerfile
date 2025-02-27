FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係ファイルのコピーとインストール
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# アプリケーションファイルのコピー
COPY ./app /app/
COPY ./scripts /app/scripts/

# 必要なディレクトリの作成
RUN mkdir -p /app/models /app/data

# スクリプトに実行権限を付与
RUN chmod +x /app/scripts/download_models.sh

# Hugging Face Hub認証用の環境変数（オプション）
ENV HUGGINGFACE_TOKEN=""

# ポートの公開
EXPOSE 8000

# 起動コマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 