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

# python3.11をインストール
RUN apt update && apt install python3.11 -y \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係ファイルのコピーとインストール
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install runpod

# アプリケーションファイルのコピー
COPY ./app /app/

# ポートの公開
EXPOSE 8000

# 起動コマンド
CMD ["python3", "-u", "handler.py"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 