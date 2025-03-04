# アニメ画像生成システム（SotaiGenerator）

## 1. プロジェクト概要

本システムは、ユーザーが入力した自然言語（主に日本語）から、**Danbooruタグを自動生成**し、それを用いて**アニメ調の画像を生成**するWebサービスです。以下の主要コンポーネントで構成されています：

1. **RAG (Retrieval-Augmented Generation)**：
   - 日本語入力から関連するDanbooruタグを検索・抽出
   - LangChainとベクトルデータベース（FAISS）を使用

2. **Dart (Danbooru Tags Transformer)**：
   - RAGで抽出したタグ候補を整形・補完
   - タグの組み合わせを最適化

3. **画像生成**：
   - Stable Diffusion系モデル（Animagineなど）を使用
   - 生成されたタグセットを元に高品質なアニメ画像を生成

4. **Gradioインターフェース**：
   - 直感的なWebインターフェース
   - タグ生成と画像生成の統合された操作環境
   - 詳細なパラメータ調整機能

## 2. システムアーキテクチャ

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │
│  ユーザー入力  ├───►│   RAG処理   ├───►│  Dart処理   ├───►│  画像生成   │
│  (日本語文)  │    │ (タグ候補)   │    │ (タグ補完)   │    │            │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          ▲                  ▲                  ▲
                          │                  │                  │
                          ▼                  ▼                  ▼
                   ┌─────────────────────────────────────────────┐
                   │                                             │
                   │              データストレージ                 │
                   │  - FAISS (タグベクトルDB)                    │
                   │  - Dartモデル                                │
                   │  - Animagineモデル                          │
                   │                                             │
                   └─────────────────────────────────────────────┘
```

## 3. ディレクトリ構造

```
/
├── app/                    # メインアプリケーションディレクトリ
│   ├── api/                # API関連ファイル
│   │   ├── __init__.py
│   │   ├── routes.py       # APIエンドポイント定義
│   │   └── schemas.py      # リクエスト/レスポンススキーマ
│   ├── core/               # コア機能
│   │   ├── __init__.py
│   │   ├── config.py       # 設定管理
│   │   └── errors.py       # エラー定義
│   ├── services/           # サービス実装
│   │   ├── __init__.py
│   │   ├── rag_service.py  # RAGによるタグ抽出
│   │   ├── dart_service.py # Dartによるタグ補完
│   │   └── image_service.py # 画像生成
│   ├── utils/              # ユーティリティ
│   │   ├── __init__.py
│   │   └── tag_utils.py    # タグ処理ユーティリティ
│   └── main.py             # アプリケーションエントリポイント
├── scripts/                # 各種スクリプト
│   ├── download_dataset.py # データセットダウンロード
│   ├── download_models.sh  # モデルダウンロード
│   ├── prepare_vector_store.py # ベクトルストア作成
│   └── gradio_runpod_interface.py # Gradioインターフェース
├── data/                   # データディレクトリ
│   └── danbooru-wiki-2024_df.pkl # Danbooruタグデータセット
├── app/models/             # モデルファイル
│   ├── dart/               # Dartモデル
│   │   └── dart_model.safetensors # タグ補完モデル
│   └── animagine/          # 画像生成モデル
├── app/data/               # アプリケーションデータ
│   └── faiss/              # FAISSインデックス
│       ├── danbooru_tags.faiss  # タグベクトルインデックス
│       └── danbooru_documents.txt # タグドキュメント
├── .env                    # 環境変数設定
├── Dockerfile              # Dockerコンテナ定義
└── requirements.txt        # 依存パッケージ
```

## 4. 主要コンポーネントの詳細

### 4.1 RAGサービス (rag_service.py)

**役割**：
- ユーザーの日本語入力を解析し、キーワードを抽出
- 抽出したキーワードに基づいてFAISSからDanbooruタグ候補を取得

**主要機能**：
- `extract_keywords`: 日本語入力からキーワード抽出
- `retrieve_tags`: キーワードからタグ候補検索
- `generate_tag_candidates`: タグ候補生成の統合メソッド

**使用技術**：
- LangChain
- FAISS (ベクトルデータベース)
- OpenAI Embeddings (埋め込みモデル)

### 4.2 Dartサービス (dart_service.py)

**役割**：
- RAGで抽出したタグ候補を適切なフォーマットに整形
- Dartモデルを使用して、最終的なタグセットを生成

**主要機能**：
- `_format_input_for_dart`: Dart入力フォーマットへの変換
- `generate_final_tags`: 最終タグ生成

**使用技術**：
- Transformers (HuggingFace)
- PyTorch
- Dart-v2-moe-sft モデル

### 4.3 画像生成サービス (image_service.py)

**役割**：
- タグセットを元に高品質なアニメ画像を生成

**主要機能**：
- `generate_image`: タグからアニメ画像を生成

**使用技術**：
- Diffusers
- Stable Diffusion (Animagine)
- PyTorch

### 4.4 Gradioインターフェース (gradio_runpod_interface.py)

**役割**：
- ユーザーフレンドリーなWebインターフェースの提供
- タグ生成と画像生成の統合された操作環境
- 詳細なパラメータ調整機能の提供

**主要機能**：
- `create_ui`: Gradioインターフェースの構築
- `call_runpod`: RunPod APIとの通信処理

**使用技術**：
- Gradio
- RunPod API
- Python Imaging Library (PIL)

## 5. セットアップ手順

### 5.1 前提条件

- CUDA対応のNVIDIA GPU
- Docker (GPUサポート)
- Python 3.10+

### 5.2 環境構築

1. **リポジトリのクローン**:

```bash
git clone https://github.com/yourusername/SotaiGenerator.git
cd SotaiGenerator
```

2. **仮想環境の作成** (Docker を使用しない場合):

```bash
python -m venv venv
source venv/bin/activate  # Linuxの場合
venv\Scripts\activate     # Windowsの場合
pip install -r requirements.txt
```

3. **データセットのダウンロード**:

```bash
python scripts/download_dataset.py
```

4. **ベクトルストアの作成**:

```bash
python scripts/prepare_vector_store.py
```

5. **モデルのダウンロード**:

```bash
bash scripts/download_models.sh
```

### 5.3 Dockerでの実行

1. **イメージのビルド**:

```bash
docker build -t anime-generator .
```

2. **コンテナの実行**:

```bash
docker run -p 8000:8000 --gpus all -v ./app/models:/app/models -v ./app/data:/app/data anime-generator
```

### 5.4 環境変数の設定

`.env`ファイルを作成し、以下の設定を行います：

```bash
# アプリケーション設定
DEBUG=True

# モデルパス設定
DART_MODEL_PATH=./app/models/dart
SD_MODEL_PATH=./app/models/animagine

# RAG設定
VECTOR_DB_PATH=./app/data/faiss

# OpenAI API設定（RAGの埋め込みに必要）
OPENAI_API_KEY=your-openai-api-key

# 画像生成パラメータ
DEFAULT_STEPS=20
DEFAULT_CFG_SCALE=7.0
DEFAULT_WIDTH=512
DEFAULT_HEIGHT=768
```

## 6. 使用方法

### 6.1 Gradioインターフェースの起動

```bash
cd scripts
python gradio_runpod_interface.py
```

### 6.2 インターフェースの使用方法

1. Webブラウザで`http://localhost:7860`にアクセス
2. プロンプト入力欄に生成したい画像の説明を入力
3. 必要に応じて詳細設定を調整
   - CFGスケール
   - ステップ数
   - 画像サイズ
   - 生成枚数
   - ネガティブプロンプト
4. 「生成開始」ボタンをクリック
5. 生成結果が表示されるまで待機

## 7. 拡張方法

### 7.1 RAGの拡張

- FAISSの代わりにElasticsearchを使用する場合は、`rag_service.py`の`VectorStore`実装を変更します。
- 新しい埋め込みモデルを使用する場合は、`CustomEmbeddings`クラスを修正します。

### 7.2 Dartモデルの変更

- 新しいDartモデルを使用するには、`dart_service.py`の`__init__`メソッドを更新します。
- タグ整形の処理を変更する場合は、`_format_input_for_dart`メソッドを修正します。

### 7.3 画像生成モデルの変更

- 別の画像生成モデル（NovelAIなど）に切り替える場合は、`image_service.py`の`__init__`メソッドを更新します。
- 新しいモデル用のパラメータを追加する場合は、`generate_image`メソッドの引数と処理を変更します。

### 7.4 新しいエンドポイントの追加

- `routes.py`に新しいエンドポイントを定義します。
- 対応するスキーマを`schemas.py`に追加します。

## 8. トラブルシューティング

### 8.1 一般的な問題と解決策

#### モデルの読み込みエラー

**症状**: `No such file or directory: './app/models/dart/dart_model.safetensors'`

**解決策**: 
- モデルファイルが正しいパスに存在するか確認
- `download_models.sh`スクリプトを実行してモデルをダウンロード

#### CUDA関連エラー

**症状**: `CUDA error: no kernel image is available for execution on the device`

**解決策**:
- CUDA互換性の確認: `nvidia-smi`で確認
- PyTorchのCUDAバージョンを確認: `print(torch.version.cuda)`
- 必要に応じてDockerfileのCUDAバージョンを変更

#### メモリ不足エラー

**症状**: `CUDA out of memory`

**解決策**:
- バッチサイズを小さくする
- 画像解像度を下げる
- 低精度計算（fp16/int8）を使用する

### 8.2 特定コンポーネントの問題

#### RAG関連

**症状**: `Vector store not initialized` または `FAISS file not found`

**解決策**:
- `prepare_vector_store.py`を実行してベクトルストアを作成
- パス設定を確認

#### Dart関連

**症状**: `Invalid tag format` または `Token not found`

**解決策**:
- タグフォーマットを確認
- トークナイザーファイルの存在を確認

#### 画像生成関連

**症状**: `Blank image generated` または `Low quality image`

**解決策**:
- CFGスケールを調整（通常7-9が適切）
- ステップ数を増やす（20-30程度）
- ネガティブプロンプトを設定

## 9. ライセンス情報

- **ソースコード**: MIT License
- **使用モデル**:
  - Dartモデル: [p1atdev/dart-v2-moe-sft](https://huggingface.co/p1atdev/dart-v2-moe-sft) - オリジナルライセンスに従う
  - Animagineモデル: [Linaqruf/animagine-xl-3.0](https://huggingface.co/Linaqruf/animagine-xl-3.0) - オリジナルライセンスに従う
- **データセット**:
  - Danbooru Wiki: [isek-ai/danbooru-wiki-2024](https://huggingface.co/datasets/isek-ai/danbooru-wiki-2024) - オリジナルライセンスに従う

## 10. リファレンス

- [日本語をAIの呪文に変換するLLMシステムの開発](https://note.com/tori29umai/n/n5006040bf465)
- [Dart + 画像生成を組み合わせた実装例](https://zenn.dev/mattyamonaca/articles/8afa9d5067577c)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/index) 