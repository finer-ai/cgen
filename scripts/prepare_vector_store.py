import pandas as pd
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# 設定
OUTPUT_DIR = "./app/data/faiss"
DATASET_PATH = "./data/danbooru-wiki-2024_df.pkl"

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CustomEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return np.array(self.model.encode(texts), dtype=np.float32)

def load_danbooru_tags(file_path):
    print(f"データセット読み込み中: {file_path}")
    df = pd.read_pickle(file_path)
    print(f"読み込み完了: {len(df)}件のタグデータ")
    return df

def create_vector_store(df):
    print("ベクトルストア作成開始...")
    
    # 埋め込みモデル読み込み
    print("埋め込みモデル読み込み中...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    embeddings = CustomEmbeddings(model)
    
    # ドキュメント作成
    documents = []
    print("ドキュメント作成中...")
    for _, row in df.iterrows():
        tag = row['title']
        text = row['body']
        other_names = row['other_names']
        doc = f"{tag}: {text} (Other names: {', '.join(other_names) if isinstance(other_names, list) else other_names})"
        documents.append(doc)
    
    # ベクトル生成
    print("ベクトル生成中...")
    vectors = []
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_vectors = embeddings.embed_documents(batch)
        vectors.extend(batch_vectors)
        print(f"処理中: {i+len(batch)}/{len(documents)}")
    
    # FAISSインデックス作成
    print("FAISSインデックス作成中...")
    vectors_np = np.array(vectors)
    index = faiss.IndexFlatL2(vectors_np.shape[1])
    index.add(vectors_np)
    
    # 保存
    index_path = Path(OUTPUT_DIR) / "danbooru_tags.faiss"
    docs_path = Path(OUTPUT_DIR) / "danbooru_documents.txt"
    
    print(f"インデックス保存中: {index_path}")
    faiss.write_index(index, str(index_path))
    
    print(f"ドキュメント保存中: {docs_path}")
    with open(docs_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")
    
    print("ベクトルストア作成完了")
    return index, documents

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"エラー: データセットファイルが見つかりません: {DATASET_PATH}")
        print("データセットを準備してください。")
        exit(1)
        
    df = load_danbooru_tags(DATASET_PATH)
    index, documents = create_vector_store(df)
    print(f"完了: {len(documents)}件のタグデータをインデックス化しました") 