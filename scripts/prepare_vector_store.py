import pandas as pd
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 設定
OUTPUT_DIR = "./app/data/faiss"
DATASET_PATH = "./app/data/danbooru-wiki-2024_df.pkl"

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_danbooru_tags(file_path):
    print(f"データセット読み込み中: {file_path}")
    df = pd.read_pickle(file_path)
    print(f"読み込み完了: {len(df)}件のタグデータ")
    return df

def create_vector_store(df):
    print("ベクトルストア作成開始...")
    
    # 埋め込みモデル読み込み
    print("埋め込みモデル読み込み中...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # ドキュメント作成
    documents = []
    print("ドキュメント作成中...")
    for _, row in df.iterrows():
        tag = row['tag']
        text = row['body']
        other_names = row['other_names']
        doc = f"{tag}"
        # doc = f"{tag}: {text} (Other names: {', '.join(other_names) if isinstance(other_names, list) else other_names})"
        documents.append(doc)
    
    # FAISSベクトルストア作成
    print("FAISSベクトルストア作成中...")
    vector_store = FAISS.from_texts(
        texts=documents,
        embedding=embeddings
    )
    
    # 保存
    print(f"ベクトルストア保存中: {OUTPUT_DIR}")
    vector_store.save_local(OUTPUT_DIR)
    
    print("ベクトルストア作成完了")
    return vector_store, documents

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"エラー: データセットファイルが見つかりません: {DATASET_PATH}")
        print("データセットを準備してください。")
        exit(1)
        
    df = load_danbooru_tags(DATASET_PATH)
    vector_store, documents = create_vector_store(df)
    print(f"完了: {len(documents)}件のタグデータをインデックス化しました") 