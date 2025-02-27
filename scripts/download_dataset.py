import pandas as pd
from datasets import load_dataset
import os

# 出力ディレクトリ
os.makedirs("./data", exist_ok=True)

print("Danbooruタグデータセットをダウンロード中...")
# Hugging Faceからデータセットをダウンロード
dataset = load_dataset("isek-ai/danbooru-wiki-2024", split="train")

# DataFrameに変換
df = pd.DataFrame(dataset)

# Pickleファイルとして保存
output_path = "./data/danbooru-wiki-2024_df.pkl"
print(f"データセットを保存中: {output_path}")
df.to_pickle(output_path)

print(f"完了：{len(df)}件のタグデータをダウンロードしました") 