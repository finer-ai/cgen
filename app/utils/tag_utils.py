from typing import List
import re

def clean_tags(tags: List[str]) -> List[str]:
    """タグリストをクリーニング
    
    - 重複を削除
    - 空白や特殊文字を処理
    - 不要タグをフィルタリング
    """
    # 空白文字とカンマを処理
    processed_tags = []
    for tag in tags:
        tag = tag.strip()
        # カンマを含むタグは分割
        if "," in tag:
            split_tags = [t.strip() for t in tag.split(",")]
            processed_tags.extend(split_tags)
        else:
            processed_tags.append(tag)
    
    # 空の要素を除去
    processed_tags = [tag for tag in processed_tags if tag]
    
    # 重複を削除して整列
    unique_tags = list(dict.fromkeys(processed_tags))
    
    # 不要なタグをフィルタリング（例: タグではない説明文などを除去）
    filtered_tags = [
        tag for tag in unique_tags 
        if not tag.startswith("http") and not len(tag) > 50
    ]
    
    return filtered_tags

def format_dart_output(generated_text: str) -> List[str]:
    """Dartの出力をタグリストにパース"""
    # 出力からタグ部分を抽出
    match = re.search(r"<general>(.*?)</general>", generated_text)
    if match:
        tags_text = match.group(1)
        # カンマまたはスペースで区切られたタグを分割
        raw_tags = re.split(r",|\s+", tags_text)
        # 空のタグを除去
        tags = [tag.strip() for tag in raw_tags if tag.strip()]
        return tags
    else:
        # タグが見つからない場合は空リストを返す
        return [] 