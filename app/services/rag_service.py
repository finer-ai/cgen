import os
import re
from pathlib import Path
import pandas as pd

from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

from core.config import settings
from core.errors import RAGError
from utils.tag_utils import clean_tags

class RAGService:
    """RAGによるタグ候補抽出サービス"""

    def __init__(self):
        # 埋め込みモデル設定（retrieval用）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # ベクトルDBの読み込み／初期化（settings.VECTOR_DB_PATHを使用）
        if os.path.exists(settings.VECTOR_DB_PATH):
            self.vector_store = FAISS.load_local(
                settings.VECTOR_DB_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True  # 信頼できるローカルファイルの場合のみTrueに設定
            )
        else:
            raise RAGError("ベクトルDBが見つかりません。初期化が必要です。")
        
        # Retrieverの設定（検索件数k）
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        
        # LLMの設定（現状はOpenAIを利用、将来的にLlamaへの切り替え可能）
        self.llm = OpenAI(temperature=0.1, openai_api_key=settings.OPENAI_API_KEY)
        
        # キーワード（＝タグ候補）抽出用LLMChain（シーン説明からカンマ区切りのタグ候補を生成）
        rich_description_template = """
Generate a detailed comma-separated list of up to 20 Danbooru tags for this scene. Include character count, group tags, and all relevant details:
- Use 'solo' for a single character (human, humanoid, or non-humanoid)
- Use '1boy' or '1girl' for a single human or humanoid character (including orcs, elves, etc.) along with 'solo'
- Use '1other' for a single non-humanoid character along with 'solo'
- Use appropriate tags for multiple characters ('2boys', 'multiple_girls', etc.)
- Include species tags (e.g., 'orc', 'elf') along with character count tags when applicable
Scene description: {scene_description}
Tags (up to 20):
"""
        self.rich_description_prompt = PromptTemplate(
            template=rich_description_template,
            input_variables=["scene_description"]
        )
        self.rich_description_chain = RunnableSequence(
            self.rich_description_prompt | self.llm | StrOutputParser()
        )
        
        # タグ補正用LLMChain：retrieverで取得した文脈を元に、入力タグの正確なマッチを行う
        tag_response_template = """
You are a Danbooru tag matcher. Find exact matches in the context for the input.
Rules:
1. Return only exact matches from the context.
2. Check main tags and "Other names".
3. If no match, return input as is.
4. Provide one tag output per input.
5. Match tag names or other names, not descriptions.
6. For character counts and groups:
- Use 'solo' for a single character (human, humanoid, or non-humanoid)
- Use '1boy' for a single male character or humanoid male-presenting character (including orcs, elves, etc.)
- Use '1girl' for a single female character or humanoid female-presenting character
- Use '1other' for a single non-humanoid character
- For other character counts, use appropriate tags like '2boys', 'multiple_girls', etc.
7. Prioritize character count and group tags over other matches if applicable.
8. When using '1boy', '1girl', or '1other', also include the 'solo' tag.
9. Do not use square brackets or double square brackets around tags.
10. Match all relevant tags, including those for background, clothing, expressions, and actions.
11. For humanoid characters like orcs, elves, etc., use appropriate character count tags ('1boy', '1girl', etc.) along with their species tag.
Context: {context}
Input: {query}
Output:
"""
        self.tag_response_prompt = PromptTemplate(
            template=tag_response_template,
            input_variables=["context", "query"]
        )
        self.tag_response_chain = self.tag_response_prompt | self.llm | StrOutputParser()
    
    async def extract_keywords(self, query: str) -> list:
        """
        シーン説明からDanbooruタグ候補（キーワード）を抽出する。
        出力はカンマ区切りの文字列を分割してList[str]に変換。
        """
        result = await self.rich_description_chain.ainvoke({"scene_description": query})
        # カンマで分割し、前後の空白を除去
        keywords = [word.strip() for word in result.split(",") if word.strip()]
        return keywords

    async def retrieve_tags(self, keywords: list) -> list:
        """
        キーワードごとにretrieverから関連文脈を取得し、
        tag_response_chainで各キーワードの正確なタグを補正する。
        """
        refined_tags = []
        seen = set()
        for keyword in keywords:
            docs = self.retriever.invoke(keyword)
            page_contents = [doc.page_content for doc in docs]
            tags = [page_content.split(':')[0] for page_content in page_contents]
            print('tags', tags)
            context_str = ", ".join(tags)
            result = await self.tag_response_chain.ainvoke({"context": context_str, "query": keyword})
            cleaned_tag = re.sub(r'\[+|\]+', '', result.strip())


            # 追加ロジック：orcかつmaleの場合、'1boy'を補完
            if 'orc' in keyword.lower() and 'male' in keyword.lower() and '1boy' not in cleaned_tag:
                cleaned_tag = '1boy, ' + cleaned_tag
            if '1boy' in cleaned_tag and 'solo' not in cleaned_tag:
                cleaned_tag = 'solo, ' + cleaned_tag
            if cleaned_tag and cleaned_tag not in seen:
                refined_tags.append(cleaned_tag)
                seen.add(cleaned_tag)
        # 整形（重複除去など既存のclean_tags関数も利用可能）
        return clean_tags(refined_tags)
    
    async def generate_tag_candidates(self, prompt: str) -> list:
        """
        プロンプト（シーン説明）からタグ候補を生成する統合メソッド。
        1. extract_keywordsでキーワード（タグ候補）を抽出
        2. retrieve_tagsで各キーワードに対し補正を実施
        3. 最終的なタグ候補リストを返す
        """
        keywords = await self.extract_keywords(prompt)
        print('keywords', keywords)
        candidates = await self.retrieve_tags(keywords)
        print('candidates', candidates)
        return candidates
