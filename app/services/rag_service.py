import os
import json
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from core.config import settings
from core.errors import RAGError
from utils.tag_utils import clean_tags

class KeywordsOutput(BaseModel):
    """キーワード抽出の出力形式"""
    keywords: List[str] = Field(description="抽出されたキーワードのリスト")

class RAGService:
    """RAGによるタグ候補抽出サービス"""
    
    def __init__(self):
        """初期化"""
        # 埋め込みモデル設定
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # ベクトルDB読み込み
        if os.path.exists(settings.VECTOR_DB_PATH):
            self.vector_store = FAISS.load_local(
                settings.VECTOR_DB_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True  # 信頼できるローカルファイルの場合のみTrueに設定
            )
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 10}
            )
        else:
            raise RAGError("ベクトルDBが見つかりません。初期化が必要です。")
        
        # LLM設定
        self.llm = OpenAI(temperature=0.1, openai_api_key=settings.OPENAI_API_KEY)
        
        # 出力パーサーの設定
        self.parser = PydanticOutputParser(pydantic_object=KeywordsOutput)
        
        # キーワード抽出用プロンプト
        self.keyword_prompt = PromptTemplate(
            input_variables=["query", "format_instructions"],
            template="""
            以下の説明文から、画像生成に役立つキーワードを抽出してください。
            説明文は日本語に限りませんが、出力は英語にしてください。
            
            説明文: {query}
            
            {format_instructions}
            """
        )
        
        self.keyword_chain = LLMChain(
            llm=self.llm,
            prompt=self.keyword_prompt,
            output_parser=self.parser
        )
    
    async def extract_keywords(self, query: str) -> List[str]:
        """ユーザー入力からキーワードを抽出"""
        try:
            # パーサーのフォーマット指示を取得
            format_instructions = self.parser.get_format_instructions()
            
            # キーワード抽出を実行
            result = await self.keyword_chain.arun(
                query=query,
                format_instructions=format_instructions
            )
            
            # 結果は既にパース済みのKeywordsOutputオブジェクト
            return result.keywords
            
        except Exception as e:
            raise RAGError(f"キーワード抽出中にエラーが発生しました: {str(e)}")
    
    async def retrieve_tags(self, keywords: List[str]) -> List[str]:
        """キーワードに関連するタグ候補を取得"""
        try:
            # キーワードを結合してクエリ作成
            query = " ".join(keywords)
            # ベクトル検索実行
            docs = self.retriever.get_relevant_documents(query)
            
            # 取得したドキュメントからタグを抽出
            all_tags = []
            for doc in docs:
                content = doc.page_content
                # タグ情報を抽出 (ここでは簡易的に実装)
                tags = [line.split(":")[0].strip() for line in content.split("\n") 
                        if ":" in line and not line.startswith("#")]
                all_tags.extend(tags)
            
            # 重複を除去し、整形
            return clean_tags(all_tags)
        except Exception as e:
            raise RAGError(f"タグ候補取得中にエラーが発生しました: {str(e)}")
    
    async def generate_tag_candidates(self, prompt: str) -> List[str]:
        """プロンプトからタグ候補を生成する統合メソッド"""
        # キーワード抽出
        keywords = await self.extract_keywords(prompt)
        print('keywords', keywords)
        # 関連タグ取得
        candidates = await self.retrieve_tags(keywords)
        print('candidates', candidates)
        return candidates 