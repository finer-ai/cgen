import os
import json
from typing import List, Dict, Any
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import VectorStoreRetriever

from core.config import settings
from core.errors import RAGError
from utils.tag_utils import clean_tags

class RAGService:
    """RAGによるタグ候補抽出サービス"""
    
    def __init__(self):
        """初期化"""
        # 埋め込みモデル設定
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        
        # ベクトルDB読み込み
        if os.path.exists(settings.VECTOR_DB_PATH):
            self.vector_store = FAISS.load_local(
                settings.VECTOR_DB_PATH, 
                self.embeddings
            )
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 10}
            )
        else:
            raise RAGError("ベクトルDBが見つかりません。初期化が必要です。")
        
        # LLM設定
        self.llm = OpenAI(temperature=0.1, openai_api_key=settings.OPENAI_API_KEY)
        
        # キーワード抽出用プロンプト
        self.keyword_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            以下の説明文から、画像生成に役立つキーワードを抽出してください。
            日本語と英語の両方を含めて構いません。
            
            説明文: {query}
            
            抽出したキーワードをJSON形式で返してください:
            {"keywords": ["キーワード1", "キーワード2", ...]}
            """
        )
        
        self.keyword_chain = LLMChain(llm=self.llm, prompt=self.keyword_prompt)
    
    async def extract_keywords(self, query: str) -> List[str]:
        """ユーザー入力からキーワードを抽出"""
        try:
            result = await self.keyword_chain.arun(query=query)
            # JSON部分を抽出
            json_str = result.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)["keywords"]
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
        # 関連タグ取得
        candidates = await self.retrieve_tags(keywords)
        return candidates 