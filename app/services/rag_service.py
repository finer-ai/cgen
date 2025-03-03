import os
import re
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

from core.config import settings
from core.errors import RAGError
from utils.tag_utils import clean_tags
from utils.llm_utils import load_llm
from tqdm import tqdm

class RAGService:
    """RAGによるタグ候補抽出サービス"""

    def __init__(self, use_local_llm=False):
        # GPUが利用可能な場合はGPUを使用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # LLMの設定
        self.llm = load_llm(use_local_llm=use_local_llm)

        # タグ候補生成用LLMChain：シーン説明からカンマ区切りのタグ候補を生成
        self.set_tag_candidate_generation_template("""
Generate Danbooru tags from the prompt. The prompt may either be a question or instruction requesting a scene to be drawn, or a direct description of the desired scene without a question.
First, extract the scene. Then, generate up to 20 Danbooru tags, separated by commas, detailing all relevant aspects including character count, group tags, and other necessary details.

Important character count rules:
- Unless explicitly specified otherwise in the prompt, assume there is only one character.
- For a single character, use exactly one character tag (e.g., '1girl', '1boy', or '1other') along with 'solo'.
- For multiple characters, use tags that match the exact number of characters (e.g., '2girls', '2boys', 'multiple_girls').
- When gender is not explicitly specified in the prompt, prioritize using '1girl' over '1boy'.

Additional rules:
- If there is a single human or humanoid character (including orcs, elves, etc.), use 'solo' along with '1boy' or '1girl'.
- If there is a single non-humanoid character, use 'solo' along with '1other'.
- If applicable, combine species tags (e.g., 'orc', 'elf') with character count tags.

Prompt: {prompt}
Tags (up to 20):""")

        # タグ補正用LLMChain：retrieverで取得した文脈を元に、入力タグの正確なマッチを行う
        self.set_tag_normalization_template("""
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
Output:""")

    def set_tag_candidate_generation_template(self, template: str):
        self.tag_candidate_generation_prompt = PromptTemplate(
            template=template,
            input_variables=["prompt"]
        )
        self.tag_candidate_generation_chain = RunnableSequence(
            self.tag_candidate_generation_prompt | self.llm | StrOutputParser()
        )

    def set_tag_normalization_template(self, template: str):
        self.tag_normalization_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "query"]
        )
        self.tag_normalization_chain = self.tag_normalization_prompt | self.llm.bind(stop=["\n"])
        
    async def extract_elements(self, query: str) -> list:
        """
        シーン説明からDanbooruタグ候補（キーワード）を抽出する。
        出力はカンマ区切りの文字列を分割してList[str]に変換。
        """
        result = await self.tag_candidate_generation_chain.ainvoke({"prompt": query})
        # カンマで分割し、前後の空白を除去
        elements = [word.strip() for word in result.split(",") if word.strip()]
        return elements

    async def retrieve_tags(self, elements: list) -> list:
        """
        キーワードごとにretrieverから関連文脈を取得し、
        tag_normalization_chainで各キーワードの正確なタグを補正する。
        """
        refined_tags = []
        seen = set()
        pbar = tqdm(elements, desc="タグマッチング中")
        import random
        for element in pbar:
            docs = self.retriever.invoke(element.replace('_', ' '))
            context = [doc.page_content for doc in docs]
            context.reverse() # 最後の要素が重視されがちなので、逆順にする

            context_str = ", ".join(context)
            tag_message = await self.tag_normalization_chain.ainvoke({"query": element, "context": context_str})
            tag = tag_message.content if hasattr(tag_message, 'content') else str(tag_message)
            
            # Remove ** from the tag and remove square brackets
            cleaned_tag = re.sub(r'\[+|\]+', '', tag.strip())
            cleaned_tag = re.sub(r'\*+', '', cleaned_tag)
            cleaned_tag = re.sub(r'^-+', '', cleaned_tag)
            cleaned_tag = cleaned_tag.replace('_', ' ').split(':')[0].split(',')[0].strip()
            
            # pbar.write(f'{element} -> {tag} -> {cleaned_tag}')

            if cleaned_tag and cleaned_tag not in seen:
                if 'style' in cleaned_tag:
                    continue
                refined_tags.append(cleaned_tag)
                seen.add(cleaned_tag)

        return clean_tags(refined_tags)
    
    async def generate_tag_candidates(self, prompt: str) -> list:
        """
        プロンプト（シーン説明）からタグ候補を生成する統合メソッド。
        1. extract_elementsでキーワード（タグ候補）を抽出
        2. retrieve_tagsで各キーワードに対し補正を実施
        3. 最終的なタグ候補リストを返す
        """
        prompt_elements = await self.extract_elements(prompt)
        candidates = await self.retrieve_tags(prompt_elements)
        return candidates
