from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from core.config import settings
from core.errors import DartError
from utils.tag_utils import clean_tags, format_dart_output

class DartService:
    """Dartによるタグ補完サービス"""
    
    def __init__(self, llm):
        """初期化
        
        Args:
            llm (LangChainLLM): 使用するLLMオブジェクト
        """
        # Dartモデルとトークナイザーの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(settings.DART_REPO_ID, device_map="auto")
        # pad_tokenが設定されていない場合はeos_tokenを使用
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # GPUが利用可能な場合はGPUを使用
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.DART_REPO_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.llm = llm
        
        # タグフィルタリング用のプロンプトテンプレート
        filter_template = """
From the tag list below, please remove tags that significantly deviate from the given context,
and keep only the tags that are relevant.
Please output the filtered tags as a comma-separated list.
Note that the context represents the content we want to include in the prompt, so please keep the tags that align with the context.

# Example Start ####
Example Context: 女の子がジャンプしているポーズを描いてください。セーラー服ではなくブレザーを着ている。
Example Tag List: original, 1girl, jumping, embarrassed, jacket, solo, outdoors, day, long hair, skirt, kneehighs, smile, looking at viewer, miniskirt, monochrome, monster, open clothes, open jacket, open mouth, pleated skirt, shoes, socks, teeth, upper teeth only, wide shot, wind, wind lift
Example Output: original, 1girl, jumping, jacket, solo, long hair, skirt, smile, looking at viewer, miniskirt, open jacket, shoes, socks
# Example End ####
Note: (deleted tags: embarrassed, outdoors, day, kneehighs, monochrome, monster, open clothes, open mouth, pleated skirt, teeth, upper teeth only, wide shot, wind, wind lift)

Context: {context_prompt}

Tag List: {tags_str}

Output:"""

        # タグの重み付け用のプロンプトテンプレート
        weight_template = """
Please analyze the context and add weights ONLY when there are explicit or strongly implied modifiers in the context.
Be aggressive in applying weights when modifiers are present, but DO NOT add weights when there are no modifiers.

Important rules for weight application:
1. NO WEIGHTS by default:
   - If a tag has no direct or implied modifier, leave it as is without weights
   - Do not add weights based on assumptions or general knowledge
   - Only add weights when there is clear evidence in the context

2. Direct modifiers - Apply to ALL related tags when present:
   - Low intensity (:0.3): ちょっと (a little), 少し (slightly), やや (somewhat), 微妙に (subtly), 控えめに (moderately)
   - High intensity (:1.7): すごく (very), とても (extremely), 非常に (highly), かなり (considerably), めちゃくちゃ (incredibly)

3. Mood/Atmosphere modifiers - Apply ONLY when explicitly mentioned or strongly implied:
   When the context explicitly mentions these moods, apply weights to ALL related tags:
   
   a) Suggestive/Erotic mood:
      - If words like エッチ, セクシー, 色気 appear:
        * Apply :0.3 to: sexually_suggestive, revealing_clothes, suggestive_pose, etc.
        * Also apply :0.3 to related tags: thighhighs, miniskirt, bare_shoulders, etc.
   
   b) Cute/Innocent mood:
      - If words like 可愛い, innocent appear:
        * Apply :1.7 to: cute, innocent, pure, etc.
        * Also apply :1.7 to related tags: smile, flower, pastel_colors, etc.
   
   c) Intensity modifiers stack with mood:
      - "ちょっとエッチ" = (sexually_suggestive:0.3)
      - "すごくエッチ" = (sexually_suggestive:1.7)

4. Context Analysis - Strict Rules:
   - Only consider explicit modifiers or very strong contextual implications
   - Do not add weights based on subtle implications or assumptions
   - When in doubt, leave the tag without weights

# Examples ####
Example 1 (With modifiers):
Context: ちょっとエッチな感じで女の子がジャンプしているポーズを描いてください。
Tags: original, 1girl, jumping, solo, sexually_suggestive, skirt, smile, thighhighs
Output: original, 1girl, jumping, solo, (sexually_suggestive:0.3), (skirt:0.3), smile, (thighhighs:0.3)

Example 2 (No modifiers):
Context: 女の子がジャンプしているポーズを描いてください。制服を着ています。
Tags: original, 1girl, jumping, solo, school_uniform, skirt, smile
Output: original, 1girl, jumping, solo, school_uniform, skirt, smile

Example 3 (Mixed modifiers):
Context: すごく可愛らしい女の子が、ちょっとセクシーな制服を着て走っています。
Tags: original, 1girl, running, school_uniform, cute, skirt, thighhighs, smile
Output: original, 1girl, running, (school_uniform:0.3), (cute:1.7), (skirt:0.3), (thighhighs:0.3), (smile:1.7)

Example 4 (Implicit but strong):
Context: 制服がびしょ濡れで、体にぴったりと張り付いている女の子が立っています。
Tags: original, 1girl, standing, school_uniform, wet_clothes, transparent, smile
Output: original, 1girl, standing, school_uniform, (wet_clothes:0.3), (transparent:0.3), smile

Example 5 (No clear modifiers):
Context: 放課後の教室で本を読んでいる女の子を描いてください。
Tags: original, 1girl, classroom, reading, book, school_uniform, afternoon
Output: original, 1girl, classroom, reading, book, school_uniform, afternoon

Context: {context_prompt}

Tags: {tags_str}

Output:"""

        self.filter_prompt = PromptTemplate(
            template=filter_template,
            input_variables=["context_prompt", "tags_str"]
        )

        self.weight_prompt = PromptTemplate(
            template=weight_template,
            input_variables=["context_prompt", "tags_str"]
        )
        
        # フィルタリングチェーン
        self.filter_chain = RunnableSequence(
            self.filter_prompt | self.llm | StrOutputParser()
        )

        # 重み付けチェーン
        self.weight_chain = RunnableSequence(
            self.weight_prompt | self.llm | StrOutputParser()
        )
    
    def _format_input_for_dart(self, tag_candidates: List[str]) -> str:
        """Dart入力用フォーマットに変換"""
        # タグをカンマ区切りの文字列に変換
        tags_str = ", ".join(tag_candidates)
        
        # Dartの入力フォーマットに変換
        formatted_input = (
            f"<|bos|>"
            f"<copyright>original</copyright>"
            f"<character></character>"
            f"<|rating:general|><|aspect_ratio:tall|><|length:long|>"
            f"<general>{tags_str}"
        )
        return formatted_input
    
    async def generate_final_tags(self, tag_candidates: List[str]) -> List[str]:
        # """タグ候補からDartを使用して最終タグを生成"""
        # 入力フォーマット作成
        dart_input = self._format_input_for_dart(tag_candidates)
        # トークン化
        inputs = self.tokenizer(
            dart_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # 一般的なトランスフォーマーモデルの標準的な長さ
            return_attention_mask=True,
        ).to(self.model.device)

        # 補完生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                top_k=50,
                max_new_tokens=64,
                num_beams=1,  # do_sample=Trueの場合は1に設定
                repetition_penalty=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # デコード
        generated_text = ", ".join([tag for tag in self.tokenizer.batch_decode(outputs[0], skip_special_tokens=True) if tag.strip() != ""])
        
        # 結果をパースしてタグリスト取得
        final_tags = format_dart_output(generated_text)
        
        # タグのクリーニング（重複除去など）
        return clean_tags(final_tags)

    async def filter_tags_by_context(self, tags_str: str, context_prompt: str) -> List[str]:
        """コンテキストに基づいてタグをフィルタリングし、重み付けを行う

        Args:
            tags_str (str): カンマ区切りのタグ文字列
            context_prompt (str): フィルタリングの基準となるコンテキスト

        Returns:
            List[str]: フィルタリングと重み付けが適用されたタグリスト
        """
        # Step 1: フィルタリング - コンテキストに関連のないタグを除去
        filtered_result = await self.filter_chain.ainvoke({
            "context_prompt": context_prompt,
            "tags_str": tags_str
        })
        
        # フィルタリング結果をトリムしてリスト化
        filtered_tags = [tag.replace('_', ' ').strip() for tag in filtered_result.split(",") if tag.strip()]
        # Step 2: 重み付け - フィルタリングされたタグに対して重みを適用
        weighted_result = await self.weight_chain.ainvoke({
            "context_prompt": context_prompt,
            "tags_str": ", ".join(filtered_tags)
        })

        # 重み付けされた結果をリスト化
        weighted_tags = [tag.replace('_', ' ').strip() for tag in weighted_result.split(",") if tag.strip()]
        # タグのクリーニング（重複除去など）
        return clean_tags(weighted_tags)
        