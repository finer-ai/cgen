import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from core.config import settings

def load_llm(use_local_llm: bool = False):
    """LLMを初期化する

    Args:
        use_local_llm (bool, optional): ローカルのLLMを使用するかどうか. Defaults to False.

    Returns:
        Union[HuggingFacePipeline, ChatOpenAI]: 初期化されたLLMインスタンス
    """
    if use_local_llm:
        # モデルとトークナイザーの読み込み
        model = AutoModelForCausalLM.from_pretrained(
            settings.MISTRAL_REPO_ID,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(settings.MISTRAL_REPO_ID)
        # pad_tokenが設定されていない場合はeos_tokenを使用
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # パイプラインの作成
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            do_sample=True,  # サンプリングを有効化
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            torch_dtype=torch.bfloat16  # 型を明示的に指定
        )
        
        # LangChainのパイプラインとして設定
        return HuggingFacePipeline(pipeline=pipe)
    else:
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY
        ) 