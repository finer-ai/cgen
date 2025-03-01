import os
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional

class Settings(BaseSettings):
    """アプリケーション設定"""
    # アプリ基本設定
    APP_NAME: str = "AnimeImageGenerator"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # モデルパス設定
    DART_MODEL_PATH: str = os.getenv("DART_MODEL_PATH", "/app/models/dart")
    DART_REPO_ID: str = os.getenv("DART_REPO_ID", "p1atdev/dart-v2-moe-sft")
    SD_MODEL_PATH: str = os.getenv("SD_MODEL_PATH", "/app/models/animagine-xl-4.0.safetensors")
    SD_REPO_ID: str = os.getenv("SD_REPO_ID", "Linaqruf/animagine-xl-4.0")
    LLAMA_MODEL_PATH: str = os.getenv("LLAMA_MODEL_PATH", "/app/models/llama3.1")
    LLAMA_REPO_ID: str = os.getenv("LLAMA_REPO_ID", "meta-llama/Llama-3.1-8B-Instruct")
    MISTRAL_MODEL_PATH: str = os.getenv("MISTRAL_MODEL_PATH", "./models/mistral")
    MISTRAL_REPO_ID: str = os.getenv("MISTRAL_REPO_ID", "mistralai/Mistral-7B-v0.1")

    # RAG設定
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "/app/data/faiss")
    
    # APIキー設定（オプション）
    API_KEY: Optional[str] = os.getenv("API_KEY")
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "False").lower() == "true"
    
    # LLM設定
    LLM_TYPE: str = os.getenv("LLM_TYPE", "openai")  # openai, llama, etc.
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

    class Config:
        env_file = ".env"

settings = Settings() 