import os
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional, ClassVar
from pydantic import ConfigDict

class Settings(BaseSettings):
    """アプリケーション設定"""
    model_config = ConfigDict(env_file=".env")

    # アプリ基本設定
    APP_NAME: str = "AnimeImageGenerator"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # モデルパス設定
    root_path: str = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    print(f"root_path: {root_path}")
    MODEL_DIR: ClassVar[str] = f"{root_path}/models"
    DART_MODEL_PATH: str = os.getenv("DART_MODEL_PATH", f"{MODEL_DIR}/dart")
    DART_REPO_ID: str = os.getenv("DART_REPO_ID", "p1atdev/dart-v2-moe-sft")
    SD_MODEL_PATH: str = os.getenv("SD_MODEL_PATH", "cagliostrolab/animagine-xl-4.0")
    # SD_REPO_ID: str = os.getenv("SD_REPO_ID", "Linaqruf/animagine-xl-4.0")
    LLAMA_MODEL_PATH: str = os.getenv("LLAMA_MODEL_PATH", f"{MODEL_DIR}/llama3.1")
    LLAMA_REPO_ID: str = os.getenv("LLAMA_REPO_ID", "meta-llama/Llama-3.1-8B-Instruct")
    MISTRAL_MODEL_PATH: str = os.getenv("MISTRAL_MODEL_PATH", f"{MODEL_DIR}/mistral")
    MISTRAL_REPO_ID: str = os.getenv("MISTRAL_REPO_ID", "mistralai/Mistral-7B-v0.1")

    # RAG設定
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", f"{root_path}/data/faiss")
    
    # APIキー設定（オプション）
    API_KEY: Optional[str] = os.getenv("API_KEY")
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "False").lower() == "true"
    
    # LLM設定
    LLM_TYPE: str = os.getenv("LLM_TYPE", "openai")  # openai, llama, etc.
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

    # モデルパス
    SD15_MODEL_PATH: str = f"{MODEL_DIR}/LoRAMergeModel_animepose_outline_sotai.fp16.safetensors"
    CONTROLNET_CONFIGS: List[Dict[str, Any]] = [
        {
            "path": f"{MODEL_DIR}/Sotai_1K_ControlNet-epoch=989.ckpt",
            "conditioning_scale": 1.4
        },
        {
            "path": f"{MODEL_DIR}/Sotai_sketch_ControlNet_epoch=0288_train_loss_epoch=9.7849e-03.ckpt",
            "conditioning_scale": 1.3
        }
    ]
    
    # その他の設定
    DEVICE: str = "cuda"
    TORCH_DTYPE: str = "float16"

    # モデルのパスとURL設定
    MODEL_CONFIGS: List[Dict[str, Any]] = [
        {
            "name": "animagine-xl-4.0",
            "path": "cagliostrolab/animagine-xl-4.0",
            "requires_auth": False
        },
        {
            "name": "sotai-sd-model",
            "path": f"{MODEL_DIR}/LoRAMergeModel_animepose_outline_sotai.fp16.safetensors",
            "url": "https://huggingface.co/yeq6x/webui-models/resolve/main/Stable-diffusion/LoRAMergeModel_animepose_outline_sotai.fp16.safetensors",
            "requires_auth": True
        },
        {
            "name": "sotai-sketch-controlnet",
            "path": f"{MODEL_DIR}/Sotai_sketch_ControlNet_epoch=0288_train_loss_epoch=9.7849e-03.ckpt",
            "url": "https://huggingface.co/yeq6x/webui-models/resolve/main/ControlNet/Sotai_sketch_ControlNet_epoch%3D0288_train_loss_epoch%3D9.7849e-03.ckpt",
            "requires_auth": True
        },
        {
            "name": "sotai-1k-controlnet",
            "path": f"{MODEL_DIR}/Sotai_1K_ControlNet-epoch=989.ckpt",
            "url": "https://huggingface.co/yeq6x/webui-models/resolve/main/ControlNet/Sotai_1K_ControlNet-epoch%3D989.ckpt",
            "requires_auth": True
        },
        {
            "name": "anime-seg",
            "path": f"{MODEL_DIR}/anime-seg/isnetis.onnx",
            "url": "https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.onnx",
            "requires_auth": False
        }
    ]

settings = Settings() 