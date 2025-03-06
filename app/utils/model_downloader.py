import os
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download
from core.config import settings
import logging

logger = logging.getLogger(__name__)

def download_file(url: str, path: str, requires_auth: bool = False) -> None:
    """ファイルをダウンロードする

    Args:
        url (str): ダウンロードURL
        path (str): 保存先パス
        requires_auth (bool, optional): 認証が必要かどうか. Defaults to False.
    """
    headers = {}
    if requires_auth:
        if not settings.HF_TOKEN:
            raise ValueError("HF_TOKEN is not set")
        headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1KB

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f, tqdm(
        desc=os.path.basename(path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def ensure_models_downloaded() -> None:
    """必要なモデルファイルが存在しない場合はダウンロードする"""
    for model_config in settings.MODEL_CONFIGS:
        model_path = model_config["path"]
        
        # パスに拡張子がない場合は、Hugging Faceのリポジトリとして扱う
        if not os.path.splitext(model_path)[1]:
            logger.info(f"Downloading repository {model_config['name']}...")
            try:
                if model_config["requires_auth"] and not settings.HF_TOKEN:
                    raise ValueError("HF_TOKEN is not set")
                
                snapshot_download(
                    repo_id=model_path,
                    token=settings.HF_TOKEN if model_config["requires_auth"] else None,
                    local_dir=os.path.join("models", model_config["name"]),
                    local_dir_use_symlinks=False
                )
                logger.info(f"Successfully downloaded repository {model_config['name']}")
            except Exception as e:
                logger.error(f"Failed to download repository {model_config['name']}: {str(e)}")
                raise
        else:
            # 通常のファイルダウンロード
            if not os.path.exists(model_path):
                logger.info(f"Downloading {model_config['name']}...")
                try:
                    download_file(
                        url=model_config["url"],
                        path=model_path,
                        requires_auth=model_config["requires_auth"]
                    )
                    logger.info(f"Successfully downloaded {model_config['name']}")
                except Exception as e:
                    logger.error(f"Failed to download {model_config['name']}: {str(e)}")
                    raise 