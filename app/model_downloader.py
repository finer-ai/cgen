import logging
from utils.model_downloader import ensure_models_downloaded

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """必要なモデルファイルをダウンロードする"""
    logger.info("Checking and downloading required models...")
    ensure_models_downloaded()
    logger.info("Model download completed") 
    
if __name__ == "__main__":
    download_models()
