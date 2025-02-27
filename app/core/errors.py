class RAGError(Exception):
    """RAG処理中のエラー"""
    pass

class DartError(Exception):
    """Dart処理中のエラー"""
    pass

class ImageGenerationError(Exception):
    """画像生成中のエラー"""
    pass

class ConfigError(Exception):
    """設定関連のエラー"""
    pass 