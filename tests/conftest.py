import pytest
import sys
import os
from pathlib import Path

# プロジェクトルートへのパスを追加
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# 環境変数の設定
os.environ["TESTING"] = "True" 