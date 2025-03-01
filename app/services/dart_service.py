from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import settings
from core.errors import DartError
from utils.tag_utils import clean_tags, format_dart_output

class DartService:
    """Dartによるタグ補完サービス"""
    
    def __init__(self):
        # """初期化"""
        # try:
            # Dartモデルとトークナイザーの読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(settings.DART_REPO_ID)
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.DART_REPO_ID,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        # except Exception as e:
        #     raise DartError(f"Dartモデルの読み込みに失敗しました: {str(e)}")
    
    def _format_input_for_dart(self, tag_candidates: List[str]) -> str:
        """Dart入力用フォーマットに変換"""
        # タグをカンマ区切りの文字列に変換
        tags_str = ",".join(tag_candidates)
        
        # Dartの入力フォーマットに変換
        formatted_input = (
            f"<|bos|>"
            f"<copyright></copyright>"
            f"<character></character>"
            f"<|rating:general|><|aspect_ratio:tall|><|length:long|>"
            f"<general>{tags_str}"
        )
        return formatted_input
    
    async def generate_final_tags(self, tag_candidates: List[str]) -> List[str]:
        """タグ候補からDartを使用して最終タグを生成"""
        try:
            # 入力フォーマット作成
            dart_input = self._format_input_for_dart(tag_candidates)
            print(dart_input)
            
            # トークン化
            inputs = self.tokenizer(
                dart_input,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                return_token_type_ids=False  # token_type_idsを無効化
            ).to(self.model.device)
            
            # 補完生成
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    # attention_mask=inputs.attention_mask,
                    # max_new_tokens=256,
                    # do_sample=True,
                    # temperature=0.5,
                    # top_p=0.9,
                    # repetition_penalty=1.2,
                    # pad_token_id=self.tokenizer.eos_token_id
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=100,
                    max_new_tokens=128,
                    num_beams=1,
                )
            
            # デコード
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # 結果をパースしてタグリスト取得
            final_tags = format_dart_output(generated_text)
            
            # タグのクリーニング（重複除去など）
            return clean_tags(final_tags)
        
        except Exception as e:
            raise DartError(f"タグ補完生成中にエラーが発生しました: {str(e)}") 