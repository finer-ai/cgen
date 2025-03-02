import pytest
from unittest.mock import Mock, patch, MagicMock
import base64
from PIL import Image
import io
from app.services.bodyline_service import BodylineService
from app.core.config import settings

@pytest.fixture
def mock_controlnet():
    with patch('app.services.bodyline_service.ControlNetModel') as mock:
        mock_instance = Mock()
        mock.from_single_file.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_pipeline():
    with patch('app.services.bodyline_service.StableDiffusionControlNetPipeline') as mock:
        mock_instance = Mock()
        mock.from_single_file.return_value = mock_instance
        mock_instance.to.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_image():
    """テスト用のダミー画像を作成"""
    img = Image.new('RGB', (100, 100), color='white')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@pytest.fixture
def bodyline_service(mock_controlnet, mock_pipeline):
    return BodylineService()

class TestBodylineService:
    async def test_init(self, mock_controlnet, mock_pipeline):
        """初期化のテスト"""
        service = BodylineService()
        
        # ControlNetの初期化を確認
        assert service.controlnet == mock_controlnet
        
        # パイプラインの初期化を確認
        assert service.pipeline == mock_pipeline
        assert mock_pipeline.to.called_with(settings.DEVICE)

    async def test_resize_for_controlnet(self, bodyline_service, sample_image):
        """resize_for_controlnetメソッドのテスト"""
        result = await bodyline_service.resize_for_controlnet(sample_image)
        
        # 結果がPIL.Imageインスタンスであることを確認
        assert isinstance(result, Image.Image)
        
        # サイズが512x512であることを確認
        assert result.size == (512, 512)

    async def test_generate_bodyline(self, bodyline_service):
        """generate_bodylineメソッドのテスト"""
        # モック画像を作成
        test_image = Image.new('RGB', (512, 512), color='white')
        
        # パイプラインの出力をモック
        mock_output = MagicMock()
        mock_output.images = [test_image]
        bodyline_service.pipeline.return_value = mock_output
        
        result = await bodyline_service.generate_bodyline(
            control_image=test_image,
            prompt="test prompt",
            negative_prompt="test negative",
            num_inference_steps=20,
            guidance_scale=7.0
        )
        
        # 結果の構造を確認
        assert "image" in result
        assert "parameters" in result
        assert result["image"].startswith("data:image/png;base64,")
        
        # パラメータを確認
        params = result["parameters"]
        assert params["prompt"] == "test prompt"
        assert params["negative_prompt"] == "test negative"
        assert params["num_inference_steps"] == 20
        assert params["guidance_scale"] == 7.0
        
        # パイプラインが正しいパラメータで呼び出されたことを確認
        bodyline_service.pipeline.assert_called_once_with(
            prompt="test prompt",
            negative_prompt="test negative",
            image=test_image,
            num_inference_steps=20,
            guidance_scale=7.0
        ) 