import pytest
from unittest.mock import Mock, patch, MagicMock
import base64
from PIL import Image
import io
from services.bodyline_service import BodylineService, StableDiffusionControlNetPipeline
from core.config import settings
import os
from unittest.mock import call

@pytest.fixture
def mock_base_pipeline():
    with patch('services.bodyline_service.StableDiffusionPipeline') as mock:
        mock_instance = Mock()
        mock.from_single_file.return_value = mock_instance
        mock_instance.to.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_controlnet():
    with patch('services.bodyline_service.ControlNetModel') as mock:
        mock_instance = Mock()
        mock.from_single_file.return_value = mock_instance
        mock_instance.to.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_pipeline():
    with patch('services.bodyline_service.StableDiffusionControlNetPipeline') as mock_cls:
        mock_instance = Mock()
        mock_instance.to.return_value = mock_instance
        mock_cls.return_value = mock_instance
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
def bodyline_service(mock_base_pipeline, mock_controlnet, mock_pipeline):
    with patch('services.bodyline_service.StableDiffusionControlNetPipeline') as pipeline_cls:
        pipeline_cls.return_value = mock_pipeline
        pipeline_cls.return_value.to.return_value = mock_pipeline
        # テスト用のモデル設定
        test_configs = [
            {"path": "model1.pth", "conditioning_scale": 1.4},
            {"path": "model2.pth", "conditioning_scale": 1.3}
        ]
        with patch('core.config.settings.CONTROLNET_CONFIGS', test_configs):
    return BodylineService()

class TestBodylineService:
    @pytest.mark.asyncio
    async def test_init(self, mock_base_pipeline, mock_controlnet, mock_pipeline):
        """初期化のテスト"""
        test_configs = [
            {"path": "model1.pth", "conditioning_scale": 1.4},
            {"path": "model2.pth", "conditioning_scale": 1.3}
        ]
        with patch('core.config.settings.CONTROLNET_CONFIGS', test_configs):
        service = BodylineService()
        
            # ControlNetモデルの初期化を確認
            assert len(service.controlnet_models) == 2
            for model in service.controlnet_models:
                assert model == mock_controlnet
            
            # スケールの確認
            assert service.controlnet_scales == [1.4, 1.3]
        
        # パイプラインの初期化を確認
        assert service.pipeline == mock_pipeline
        assert mock_pipeline.to.called_with(settings.DEVICE)

            # パイプラインが正しいパラメータで作成されたことを確認
            mock_pipeline.assert_has_calls([
                call(
                    vae=mock_base_pipeline.vae,
                    text_encoder=mock_base_pipeline.text_encoder,
                    tokenizer=mock_base_pipeline.tokenizer,
                    unet=mock_base_pipeline.unet,
                    scheduler=mock_base_pipeline.scheduler,
                    safety_checker=mock_base_pipeline.safety_checker,
                    feature_extractor=mock_base_pipeline.feature_extractor,
                    controlnet=service.controlnet_models
                )
            ])

    def test_calculate_resize_dimensions(self, bodyline_service):
        """calculate_resize_dimensionsメソッドのテスト"""
        # 横長画像のテスト
        landscape_image = Image.new('RGB', (800, 400))
        landscape_size = bodyline_service.calculate_resize_dimensions(landscape_image, 512)
        assert landscape_size == (512, 256)
        
        # 縦長画像のテスト
        portrait_image = Image.new('RGB', (400, 800))
        portrait_size = bodyline_service.calculate_resize_dimensions(portrait_image, 512)
        assert portrait_size == (256, 512)
        
        # 正方形画像のテスト
        square_image = Image.new('RGB', (600, 600))
        square_size = bodyline_service.calculate_resize_dimensions(square_image, 512)
        assert square_size == (512, 512)

    @pytest.mark.asyncio
    async def test_resize_for_controlnet(self, bodyline_service, sample_image):
        """resize_for_controlnetメソッドのテスト"""
        # デフォルトサイズのテスト
        result = await bodyline_service.resize_for_controlnet(sample_image)
        assert isinstance(result, Image.Image)
        assert result.size == (512, 512)

        # カスタムサイズのテスト
        custom_size = (256, 256)
        result_custom = await bodyline_service.resize_for_controlnet(sample_image, target_size=custom_size)
        assert isinstance(result_custom, Image.Image)
        assert result_custom.size == custom_size

    @pytest.mark.asyncio
    async def test_generate_bodyline(self, bodyline_service):
        """generate_bodylineメソッドのテスト"""
        # モック画像を作成
        test_image = Image.new('RGB', (512, 512), color='white')
        output_size = (768, 768)
        
        # パイプラインの出力をモック
        mock_output = MagicMock()
        mock_output.images = [Image.new('RGB', output_size, color='white')]
        bodyline_service.pipeline.return_value = mock_output
        
        result = await bodyline_service.generate_bodyline(
            control_image=test_image,
            prompt="test prompt",
            negative_prompt="test negative",
            num_inference_steps=20,
            guidance_scale=7.0,
            output_size=output_size
        )
        
        # 結果の構造を確認
        assert "image" in result
        assert "parameters" in result
        assert isinstance(result["image"], str)  # Base64文字列であることを確認
        
        # パラメータを確認
        params = result["parameters"]
        assert params["prompt"] == "test prompt"
        assert params["negative_prompt"] == "test negative"
        assert params["num_inference_steps"] == 20
        assert params["guidance_scale"] == 7.0
        assert params["output_size"] == output_size
        
        # パイプラインが正しいパラメータで呼び出されたことを確認
        bodyline_service.pipeline.assert_called_once_with(
            prompt="test prompt",
            negative_prompt="test negative",
            image=[test_image] * 2,  # 2つのControlNetモデル用
            num_inference_steps=20,
            guidance_scale=7.0,
            width=output_size[0],
            height=output_size[1],
            denoising_strength=0.13,
            num_images_per_prompt=1,
            guess_mode=[True] * 2,
            controlnet_conditioning_scale=[1.4, 1.3],
            guidance_start=[0.0] * 2,
            guidance_end=[1.0] * 2
        )

        # 生成された画像を保存
        # Base64文字列を画像として保存
        image_bytes = base64.b64decode(result["image"])
        test_output_dir = "tests"
        os.makedirs(test_output_dir, exist_ok=True)
        output_path = os.path.join(test_output_dir, "test_generated_bodyline.png")
        
        # BytesIOを使用して画像を保存
        image = Image.open(io.BytesIO(image_bytes))
        with io.BytesIO() as bio:
            image.save(bio, format='PNG')
            with open(output_path, 'wb') as f:
                f.write(bio.getvalue())
        
        print(f"Generated image saved to: {output_path}")
        
        # 生成された画像のサイズを確認（8ピクセル以内の誤差を許容）
        actual_width, actual_height = image.size
        expected_width, expected_height = output_size
        assert abs(actual_width - expected_width) <= 8, f"Width difference is too large: {abs(actual_width - expected_width)}"
        assert abs(actual_height - expected_height) <= 8, f"Height difference is too large: {abs(actual_height - expected_height)}"

@pytest.mark.integration
@pytest.mark.slow
class TestBodylineServiceIntegration:
    """実際のモデルを使用する統合テスト"""
    
    @pytest.fixture
    def bodyline_service(self):
        """実際のモデルを使用するBodylineServiceインスタンス"""
        return BodylineService()
    
    @pytest.mark.asyncio
    async def test_generate_bodyline_with_real_model(self, bodyline_service):
        """実際のモデルを使用してボディライン生成をテスト"""
        # テスト用の入力画像を作成
        test_image = Image.open("tests/data/test_pose.jpg").convert('RGB')
        
        # 長辺768ピクセルにリサイズ
        output_size = bodyline_service.calculate_resize_dimensions(test_image, 768)
        input_size = bodyline_service.calculate_resize_dimensions(test_image, 512)
        test_image = test_image.resize(input_size, Image.Resampling.LANCZOS)
        print(f"Input image dimensions: {test_image.size}")

        prompt = "anime pose, girl, (white background:1.5), (monochrome:1.5), full body, sketch, eyes, breasts, (slim legs, skinny legs:1.2)"
        # 実際のモデルを使用して画像生成
        result = await bodyline_service.generate_bodyline(
            control_image=test_image,
            prompt=prompt,
            negative_prompt=f"(wings:1.6), (clothes:1.4), (garment:1.4), (lighting:1.4), (gray:1.4), (missing limb:1.4), (extra line:1.4), (extra limb:1.4), (extra arm:1.4), (extra legs:1.4), (hair:1.4), (bangs:1.4), (fringe:1.4), (forelock:1.4), (front hair:1.4), (fill:1.4), (ink pool:1.6)",
            num_inference_steps=20,  # テスト用に少ない推論ステップ数
            guidance_scale=8,
            output_size=output_size
        )
        
        # 結果の構造を確認
        assert "image" in result
        assert "parameters" in result
        assert isinstance(result["image"], str)
        
        # 生成された画像を保存
        image_bytes = base64.b64decode(result["image"])
        test_output_dir = "tests"
        os.makedirs(test_output_dir, exist_ok=True)
        output_path = os.path.join(test_output_dir, "test_generated_bodyline_real.png")
        
        # BytesIOを使用して画像を保存
        image = Image.open(io.BytesIO(image_bytes))
        with io.BytesIO() as bio:
            image.save(bio, format='PNG')
            with open(output_path, 'wb') as f:
                f.write(bio.getvalue())
        
        print(f"Real model generated image saved to: {output_path}")
        print(f"Output image dimensions: {image.size}")
        
        # 画像のサイズと形式を確認（8ピクセル以内の誤差を許容）
        actual_width, actual_height = image.size
        expected_width, expected_height = output_size
        assert abs(actual_width - expected_width) <= 8, f"Width difference is too large: {abs(actual_width - expected_width)}"
        assert abs(actual_height - expected_height) <= 8, f"Height difference is too large: {abs(actual_height - expected_height)}"
        assert image.mode in ['RGB', 'RGBA'] 