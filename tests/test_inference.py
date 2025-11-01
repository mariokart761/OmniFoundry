"""
推論引擎模組測試
"""

import pytest
from omnifoundry.core.inference import (
    InferenceEngine,
    TransformersEngine,
)


class TestInferenceEngine:
    """推論引擎基類測試"""
    
    def test_init(self):
        """測試初始化"""
        # 基類不能直接實例化使用
        engine = InferenceEngine(model_id="test-model")
        assert engine.model_id == "test-model"
        assert engine.backend == "auto"
        assert engine.is_loaded is False
    
    def test_not_implemented(self):
        """測試基類的未實作方法"""
        engine = InferenceEngine(model_id="test-model")
        
        with pytest.raises(NotImplementedError):
            engine.load_model()
        
        with pytest.raises(NotImplementedError):
            engine.infer("test input")
        
        with pytest.raises(NotImplementedError):
            engine.stream("test input")


class TestTransformersEngine:
    """Transformers 引擎測試"""
    
    def test_init(self):
        """測試初始化"""
        engine = TransformersEngine(model_id="gpt2", auto_config=False)
        assert engine.model_id == "gpt2"
        assert engine.backend == "transformers"
        assert engine.is_loaded is False
    
    def test_auto_configure(self):
        """測試自動配置"""
        engine = TransformersEngine(model_id="gpt2", auto_config=True)
        
        # 應該有自動設定的配置
        assert "device" in engine.config
        assert "dtype" in engine.config
        assert "quantization" in engine.config
    
    def test_manual_configure(self):
        """測試手動配置"""
        engine = TransformersEngine(
            model_id="gpt2",
            auto_config=False,
            device="cpu",
            dtype="float32",
            quantization=None,
        )
        
        # 檢查手動設定是否生效
        assert "device" in engine.config
        assert "dtype" in engine.config
    
    @pytest.mark.slow
    def test_load_model(self):
        """測試模型載入（需要網路，標記為 slow）"""
        engine = TransformersEngine(
            model_id="gpt2",
            auto_config=False,
            device="cpu",
            dtype="float32",
        )
        
        engine.load_model()
        
        assert engine.is_loaded is True
        assert engine.model is not None
        assert engine.tokenizer is not None
        assert engine.device == "cpu"
        
        engine.unload_model()
        assert engine.is_loaded is False
    
    @pytest.mark.slow
    def test_inference(self):
        """測試推論（需要網路，標記為 slow）"""
        engine = TransformersEngine(
            model_id="gpt2",
            auto_config=False,
            device="cpu",
            dtype="float32",
        )
        
        engine.load_model()
        
        # 測試單一輸入
        result = engine.infer("Hello", max_new_tokens=5)
        assert isinstance(result, str)
        assert len(result) > 0
        
        engine.unload_model()
    
    @pytest.mark.slow
    def test_batch_inference(self):
        """測試批次推論（需要網路，標記為 slow）"""
        engine = TransformersEngine(
            model_id="gpt2",
            auto_config=False,
            device="cpu",
            dtype="float32",
        )
        
        engine.load_model()
        
        # 測試批次輸入
        prompts = ["Hello", "World"]
        results = engine.infer(prompts, max_new_tokens=5)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
        
        engine.unload_model()
    
    def test_inference_without_load(self):
        """測試未載入模型就推論應該拋出錯誤"""
        engine = TransformersEngine(model_id="gpt2", auto_config=False)
        
        with pytest.raises(RuntimeError):
            engine.infer("test")
    
    def test_unload_model(self):
        """測試模型卸載"""
        engine = TransformersEngine(model_id="gpt2", auto_config=False)
        
        # 未載入時卸載不應該報錯
        engine.unload_model()
        
        assert engine.is_loaded is False
        assert engine.model is None
        assert engine.tokenizer is None


# 執行測試時跳過需要網路的測試
# 使用: pytest tests/test_inference.py -v
# 執行 slow 測試: pytest tests/test_inference.py -v --run-slow
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
