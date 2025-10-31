"""
硬體偵測模組測試
"""

import pytest
from omnifoundry.core.hardware import HardwareDetector


class TestHardwareDetector:
    """硬體偵測器測試"""
    
    def test_init(self):
        """測試初始化"""
        detector = HardwareDetector()
        assert detector is not None
        assert detector._cpu_info is None
        assert detector._gpu_info is None
        assert detector._memory_info is None
    
    def test_detect_cpu(self):
        """測試 CPU 偵測"""
        detector = HardwareDetector()
        cpu_info = detector.detect_cpu()
        
        assert cpu_info is not None
        assert isinstance(cpu_info, dict)
        assert "physical_cores" in cpu_info
        assert "logical_cores" in cpu_info
        assert cpu_info["physical_cores"] > 0
        assert cpu_info["logical_cores"] > 0
        assert cpu_info["logical_cores"] >= cpu_info["physical_cores"]
    
    def test_detect_gpu(self):
        """測試 GPU 偵測"""
        detector = HardwareDetector()
        gpu_info = detector.detect_gpu()
        
        assert gpu_info is not None
        assert isinstance(gpu_info, list)
        assert len(gpu_info) > 0
        
        # 至少應該有一個 "GPU"（即使是 CPU 模式）
        gpu = gpu_info[0]
        assert "vendor" in gpu
        assert "name" in gpu
    
    def test_detect_memory(self):
        """測試記憶體偵測"""
        detector = HardwareDetector()
        mem_info = detector.detect_memory()
        
        assert mem_info is not None
        assert isinstance(mem_info, dict)
        assert "total" in mem_info
        assert "available" in mem_info
        assert "used" in mem_info
        assert "_total_bytes" in mem_info
        assert mem_info["_total_bytes"] > 0
    
    def test_get_hardware_info(self):
        """測試獲取硬體資訊"""
        detector = HardwareDetector()
        hw_info = detector.get_hardware_info()
        
        assert hw_info is not None
        assert isinstance(hw_info, dict)
        assert "system" in hw_info
        assert "cpu" in hw_info
        assert "gpu" in hw_info
        assert "memory" in hw_info
        
        # 檢查系統資訊
        assert "os" in hw_info["system"]
        assert "python_version" in hw_info["system"]
    
    def test_recommend_config_llm_7b(self):
        """測試推薦配置 - LLM 7B 模型"""
        detector = HardwareDetector()
        config = detector.recommend_config(model_type="llm", model_size="7b")
        
        assert config is not None
        assert isinstance(config, dict)
        assert "device" in config
        assert "dtype" in config
        assert "batch_size" in config
        assert config["device"] in ["cpu", "cuda"]
        assert config["batch_size"] > 0
    
    def test_recommend_config_diffusion(self):
        """測試推薦配置 - Diffusion 模型"""
        detector = HardwareDetector()
        config = detector.recommend_config(model_type="diffusion")
        
        assert config is not None
        assert isinstance(config, dict)
        assert "device" in config
        assert "dtype" in config
    
    def test_cache_mechanism(self):
        """測試快取機制"""
        detector = HardwareDetector()
        
        # 第一次呼叫
        cpu_info_1 = detector.detect_cpu()
        gpu_info_1 = detector.detect_gpu()
        mem_info_1 = detector.detect_memory()
        
        # 第二次呼叫應該回傳快取的結果
        cpu_info_2 = detector.detect_cpu()
        gpu_info_2 = detector.detect_gpu()
        mem_info_2 = detector.detect_memory()
        
        assert cpu_info_1 is cpu_info_2
        assert gpu_info_1 is gpu_info_2
        assert mem_info_1 is mem_info_2

