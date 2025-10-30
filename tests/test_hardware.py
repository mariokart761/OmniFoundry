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
    
    def test_detect_cpu(self):
        """測試 CPU 偵測"""
        # TODO: 實作測試
        pass
    
    def test_detect_gpu(self):
        """測試 GPU 偵測"""
        # TODO: 實作測試
        pass
    
    def test_detect_memory(self):
        """測試記憶體偵測"""
        # TODO: 實作測試
        pass
    
    def test_get_hardware_info(self):
        """測試獲取硬體資訊"""
        # TODO: 實作測試
        pass

