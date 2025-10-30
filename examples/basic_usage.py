"""
OmniFoundry 基本使用範例
"""

# 範例 1: 硬體偵測
def example_hardware_detection():
    """硬體偵測範例"""
    from omnifoundry.core.hardware import HardwareDetector
    
    detector = HardwareDetector()
    info = detector.get_hardware_info()
    print("硬體資訊:", info)


# 範例 2: 模型下載和管理
def example_model_management():
    """模型管理範例"""
    from omnifoundry.core.model_manager import ModelManager
    
    manager = ModelManager(cache_dir="./models")
    
    # 搜尋模型
    models = manager.search_models("llama", model_type="text-generation")
    
    # 下載模型
    manager.download_model("meta-llama/Llama-2-7b-hf")
    
    # 列出本地模型
    local_models = manager.list_local_models()
    print("本地模型:", local_models)


# 範例 3: 執行推論
def example_inference():
    """推論範例"""
    from omnifoundry.core.inference import InferenceEngine
    
    # 初始化推論引擎
    engine = InferenceEngine(
        model_id="meta-llama/Llama-2-7b-hf",
        backend="auto"
    )
    
    # 載入模型
    engine.load_model()
    
    # 執行推論
    result = engine.infer(
        "Hello, how are you?",
        max_length=100,
        temperature=0.7
    )
    
    print("推論結果:", result)
    
    # 卸載模型
    engine.unload_model()


# 範例 4: 使用配置檔
def example_config():
    """配置管理範例"""
    from omnifoundry.core.config import Config
    
    # 載入配置
    config = Config("configs/default.yaml")
    config.load()
    
    # 讀取配置值
    api_port = config.get("api.port", default=8000)
    print("API 埠號:", api_port)
    
    # 設定配置值
    config.set("api.port", 8080)
    config.save()


if __name__ == "__main__":
    print("OmniFoundry 使用範例")
    print("=" * 50)
    
    # 執行範例
    # example_hardware_detection()
    # example_model_management()
    # example_inference()
    # example_config()

