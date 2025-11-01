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
    from omnifoundry.core.inference import TransformersEngine
    
    # 初始化推論引擎（使用 GPT-2 做測試）
    engine = TransformersEngine(
        model_id="gpt2",
        device="cpu",  # 或 "cuda" 如果有 GPU
        dtype="float32",
    )
    
    # 載入模型
    print("載入模型...")
    engine.load_model()
    
    # 執行推論
    result = engine.infer(
        "Once upon a time",
        max_new_tokens=50,
        temperature=0.7
    )
    
    print("推論結果:", result)
    
    # 串流推論
    print("\n串流推論:")
    print("Assistant: ", end="", flush=True)
    for token in engine.stream("The future of AI is", max_new_tokens=30):
        print(token, end="", flush=True)
    print()
    
    # 批次推論
    prompts = ["Hello", "World", "AI"]
    results = engine.infer(prompts, max_new_tokens=10)
    print("\n批次推論結果:")
    for prompt, result in zip(prompts, results):
        print(f"  {prompt} -> {result}")
    
    # 卸載模型
    engine.unload_model()
    print("\n模型已卸載")


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
    print("\n選擇要執行的範例:")
    print("1. 硬體偵測")
    print("2. 模型管理（尚未實作）")
    print("3. 推論引擎")
    print("4. 配置管理（尚未實作）")
    
    # 取消註解以執行範例
    # example_hardware_detection()
    # example_model_management()
    # example_inference()
    # example_config()
    
    print("\n請取消註解上方的函數呼叫以執行範例")

