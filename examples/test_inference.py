"""
測試 Transformers 推論引擎

使用輕量級模型測試推論功能
"""

import logging
from omnifoundry.core.inference import TransformersEngine

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_inference():
    """測試基本推論功能"""
    print("\n" + "="*60)
    print("🧪 測試 1: 基本文字生成")
    print("="*60)
    
    # 使用 GPT-2（輕量級，快速下載）
    model_id = "gpt2"
    print(f"使用模型: {model_id}")
    
    # 建立引擎（使用 CPU 模式以確保相容性）
    engine = TransformersEngine(model_id, device="cpu", dtype="float32", quantization=None)
    
    # 載入模型
    print("\n正在載入模型...")
    engine.load_model()
    
    # 執行推論
    prompt = "Once upon a time"
    print(f"\n提示詞: {prompt}")
    print("="*60)
    
    result = engine.infer(
        prompt,
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True,
    )
    
    print(f"生成結果:\n{result}")
    print("="*60)
    
    # 卸載模型
    engine.unload_model()
    print("✅ 測試 1 完成\n")


def test_stream_inference():
    """測試串流推論"""
    print("\n" + "="*60)
    print("🧪 測試 2: 串流文字生成")
    print("="*60)
    
    model_id = "gpt2"
    print(f"使用模型: {model_id}")
    
    engine = TransformersEngine(model_id, device="cpu", dtype="float32", quantization=None)
    engine.load_model()
    
    prompt = "The future of artificial intelligence is"
    print(f"\n提示詞: {prompt}")
    print("="*60)
    print("串流輸出: ", end="", flush=True)
    
    for token in engine.stream(prompt, max_new_tokens=30, temperature=0.8):
        print(token, end="", flush=True)
    
    print("\n" + "="*60)
    
    engine.unload_model()
    print("✅ 測試 2 完成\n")


def test_batch_inference():
    """測試批次推論"""
    print("\n" + "="*60)
    print("🧪 測試 3: 批次推論")
    print("="*60)
    
    model_id = "gpt2"
    print(f"使用模型: {model_id}")
    
    engine = TransformersEngine(model_id, device="cpu", dtype="float32", quantization=None)
    engine.load_model()
    
    prompts = [
        "Artificial intelligence is",
        "The best programming language is",
        "In the year 2050,",
    ]
    
    print(f"\n批次處理 {len(prompts)} 個提示詞...")
    print("="*60)
    
    results = engine.infer(
        prompts,
        max_new_tokens=20,
        temperature=0.7,
    )
    
    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"\n提示 {i}: {prompt}")
        print(f"結果: {result}")
    
    print("="*60)
    
    engine.unload_model()
    print("✅ 測試 3 完成\n")


def test_chinese_model():
    """測試中文模型（如果可用）"""
    print("\n" + "="*60)
    print("🧪 測試 4: 中文模型推論")
    print("="*60)
    
    # 使用 Qwen2.5-0.5B（輕量級中文模型）
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"使用模型: {model_id}")
    print("注意: 此模型約 1GB，首次下載可能需要時間")
    
    try:
        # 建立引擎，手動設定不使用量化（小模型）
        engine = TransformersEngine(
            model_id,
            trust_remote_code=True,
            quantization=None,  # 不使用量化
        )
        
        engine.load_model()
        
        # 測試對話功能
        messages = [
            {"role": "system", "content": "你是一個有幫助的 AI 助手。"},
            {"role": "user", "content": "請用一句話介紹人工智慧。"},
        ]
        
        print("\n對話測試:")
        print("="*60)
        print(f"User: {messages[1]['content']}")
        
        response = engine.chat(messages, max_new_tokens=100, temperature=0.7)
        print(f"Assistant: {response}")
        print("="*60)
        
        engine.unload_model()
        print("✅ 測試 4 完成\n")
        
    except Exception as e:
        print(f"⚠️  測試 4 跳過: {e}")
        print("可能是因為模型下載失敗或記憶體不足\n")


def test_custom_config():
    """測試自訂配置"""
    print("\n" + "="*60)
    print("🧪 測試 5: 自訂配置")
    print("="*60)
    
    model_id = "gpt2"
    print(f"使用模型: {model_id}")
    
    # 手動設定配置（不自動偵測）
    engine = TransformersEngine(
        model_id,
        auto_config=False,  # 不自動配置
        device="cpu",       # 強制使用 CPU
        dtype="float32",    # 使用 float32
        quantization=None,  # 不使用量化
    )
    
    print(f"\n配置: device={engine.config['device']}, "
          f"dtype={engine.config['dtype']}, "
          f"quantization={engine.config['quantization']}")
    
    engine.load_model()
    
    result = engine.infer(
        "Hello, I am a",
        max_new_tokens=10,
        temperature=0.5,
    )
    
    print(f"\n生成結果: {result}")
    print("="*60)
    
    engine.unload_model()
    print("✅ 測試 5 完成\n")


def main():
    """主函數"""
    print("\n" + "🚀 "* 20)
    print("OmniFoundry Transformers 引擎測試")
    print("🚀 "* 20)
    
    try:
        # 執行測試
        test_basic_inference()
        test_stream_inference()
        test_batch_inference()
        test_custom_config()
        
        # 中文模型測試（可選）
        user_input = input("\n是否測試中文模型？(y/n，需要下載約 1GB): ")
        if user_input.lower() == 'y':
            test_chinese_model()
        
        print("\n" + "="*60)
        print("🎉 所有測試完成！")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  測試被用戶中斷")
    except Exception as e:
        print(f"\n\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

