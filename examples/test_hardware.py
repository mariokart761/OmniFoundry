"""
測試硬體偵測模組的範例腳本
"""

from omnifoundry.core.hardware import HardwareDetector


def main():
    """主程式"""
    print("正在偵測硬體資訊...\n")
    
    # 建立硬體偵測器
    detector = HardwareDetector()
    
    # 顯示完整硬體資訊
    detector.print_hardware_info()
    
    # 測試推薦配置功能
    print("\n" + "="*60)
    print("📋 推薦配置測試")
    print("="*60)
    
    # 測試不同模型大小的推薦配置
    test_cases = [
        ("llm", "7b", "LLM 7B 模型"),
        ("llm", "13b", "LLM 13B 模型"),
        ("llm", "70b", "LLM 70B 模型"),
        ("diffusion", "sd-1.5", "Stable Diffusion 1.5"),
    ]
    
    for model_type, model_size, description in test_cases:
        print(f"\n【{description}】")
        config = detector.recommend_config(model_type=model_type, model_size=model_size)
        
        print(f"  裝置: {config['device']}")
        print(f"  資料型別: {config['dtype']}")
        print(f"  量化: {config['quantization'] or '無'}")
        print(f"  批次大小: {config['batch_size']}")
        print(f"  最大長度: {config.get('max_length', 'N/A')}")
        print(f"  Flash Attention: {'是' if config.get('use_flash_attention') else '否'}")
        print(f"  CPU Offload: {'是' if config.get('cpu_offload') else '否'}")
    
    print("\n" + "="*60 + "\n")
    
    # 顯示原始資料（用於除錯）
    print("\n【原始資料（JSON 格式）】")
    import json
    hw_info = detector.get_hardware_info()
    print(json.dumps(hw_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

