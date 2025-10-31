# 硬體偵測模組說明

## 概述

硬體偵測模組 (`omnifoundry.core.hardware`) 提供自動偵測系統硬體配置的功能，包括 CPU、GPU 和記憶體資訊，並根據硬體規格自動推薦最佳的模型執行配置。

## 功能特性

### 1. CPU 偵測

自動偵測以下 CPU 資訊：
- 處理器品牌和型號
- 架構 (x86_64, ARM 等)
- 實體核心數
- 邏輯核心數 (含超執行緒)
- CPU 頻率 (最大、最小、當前)
- CPU 指令集 (SSE, AVX, AVX2, AVX-512 等)

### 2. GPU 偵測

支援多種 GPU 平台：

#### NVIDIA GPU (CUDA)
- 透過 PyTorch 或 GPUtil 自動偵測
- 顯示 GPU 型號、VRAM 容量
- 顯示當前可用記憶體
- 顯示計算能力 (Compute Capability)
- 顯示 GPU 負載和溫度

#### AMD GPU (ROCm)
- 透過 PyTorch ROCm 版本偵測
- 顯示 GPU 型號和可用性

#### Intel GPU (OneAPI)
- 透過 PyTorch Intel 擴展偵測
- 顯示 GPU 型號和可用性

### 3. 記憶體偵測

自動偵測以下記憶體資訊：
- 總記憶體容量
- 可用記憶體
- 已使用記憶體和使用率
- Swap 記憶體資訊

### 4. 智能配置推薦

根據硬體規格和模型類型自動推薦最佳配置：

#### 推薦參數
- **device**: 使用的裝置 (cpu, cuda)
- **dtype**: 資料型別 (float32, float16, bfloat16)
- **quantization**: 量化等級 (None, int8, int4)
- **batch_size**: 批次大小
- **max_length**: 最大序列長度
- **use_flash_attention**: 是否使用 Flash Attention
- **cpu_offload**: 是否使用 CPU 卸載

#### 支援的模型類型
- **llm**: 大型語言模型 (7B, 13B, 70B 等)
- **diffusion**: 圖像生成模型 (Stable Diffusion 等)

## 使用方式

### Python API

```python
from omnifoundry.core.hardware import HardwareDetector

# 建立硬體偵測器
detector = HardwareDetector()

# 偵測 CPU 資訊
cpu_info = detector.detect_cpu()
print(cpu_info)

# 偵測 GPU 資訊
gpu_info = detector.detect_gpu()
print(gpu_info)

# 偵測記憶體資訊
memory_info = detector.detect_memory()
print(memory_info)

# 獲取完整硬體資訊
hw_info = detector.get_hardware_info()
print(hw_info)

# 友善格式印出硬體資訊
detector.print_hardware_info()

# 獲取推薦配置
config = detector.recommend_config(model_type="llm", model_size="7b")
print(config)
```

### CLI 命令

```bash
# 顯示硬體資訊
omnifoundry info

# 以 JSON 格式顯示硬體資訊
omnifoundry info --json-output

# 獲取模型推薦配置
omnifoundry info --recommend llm:7b
omnifoundry info --recommend llm:13b
omnifoundry info --recommend diffusion:sd-1.5

# 以 JSON 格式顯示推薦配置
omnifoundry info --recommend llm:7b --json-output
```

## 配置推薦邏輯

### NVIDIA GPU 環境

#### 7B 模型
- **16GB+ VRAM**: Float16, 無量化, batch_size=8, Flash Attention
- **8GB+ VRAM**: Float16, Int8 量化, batch_size=4
- **<8GB VRAM**: Float16, Int4 量化, batch_size=2

#### 13B 模型
- **24GB+ VRAM**: Float16, 無量化, batch_size=4
- **12GB+ VRAM**: Float16, Int8 量化, batch_size=2
- **<12GB VRAM**: Float16, Int4 量化, batch_size=1

#### 70B 模型
- **80GB+ VRAM**: Float16, 無量化, batch_size=4
- **48GB+ VRAM**: Float16, Int8 量化, batch_size=2
- **<48GB VRAM**: Float16, Int4 量化, batch_size=1, CPU Offload

### CPU 環境

- **32GB+ RAM**: Float32, batch_size=2, Int8 量化
- **16GB+ RAM**: Float32, batch_size=1, Int8 量化
- **<16GB RAM**: Float32, batch_size=1, Int4 量化

## 快取機制

硬體偵測結果會在首次執行後快取在記憶體中，避免重複偵測造成的效能損耗。

## 依賴項

### 必需依賴
- `psutil`: CPU 和記憶體偵測
- `platform`: 系統資訊

### 可選依賴
- `py-cpuinfo`: 詳細的 CPU 資訊
- `GPUtil`: NVIDIA GPU 詳細資訊
- `torch`: GPU 偵測和 CUDA 支援

## 測試

```bash
# 執行單元測試
pytest tests/test_hardware.py -v

# 執行測試腳本
python examples/test_hardware.py
```

## 範例輸出

```
============================================================
🖥️  系統硬體資訊
============================================================

【系統】
  作業系統: Windows Windows-11-10.0.26100-SP0
  Python 版本: 3.12.3

【CPU】
  處理器: AMD Ryzen 7 7800X3D 8-Core Processor
  架構: AMD64
  實體核心: 8
  邏輯核心: 16
  當前頻率: 4201.00 MHz

【GPU】
  GPU 0: NVIDIA GeForce RTX 3070
    廠商: NVIDIA
    顯存: 8.00 GB (可用: 5.58 GB)
    負載: 7.0%

【記憶體】
  總容量: 63.16 GB
  可用: 29.32 GB
  已用: 33.84 GB (53.6%)

============================================================
```

## 未來改進

- [ ] 支援更多 GPU 平台 (Apple Silicon, 其他 ARM GPU)
- [ ] 更精確的記憶體需求估算
- [ ] 支援多 GPU 配置推薦
- [ ] 自動偵測最佳的推論引擎
- [ ] 效能基準測試整合

