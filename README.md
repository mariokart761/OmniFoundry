# OmniFoundry

> 開源模型推論環境集成程式

一個自動偵測硬體、智能選擇推論引擎的開源 AI 模型管理與部署工具。

## 特色功能

- 🔍 **自動硬體偵測** - 自動識別 CPU、GPU（NVIDIA/AMD/Intel）和記憶體配置
- 🎯 **智能引擎選擇** - 根據硬體和模型自動選擇最佳推論引擎
- 📦 **模型管理** - 從 Hugging Face Hub 搜尋、下載和管理模型
- 🚀 **多引擎支援** - 支援 Transformers、llama.cpp、ONNX Runtime、Diffusers 等
- 🌐 **API 服務** - 提供 OpenAI 相容的 REST API
- 🖥️ **CLI 工具** - 簡單易用的命令列介面
- 🐳 **Docker 容器化** - 開箱即用的容器映像

## 支援的模型類型

- 文字生成（LLM）- GPT、LLaMA、Mistral 等
- 圖像生成 - Stable Diffusion、DALL-E 等
- 語音識別/合成 - Whisper、Bark 等
- 多模態 - CLIP、LLaVA 等

## 安裝

### 從 PyPI 安裝（即將推出）

```bash
pip install omnifoundry
```

### 從原始碼安裝

```bash
git clone https://github.com/yourusername/omnifoundry.git
cd omnifoundry
pip install -e .
```

### 安裝額外依賴

```bash
# 安裝 llama.cpp 支援
pip install omnifoundry[llama-cpp]

# 安裝 ONNX Runtime 支援
pip install omnifoundry[optimum]

# 安裝所有可選依賴
pip install omnifoundry[all]
```

## 快速開始

### 檢查硬體資訊

```bash
omnifoundry info
```

### 搜尋和下載模型

```bash
# 搜尋模型
omnifoundry list

# 下載模型
omnifoundry download meta-llama/Llama-2-7b-hf
```

### 執行推論

```bash
omnifoundry run meta-llama/Llama-2-7b-hf --prompt "Hello, how are you?"
```

### 啟動 API 服務

```bash
omnifoundry serve meta-llama/Llama-2-7b-hf
```

然後可以使用 OpenAI 相容的 API：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Docker 使用

```bash
# 建構映像
docker build -t omnifoundry .

# 執行容器（CPU 版本）
docker run -p 8000:8000 omnifoundry

# 執行容器（GPU 版本）
docker run --gpus all -p 8000:8000 omnifoundry
```

或使用 docker-compose：

```bash
docker-compose up
```

## 配置

在 `configs/default.yaml` 中設定預設配置：

```yaml
hardware:
  auto_detect: true
  force_cpu: false
  gpu_memory_fraction: 0.9

models:
  cache_dir: "./models"
  auto_download: true

inference:
  max_length: 2048
  temperature: 0.7
  
api:
  host: "0.0.0.0"
  port: 8000
```

## Python API 使用

```python
from omnifoundry import OmniFoundry

# 初始化
foundry = OmniFoundry()

# 載入模型
model = foundry.load_model("meta-llama/Llama-2-7b-hf")

# 執行推論
result = model.generate("Hello, how are you?")
print(result)
```

## 專案架構

```
OmniFoundry/
├── omnifoundry/              # 主要 Python 套件
│   ├── core/                 # 核心功能模組
│   │   ├── hardware.py       # 硬體自動偵測
│   │   ├── model_manager.py  # 模型下載與管理
│   │   ├── inference.py      # 推論引擎整合
│   │   └── config.py         # 配置管理
│   ├── api/                  # FastAPI REST API
│   ├── cli/                  # 命令列介面
│   └── utils/                # 工具函數
├── docker/                   # Docker 相關
├── configs/                  # 配置範例
├── examples/                 # 使用範例
└── tests/                    # 測試
```

## 開發

### 安裝開發依賴

```bash
pip install -e .[dev]
```

### 執行測試

```bash
pytest tests/
```

### 程式碼格式化

```bash
black omnifoundry/
flake8 omnifoundry/
```

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 授權

MIT License

## 聯絡方式

- GitHub: https://github.com/yourusername/omnifoundry
- Issues: https://github.com/yourusername/omnifoundry/issues

## 致謝

感謝以下專案提供的靈感和技術支援：
- Hugging Face Transformers
- llama.cpp
- ONNX Runtime
- FastAPI

