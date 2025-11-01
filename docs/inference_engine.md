# Transformers 推論引擎說明

## 概述

Transformers 推論引擎 (`omnifoundry.core.inference`) 提供強大且易用的模型推論功能，支援文字生成、對話、批次處理和串流輸出。

## 功能特性

### 1. 自動硬體配置

- 自動偵測 CPU/GPU 並選擇最佳設備
- 自動選擇資料型別（float32/float16/bfloat16）
- 智能量化選擇（8-bit/4-bit）
- 根據硬體自動調整批次大小

### 2. 多種推論模式

#### 基本文字生成
```python
result = engine.infer("Once upon a time", max_new_tokens=50)
```

#### 串流生成（逐 token 輸出）
```python
for token in engine.stream("Hello", max_new_tokens=30):
    print(token, end="", flush=True)
```

#### 批次推論
```python
prompts = ["Question 1?", "Question 2?", "Question 3?"]
results = engine.infer(prompts, max_new_tokens=20)
```

#### 對話模式
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
]
response = engine.chat(messages, max_new_tokens=100)
```

### 3. 靈活配置

支援自動和手動兩種配置模式：

#### 自動配置（推薦）
```python
engine = TransformersEngine("gpt2")  # 自動偵測硬體
```

#### 手動配置
```python
engine = TransformersEngine(
    "gpt2",
    auto_config=False,
    device="cpu",
    dtype="float32",
    quantization=None,
)
```

### 4. 記憶體管理

- 自動卸載模型釋放記憶體
- GPU 記憶體使用監控
- 支援 CPU offload

## 使用範例

### 完整流程

```python
from omnifoundry.core.inference import TransformersEngine

# 1. 建立引擎
engine = TransformersEngine("gpt2")

# 2. 載入模型
engine.load_model()

# 3. 執行推論
result = engine.infer(
    "Once upon a time",
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
)
print(result)

# 4. 卸載模型
engine.unload_model()
```

### 串流生成範例

```python
engine = TransformersEngine("gpt2")
engine.load_model()

print("Assistant: ", end="", flush=True)
for token in engine.stream("The future of AI is", max_new_tokens=30):
    print(token, end="", flush=True)
print()

engine.unload_model()
```

### 對話範例

```python
engine = TransformersEngine("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
engine.load_model()

messages = [
    {"role": "system", "content": "你是一個有幫助的 AI 助手。"},
    {"role": "user", "content": "什麼是深度學習？"},
]

response = engine.chat(messages, max_new_tokens=200, temperature=0.7)
print(f"Assistant: {response}")

engine.unload_model()
```

## 支援的模型

### 文字生成模型（Causal LM）
- GPT-2
- GPT-Neo / GPT-J
- LLaMA / LLaMA 2
- Qwen / Qwen2
- Mistral
- Phi
- 等等...

### Seq2Seq 模型
設定 `model_type="seq2seq"`：
- T5
- BART
- mT5
- 等等...

## 生成參數

### 常用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `max_new_tokens` | 100 | 最大生成 token 數 |
| `temperature` | 0.7 | 溫度參數（0.0-2.0），越高越隨機 |
| `top_p` | 0.9 | Nucleus sampling 參數 |
| `top_k` | 50 | Top-k sampling 參數 |
| `repetition_penalty` | 1.0 | 重複懲罰（>1.0 減少重複） |
| `do_sample` | True | 是否使用取樣 |

### 進階參數

- `num_beams`: Beam search 寬度
- `early_stopping`: 是否提前停止
- `no_repeat_ngram_size`: 避免 n-gram 重複
- `length_penalty`: 長度懲罰

## 配置選項

### 模型載入配置

| 參數 | 說明 |
|------|------|
| `device` | 設備選擇：`auto`, `cpu`, `cuda` |
| `dtype` | 資料型別：`auto`, `float32`, `float16`, `bfloat16` |
| `quantization` | 量化方式：`None`, `int8`, `int4` |
| `trust_remote_code` | 是否信任遠端程式碼（某些模型需要） |
| `model_type` | 模型類型：`causal-lm`, `seq2seq` |

### 效能優化

1. **使用 GPU**：自動使用 CUDA 加速（如果可用）
2. **使用 float16**：在 GPU 上使用 float16 提升速度
3. **量化**：使用 int8/int4 量化減少記憶體使用
4. **批次處理**：一次處理多個輸入提升吞吐量

## 測試結果

### ✅ 測試通過

1. **基本文字生成** - 成功
2. **串流推論** - 成功
3. **批次推論** - 成功
4. **自訂配置** - 成功
5. **對話模式** - 成功（支援 chat template 的模型）

### 測試模型

- **GPT-2** (124M 參數) - 輕量級英文模型 ✅
- **Qwen2.5-0.5B-Instruct** - 輕量級中文模型 ✅

### 效能指標

在 CPU 模式下（AMD Ryzen 7 7800X3D）：
- 模型載入時間：~2秒（GPT-2）
- 推論速度：~10-15 tokens/秒
- 記憶體使用：~500MB（GPT-2）

## 注意事項

### Windows + CUDA

在 Windows 環境下，如果遇到 CUDA 問題：
1. 確保安裝了支援 CUDA 的 PyTorch 版本
2. 或者使用 `device="cpu"` 強制使用 CPU 模式

### 量化支援

- `int8`/`int4` 量化需要 `bitsandbytes` 庫
- Windows 上可能需要特殊版本的 bitsandbytes
- 如果量化失敗，系統會自動降級為不使用量化

### 記憶體管理

推薦做法：
```python
engine = TransformersEngine(model_id)
try:
    engine.load_model()
    result = engine.infer(prompt)
finally:
    engine.unload_model()  # 確保釋放記憶體
```

## 錯誤處理

常見錯誤及解決方案：

1. **模型未載入**
   ```python
   RuntimeError: 模型尚未載入，請先呼叫 load_model()
   ```
   解決：先呼叫 `engine.load_model()`

2. **記憶體不足**
   - 使用量化：`quantization="int8"`
   - 使用較小的模型
   - 減少 `max_new_tokens`

3. **CUDA 不可用**
   - 使用 `device="cpu"`
   - 安裝支援 CUDA 的 PyTorch

## 未來計劃

- [ ] 支援 llama.cpp 後端（CPU 優化）
- [ ] 支援 ONNX Runtime 後端
- [ ] 支援 vLLM 後端（高效能推論）
- [ ] 支援圖像生成模型（Stable Diffusion）
- [ ] 支援多 GPU 推論
- [ ] 支援模型量化工具

## API 參考

完整的 API 文檔請參閱：`omnifoundry/core/inference.py`

主要類別：
- `InferenceEngine`: 推論引擎基類
- `TransformersEngine`: Transformers 推論引擎
- `LlamaCppEngine`: Llama.cpp 引擎（待實作）
- `OptimumEngine`: ONNX Runtime 引擎（待實作）
- `DiffusersEngine`: 圖像生成引擎（待實作）

