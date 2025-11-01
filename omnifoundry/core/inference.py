"""
推論引擎整合模組

整合多種推論後端：
- transformers (預設，最通用)
- llama-cpp-python (輕量級)
- optimum (ONNX Runtime)
- diffusers (圖像生成)
"""

import logging
from typing import Dict, Any, Optional, List, Union, Iterator
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

from .hardware import HardwareDetector

logger = logging.getLogger(__name__)


class InferenceEngine:
    """推論引擎基類"""
    
    def __init__(self, model_id: str, backend: str = "auto", **kwargs):
        """
        初始化推論引擎
        
        Args:
            model_id: 模型 ID（Hugging Face 模型或本地路徑）
            backend: 推論後端（auto, transformers, llama-cpp, etc.）
            **kwargs: 額外配置參數
        """
        self.model_id = model_id
        self.backend = backend
        self.config = kwargs
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
        
    def load_model(self) -> None:
        """載入模型"""
        raise NotImplementedError("子類必須實作 load_model 方法")
    
    def infer(self, input_data: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        執行推論
        
        Args:
            input_data: 輸入資料（文字或文字列表）
            **kwargs: 推論參數
            
        Returns:
            推論結果
        """
        raise NotImplementedError("子類必須實作 infer 方法")
    
    def stream(self, input_data: str, **kwargs) -> Iterator[str]:
        """
        串流推論（逐 token 輸出）
        
        Args:
            input_data: 輸入文字
            **kwargs: 推論參數
            
        Yields:
            生成的 token
        """
        raise NotImplementedError("此引擎不支援串流推論")
    
    def unload_model(self) -> None:
        """卸載模型，釋放記憶體"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # 清理 GPU 記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info(f"模型 {self.model_id} 已卸載")
    
    def __del__(self):
        """析構函數，確保模型被卸載"""
        if self.is_loaded:
            self.unload_model()


class TransformersEngine(InferenceEngine):
    """Transformers 推論引擎"""
    
    def __init__(self, model_id: str, **kwargs):
        """
        初始化 Transformers 引擎
        
        Args:
            model_id: 模型 ID
            **kwargs: 配置參數
                - device: 設備 (auto, cpu, cuda)
                - dtype: 資料型別 (auto, float32, float16, bfloat16)
                - quantization: 量化方式 (None, int8, int4)
                - trust_remote_code: 是否信任遠端程式碼
                - use_flash_attention: 是否使用 Flash Attention
                - model_type: 模型類型 (causal-lm, seq2seq)
        """
        super().__init__(model_id, backend="transformers", **kwargs)
        
        # 自動偵測硬體並推薦配置
        if kwargs.get("auto_config", True):
            self._auto_configure()
        else:
            self._manual_configure(kwargs)
    
    def _auto_configure(self):
        """自動配置推論參數"""
        logger.info("自動偵測硬體配置...")
        
        detector = HardwareDetector()
        hw_config = detector.recommend_config(model_type="llm")
        
        # 套用推薦配置
        self.config.setdefault("device", hw_config["device"])
        self.config.setdefault("dtype", hw_config["dtype"])
        self.config.setdefault("quantization", hw_config["quantization"])
        self.config.setdefault("batch_size", hw_config["batch_size"])
        self.config.setdefault("use_flash_attention", hw_config.get("use_flash_attention", False))
        
        logger.info(f"自動配置完成: device={self.config['device']}, "
                   f"dtype={self.config['dtype']}, quantization={self.config['quantization']}")
    
    def _manual_configure(self, kwargs):
        """手動配置推論參數"""
        self.config.setdefault("device", "auto")
        self.config.setdefault("dtype", "auto")
        self.config.setdefault("quantization", None)
        self.config.setdefault("batch_size", 1)
        self.config.setdefault("use_flash_attention", False)
    
    def load_model(self) -> None:
        """載入模型"""
        if self.is_loaded:
            logger.warning(f"模型 {self.model_id} 已經載入")
            return
        
        logger.info(f"正在載入模型: {self.model_id}")
        
        try:
            # 載入 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.config.get("trust_remote_code", False),
            )
            
            # 如果沒有 pad_token，設為 eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 設定設備
            device = self.config.get("device", "auto")
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            # 設定資料型別
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "auto": torch.float16 if self.device == "cuda" else torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.get("dtype", "auto"), torch.float32)
            
            # 設定量化配置
            quantization_config = None
            quantization = self.config.get("quantization")
            
            if quantization == "int8":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                    )
                    logger.info("使用 8-bit 量化")
                except Exception as e:
                    logger.warning(f"8-bit 量化不可用: {e}")
            elif quantization == "int4":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("使用 4-bit 量化")
                except Exception as e:
                    logger.warning(f"4-bit 量化不可用: {e}")
            
            # 載入模型
            model_kwargs = {
                "trust_remote_code": self.config.get("trust_remote_code", False),
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch_dtype,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            
            # 根據模型類型選擇正確的 AutoModel 類別
            model_type = self.config.get("model_type", "causal-lm")
            
            if model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id, **model_kwargs
                )
            else:  # causal-lm (預設)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, **model_kwargs
                )
            
            # 如果不使用 device_map，手動移到設備
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 設為評估模式
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"模型載入完成，設備: {self.device}, 資料型別: {torch_dtype}")
            
            # 顯示記憶體使用情況
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU 記憶體使用: {allocated:.2f} GB (已分配), {reserved:.2f} GB (已保留)")
            
        except Exception as e:
            logger.error(f"載入模型時發生錯誤: {e}")
            raise
    
    def infer(
        self,
        input_data: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        執行推論
        
        Args:
            input_data: 輸入文字或文字列表
            max_new_tokens: 最大生成 token 數
            temperature: 溫度參數
            top_p: Top-p 取樣
            top_k: Top-k 取樣
            repetition_penalty: 重複懲罰
            do_sample: 是否使用取樣
            **kwargs: 其他生成參數
            
        Returns:
            生成的文字或文字列表
        """
        if not self.is_loaded:
            raise RuntimeError("模型尚未載入，請先呼叫 load_model()")
        
        # 處理單一輸入
        is_single = isinstance(input_data, str)
        if is_single:
            input_data = [input_data]
        
        # Tokenize 輸入
        inputs = self.tokenizer(
            input_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=kwargs.get("max_length", 2048),
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # 解碼輸出
        generated_texts = []
        for i, output in enumerate(outputs):
            # 只保留新生成的部分
            input_length = inputs["input_ids"][i].shape[0]
            generated = output[input_length:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts[0] if is_single else generated_texts
    
    def stream(
        self,
        input_data: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> Iterator[str]:
        """
        串流推論（逐 token 輸出）
        
        Args:
            input_data: 輸入文字
            max_new_tokens: 最大生成 token 數
            temperature: 溫度參數
            top_p: Top-p 取樣
            top_k: Top-k 取樣
            repetition_penalty: 重複懲罰
            **kwargs: 其他生成參數
            
        Yields:
            生成的文字片段
        """
        if not self.is_loaded:
            raise RuntimeError("模型尚未載入，請先呼叫 load_model()")
        
        # Tokenize 輸入
        inputs = self.tokenizer(
            input_data,
            return_tensors="pt",
            truncation=True,
            max_length=kwargs.get("max_length", 2048),
        ).to(self.device)
        
        # 設定串流器
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # 生成參數
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # 在背景執行緒中生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 逐個 yield token
        for text in streamer:
            yield text
        
        thread.join()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        對話模式推論
        
        Args:
            messages: 對話訊息列表，格式：[{"role": "user", "content": "..."}]
            max_new_tokens: 最大生成 token 數
            temperature: 溫度參數
            **kwargs: 其他參數
            
        Returns:
            助手回覆
        """
        if not self.is_loaded:
            raise RuntimeError("模型尚未載入，請先呼叫 load_model()")
        
        # 如果 tokenizer 支援 chat template，使用它
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"使用 chat template 失敗: {e}，改用簡單格式")
                prompt = self._format_chat_simple(messages)
        else:
            prompt = self._format_chat_simple(messages)
        
        # 執行推論
        response = self.infer(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response
    
    def _format_chat_simple(self, messages: List[Dict[str, str]]) -> str:
        """簡單的對話格式化"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        formatted.append("Assistant:")
        return "\n".join(formatted)


class LlamaCppEngine(InferenceEngine):
    """Llama.cpp 推論引擎"""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, backend="llama-cpp", **kwargs)
        logger.info("Llama.cpp 引擎尚未實作")
    
    def load_model(self):
        raise NotImplementedError("Llama.cpp 引擎尚未實作")
    
    def infer(self, input_data, **kwargs):
        raise NotImplementedError("Llama.cpp 引擎尚未實作")


class OptimumEngine(InferenceEngine):
    """Optimum (ONNX) 推論引擎"""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, backend="optimum", **kwargs)
        logger.info("Optimum 引擎尚未實作")
    
    def load_model(self):
        raise NotImplementedError("Optimum 引擎尚未實作")
    
    def infer(self, input_data, **kwargs):
        raise NotImplementedError("Optimum 引擎尚未實作")


class DiffusersEngine(InferenceEngine):
    """Diffusers 推論引擎"""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, backend="diffusers", **kwargs)
        logger.info("Diffusers 引擎尚未實作")
    
    def load_model(self):
        raise NotImplementedError("Diffusers 引擎尚未實作")
    
    def infer(self, input_data, **kwargs):
        raise NotImplementedError("Diffusers 引擎尚未實作")

