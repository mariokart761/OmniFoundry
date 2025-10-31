"""
硬體自動偵測模組

負責偵測系統硬體配置，包括：
- CPU 類型、核心數、記憶體大小
- NVIDIA/AMD/Intel GPU 及 VRAM
- CUDA/ROCm/OneAPI 可用性
- 根據硬體自動推薦配置
"""

import platform
import psutil
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class HardwareDetector:
    """硬體偵測器"""
    
    def __init__(self):
        """初始化硬體偵測器"""
        self._cpu_info = None
        self._gpu_info = None
        self._memory_info = None
        
    def detect_cpu(self) -> Dict[str, Any]:
        """
        偵測 CPU 資訊
        
        Returns:
            包含 CPU 資訊的字典
        """
        if self._cpu_info is not None:
            return self._cpu_info
            
        try:
            # 基本 CPU 資訊
            cpu_info = {
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": None,
                "min_frequency": None,
                "current_frequency": None,
            }
            
            # 取得 CPU 頻率資訊
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    cpu_info["max_frequency"] = f"{cpu_freq.max:.2f} MHz"
                    cpu_info["min_frequency"] = f"{cpu_freq.min:.2f} MHz"
                    cpu_info["current_frequency"] = f"{cpu_freq.current:.2f} MHz"
            except Exception as e:
                logger.warning(f"無法獲取 CPU 頻率資訊: {e}")
            
            # 嘗試使用 cpuinfo 獲取更詳細資訊
            try:
                import cpuinfo
                cpu_data = cpuinfo.get_cpu_info()
                cpu_info["brand"] = cpu_data.get("brand_raw", "Unknown")
                cpu_info["vendor"] = cpu_data.get("vendor_id_raw", "Unknown")
                cpu_info["flags"] = cpu_data.get("flags", [])
            except ImportError:
                logger.info("py-cpuinfo 未安裝，使用基本 CPU 資訊")
                cpu_info["brand"] = platform.processor()
                cpu_info["vendor"] = "Unknown"
                cpu_info["flags"] = []
            except Exception as e:
                logger.warning(f"獲取詳細 CPU 資訊時出錯: {e}")
                cpu_info["brand"] = platform.processor()
                cpu_info["vendor"] = "Unknown"
                cpu_info["flags"] = []
            
            self._cpu_info = cpu_info
            return cpu_info
            
        except Exception as e:
            logger.error(f"偵測 CPU 時發生錯誤: {e}")
            return {
                "error": str(e),
                "physical_cores": 0,
                "logical_cores": 0,
            }
    
    def detect_gpu(self) -> List[Dict[str, Any]]:
        """
        偵測 GPU 資訊
        
        Returns:
            GPU 資訊列表
        """
        if self._gpu_info is not None:
            return self._gpu_info
            
        gpus = []
        
        # 偵測 NVIDIA GPU (CUDA)
        nvidia_gpus = self._detect_nvidia_gpu()
        gpus.extend(nvidia_gpus)
        
        # 偵測 AMD GPU (ROCm)
        amd_gpus = self._detect_amd_gpu()
        gpus.extend(amd_gpus)
        
        # 偵測 Intel GPU (OneAPI)
        intel_gpus = self._detect_intel_gpu()
        gpus.extend(intel_gpus)
        
        # 如果沒有偵測到任何 GPU
        if not gpus:
            logger.info("未偵測到獨立 GPU，將使用 CPU 模式")
            gpus.append({
                "id": -1,
                "name": "CPU (無 GPU)",
                "vendor": "CPU",
                "memory_total": 0,
                "memory_free": 0,
                "compute_capability": None,
                "available": True,
            })
        
        self._gpu_info = gpus
        return gpus
    
    def _detect_nvidia_gpu(self) -> List[Dict[str, Any]]:
        """偵測 NVIDIA GPU"""
        gpus = []
        
        # 方法 1: 使用 PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                cuda_available = True
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory / (1024**3)  # 轉換為 GB
                    
                    gpu_info = {
                        "id": i,
                        "name": props.name,
                        "vendor": "NVIDIA",
                        "memory_total": f"{memory_total:.2f} GB",
                        "memory_free": None,  # 稍後更新
                        "compute_capability": f"{props.major}.{props.minor}",
                        "cuda_available": True,
                        "available": True,
                    }
                    
                    # 嘗試獲取當前可用記憶體
                    try:
                        memory_free = torch.cuda.mem_get_info(i)[0] / (1024**3)
                        gpu_info["memory_free"] = f"{memory_free:.2f} GB"
                    except:
                        pass
                    
                    gpus.append(gpu_info)
                    
                logger.info(f"透過 PyTorch 偵測到 {device_count} 個 NVIDIA GPU")
                return gpus
        except ImportError:
            logger.debug("PyTorch 未安裝，無法透過 PyTorch 偵測 NVIDIA GPU")
        except Exception as e:
            logger.debug(f"透過 PyTorch 偵測 NVIDIA GPU 時發生錯誤: {e}")
        
        # 方法 2: 使用 GPUtil
        try:
            import GPUtil
            gpu_list = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpu_list):
                gpu_info = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "vendor": "NVIDIA",
                    "memory_total": f"{gpu.memoryTotal / 1024:.2f} GB",
                    "memory_free": f"{gpu.memoryFree / 1024:.2f} GB",
                    "memory_used": f"{gpu.memoryUsed / 1024:.2f} GB",
                    "gpu_load": f"{gpu.load * 100:.1f}%",
                    "temperature": f"{gpu.temperature}°C",
                    "compute_capability": None,
                    "cuda_available": True,
                    "available": True,
                }
                gpus.append(gpu_info)
            
            if gpus:
                logger.info(f"透過 GPUtil 偵測到 {len(gpus)} 個 NVIDIA GPU")
            return gpus
            
        except ImportError:
            logger.debug("GPUtil 未安裝，無法透過 GPUtil 偵測 NVIDIA GPU")
        except Exception as e:
            logger.debug(f"透過 GPUtil 偵測 NVIDIA GPU 時發生錯誤: {e}")
        
        return gpus
    
    def _detect_amd_gpu(self) -> List[Dict[str, Any]]:
        """偵測 AMD GPU (ROCm)"""
        gpus = []
        
        try:
            import torch
            if hasattr(torch, 'hip') and torch.hip.is_available():
                device_count = torch.hip.device_count()
                
                for i in range(device_count):
                    gpu_info = {
                        "id": i,
                        "name": torch.hip.get_device_name(i),
                        "vendor": "AMD",
                        "memory_total": None,
                        "memory_free": None,
                        "rocm_available": True,
                        "available": True,
                    }
                    gpus.append(gpu_info)
                
                logger.info(f"偵測到 {device_count} 個 AMD GPU (ROCm)")
        except Exception as e:
            logger.debug(f"偵測 AMD GPU 時發生錯誤: {e}")
        
        return gpus
    
    def _detect_intel_gpu(self) -> List[Dict[str, Any]]:
        """偵測 Intel GPU (OneAPI)"""
        gpus = []
        
        try:
            import torch
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device_count = torch.xpu.device_count()
                
                for i in range(device_count):
                    gpu_info = {
                        "id": i,
                        "name": f"Intel GPU {i}",
                        "vendor": "Intel",
                        "memory_total": None,
                        "memory_free": None,
                        "oneapi_available": True,
                        "available": True,
                    }
                    gpus.append(gpu_info)
                
                logger.info(f"偵測到 {device_count} 個 Intel GPU (OneAPI)")
        except Exception as e:
            logger.debug(f"偵測 Intel GPU 時發生錯誤: {e}")
        
        return gpus
    
    def detect_memory(self) -> Dict[str, Any]:
        """
        偵測記憶體資訊
        
        Returns:
            包含記憶體資訊的字典
        """
        if self._memory_info is not None:
            return self._memory_info
            
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_info = {
                "total": f"{mem.total / (1024**3):.2f} GB",
                "available": f"{mem.available / (1024**3):.2f} GB",
                "used": f"{mem.used / (1024**3):.2f} GB",
                "percent_used": f"{mem.percent}%",
                "swap_total": f"{swap.total / (1024**3):.2f} GB",
                "swap_used": f"{swap.used / (1024**3):.2f} GB",
                "swap_percent": f"{swap.percent}%",
                # 原始值（用於計算）
                "_total_bytes": mem.total,
                "_available_bytes": mem.available,
            }
            
            self._memory_info = memory_info
            return memory_info
            
        except Exception as e:
            logger.error(f"偵測記憶體時發生錯誤: {e}")
            return {
                "error": str(e),
                "total": "0 GB",
                "available": "0 GB",
            }
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        獲取完整硬體資訊
        
        Returns:
            包含所有硬體資訊的字典
        """
        return {
            "system": {
                "os": platform.system(),
                "os_version": platform.version(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "cpu": self.detect_cpu(),
            "gpu": self.detect_gpu(),
            "memory": self.detect_memory(),
        }
    
    def recommend_config(self, model_type: str = "llm", model_size: str = "7b") -> Dict[str, Any]:
        """
        根據硬體推薦配置
        
        Args:
            model_type: 模型類型 (llm, diffusion, etc.)
            model_size: 模型大小 (7b, 13b, 70b, etc.)
            
        Returns:
            推薦的配置字典
        """
        cpu_info = self.detect_cpu()
        gpu_info = self.detect_gpu()
        memory_info = self.detect_memory()
        
        # 解析記憶體大小（GB）
        total_ram_gb = memory_info.get("_total_bytes", 0) / (1024**3)
        
        # 檢查是否有可用的 GPU
        has_gpu = any(gpu.get("vendor") != "CPU" for gpu in gpu_info)
        
        # 基礎配置
        config = {
            "device": "cpu",
            "dtype": "float32",
            "quantization": None,
            "batch_size": 1,
            "max_length": 2048,
            "use_flash_attention": False,
            "cpu_offload": False,
        }
        
        # 根據 GPU 情況調整
        if has_gpu:
            nvidia_gpus = [g for g in gpu_info if g.get("vendor") == "NVIDIA"]
            
            if nvidia_gpus:
                # 使用第一個 NVIDIA GPU
                gpu = nvidia_gpus[0]
                config["device"] = "cuda"
                
                # 解析 VRAM 大小
                vram_str = gpu.get("memory_total", "0 GB")
                try:
                    vram_gb = float(vram_str.split()[0])
                except:
                    vram_gb = 0
                
                # 根據 VRAM 和模型大小推薦配置
                if model_type == "llm":
                    if "70b" in model_size.lower() or "65b" in model_size.lower():
                        # 大型模型 (70B)
                        if vram_gb >= 80:
                            config["dtype"] = "float16"
                            config["quantization"] = None
                            config["batch_size"] = 4
                        elif vram_gb >= 48:
                            config["dtype"] = "float16"
                            config["quantization"] = "int8"
                            config["batch_size"] = 2
                        else:
                            config["dtype"] = "float16"
                            config["quantization"] = "int4"
                            config["batch_size"] = 1
                            config["cpu_offload"] = True
                    
                    elif "13b" in model_size.lower():
                        # 中型模型 (13B)
                        if vram_gb >= 24:
                            config["dtype"] = "float16"
                            config["quantization"] = None
                            config["batch_size"] = 4
                        elif vram_gb >= 12:
                            config["dtype"] = "float16"
                            config["quantization"] = "int8"
                            config["batch_size"] = 2
                        else:
                            config["dtype"] = "float16"
                            config["quantization"] = "int4"
                            config["batch_size"] = 1
                    
                    else:
                        # 小型模型 (7B 或更小)
                        if vram_gb >= 16:
                            config["dtype"] = "float16"
                            config["quantization"] = None
                            config["batch_size"] = 8
                            config["use_flash_attention"] = True
                        elif vram_gb >= 8:
                            config["dtype"] = "float16"
                            config["quantization"] = "int8"
                            config["batch_size"] = 4
                        else:
                            config["dtype"] = "float16"
                            config["quantization"] = "int4"
                            config["batch_size"] = 2
                
                elif model_type == "diffusion":
                    # 圖像生成模型
                    if vram_gb >= 12:
                        config["dtype"] = "float16"
                        config["batch_size"] = 4
                    elif vram_gb >= 8:
                        config["dtype"] = "float16"
                        config["batch_size"] = 2
                    else:
                        config["dtype"] = "float16"
                        config["batch_size"] = 1
        
        # 純 CPU 模式配置
        else:
            config["device"] = "cpu"
            
            if total_ram_gb >= 32:
                config["dtype"] = "float32"
                config["batch_size"] = 2
            elif total_ram_gb >= 16:
                config["dtype"] = "float32"
                config["batch_size"] = 1
                config["quantization"] = "int8"
            else:
                config["dtype"] = "float32"
                config["batch_size"] = 1
                config["quantization"] = "int4"
            
            # CPU 建議使用量化模型
            if not config["quantization"]:
                config["quantization"] = "int8"
        
        return config
    
    def print_hardware_info(self):
        """以友善格式印出硬體資訊"""
        info = self.get_hardware_info()
        
        print("\n" + "="*60)
        print("🖥️  系統硬體資訊")
        print("="*60)
        
        # 系統資訊
        print("\n【系統】")
        print(f"  作業系統: {info['system']['os']} {info['system']['platform']}")
        print(f"  Python 版本: {info['system']['python_version']}")
        
        # CPU 資訊
        print("\n【CPU】")
        cpu = info['cpu']
        print(f"  處理器: {cpu.get('brand', 'Unknown')}")
        print(f"  架構: {cpu.get('architecture', 'Unknown')}")
        print(f"  實體核心: {cpu.get('physical_cores', 0)}")
        print(f"  邏輯核心: {cpu.get('logical_cores', 0)}")
        if cpu.get('current_frequency'):
            print(f"  當前頻率: {cpu['current_frequency']}")
        
        # GPU 資訊
        print("\n【GPU】")
        gpus = info['gpu']
        if gpus:
            for i, gpu in enumerate(gpus):
                if gpu.get('vendor') == 'CPU':
                    print(f"  未偵測到獨立 GPU")
                else:
                    print(f"  GPU {i}: {gpu.get('name', 'Unknown')}")
                    print(f"    廠商: {gpu.get('vendor', 'Unknown')}")
                    if gpu.get('memory_total'):
                        print(f"    顯存: {gpu['memory_total']}", end="")
                        if gpu.get('memory_free'):
                            print(f" (可用: {gpu['memory_free']})", end="")
                        print()
                    if gpu.get('compute_capability'):
                        print(f"    計算能力: {gpu['compute_capability']}")
                    if gpu.get('gpu_load'):
                        print(f"    負載: {gpu['gpu_load']}")
        else:
            print("  未偵測到 GPU")
        
        # 記憶體資訊
        print("\n【記憶體】")
        mem = info['memory']
        print(f"  總容量: {mem.get('total', 'Unknown')}")
        print(f"  可用: {mem.get('available', 'Unknown')}")
        print(f"  已用: {mem.get('used', 'Unknown')} ({mem.get('percent_used', '0%')})")
        
        print("\n" + "="*60 + "\n")

