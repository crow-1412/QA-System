import os
import torch

# device config
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
num_gpus = torch.cuda.device_count()

# GPU memory config
GPU_MEMORY_FRACTION = 0.85  # 设置GPU内存使用率
BATCH_SIZE = 8  # 减小批处理大小
TENSOR_PARALLEL_SIZE = 1  # 暂时关闭张量并行

# 并行处理配置
PARALLEL_CONFIG = {
    "use_data_parallel": False,  # 暂时关闭数据并行
    "use_model_parallel": False,  # 暂时关闭模型并行
    "batch_size_per_gpu": BATCH_SIZE // max(num_gpus, 1),  # 每个GPU的批大小
    "gradient_accumulation_steps": 4,  # 增加梯度累积步数
    "max_parallel_workers": min(8, os.cpu_count() or 1)  # 减少最大并行工作进程数
}

# 模型路径配置
MODEL_PATHS = {
    "m3e": "/root/autodl-tmp/pre_train_model/m3e-large",
    "bge": "/root/autodl-tmp/pre_train_model/bge-large-zh-v1.5",
    "gte": "/root/autodl-tmp/pre_train_model/gte-large-zh",
    "bce": "/root/autodl-tmp/pre_train_model/bce-embedding-base_v1",
    "bge_reranker": "/root/autodl-tmp/pre_train_model/bge-reranker-large",
    "bce_reranker": "/root/autodl-tmp/pre_train_model/bce-reranker-base_v1"
}

# model cache config
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')

# LLM model paths
LLM_MODEL_PATHS = {
    "chatglm3": "/root/autodl-tmp/pre_train_model/chatglm3-6b",
    "qwen15": "/root/autodl-tmp/pre_train_model/Qwen-7B-Chat",
    "baichuan2": "/root/autodl-tmp/pre_train_model/Baichuan2-7B-Chat"
}

# LLM config
LLM_CONFIG = {
    "chatglm3": {
        "model_path": "/root/autodl-tmp/pre_train_model/chatglm3-6b",
        "device": "cuda",
        "use_flash_attention": True,
        "load_in_8bit": True,
        "tensor_parallel_size": 2,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.8,
        "dtype": "float16"
    },
    "qwen": {
        "model_path": "/root/autodl-tmp/pre_train_model/Qwen1.5-7B-Chat",
        "device": "cuda",
        "use_flash_attention": True,
        "load_in_8bit": True,
        "tensor_parallel_size": 2,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.8,
        "dtype": "float16"
    },
    "baichuan": {
        "model_path": "/root/autodl-tmp/pre_train_model/Baichuan2-7B-Chat",
        "device": "cuda",
        "use_flash_attention": True,
        "load_in_8bit": True,
        "tensor_parallel_size": 2,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.8,
        "dtype": "float16"
    }
}

# 默认使用的LLM模型
DEFAULT_LLM = "chatglm3"

# vector storage config
VECTOR_STORE_PATH='./vector_store'
COLLECTION_NAME='my_collection'

# 检索器权重配置
OPTIMIZED_RETRIEVER_WEIGHTS = {
    'bm25': 0.35,
    'tfidf': 0.15,
    'faiss': 0.25,
    'bge': 0.35,
    'gte': 0.25,
    'bce': 0.15
}

# 重排序器权重配置
OPTIMIZED_RERANKER_WEIGHTS = {
    'bge': 0.6,
    'bce': 0.4
}

# GPU性能优化配置
GPU_CONFIG = {
    'batch_size': 8,  # 减小批处理大小
    'tensor_parallel_size': 1,  # 暂时关闭张量并行
    'memory_fraction': 0.85,  # GPU显存使用比例
    'enable_cuda_graph': False,  # 暂时关闭CUDA图优化
    'use_fp16': True,  # 启用FP16
    'use_8bit': True,  # 启用8位量化
    'max_seq_length': 512,  # 限制序列长度
    'gradient_checkpointing': True  # 启用梯度检查点
}

# CUDA优化设置
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    
    # 设置CUDA内存分配器
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,"
        "expandable_segments:True,"
        "garbage_collection_threshold:0.8"
    )
    
    # 启用自动混合精度
    torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
    
# 模型加载优化配置
MODEL_LOADING_CONFIG = {
    'device_map': 'auto',  # 自动设备映射
    'offload_folder': 'offload',  # 设置模型卸载文件夹
    'torch_dtype': torch.float16,  # 使用FP16
    'low_cpu_mem_usage': True,  # 低CPU内存使用
    'use_cache': True  # 启用缓存
}
