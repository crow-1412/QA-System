import os

import torch

# device config
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
num_gpus = torch.cuda.device_count()

# GPU memory config
GPU_MEMORY_FRACTION = 0.7  # 使用70%的GPU内存
BATCH_SIZE = 16  # 减小批处理大小
TENSOR_PARALLEL_SIZE = 1  # 设置为1，避免多GPU通信问题

# model cache config
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')


# vector storage config
VECTOR_STORE_PATH='./vector_store'
COLLECTION_NAME='my_collection'

# CUDA optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()  # 清理GPU缓存
