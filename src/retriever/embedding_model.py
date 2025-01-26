from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import torch
import logging

logger = logging.getLogger(__name__)

class SentenceTransformerMy(Embeddings):
    """封装SentenceTransformer的Embeddings类"""
    
    encode_kwargs = dict()
    multi_process = False
    show_progress = True
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """初始化embedding模型
        
        Args:
            model_path: 模型路径
            device: 设备类型，默认为cuda
        """
        logger.info(f"正在从{model_path}初始化SentenceTransformer...")
        try:
            self.device = device
            if device == "cuda" and torch.cuda.is_available():
                self.client = SentenceTransformer(model_path, device=device)
                self.client.half()  # 使用半精度
            else:
                self.client = SentenceTransformer(model_path, device="cpu")
                
            logger.info("SentenceTransformer初始化成功")
            
            # 清理GPU缓存
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
            
    def to(self, device: str):
        """将模型移动到指定设备
        
        Args:
            device: 目标设备
        """
        self.device = device
        self.client.to(device)
        return self
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """计算文档嵌入向量
        
        Args:
            texts: 要嵌入的文本列表
            
        Returns:
            List[List[float]]: 每个文本的嵌入向量列表
        """
        logger.info(f"正在编码{len(texts)}个文档...")
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        try:
            if self.multi_process:
                logger.info("使用多进程编码...")
                pool = self.client.start_multi_process_pool()
                embeddings = self.client.encode_multi_process(texts, pool)
                self.client.stop_multi_process_pool(pool)
            else:
                logger.info("使用单进程编码...")
                embeddings = self.client.encode(
                    texts, 
                    show_progress_bar=self.show_progress, 
                    batch_size=32,
                    **self.encode_kwargs
                )
            logger.info("文档编码完成")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"编码文档失败: {str(e)}")
            raise
        
    def embed_query(self, text: str) -> list[float]:
        """计算查询文本的嵌入向量
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            List[float]: 文本的嵌入向量
        """
        logger.info("正在编码查询...")
        try:
            result = self.embed_documents([text])[0]
            logger.info("查询编码完成")
            return result
        except Exception as e:
            logger.error(f"编码查询失败: {str(e)}")
            raise 