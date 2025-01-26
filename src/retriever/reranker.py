from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import jieba
from src.retriever.stopwords import stopwords

logger = logging.getLogger(__name__)

class BGEReranker:
    def __init__(self, config: Dict[str, Any]):
        """初始化BGE重排序器
        
        Args:
            config: 配置字典，包含model_path、device等参数
        """
        if isinstance(config, str):
            # 向后兼容：如果传入的是字符串，将其视为model_path
            config = {"model_path": config}
            
        self.model_path = config["model_path"]
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("batch_size", 32)
        
        # 初始化模型
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
        
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """使用BGE模型重排序文档列表"""
        try:
            scores = []
            for doc in documents:
                score = self._compute_score(query, doc.get("content", ""))
                scores.append(score)
            return scores
        except Exception as e:
            logger.error(f"BGE重排序失败: {str(e)}")
            return [0.0] * len(documents)
            
    def _compute_score(self, query: str, doc_text: str) -> float:
        """使用BGE特定的评分方法"""
        try:
            inputs = self.tokenizer([query, doc_text], padding=True, truncation=True, 
                                  return_tensors="pt", max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]
                # 使用L2归一化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                score = torch.matmul(embeddings[0], embeddings[1].T).item()
                return (score + 1) / 2  # 将分数映射到[0,1]范围
        except Exception as e:
            logger.error(f"BGE评分计算失败: {str(e)}")
            return 0.0
            
class BCEReranker:
    def __init__(self, config: Dict[str, Any]):
        """初始化BCE重排序器
        
        Args:
            config: 配置字典，包含model_path、device等参数
        """
        if isinstance(config, str):
            # 向后兼容：如果传入的是字符串，将其视为model_path
            config = {"model_path": config}
            
        self.model_path = config["model_path"]
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("batch_size", 32)
        
        # 初始化模型
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
        
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """使用BCE模型重排序文档列表"""
        try:
            scores = []
            for doc in documents:
                score = self._compute_score(query, doc.get("content", ""))
                scores.append(score)
            return scores
        except Exception as e:
            logger.error(f"BCE重排序失败: {str(e)}")
            return [0.0] * len(documents)
            
    def _compute_score(self, query: str, doc_text: str) -> float:
        """使用BCE特定的评分方法"""
        try:
            inputs = self.tokenizer([query, doc_text], padding=True, truncation=True, 
                                  return_tensors="pt", max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用平均池化
                mask = inputs["attention_mask"]
                embeddings = (outputs.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
                # 计算相似度
                query_embedding = embeddings[0]
                doc_embedding = embeddings[1]
                score = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=0).item()
                return (score + 1) / 2  # 将分数映射到[0,1]范围
        except Exception as e:
            logger.error(f"BCE评分计算失败: {str(e)}")
            return 0.0 

class MultiReranker:
    def __init__(self, model_paths: Dict[str, str]):
        self.bge_reranker = BGEReranker({
            "model_path": model_paths.get("bge_reranker", ""),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        })
        self.bce_reranker = BCEReranker({
            "model_path": model_paths.get("bce_reranker", ""),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        })
        
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            bge_scores = self.bge_reranker.rerank(query, documents)
            bce_scores = self.bce_reranker.rerank(query, documents)
            
            # 计算关键词匹配度分数
            keyword_scores = []
            query_words = set(jieba.cut(query)) - stopwords
            for doc in documents:
                content = doc.get("content", "")
                doc_words = set(jieba.cut(content)) - stopwords
                # 计算关键词覆盖率
                keyword_overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
                # 计算关键词密度
                keyword_density = len(query_words & doc_words) / len(doc_words) if doc_words else 0
                # 综合分数
                keyword_scores.append(0.7 * keyword_overlap + 0.3 * keyword_density)
            
            # 调整分数合并策略 (0.4 * BGE + 0.3 * BCE + 0.3 * 关键词)
            final_scores = [0.4 * bge + 0.3 * bce + 0.3 * kw 
                          for bge, bce, kw in zip(bge_scores, bce_scores, keyword_scores)]
            
            # 根据分数排序文档
            scored_docs = list(zip(documents, final_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 记录重排序结果
            logger.info(f"重排序后分数: {[f'{score:.3f}' for _, score in scored_docs[:5]]}")
            
            return [doc for doc, _ in scored_docs]
            
        except Exception as e:
            logger.error(f"多重排序失败: {str(e)}")
            return documents 