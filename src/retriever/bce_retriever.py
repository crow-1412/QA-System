from sentence_transformers import SentenceTransformer
from src.retriever.embedding_model import SentenceTransformerMy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import numpy as np
import torch
from typing import List, Dict, Any
import logging
import faiss

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BCERetriever:
    def __init__(self, documents: List[Dict[str, Any]], model_path: str = None):
        """初始化BCE检索器
        
        Args:
            documents: 文档列表,每个文档是包含content字段的字典
            model_path: 模型路径
        """
        self.documents = documents
        self.model_path = model_path
        
        try:
            # 初始化embedding模型
            self.model = SentenceTransformerMy(model_path)
            
            # 编码文档并创建索引
            self._update_faiss_index()
            
        except Exception as e:
            logger.error(f"初始化BCE检索器时出错: {str(e)}")
            raise

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """检索最相关的k个文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            list: 包含文档和相似度分数的列表
        """
        try:
            if not self.index:
                logger.warning("FAISS索引未初始化")
                return []
                
            # 编码查询
            query_vector = np.array(self.model.embed_query(query)).reshape(1, -1)
            
            # 搜索相似文档
            scores, indices = self.index.search(query_vector, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    # 处理不同类型的文档对象
                    if hasattr(doc, 'page_content'):
                        content = doc.page_content
                    elif isinstance(doc, dict):
                        content = doc.get("content", "")
                    else:
                        content = str(doc)
                        
                    result = {
                        "content": content,
                        "score": float(score)
                    }
                    results.append(result)
                    
            return results
            
        except Exception as e:
            logger.error(f"BCE检索失败: {str(e)}")
            return []

    def _encode_documents(self) -> np.ndarray:
        """将所有文档编码为向量"""
        try:
            contents = []
            for doc in self.documents:
                if isinstance(doc, dict):
                    content = doc.get("content", "")
                else:
                    content = str(doc)
                contents.append(content)
                
            if not contents:
                logger.warning("没有文档内容可供编码")
                return np.array([])
                
            # 批量编码
            embeddings = np.array(self.model.embed_documents(contents))
            return embeddings
            
        except Exception as e:
            logger.error(f"文档编码失败: {str(e)}")
            return np.array([])

    def _update_faiss_index(self):
        """更新FAISS索引"""
        try:
            # 编码所有文档
            embeddings = self._encode_documents()
            
            if len(embeddings) > 0:
                # 创建FAISS索引
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings)
                logger.info(f"已创建包含{len(embeddings)}个文档的FAISS索引")
            else:
                logger.warning("没有文档可供索引")
                self.index = None
                
        except Exception as e:
            logger.error(f"更新FAISS索引失败: {str(e)}")
            self.index = None

    def GetTopK(self, query, k=10):
        """检索与查询最相关的k个文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            list: 包含(Document, score)元组的列表
        """
        print(f"\nSearching top {k} documents for query: {query}")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            print(f"Found {len(results)} matching documents")
            return [(Document(page_content=doc.page_content), score) for doc, score in results]
        except Exception as e:
            print(f"Error in GetTopK: {str(e)}")
            return []
            
    def retrieve(self, query: str, k: int = 10) -> list:
        """retrieve方法，作为GetTopK的另一个别名
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            list: 包含(Document, score)元组的列表
        """
        return self.GetTopK(query, k)