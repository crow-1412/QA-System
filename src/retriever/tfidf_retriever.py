from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document
import numpy as np
import jieba
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class TFIDFRetriever:
    def __init__(self, documents: List[Dict[str, Any]]):
        """初始化TF-IDF检索器
        
        Args:
            documents: 文档列表,每个文档是包含content字段的字典
        """
        self.documents = documents
        
        try:
            # 提取文档内容
            contents = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif isinstance(doc, dict):
                    content = doc.get("content", "")
                else:
                    content = str(doc)
                contents.append(content)
            
            if not contents:
                logger.warning("没有文档内容可供向量化")
                self.vectorizer = None
                self.doc_vectors = None
                return
                
            # 创建TF-IDF向量化器
            self.vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=1.0,
                token_pattern=r"(?u)\b\w+\b"
            )
            
            # 对文档进行向量化
            self.doc_vectors = self.vectorizer.fit_transform(contents)
            logger.info(f"已完成{len(contents)}个文档的TF-IDF向量化")
            
        except Exception as e:
            logger.error(f"初始化TF-IDF检索器时出错: {str(e)}")
            self.vectorizer = None
            self.doc_vectors = None
            
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """检索最相关的k个文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            list: 包含文档和相似度分数的列表
        """
        try:
            if self.vectorizer is None or self.doc_vectors is None:
                logger.warning("TF-IDF向量化器未初始化")
                return []
                
            # 对查询进行向量化
            query_vector = self.vectorizer.transform([query])
            
            # 计算相似度
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # 获取前k个最相似的文档索引
            top_k_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for idx in top_k_indices:
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
                        "score": float(similarities[idx])
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF检索失败: {str(e)}")
            return []
        
    def GetTopK(self, query, k=10):
        """检索与查询最相关的k个文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            list: 包含(Document, score)元组的列表，按相关性得分降序排序
        """
        # 预处理查询
        processed_query = self._preprocess_text(query)
        print(f"\nProcessed query: {processed_query}")
        
        try:
            # 转换查询文本为TF-IDF向量
            query_vec = self.vectorizer.transform([processed_query])
            
            # 计算余弦相似度
            scores = np.asarray(query_vec.dot(self.doc_vectors.T).toarray()[0])
            
            # 获取前k个最相关文档的索引
            top_k_indices = np.argsort(scores)[-k:][::-1]
            
            # 过滤掉相似度为0的结果
            results = []
            for idx in top_k_indices:
                if scores[idx] > 0:
                    results.append((self.documents[idx], float(scores[idx])))
                    print(f"Found document with score {scores[idx]}: {self.documents[idx]['content'][:100]}...")
                    
            print(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            print(f"Error in GetTopK: {str(e)}")
            return [] 

    def _tokenize(self, text: str) -> List[str]:
        """使用jieba分词"""
        return list(jieba.cut(text))
        
    def _preprocess_text(self, text: str) -> str:
        """预处理文本
        
        1. 移除换行符
        2. 移除多余空格
        3. 转换为小写
        """
        return ' '.join(text.replace('\n', ' ').lower().split()) 