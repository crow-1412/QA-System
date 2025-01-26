#!/usr/bin/env python
# coding: utf-8

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.data_process import DataProcess
import jieba
import numpy as np

class BM25:
    def __init__(self, data):
        """初始化BM25检索器
        
        Args:
            data: 包含Document对象的列表
        """
        print("\nInitializing BM25 retriever...")
        self.docs = data
        
        # 预处理文档
        self.processed_docs = []
        for doc in data:
            if isinstance(doc, Document):
                text = doc.page_content
            else:
                text = str(doc)
            # 使用jieba分词
            tokens = list(jieba.cut(self._preprocess_text(text)))
            self.processed_docs.append(tokens)
            
        # 初始化BM25模型
        self.bm25 = BM25Okapi(self.processed_docs)
        print(f"BM25 initialized with {len(self.processed_docs)} documents")
        
    def _preprocess_text(self, text):
        """预处理文本
        
        1. 移除换行符
        2. 移除多余空格
        3. 转换为小写
        """
        return ' '.join(text.replace('\n', ' ').lower().split())
        
    def GetBM25TopK(self, query, k=10):
        """获取BM25得分最高的k个文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            list: 包含(Document, score)元组的列表
        """
        try:
            # 预处理查询
            processed_query = self._preprocess_text(query)
            print(f"\nProcessed query: {processed_query}")
            
            # 分词
            tokenized_query = list(jieba.cut(processed_query))
            print(f"Tokenized query: {tokenized_query}")
            
            # 计算BM25得分
            scores = self.bm25.get_scores(tokenized_query)
            
            # 获取前k个最相关文档的索引
            top_k_indices = np.argsort(scores)[-k:][::-1]
            
            # 过滤掉得分为0的结果
            results = []
            for idx in top_k_indices:
                if scores[idx] > 0:
                    results.append((self.docs[idx], float(scores[idx])))
                    print(f"Found document with score {scores[idx]}: {''.join(self.processed_docs[idx][:50])}...")
                    
            print(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            print(f"Error in GetBM25TopK: {str(e)}")
            return []
            
    def search(self, query: str, k: int = 10) -> list:
        """search方法，作为GetBM25TopK的别名
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            list: 包含(Document, score)元组的列表
        """
        return self.GetBM25TopK(query, k)
        
    def retrieve(self, query: str, k: int = 10) -> list:
        """retrieve方法，作为GetBM25TopK的别名
        
        Args:
            query: 查询文本 
            k: 返回的文档数量
            
        Returns:
            list: 包含(Document, score)元组的列表
        """
        return self.GetBM25TopK(query, k)

if __name__ == "__main__":

    # bm2.5
    dp =  DataProcess(pdf_path = "./data/train_a.pdf")
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data
    bm25 = BM25(data)
    res = bm25.GetBM25TopK("座椅加热", 6)
    print(res)
