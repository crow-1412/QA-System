#!/usr/bin/env python
# coding: utf-8

import os
import torch
from sentence_transformers import SentenceTransformer, models
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from pdf_parse import DataProcess


class FaissRetriever(object):
    def __init__(self, model, data):
        """初始化FAISS检索器
        
        Args:
            model: HuggingFace embedding模型的路径或SentenceTransformer对象
            data: 待索引的文本数据列表
        """
        try:
            if isinstance(model, str):
                self.model = SentenceTransformer(model)
            else:
                self.model = model
            
            # 将模型移动到GPU（如果可用）
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            
            # 构建 Document 对象列表
            docs = []
            for idx, line in enumerate(data):
                line = line.strip("\n").strip()
                words = line.split("\t")
                docs.append(Document(page_content=words[0], metadata={"id": idx}))
                
            # 使用模型编码文档
            print("Encoding documents...")
            embeddings = self.model.encode([doc.page_content for doc in docs], 
                                         normalize_embeddings=True,
                                         show_progress_bar=True,
                                         batch_size=32)
            
            # 创建 FAISS 索引
            print("Creating FAISS index...")
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=[(doc.page_content, emb) for doc, emb in zip(docs, embeddings)],
                embedding=self.model,
                metadatas=[doc.metadata for doc in docs]
            )
            
            # 释放资源
            del embeddings
            if torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
        except Exception as e:
            print(f"Error initializing FaissRetriever: {str(e)}")
            raise

    def GetTopK(self, query, k):
        """获取top-K分数最高的文档块"""
        try:
            context = self.vector_store.similarity_search_with_score(query, k=k)
            return context
        except Exception as e:
            print(f"Error in GetTopK: {str(e)}")
            return []

    def GetvectorStore(self):
        """返回faiss向量检索对象"""
        return self.vector_store

if __name__ == "__main__":
    base = "."
    model_name = base + "/pre_train_model/m3e-large" #text2vec-large-chinese
    dp =  DataProcess(pdf_path = base + "/data/train_a.pdf")
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

    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("如何预防新冠肺炎", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("交通事故如何处理", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利集团的董事长是谁", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利汽车语音组手叫什么", 6)
    print(faiss_ans)
