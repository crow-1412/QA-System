from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import numpy as np
import torch

class SentenceTransformerMy(Embeddings):
    encode_kwargs = dict()
    multi_process = False
    show_progress = True
    
    def __init__(self, model_path, **kwargs):
        """初始化embedding模型
        
        Args:
            model_path: 模型路径
            **kwargs: 其他参数，如device等
        """
        print(f"\nInitializing SentenceTransformer from {model_path}...")
        try:
            self.client = SentenceTransformer(model_path, **kwargs)
            print("SentenceTransformer initialized successfully")
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """计算文档嵌入向量
        
        Args:
            texts: 要嵌入的文本列表
            
        Returns:
            List[List[float]]: 每个文本的嵌入向量列表
        """
        print(f"\nEmbedding {len(texts)} documents...")
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        try:
            if self.multi_process:
                print("Using multi-process encoding...")
                pool = self.client.start_multi_process_pool()
                embeddings = self.client.encode_multi_process(texts, pool)
                self.client.stop_multi_process_pool(pool)
            else:
                print("Using single-process encoding...")
                embeddings = self.client.encode(
                    texts, 
                    show_progress_bar=self.show_progress, 
                    batch_size=32,
                    **self.encode_kwargs
                )
            print("Document embedding completed")
            return embeddings.tolist()
        except Exception as e:
            print(f"Error embedding documents: {str(e)}")
            raise
        
    def embed_query(self, text: str) -> list[float]:
        """计算查询文本的嵌入向量
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            List[float]: 文本的嵌入向量
        """
        print("\nEmbedding query...")
        try:
            result = self.embed_documents([text])[0]
            print("Query embedding completed")
            return result
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            raise

class GTERetriever:
    def __init__(self, model_path, data):
        """初始化GTE检索器
        
        Args:
            model_path: GTE模型路径
            data: 包含Document对象的列表
        """
        self.model = SentenceTransformerMy(model_path, device="cuda")
        self.docs = data
        self.vector_store = self._create_vector_store(data)
        
    def _create_vector_store(self, data):
        """创建向量存储
        
        Args:
            data: Document对象列表
            
        Returns:
            FAISS: 向量存储对象
        """
        try:
            print("\nProcessing documents...")
            # 构建Document对象列表
            docs = []
            for idx, doc in enumerate(data):
                if isinstance(doc, Document):
                    docs.append(Document(page_content=doc.page_content, metadata={"id": idx}))
                else:
                    docs.append(Document(page_content=str(doc), metadata={"id": idx}))
            print(f"Processed {len(docs)} documents")
            
            # 使用FAISS创建向量索引
            print("\nCreating FAISS index...")
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            
            # 使用from_texts方法创建向量存储
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.model,
                metadatas=metadatas
            )
            print("FAISS index created successfully")
            
            # 释放GPU缓存
            print("\nClearing GPU cache...")
            torch.cuda.empty_cache()
            
            return vector_store
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise
        
    def GetTopK(self, query, k=10):
        """检索与查询最相关的k个文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            list: 包含(Document, score)元组的列表
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [(Document(page_content=doc.page_content), score) for doc, score in results] 