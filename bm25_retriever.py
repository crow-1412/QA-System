#!/usr/bin/env python
# coding: utf-8

from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from pdf_parse import DataProcess
import jieba

class BM25(object):
    def __init__(self, documents):
        """
        初始化BM25检索器
        参数:
            documents: 包含文档内容的列表
        """
        try:
            # 初始化两个文档列表
            docs = []        # 存储分词后的文档
            full_docs = []   # 存储原始文档
            
            # 遍历处理每个输入文档
            for idx, line in enumerate(documents):
                try:
                    # 清理文本
                    line = line.strip("\n").strip()
                    
                    # 跳过过短的文本
                    if len(line) < 5:
                        continue
                    
                    # 分词处理
                    words = line.split("\t")
                    if not words:  # 确保分割后的文本非空
                        continue
                        
                    content = words[0]
                    tokens = " ".join(jieba.cut_for_search(content))
                    
                    # 创建Document对象
                    doc = Document(page_content=tokens, metadata={"id": idx})
                    full_doc = Document(page_content=content, metadata={"id": idx})
                    
                    docs.append(doc)
                    full_docs.append(full_doc)
                    
                except Exception as e:
                    print(f"Error processing document {idx}: {str(e)}")
                    continue
            
            # 保存文档集到实例变量
            self.documents = docs
            self.full_documents = full_docs
            
            # 初始化BM25检索器
            self.retriever = self._init_bm25()
            
        except Exception as e:
            print(f"Error initializing BM25: {str(e)}")
            raise

    def _init_bm25(self):
        """初始化BM25检索器"""
        try:
            return BM25Retriever.from_documents(self.documents)
        except Exception as e:
            print(f"Error initializing BM25 retriever: {str(e)}")
            raise

    def GetBM25TopK(self, query, topk):
        """获取得分最高的topk个文档"""
        try:
            self.retriever.k = topk
            # 对查询进行分词
            processed_query = " ".join(jieba.cut_for_search(query))
            
            # 使用新的invoke方法替代get_relevant_documents
            ans_docs = self.retriever.invoke(processed_query)
            
            # 获取原始文档
            ans = []
            for doc in ans_docs:
                try:
                    doc_id = doc.metadata.get("id")
                    if doc_id is not None and doc_id < len(self.full_documents):
                        ans.append(self.full_documents[doc_id])
                except Exception as e:
                    print(f"Error processing result document: {str(e)}")
                    continue
                    
            return ans
            
        except Exception as e:
            print(f"Error in GetBM25TopK: {str(e)}")
            return []

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
