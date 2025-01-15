#!/usr/bin/env python
# coding: utf-8


from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from pdf_parse import DataProcess
import jieba

class BM25(object):

    # 遍历文档，首先做分词，然后把分词后的文档和全文文档建立索引和映射关系 
    def __init__(self, documents):
        """
        初始化BM25检索器
        参数:
            documents: 包含文档内容的列表
        功能:
            1. 对输入文档进行预处理和分词
            2. 构建两个文档集:
               - docs: 存储分词后的文档,用于BM25检索
               - full_docs: 存储原始文档,用于返回检索结果
            3. 初始化BM25检索器
        """
        # 初始化两个文档列表
        docs = []        # 存储分词后的文档
        full_docs = []   # 存储原始文档
        
        # 遍历处理每个输入文档
        for idx, line in enumerate(documents):
            # 清理文本,去除首尾空白和换行符
            line = line.strip("\n").strip()
            
            # 跳过过短的文本
            if(len(line)<5):
                continue
                
            # 对文本进行分词,并用空格连接
            tokens = " ".join(jieba.cut_for_search(line))
            
            # 将分词结果添加到docs列表
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            
            # 按tab分割文本
            words = line.split("\t")
            
            # 将原始文本添加到full_docs列表
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
            
        # 保存文档集到实例变量
        self.documents = docs
        self.full_documents = full_docs
        
        # 初始化BM25检索器
        self.retriever = self._init_bm25()

    # 初始化BM25的知识库
    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    # 获得得分在topk的文档和分数
    def GetBM25TopK(self, query, topk):
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        # 遍历检索结果,将原始文档添加到结果列表
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans

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
