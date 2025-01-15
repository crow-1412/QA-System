#!/usr/bin/env python
# coding: utf-8

import json
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
import time
import re

from vllm_model import ChatLLM
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess
from sentence_transformers import SentenceTransformer

def process_docs(docs, max_length=4000, max_docs=6):
    """处理文档列表，返回拼接后的内容
    
    Args:
        docs: 文档列表，可以是(doc, score)元组列表或doc列表
        max_length: 最大长度限制
        max_docs: 最多使用的文档数量
    
    Returns:
        str: 拼接后的文档内容
    """
    result = ""
    cnt = 0
    for item in docs:
        doc = item[0] if isinstance(item, tuple) else item
        cnt += 1
        if len(result + doc.page_content) > max_length:
            break
        result += doc.page_content
        if cnt >= max_docs:
            break
    return result

# 获取Langchain的工具链 
def get_qa_chain(llm, vector_store, prompt_template):

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 10}), prompt=prompt)

# 构造提示，根据merged faiss和bm25的召回结果返回答案
def get_emb_bm25_merge(faiss_context, bm25_context, query):
    max_length = 2500
    emb_ans = process_docs(faiss_context, max_length)
    bm25_ans = process_docs(bm25_context, max_length)

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案"，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                2: {bm25_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, bm25_ans = bm25_ans, question = query)
    return prompt_template


def get_rerank(emb_ans, query):

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案" ，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, question = query)
    return prompt_template


def question(text, llm, vector_store, prompt_template):

    chain = get_qa_chain(llm, vector_store, prompt_template)

    response = chain({"query": text})
    return response

def reRank(rerank, top_k, query, bm25_ans, faiss_ans):
    """对Faiss和BM25召回的结果进行重排序
    
    Args:
        rerank: 重排序模型对象
        top_k: 重排序后需要返回的文档数量
        query: 用户查询文本
        bm25_ans: BM25召回的文档列表
        faiss_ans: Faiss召回的文档列表,每个元素是(doc, score)元组
    
    Returns:
        str: 重排序后拼接的文档内容字符串
        
    主要步骤:
        1. 将Faiss和BM25召回的文档合并到一个列表
        2. 使用重排序模型对合并后的文档进行重新排序
        3. 取排序后的前top_k个文档
        4. 将文档内容拼接,控制总长度不超过max_length
    """
    # 初始化文档列表和最大长度限制
    items = []
    max_length = 4000
    
    # 将Faiss召回结果中的文档添加到列表
    for doc, score in faiss_ans:
        items.append(doc)
    
    # 将BM25召回的文档添加到列表
    items.extend(bm25_ans)
    
    # 使用重排序模型对文档重新排序
    rerank_ans = rerank.predict(query, items)
    
    # 取排序后的前top_k个文档
    rerank_ans = rerank_ans[:top_k]
    
    # 拼接重排序后的文档内容
    emb_ans = ""
    for doc in rerank_ans:
        # 如果添加当前文档会超过最大长度限制,则停止添加
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
        
    return emb_ans

if __name__ == "__main__":

    start = time.time()

    base = "."
    qwen7 = base + "/pre_train_model/Qwen-7B-Chat"
    m3e =  base + "/pre_train_model/m3e-large"
    bge_reranker_large = base + "/pre_train_model/bge-reranker-large"

    # 解析pdf文档，构造数据
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
    print("data load ok")

    # Faiss召回
    model_path = '/root/project/pre_train_model/m3e-large'
    faissretriever = FaissRetriever(model_path, data)
    vector_store = faissretriever.vector_store
    print("faissretriever load ok")

    # BM25召回
    bm25 = BM25(data)
    print("bm25 load ok")

    # LLM大模型
    llm = ChatLLM(qwen7)
    print("llm qwen load ok")

    # reRank模型
    rerank = reRankLLM(bge_reranker_large)
    print("rerank model load ok")

    # 对每一条测试问题，做答案生成处理
    with open(base + "/data/test_question.json", "r") as f:
        jdata = json.loads(f.read())
        print(len(jdata))
        max_length = 4000
        for idx, line in enumerate(jdata):
            query = line["question"]

            # faiss召回topk
            # 使用faiss向量检索获取与query最相似的15个文档及其相似度分数
            faiss_context = faissretriever.GetTopK(query, 15)
            
            # 初始化最小相似度分数为0
            faiss_min_score = 0.0
            
            # 如果检索到了文档,获取第一个文档(最相似)的相似度分数
            if(len(faiss_context) > 0):
                faiss_min_score = faiss_context[0][1]
                
            # 处理检索到的文档,将其截断到max_length长度,作为后续生成的上下文
            emb_ans = process_docs(faiss_context, max_length)

            # bm2.5召回topk
            bm25_context = bm25.GetBM25TopK(query, 15)
            bm25_ans = process_docs(bm25_context, max_length)

            # 构造合并bm25召回和向量召回的prompt
            emb_bm25_merge_inputs = get_emb_bm25_merge(faiss_context, bm25_context, query)

            # 构造bm25召回的prompt
            bm25_inputs = get_rerank(bm25_ans, query)

            # 构造向量召回的prompt
            emb_inputs = get_rerank(emb_ans, query)

            # rerank召回的候选，并按照相关性得分排序
            rerank_ans = reRank(rerank, 6, query, bm25_context, faiss_context)
            # 构造得到rerank后生成答案的prompt
            rerank_inputs = get_rerank(rerank_ans, query)

            batch_input = []
            batch_input.append(emb_bm25_merge_inputs)
            batch_input.append(bm25_inputs)
            batch_input.append(emb_inputs)
            batch_input.append(rerank_inputs)
            # 执行batch推理
            batch_output = llm.infer(batch_input)
            line["answer_1"] = batch_output[0] # 合并两路召回的结果
            line["answer_2"] = batch_output[1] # bm召回的结果
            line["answer_3"] = batch_output[2] # 向量召回的结果
            line["answer_4"] = batch_output[3] # 多路召回重排序后的结果
            line["answer_5"] = emb_ans
            line["answer_6"] = bm25_ans
            line["answer_7"] = rerank_ans
            # 如果faiss检索跟query的距离高于500，输出无答案
            if(faiss_min_score >500):
                line["answer_5"] = "无答案"
            else:
                line["answer_5"] = str(faiss_min_score)

        # 保存结果，生成submission文件
        json.dump(jdata, open(base + "/data/result.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        end = time.time()
        print("cost time: " + str(int(end-start)/60))
