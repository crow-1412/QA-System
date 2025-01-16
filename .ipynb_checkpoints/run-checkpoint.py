#!/usr/bin/env python
# coding: utf-8

import json
import time
from langchain_core.documents import Document
from vllm_model import ChatLLM
from pdf_parse import DataProcess
from multi_retriever import MultiRetriever, MultiReranker

# 模型路径常量
MODEL_PATHS = {
    "qwen": "/root/autodl-tmp/pre_train_model/Qwen-7B-Chat",
    "m3e": "/root/autodl-tmp/pre_train_model/m3e-large",
    "bge": "/root/autodl-tmp/pre_train_model/bge-large-zh-v1.5",
    "gte": "/root/autodl-tmp/pre_train_model/gte-large-zh",
    "bce": "/root/autodl-tmp/pre_train_model/bce-embedding-base_v1",
    "bge_reranker": "/root/autodl-tmp/pre_train_model/bge-reranker-large",
    "bce_reranker": "/root/autodl-tmp/pre_train_model/bce-reranker-base_v1"
}

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

def get_rerank(emb_ans, query):
    """构造重排序的prompt模板
    
    Args:
        emb_ans: 文档内容
        query: 查询问题
    
    Returns:
        str: 格式化后的prompt
    """
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                            如果无法从中得到答案，请说 "无答案"或"无答案" ，不允许在答案中添加编造成分，答案请使用中文。
                            已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                            1: {emb_ans}
                            问题:
                            {question}""".format(emb_ans=emb_ans, question=query)
    return prompt_template

def main():
    """主函数"""
    start = time.time()
    base = "."

    # 解析pdf文档，构造数据
    dp = DataProcess(pdf_path=base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    print(len(dp.data))
    data = dp.data
    print("data load ok")

    # 初始化多路召回器和重排序器
    multi_retriever = MultiRetriever(data, MODEL_PATHS)
    multi_reranker = MultiReranker(MODEL_PATHS)
    print("retriever and reranker load ok")

    # LLM大模型
    llm = ChatLLM(MODEL_PATHS["qwen"])
    print("llm qwen load ok")

    # 处理测试问题
    with open(base + "/data/test_question.json", "r") as f:
        jdata = json.loads(f.read())
        print(len(jdata))
        max_length = 4000
        
        for idx, line in enumerate(jdata):
            query = line["question"]

            try:
                # 使用多路召回获取结果
                merged_results = multi_retriever.get_merged_results(query, k=15)
                
                # 使用多重排序器进行重排序
                reranked_docs = multi_reranker.rerank(query, [doc for doc, _ in merged_results])
                
                # 处理召回的文档
                merged_context = process_docs(merged_results, max_length)
                reranked_context = process_docs(reranked_docs, max_length)

                # 构造prompt并执行推理
                merged_prompt = get_rerank(merged_context, query)
                reranked_prompt = get_rerank(reranked_context, query)
                batch_input = [merged_prompt, reranked_prompt]
                batch_output = llm.infer(batch_input)
                
                # 保存结果
                line["answer_1"] = batch_output[0]  # 多路召回的结果
                line["answer_2"] = batch_output[1]  # 重排序后的结果
                line["merged_context"] = merged_context
                line["reranked_context"] = reranked_context
                
            except Exception as e:
                print(f"Error processing question {idx}: {str(e)}")
                line["answer_1"] = "处理出错"
                line["answer_2"] = "处理出错"
                continue

        # 保存结果
        json.dump(jdata, open(base + "/data/result.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        end = time.time()
        print("cost time: " + str(int(end-start)/60))

if __name__ == "__main__":
    main()
