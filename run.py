#!/usr/bin/env python
# coding: utf-8

import json
import time
import torch
from langchain_core.documents import Document
from vllm_model import ChatLLM
from pdf_parse import DataProcess
from multi_retriever import MultiRetriever, MultiReranker
from config import GPU_MEMORY_FRACTION, BATCH_SIZE, TENSOR_PARALLEL_SIZE

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

def process_docs(docs, max_length=400, max_docs=4):
    """处理文档列表，返回拼接后的内容
    
    Args:
        docs: 文档列表，可以是(doc, score)元组列表或doc列表
        max_length: 最大长度限制，默认400 tokens
        max_docs: 最多使用的文档数量，默认4个
    
    Returns:
        str: 拼接后的文档内容
    """
    try:
        result = ""
        cnt = 0
        for item in docs:
            doc = item[0] if isinstance(item, tuple) else item
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            
            # 添加长度惩罚
            length_penalty = min(1.0, 200 / (len(content) + 1e-6))
            
            # 如果单个文档就超过长度限制，进行截断
            if len(content) > max_length:
                content = content[:max_length]
                
            # 检查添加当前文档是否会超出长度限制
            if len(result + content) > max_length:
                break
                
            # 根据长度惩罚调整内容
            content = content[:int(len(content) * length_penalty)]
            result += content
            cnt += 1
            if cnt >= max_docs:
                break
                
        return result
    except Exception as e:
        print(f"Error in process_docs: {str(e)}")
        return ""

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
    try:
        # 设置GPU内存使用限制
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, i)
                
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
        
        # 将字符串数据转换为Document对象
        data = [Document(page_content=text) if isinstance(text, str) else text for text in dp.data]
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
            
            for idx, line in enumerate(jdata):
                query = line["question"]
                print(f"\nProcessing question {idx}: {query}")

                try:
                    # 使用多路召回获取结果，增加召回数量
                    merged_results = multi_retriever.get_merged_results(query, k=12)
                    
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 使用多重排序器进行重排序，增加重排序数量
                    reranked_docs = multi_reranker.rerank(query, [doc for doc, _ in merged_results], top_k=6)
                    
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 处理召回的文档，使用优化后的参数
                    merged_context = process_docs(merged_results, max_length=400, max_docs=4)
                    reranked_context = process_docs(reranked_docs, max_length=400, max_docs=4)

                    # 构造prompt并执行推理
                    merged_prompt = get_rerank(merged_context, query)
                    reranked_prompt = get_rerank(reranked_context, query)
                    
                    # 分批处理以减少内存使用
                    batch_output = []
                    for prompt in [merged_prompt, reranked_prompt]:
                        try:
                            result = llm.infer([prompt])
                            if result:  # 确保结果不为空
                                batch_output.extend(result)
                            else:
                                batch_output.append("生成答案失败")
                        except Exception as e:
                            print(f"Error in inference: {str(e)}")
                            batch_output.append("生成答案失败")
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # 保存结果，添加错误处理
                    line["answer_1"] = batch_output[0] if len(batch_output) > 0 else "处理出错"
                    line["answer_2"] = batch_output[1] if len(batch_output) > 1 else "处理出错"
                    line["merged_context"] = merged_context
                    line["reranked_context"] = reranked_context
                    
                except Exception as e:
                    print(f"Error processing question {idx}: {str(e)}")
                    line["answer_1"] = "处理出错"
                    line["answer_2"] = "处理出错"
                    line["merged_context"] = ""
                    line["reranked_context"] = ""
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            # 保存结果
            json.dump(jdata, open(base + "/data/result.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)
            end = time.time()
            print("cost time: " + str(int(end-start)/60))
            
    except Exception as e:
        print(f"Critical error in main: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

if __name__ == "__main__":
    main()
