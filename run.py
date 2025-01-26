from typing import List, Dict
import os
import json
import random
import logging
from system_config import (
    MODEL_PATHS,
    PARALLEL_CONFIG,
    OPTIMIZED_RETRIEVER_WEIGHTS,
    GPU_CONFIG
)
from src.llm.llm_factory import LLMFactory
from src.knowledge.knowledge_refiner import KnowledgeRefiner, evaluate_answer_quality
from src.retriever.multi_retriever import MultiRetriever
from src.retriever.reranker import MultiReranker
from langchain.schema import Document
from src.data_process import DataProcess
import torch
import argparse
import sys


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_batch(queries: List[str], multi_retriever: MultiRetriever, knowledge_refiner: KnowledgeRefiner, multi_reranker: MultiReranker) -> List[Dict]:
    """批量处理问题
    
    Args:
        queries: 问题列表
        multi_retriever: 多检索器实例
        knowledge_refiner: 知识规整器实例
        multi_reranker: 多重排序器实例
        
    Returns:
        List[Dict]: 处理结果列表
    """
    results = []
    actual_batch_size = min(GPU_CONFIG['batch_size'], len(queries))
    
    for i in range(0, len(queries), actual_batch_size):
        batch_queries = queries[i:i + actual_batch_size]
        batch_results = []
        
        try:
            # 使用异步流处理检索和生成
            with torch.cuda.stream(knowledge_refiner.retrieval_stream):
                # 批量处理每个查询
                for query in batch_queries:
                    try:
                        # 1. 提取查询关键词并进行相关性初步检查
                        query_keywords = multi_retriever.extract_key_info([query])
                        print(f"Query keywords: {query_keywords}")
                        
                        # 2. 增强查询并检索文档
                        enhanced_query, direct_answer = knowledge_refiner.get_enhanced_query(query)
                        docs = multi_retriever.get_merged_results(enhanced_query, k=5)
                        
                        # 3. 检查检索结果的相关性
                        retrieved_docs = []
                        for doc, _ in docs:
                            if isinstance(doc, Document):
                                doc_text = doc.page_content
                            elif isinstance(doc, str):
                                doc_text = doc
                            else:
                                continue
                                
                            # 检查文档相关性
                            if any(keyword in doc_text.lower() for keyword in query_keywords):
                                retrieved_docs.append(doc)
                        
                        if not retrieved_docs:
                            print(f"No relevant documents found for query: {query}")
                            batch_results.append({
                                "query": query,
                                "error": "No relevant documents found"
                            })
                            continue
                        
                        # 4. 重排序
                        try:
                            reranked_docs = multi_reranker.rerank(query, retrieved_docs)
                            print(f"Documents reranked successfully for query: {query}")
                        except Exception as e:
                            print(f"Reranking failed for query '{query}': {str(e)}")
                            reranked_docs = retrieved_docs
                        
                        # 5. 提取文档关键词
                        doc_texts = [doc.page_content if isinstance(doc, Document) else doc for doc, _ in reranked_docs]
                        doc_keywords = multi_retriever.extract_key_info(doc_texts)
                        print(f"Document keywords: {doc_keywords}")
                        
                        # 6. 规整文档
                        refined_docs = knowledge_refiner.refine_knowledge(doc_texts)
                        
                        # 7. 生成答案
                        answer = knowledge_refiner.optimize_answer_iteratively(query, refined_docs)
                        
                        # 8. 评估答案质量
                        quality_scores = evaluate_answer_quality(query, answer, refined_docs)
                        
                        # 9. 记录结果
                        result = {
                            "query": query,
                            "direct_answer": direct_answer,
                            "query_keywords": query_keywords,
                            "doc_keywords": doc_keywords,
                            "retrieved_docs": doc_texts,
                            "answer": answer,
                            "quality_scores": quality_scores
                        }
                        batch_results.append(result)
                        
                    except Exception as e:
                        print(f"Error processing query '{query}': {str(e)}")
                        batch_results.append({
                            "query": query,
                            "error": str(e)
                        })
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 合并结果
            results.extend(batch_results)
            
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            for query in batch_queries:
                batch_results.append({
                    "query": query,
                    "error": str(e)
                })
    
    return results

def process_questions(test_data, knowledge_refiner, multi_retriever, multi_reranker):
    """处理测试问题
    
    Args:
        test_data: 测试数据
        knowledge_refiner: 知识规整器
        multi_retriever: 多检索器
        multi_reranker: 多重排序器
        
    Returns:
        list: 处理结果列表
    """
    batch_results = []
    
    try:
        for question_data in test_data:
            try:
                # 1. 获取问题
                query = question_data.get("question", "").strip()
                if not query:
                    logger.warning("跳过空问题")
                    continue
                    
                logger.info(f"开始处理问题: {query}")
                
                # 2. 增强查询并检索文档
                enhanced_query, direct_answer = knowledge_refiner.get_enhanced_query(query)
                logger.info(f"增强后的查询: {enhanced_query}")
                logger.info(f"直接回答: {direct_answer}")
                
                # 3. 检索相关文档
                retrieved_docs = multi_retriever.retrieve(enhanced_query, k=5)
                logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")
                
                # 确保检索结果是 Document 类型
                retrieved_docs = [
                    doc if isinstance(doc, Document) else Document(page_content=str(doc))
                    for doc in retrieved_docs
                ]
                
                # 4. 重排序文档
                reranked_docs = multi_reranker.rerank(enhanced_query, retrieved_docs, top_k=3)
                logger.info(f"重排序后保留 {len(reranked_docs)} 个文档")
                
                # 确保重排序结果是 Document 类型
                reranked_docs = [
                    doc if isinstance(doc, Document) else Document(page_content=str(doc))
                    for doc in reranked_docs
                ]
                
                # 5. 规整和组织知识
                refined_docs = knowledge_refiner.refine_knowledge(reranked_docs)
                logger.info(f"规整后的文档数量: {len(refined_docs)}")
                
                # 6. 生成答案
                context = "\n".join([doc.page_content for doc in refined_docs])
                answer = knowledge_refiner.generate_answer(context, query) if context else "抱歉，没有找到相关信息。"
                logger.info(f"生成答案长度: {len(answer)}")
                
                # 7. 保存结果
                result = {
                    "question": query,
                    "answer": answer,
                    "direct_answer": direct_answer,
                    "doc_count": len(refined_docs),
                    "knowledge_graph": dict(knowledge_refiner.knowledge_graph)  # 保存知识图谱
                }
                batch_results.append(result)
                logger.info("问题处理完成")
                
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}", exc_info=True)
                batch_results.append({
                    "question": query,
                    "error": str(e)
                })
                
    except Exception as e:
        logger.error(f"批处理过程出错: {str(e)}", exc_info=True)
        
    return batch_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="智能问答系统")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATHS["chatglm3"],
        help="LLM模型路径"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="运行设备"
    )
    
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="./data/train_a.pdf",
        help="PDF文档路径"
    )
    
    return parser.parse_args()

def init_system(args):
    """初始化系统组件
    
    Args:
        args: 命令行参数
        
    Returns:
        tuple: (knowledge_refiner, multi_retriever, multi_reranker)
    """
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
    # 加载PDF数据
    dp = DataProcess(pdf_path=args.pdf_path)
    
    # 使用多种参数组合解析
    print("\n使用不同参数组合解析...")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    block_data = dp.data.copy()
    print(f"ParseBlock结果数量: {len(block_data)}")
    
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    page_data = [x for x in dp.data if x not in block_data]
    print(f"ParseAllPage新增结果数量: {len(page_data)}")
    
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    rule_data = [x for x in dp.data if x not in block_data and x not in page_data]
    print(f"ParseOnePageWithRule新增结果数量: {len(rule_data)}")
    
    # 合并所有解析结果
    all_data = []
    seen = set()
    for text in block_data + page_data + rule_data:
        if isinstance(text, str) and text.strip():
            text_hash = hash(text.strip())
            if text_hash not in seen:
                seen.add(text_hash)
                all_data.append(text)
                    
    print(f"\n合并后的总文档数量: {len(all_data)}")
    
    # 转换为Document对象
    data = [Document(page_content=text) for text in all_data if isinstance(text, str)]
    
    # 初始化组件
    print("初始化系统组件...")
    knowledge_refiner = KnowledgeRefiner()
    multi_retriever = MultiRetriever(data=data, model_paths=MODEL_PATHS, weights=OPTIMIZED_RETRIEVER_WEIGHTS)
    multi_reranker = MultiReranker(model_paths=MODEL_PATHS)
    
    return knowledge_refiner, multi_retriever, multi_reranker

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 设置工作目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 初始化系统组件
        print("初始化系统组件...")
        knowledge_refiner, multi_retriever, multi_reranker = init_system(args)
        
        # 加载测试问题
        print("加载测试问题...")
        json_path = os.path.join(current_dir, "data/test_question.json")
        with open(json_path, "r", encoding='utf-8') as f:
            test_data = json.load(f)
            
        # 处理测试问题
        processed_questions = process_questions(
            test_data,
            knowledge_refiner,
            multi_retriever,
            multi_reranker
        )
        
        # 保存结果
        output_path = os.path.join(current_dir, "data/result.json")
        if processed_questions:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_questions, f, ensure_ascii=False, indent=4)
            logger.info("结果已保存到 result.json")
        else:
            logger.error("没有成功处理的问题")
            
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        return

if __name__ == "__main__":
    main()