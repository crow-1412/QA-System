import sys
import os
import json
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge.knowledge_refiner import KnowledgeRefiner, evaluate_answer_quality
from src.data_process import DataProcess
from system_config import MODEL_PATHS, OPTIMIZED_RETRIEVER_WEIGHTS, LLM_CONFIG, LLM_MODEL_PATHS

# 配置logger
logger = logging.getLogger(__name__)

def test_single_question():
    """测试单个问题的处理流程"""
    try:
        print("\n加载PDF数据...")
        data_processor = DataProcess()
        documents = data_processor.process_pdf("data/train_a.pdf")
        
        
        # 初始化配置
        config = {
            "model_path": LLM_MODEL_PATHS["chatglm3"],
            "documents": documents,
            "use_model_parallel": False,
            "device_ids": [0]
        }
        
        # 初始化知识精炼器
        knowledge_refiner = KnowledgeRefiner(config)
        
        # 测试问题
        question = "当我开启前挡风玻璃的去冰去雾模式时，车辆会有什么反应？"
        print(f"\n处理问题: {question}")
        
        # 调试输出：问题类型
        question_type = knowledge_refiner._get_question_type(question)
        print(f"问题类型: {question_type}")
        
        # 使用新的处理流程
        result = knowledge_refiner.process_query(question)
        
        # 调试输出：检查答案优化过程
        print("\n答案优化过程:")
        print(f"初始答案: {result['initial_answer']}")
        print(f"改写后的问题: {result['rewritten_query']}")
        print(f"检索到的文档数量: {result['doc_count']}")
        
        # 确保最终答案经过清理
        final_answer = result['final_answer']
        if '{content:' in final_answer:
            final_answer = knowledge_refiner._clean_response(
                knowledge_refiner._extract_final_answer(final_answer)
            )
            result['final_answer'] = final_answer
        
        print(f"\n最终答案: {final_answer}")
        
        # 评估答案质量
        quality_scores = evaluate_answer_quality(
            question=question,
            answer=final_answer,
            relevant_docs=documents
        )
        
        print("\n答案质量评分:")
        print(json.dumps(quality_scores, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    test_single_question() 