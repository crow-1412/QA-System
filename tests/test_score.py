# coding=utf-8
import json
import sys
import re
import numpy as np
from text2vec import SentenceModel, semantic_search, Similarity
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.llm_factory import LLMFactory
from system_config import DEFAULT_LLM, LLM_MODEL_PATHS

# 修改模型路径并添加路径检查
simModel_path = "/root/autodl-tmp/pre_train_model/text2vec-base-chinese"  # 相似度模型路径

# 检查模型路径是否存在
if not os.path.exists(simModel_path):
    raise FileNotFoundError(f"模型路径不存在: {simModel_path}")

try:
    print(f"正在加载相似度模型: {simModel_path}")
    simModel = SentenceModel(model_name_or_path=simModel_path, device='cuda:0')  # 使用第一张 GPU
    print("相似度模型加载成功")
except Exception as e:
    print(f"加载相似度模型失败: {str(e)}")
    raise

def calc_jaccard(list_a, list_b, threshold=0.3):
    """计算关键词的Jaccard相似度
    
    Args:
        list_a: 第一个关键词列表
        list_b: 第二个关键词列表
        threshold: 相似度阈值，默认0.3
        
    Returns:
        float: 如果相似度大于阈值返回1，否则返回0
    """
    size_a, size_b = len(list_a), len(list_b)
    list_c = [i for i in list_a if i in list_b]
    size_c = len(list_c)
    score = size_c / (size_b + 1e-6)
    if score > threshold:
        return 1
    else:
        return 0

def calc_single_score(gold, pred, keywords):
    """计算单个预测答案的得分
    
    Args:
        gold: 标准答案
        pred: 预测答案
        keywords: 关键词列表
        
    Returns:
        float: 得分
    """
    try:
        if not pred or pred.isspace():  # 处理空答案或纯空白答案
            return 0.0
            
        if gold == "无答案" and pred != gold:
            return 0.0
        elif gold == "无答案" and pred == gold:
            return 1.0
        else:
            # 计算语义相似度得分
            semantic_score = semantic_search(simModel.encode([gold]), simModel.encode([pred]), top_k=1)[0][0]['score']
            # 计算关键词匹配得分
            join_keywords = [word for word in keywords if word in pred]
            keyword_score = calc_jaccard(join_keywords, keywords)
            # 最终得分为语义得分和关键词得分的加权平均
            return 0.5 * keyword_score + 0.5 * semantic_score
    except Exception as e:
        print(f"Error calculating score: {str(e)}")
        return 0.0

def report_score(gold_path, predict_path, llm=None):
    """计算预测结果的得分"""
    try:
        # 加载标准答案和预测结果
        gold_info = json.load(open(gold_path))
        pred_info = json.load(open(predict_path))

        idx = 0
        for gold, pred in zip(gold_info, pred_info):
            # 获取问题、关键词和答案
            question = gold["question"]
            keywords = gold.get("keywords", [])
            gold_answer = gold["answer"].strip()
            
            # 从result.json中获取答案，使用统一的字段名
            main_answer = pred.get("answer", "").strip()  # 主答案
            direct_answer = pred.get("direct_answer", "").strip()  # 直接答案
            
            # 使用新的评估函数计算得分
            main_score = evaluate_answer_quality(
                query=question,
                answer=main_answer,
                docs=[gold_answer],
                llm=llm
            )
            
            direct_score = evaluate_answer_quality(
                query=question,
                answer=direct_answer,
                docs=[gold_answer],
                llm=llm
            )
            
            # 取最高分作为最终得分
            final_score = max(main_score["weighted_score"], direct_score["weighted_score"])
            
            # 记录得分和预测结果，保持字段命名一致性
            gold_info[idx]["score"] = final_score
            gold_info[idx]["answer"] = main_answer
            gold_info[idx]["direct_answer"] = direct_answer
            gold_info[idx]["answer_score"] = main_score["weighted_score"]
            gold_info[idx]["direct_answer_score"] = direct_score["weighted_score"]
            gold_info[idx]["answer_evaluation"] = main_score
            gold_info[idx]["direct_answer_evaluation"] = direct_score
            
            idx += 1
            print(f"\n问题: {question}")
            print(f"标准答案: {gold_answer}")
            print(f"主要答案: {main_answer}")
            print(f"主要答案评估详情:")
            print(json.dumps(main_score, ensure_ascii=False, indent=2))
            print(f"直接答案: {direct_answer}")
            print(f"直接答案评估详情:")
            print(json.dumps(direct_score, ensure_ascii=False, indent=2))
            print(f"最终得分: {final_score:.4f}")
            print("-" * 80)

        return gold_info
        
    except Exception as e:
        print(f"Error in report_score: {str(e)}")
        return []

def extract_score(response: str) -> float:
    """从LLM响应中提取分数并进行更细致的映射"""
    try:
        # 清理响应文本，只保留数字和小数点
        cleaned = re.sub(r'[^0-9.]', '', response.strip())
        score = float(cleaned)
        
        # 更细致的分数映射
        if score <= 2:
            return 0.2
        elif score <= 3:
            return 0.3
        elif score <= 4:
            return 0.4
        elif score <= 5:
            return 0.5
        elif score <= 6:
            return 0.6
        elif score <= 7:
            return 0.7
        elif score <= 8:
            return 0.8
        elif score <= 9:
            return 0.9
        else:
            return 1.0
    except:
        return 0.4  # 解析失败时返回较低分数

def evaluate_answer_quality(query: str, answer: str, docs: list, llm=None) -> dict:
    """改进后的答案质量评估函数"""
    try:
        if not answer or answer.isspace():
            return {
                "is_valid": False,
                "completeness": 0.0,
                "relevance": 0.0,
                "fact_consistency": 0.0,
                "clarity": 0.0,
                "weighted_score": 0.0
            }

        # 1. 要点覆盖度评估 (35%)
        prompt = f"""请将以下问题拆分为需要回答的关键要点，严格按照示例格式返回JSON数组：

示例问题1："如何预防新冠？"
示例输出1：["传播途径","个人防护措施","群体免疫措施"]

示例问题2："什么是电动车续航里程？"
示例输出2：["续航里程定义","影响因素","计算方法"]

问题：{query}
请直接返回JSON数组，不要有任何其他文字："""
        
        try:
            # 多次尝试解析要点
            for _ in range(3):
                try:
                    key_points = json.loads(llm.infer([prompt])[0])
                    if isinstance(key_points, list) and len(key_points) > 0:
                        break
                except:
                    continue
            else:  # 如果3次都失败，使用备选策略
                key_points = [query]  # 将问题本身作为唯一要点
            
            covered_points = []
            for point in key_points:
                point_check_prompt = f"""请评估答案对问题要点的覆盖程度。

要点：{point}
答案：{answer}

评分标准（0-10分）：
0-2分：完全不合格（完全未提及该要点或内容完全错误）
例如：问"如何预防感冒"，答案谈论其他无关话题

3-4分：差（略微提及但信息严重不足或有明显错误）
例如：问"如何预防感冒"，答案只说"多喝水"且有错误信息

5-6分：一般（部分回答，但存在明显不足）
例如：问"如何预防感冒"，答案只提到保暖，未提及其他关键方面

7-8分：良好（基本完整但有待改进）
例如：问"如何预防感冒"，提到保暖、营养但未提运动，有小的遗漏

9-10分：优秀（完整且准确，无明显改进空间）
例如：问"如何预防感冒"，全面且准确地提到保暖、营养、运动、休息等

请根据以上标准严格评分，直接返回分数（如：7），不要其他文字："""
                
                # 多次采样取平均
                scores = []
                for _ in range(3):  # 增加采样次数
                    try:
                        score = extract_score(llm.infer([point_check_prompt])[0])
                        if 0 <= score <= 1:
                            scores.append(score)
                    except:
                        continue
                point_score = sum(scores) / len(scores) if scores else 0.4
                covered_points.append(point_score)
            
            completeness = sum(covered_points) / len(key_points) if key_points else 0.4
        except Exception as e:
            print(f"要点覆盖度评估出错: {str(e)}")
            completeness = 0.4

        # 2. 语义相关性评估 (25%)
        semantic_check_prompt = f"""请评估答案与问题的语义相关程度。

问题：{query}
答案：{answer}

评分标准（0-10分）：
0-2分：完全不相关（答案内容与问题主题完全无关）
例如：问"如何使用洗衣机"，答案讲述电视机使用方法

3-4分：严重偏离（答案虽有极少相关内容，但主要内容偏离主题）
例如：问"如何使用洗衣机"，答案主要在推销洗衣粉品牌

5-6分：部分相关（答案部分内容相关，但有明显偏离）
例如：问"如何使用洗衣机"，答案过多讨论洗衣机品牌选择

7-8分：基本相关（答案主要内容相关，有少量不相关内容）
例如：问"如何使用洗衣机"，答案主要描述操作步骤，但包含些许无关信息

9-10分：高度相关（答案完全围绕问题主题，无偏离内容）
例如：问"如何使用洗衣机"，答案准确描述操作步骤和注意事项

请根据以上标准严格评分，直接返回分数（如：7），不要其他文字："""
        
        # 多次采样取平均
        relevance_scores = []
        for _ in range(3):  # 增加采样次数
            try:
                score = extract_score(llm.infer([semantic_check_prompt])[0])
                if 0 <= score <= 1:
                    relevance_scores.append(score)
            except:
                continue
        relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.4

        # 3. 事实一致性检查 (30%)
        docs_text = "\n".join([str(doc) for doc in docs])
        if docs_text.strip() == "无答案":
            if "无答案" in answer or "不确定" in answer or "抱歉" in answer:
                fact_consistency = 0.8  # 对于"无答案"情况的合理回应给较高分
            else:
                fact_consistency = 0.2  # 对于无依据的回答给低分
        else:
            fact_check_prompt = f"""请严格评估答案与参考文档的事实一致性。

参考文档：{docs_text}
答案：{answer}

评分标准（0-10分）：
0-2分：完全不一致（答案内容与参考文档完全矛盾或纯属编造）
例如：文档说"按A键开机"，答案说"按B键开机"

3-4分：严重不一致（答案大部分内容与参考文档矛盾或缺乏依据）
例如：文档详细描述5个步骤，答案只提到2个且有错误

5-6分：部分一致（答案部分内容有依据，但存在明显偏差）
例如：文档提供完整流程，答案只覆盖一半且有错误

7-8分：基本一致（答案主要内容有依据，存在小的偏差）
例如：文档列出10个要点，答案准确提到8个

9-10分：完全一致（答案内容完全基于参考文档，无错误信息）
例如：答案准确完整地复述了文档中的关键信息，无添加或遗漏

请根据以上标准严格评分，直接返回分数（如：7），不要其他文字："""
            
            # 多次采样取平均
            consistency_scores = []
            for _ in range(3):  # 增加采样次数
                try:
                    score = extract_score(llm.infer([fact_check_prompt])[0])
                    if 0 <= score <= 1:
                        consistency_scores.append(score)
                except:
                    continue
            fact_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.4

        # 4. 信息组织评估 (10%)
        structure_prompt = f"""请评估答案的结构清晰度和表达质量。

答案：{answer}

评分标准（0-10分）：
0-2分：结构混乱（内容杂乱无章，表达不通顺）
例如：语句零散，逻辑跳跃，没有段落划分

3-4分：结构松散（有基本段落但逻辑混乱）
例如：段落之间缺乏连贯性，存在重复或矛盾

5-6分：结构一般（有明确段落但层次不够清晰）
例如：段落划分合理但过渡不够自然

7-8分：结构清晰（层次分明，表达流畅）
例如：段落分明，层次有序，衔接自然

9-10分：结构优秀（层次清晰，表达精准，逻辑严密）
例如：段落划分合理，层次分明，衔接流畅，表达准确

请根据以上标准严格评分，直接返回分数（如：7），不要其他文字："""
        
        # 多次采样取平均
        clarity_scores = []
        for _ in range(3):  # 增加采样次数
            try:
                score = extract_score(llm.infer([structure_prompt])[0])
                if 0 <= score <= 1:
                    clarity_scores.append(score)
            except:
                continue
        clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.4

        # 5. 综合评分（调整权重）
        final_score = {
            "completeness": float(completeness),
            "relevance": float(relevance),
            "fact_consistency": float(fact_consistency),
            "clarity": float(clarity),
            "weighted_score": float(
                0.35 * completeness +  # 降低完整性权重
                0.25 * relevance +    # 提高相关性权重
                0.30 * fact_consistency +  # 保持事实一致性权重
                0.10 * clarity  # 保持清晰度权重
            )
        }

        # 6. 有效性判断（调整阈值）
        final_score["is_valid"] = all([
            completeness >= 0.5,  # 降低完整性要求
            relevance >= 0.6,     # 提高相关性要求
            fact_consistency >= 0.6 if docs_text.strip() != "无答案" else True,  # 保持事实一致性要求
            clarity >= 0.4        # 保持清晰度要求
        ])

        return final_score

    except Exception as e:
        print(f"评估答案质量时发生错误: {str(e)}")
        return {
            "is_valid": False,
            "completeness": 0.4,
            "relevance": 0.4,
            "fact_consistency": 0.4,
            "clarity": 0.4,
            "weighted_score": 0.4,
            "error": str(e)
        }

if __name__ == "__main__":
    '''
    评估预测结果
    '''
    # 初始化LLM
    model_type = DEFAULT_LLM
    model_path = LLM_MODEL_PATHS[model_type]
    llm = LLMFactory.create_llm(model_type=model_type, model_path=model_path)
    
    # 标准答案路径
    gold_path = "./data/gold.json" 
    print("Read gold from %s" % gold_path)

    # 预测文件路径
    predict_path = "./data/result.json" 
    print("Read predict file from %s" % predict_path)

    # 计算得分
    results = report_score(gold_path, predict_path, llm=llm)
    
    if results:
        # 输出最终得分
        final_score = np.mean([item["score"] for item in results])
        print("\n")
        print("="*100)
        print(f"预测问题数：{len(results)}, 预测最终得分：{final_score:.4f}")
        print("="*100)

        # 保存详细评估结果
        metric_path = "./data/metrics.json" 
        results_info = json.dumps(results, ensure_ascii=False, indent=2)
        with open(metric_path, "w", encoding='utf-8') as fd:
            fd.write(results_info)
        print(f"\n结果文件保存至{metric_path}")
    else:
        print("评估失败，请检查输入文件格式是否正确")

