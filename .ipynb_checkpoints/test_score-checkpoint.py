# coding=utf-8
import json
import sys
import re
import numpy as np
from text2vec import SentenceModel, semantic_search, Similarity


simModel_path = './pre_train_model/text2vec-base-chinese'  # 相似度模型路径
simModel = SentenceModel(model_name_or_path=simModel_path, device='cuda:0')  # 使用第一张 GPU

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

def report_score(gold_path, predict_path):
    """计算预测结果的得分
    
    Args:
        gold_path: 标准答案文件路径
        predict_path: 预测结果文件路径
        
    Returns:
        list: 包含每个问题得分的结果列表
    """
    try:
        # 加载标准答案和预测结果
        gold_info = json.load(open(gold_path))
        pred_info = json.load(open(predict_path))

        idx = 0
        for gold, pred in zip(gold_info, pred_info):
            # 获取问题、关键词和答案
            question = gold["question"]
            keywords = gold["keywords"] 
            gold_answer = gold["answer"].strip()
            
            # 获取两个预测答案
            pred1 = pred.get("answer_1", "").strip()
            pred2 = pred.get("answer_2", "").strip()
            
            # 分别计算两个答案的得分
            score1 = calc_single_score(gold_answer, pred1, keywords)
            score2 = calc_single_score(gold_answer, pred2, keywords)
            
            # 取最高分作为最终得分
            final_score = max(score1, score2)
            
            # 记录得分和预测结果
            gold_info[idx]["score"] = final_score
            gold_info[idx]["predict_1"] = pred1
            gold_info[idx]["predict_2"] = pred2
            gold_info[idx]["score_1"] = score1
            gold_info[idx]["score_2"] = score2
            
            idx += 1
            print(f"问题: {question}")
            print(f"标准答案: {gold_answer}")
            print(f"预测答案1: {pred1}, 得分: {score1:.4f}")
            print(f"预测答案2: {pred2}, 得分: {score2:.4f}")
            print(f"最终得分: {final_score:.4f}\n")

        return gold_info
        
    except Exception as e:
        print(f"Error in report_score: {str(e)}")
        return []


if __name__ == "__main__":
    '''
    评估预测结果
    '''
    # 标准答案路径
    gold_path = "./data/gold.json" 
    print("Read gold from %s" % gold_path)

    # 预测文件路径
    predict_path = "./data/result.json" 
    print("Read predict file from %s" % predict_path)

    # 计算得分
    results = report_score(gold_path, predict_path)
    
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

