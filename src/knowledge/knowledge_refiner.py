from typing import List, Dict, Any, Tuple, Union, Optional
import json
import logging
import re
from collections import defaultdict
import jieba
import torch
import os
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
from system_config import (
    LLM_CONFIG,
    MODEL_PATHS,
    OPTIMIZED_RETRIEVER_WEIGHTS,
    OPTIMIZED_RERANKER_WEIGHTS,
    GPU_CONFIG
)
from src.data_process import DataProcess
from src.retriever.multi_retriever import MultiRetriever
from src.retriever.reranker import BGEReranker, BCEReranker, MultiReranker
from .knowledge_base import KnowledgeBase
import time
import torch.nn.parallel
import torch.distributed
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

stopwords = {'的', '了', '和', '是', '就', '都', '而', '及', '与'}

def evaluate_answer_quality(question: str, answer: str, relevant_docs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    评估答案质量的独立函数
    
    Args:
        question: 原始问题
        answer: 生成的答案
        relevant_docs: 相关文档列表
        
    Returns:
        Dict[str, float]: 包含各项评分的字典
    """
    if not answer:
        return {
            "completeness": 0.0,
            "relevance": 0.0,
            "clarity": 0.0,
            "factual_consistency": 0.0,
            "overall_score": 0.0
        }
    
    # 1. 完整性评分 - 根据问题类型判断理想长度
    question_type_lengths = {
        "是什么": (50, 150),  # (最小长度, 理想长度)
        "作用": (80, 200),
        "如何": (100, 250),
        "步骤": (120, 300),
        "区别": (80, 200)
    }
    
    # 获取问题类型
    question_type = None
    for qtype in question_type_lengths:
        if qtype in question:
            question_type = qtype
            break
    
    min_len, ideal_len = question_type_lengths.get(question_type, (50, 150))
    answer_len = len(answer)
    
    if answer_len < min_len:
        completeness = answer_len / min_len
    else:
        completeness = min(1.0, 2.0 - (answer_len / ideal_len)) if answer_len > ideal_len else 1.0
    
    # 2. 相关性评分 - 考虑语义相关性
    question_words = set(jieba.cut(question)) - stopwords
    answer_words = set(jieba.cut(answer)) - stopwords
    
    # 计算关键信息覆盖度
    key_info_patterns = {
        "作用": ["作用", "功能", "用途", "目的", "效果"],
        "步骤": ["首先", "然后", "接着", "最后", "步骤"],
        "原因": ["原因", "因为", "所以", "导致", "造成"],
        "时机": ["当", "时候", "情况下", "条件", "场景"]
    }
    
    pattern_matches = 0
    total_patterns = 0
    for qtype, patterns in key_info_patterns.items():
        if qtype in question:
            total_patterns = len(patterns)
            pattern_matches = sum(1 for p in patterns if any(p in w for w in answer_words))
    
    relevance = (
        (0.6 * len(question_words & answer_words) / len(question_words)) +
        (0.4 * (pattern_matches / total_patterns if total_patterns else 1.0))
    )
    
    # 3. 清晰度评分 - 考虑句子结构和连贯性
    sentences = re.split(r'[。！？]', answer)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]
    
    # 检查句子结构完整性
    structure_score = len(valid_sentences) / max(len(sentences), 1)
    
    # 检查关键连接词的使用
    connectives = ["因此", "所以", "但是", "而且", "如果", "当"]
    has_connectives = any(c in answer for c in connectives)
    
    clarity = (0.7 * structure_score + 0.3 * (1.0 if has_connectives else 0.5))
    
    # 4. 事实一致性评分 - 考虑文档内容匹配度
    factual_consistency = 0.0
    if relevant_docs:
        # 提取文档内容
        doc_text = ""
        for doc in relevant_docs:
            if isinstance(doc, str):
                doc_text += doc
            elif isinstance(doc, dict):
                doc_text += doc.get("content", "")
        
        # 计算关键信息匹配度
        doc_words = set(jieba.cut(doc_text)) - stopwords
        key_info_matches = len(answer_words & doc_words)
        
        # 计算句子级别的匹配度
        answer_sentences = [s.strip() for s in re.split(r'[。！？]', answer) if s.strip()]
        doc_sentences = [s.strip() for s in re.split(r'[。！？]', doc_text) if s.strip()]
        
        sentence_matches = 0
        for ans_sent in answer_sentences:
            ans_words = set(jieba.cut(ans_sent)) - stopwords
            for doc_sent in doc_sentences:
                doc_sent_words = set(jieba.cut(doc_sent)) - stopwords
                if len(ans_words & doc_sent_words) / len(ans_words) > 0.5:
                    sentence_matches += 1
                    break
        
        sentence_consistency = sentence_matches / len(answer_sentences) if answer_sentences else 0.0
        factual_consistency = (0.6 * sentence_consistency + 0.4 * (key_info_matches / len(answer_words)))
    
    # 5. 综合评分 - 动态权重
    weights = {
        "completeness": 0.25,
        "relevance": 0.3,
        "clarity": 0.15,
        "factual_consistency": 0.3
    }
    
    # 根据问题类型调整权重
    if "作用" in question or "是什么" in question:
        weights["factual_consistency"] = 0.35
        weights["relevance"] = 0.25
    elif "如何" in question or "步骤" in question:
        weights["clarity"] = 0.2
        weights["completeness"] = 0.3
    
    overall_score = sum(score * weights[metric] for metric, score in {
        "completeness": completeness,
        "relevance": relevance,
        "clarity": clarity,
        "factual_consistency": factual_consistency
    }.items())
    
    return {
        "completeness": round(completeness, 3),
        "relevance": round(relevance, 3),
        "clarity": round(clarity, 3),
        "factual_consistency": round(factual_consistency, 3),
        "overall_score": round(overall_score, 3)
    }

class KnowledgeRefiner(KnowledgeBase):
    """
    精简+改进版知识优化器，用于生成和优化答案。
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化知识精炼器
        Args:
            config: 配置字典，包含模型路径等信息
        """
        if config is None:
            config = {}
        
        # 确保config中包含必要的配置
        if "model_path" not in config:
            config["model_path"] = MODEL_PATHS.get("chatglm3")
            
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # 首先设置设备
        # 检查可用的GPU数量
        num_gpus = torch.cuda.device_count()
        self.logger.info(f"检测到 {num_gpus} 个可用GPU")
        
        # 使用单GPU模式
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"使用设备: {self.device}")
        
        # 保存文档
        self.documents = config.get("documents", [])
        
        # 初始化模型相关属性
        self.model_path = config["model_path"]
        self.tokenizer = None
        self.model = None
        
        # 初始化检索器
        self.retriever = MultiRetriever({"documents": self.documents})
        
        # 初始化重排序器，传入正确的配置字典
        self.reranker = {
            'bge': BGEReranker({
                "model_path": MODEL_PATHS["bge_reranker"],
                "device": self.device,
                "batch_size": 32
            }),
            'bce': BCEReranker({
                "model_path": MODEL_PATHS["bce_reranker"],
                "device": self.device,
                "batch_size": 32
            })
        }
        
        # 初始化模型
        try:
            self.logger.warning("模型尚未加载，开始加载模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # 使用半精度
            ).to(self.device)
            self.model.eval()
            self.logger.info(f"成功加载模型: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}", exc_info=True)
            raise
            
        self._query_encoding_cache = {}
        self._doc_encoding_cache = {}
        
        # 配置日志
        self._configure_logging()
        
    def _configure_logging(self):
        """配置日志输出"""
        try:
            # 设置日志格式
            log_format = '%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
            logging.basicConfig(
                format=log_format,
                level=logging.INFO
            )
            
            # 调整日志级别
            self.logger.setLevel(logging.INFO)  # 主要流程用 INFO
            logging.getLogger('transformers').setLevel(logging.WARNING)
            logging.getLogger('torch').setLevel(logging.WARNING)
            logging.getLogger('faiss').setLevel(logging.WARNING)
            
            # 详细调试信息使用 DEBUG
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(logging.DEBUG)
            
        except Exception as e:
            print(f"配置日志失败: {str(e)}")

    def rewrite_question(self, question: str) -> Tuple[str, List[str]]:
        """重写问题并提取关键词
        Args:
            question: 原始问题
        Returns:
            重写后的问题和关键词列表
        """
        try:
            # 1. 扩充原始问题
            expand_prompt = f"""请用一句话扩充这个问题，添加相关的概念和表述方式：
问题：{question}

要求：
1. 必须以"关于"开头
2. 添加相关概念
3. 保持问题的核心含义不变

请直接给出扩充后的问题："""
            
            expanded = self._generate_response(expand_prompt)
            # 确保以"关于"开头
            if not expanded.startswith("关于"):
                expanded = "关于" + expanded
            expanded = self._shorten_text(expanded, 50)  # 限制长度
            self.logger.debug(f"[问题扩充]: {expanded}")
            
            # 2. 提取关键词
            combined_text = f"{question}\n{expanded}"
            keywords = self._extract_keywords(combined_text, top_k=5)
            self.logger.info(f"提取的关键词: {keywords}")
            
            # 3. 重写问题
            rewrite_prompt = f"""请根据以下信息重写问题：
原始问题：{question}
扩充内容：{expanded}
关键词：{', '.join(keywords)}

要求：
1. 必须以"请问"开头
2. 保持问题的核心含义
3. 使用更准确的表述
4. 确保问题简洁明了，不超过30个字
5. 必须是一个完整的问句

重写后的问题："""
            
            rewritten = self._generate_response(rewrite_prompt)
            # 确保以"请问"开头
            if not rewritten.startswith("请问"):
                rewritten = "请问" + rewritten
            rewritten = self._shorten_text(rewritten, 30)  # 限制长度
            
            return rewritten, keywords
            
        except Exception as e:
            self.logger.error(f"重写问题失败: {str(e)}")
            return question, []


    def _clean_text(self, text: str) -> str:
        """清理生成的文本，移除提示词和无关内容"""
        if not text:
            return ""
            
        # 1. 移除[gMASK]和sop等特殊token
        text = re.sub(r'\[gMASK\].*?sop\s*', '', text)
        text = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]|\[MASK\]', '', text)
        
        # 2. 移除{content:}格式内容（非贪婪匹配）
        text = re.sub(r'\{[\s\S]*?content:.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'说明[：:]\s*', '', text)
        text = re.sub(r'□\s*', '', text)
            
        # 3. 移除常见的提示词模式
        patterns = [
            r"请.*?回答[：:](.*)",
            r"答案[：:](.*)",
            r"最终答案[：:](.*)",
            r"简要回答[：:](.*)",
            r"回答[：:](.*)",
            r"总结[：:](.*)",
            r"综上所述[，,](.*)",
            r"参考[：:](.*)",
            r"原文档[：:](.*)",
            r"参考文档[：:](.*)",
            r"\{content:.*?\}",
            r"content[：:](.*)",
        ]
        
        cleaned = text
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.DOTALL)
            if match and match.groups():  # 确保有捕获组
                try:
                    group_text = match.group(1)
                    if group_text:  # 确保捕获组内容不为空
                        cleaned = group_text.strip()
                except IndexError:
                    continue  # 如果无法访问组，跳过当前模式
                
        # 4. 移除重复的问题陈述
        question_patterns = [
            r"问题[：:](.*?)[。\n]",
            r"如何.*?[。\n]",
            r".*?的判断方法.*?[。\n]",
            r"^\d+[\.、\s]+",  # 移除数字开头
        ]
        for pattern in question_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)
            
        # 5. 移除多余的空白字符和标点
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[。；，：]$', '。', cleaned)
        cleaned = re.sub(r'。+', '。', cleaned)
        
        # 6. 移除数字开头的句子中的数字
        sentences = []
        for sent in re.split(r'[。！？]', cleaned):
            sent = sent.strip()
            if sent:
                # 移除数字开头
                sent = re.sub(r'^\d+[\.、\s]*', '', sent)
                if sent.strip():  # 确保去除数字后还有内容
                    sentences.append(sent)
        
        cleaned = "。".join(sentences)
        if cleaned and not cleaned.endswith(("。", "！", "？")):
            cleaned += "。"
        
        # 7. 如果清理后为空，返回原文中的最后一个完整句子
        if not cleaned.strip():
            sentences = re.split(r'[。！？]', text)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) >= 20 and not any(token in s for token in ['[gMASK]', 'sop', '[CLS]', '[SEP]'])]
            if valid_sentences:
                return valid_sentences[-1] + "。"
            return ""
            
        return cleaned.strip()

    def _extract_final_answer(self, text: str) -> str:
        """从文本中提取最终答案，按优先级尝试不同的提取策略"""
        if not text or len(text.strip()) == 0:
            return ""
        
        # 1. 清理特殊token和JSON格式内容
        text = re.sub(r'\{[\s\S]*?content:.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\[gMASK\].*?sop\s*', '', text)
        text = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]|\[MASK\]', '', text)
        
        # 2. 移除提示性话语和代码片段
        text = re.sub(r'//.*', '', text)
        text = re.sub(r'python .*', '', text)
        
        # 3. 如果清理后的文本仍然有效，直接返回
        cleaned_text = text.strip()
        if len(cleaned_text) >= 10 and not re.search(r'\{content:', cleaned_text):
            return cleaned_text
        
        # 4. 尝试提取标记后的内容（如果需要）
        markers = [
            r"最终答案[：:](.*?)(?=\n|$)",
            r"答案[：:](.*?)(?=\n|$)",
            r"总结[：:](.*?)(?=\n|$)",
            r"综上所述[,，:：](.*?)(?=\n|$)",
            r"因此[,，:：](.*?)(?=\n|$)"
        ]
        
        for pattern in markers:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # 移除可能的JSON或重复内容
                answer = re.sub(r'\{[\s\S]*?content:.*?\}', '', answer, flags=re.DOTALL)
                answer = re.sub(r'\s+', ' ', answer)
                # 移除提示性话语
                answer = re.sub(r'//.*', '', answer)
                if len(answer) >= 10 and len(answer) <= 200:
                    return answer
        
        # 5. 提取完整的句子
        sentences = re.split(r'[。！？]', text)
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            if len(s) >= 10 and len(s) <= 200 and not any(token in s for token in ['[gMASK]', 'sop', '[CLS]', '[SEP]']):
                # 确保没有特殊格式
                if not re.search(r'\{content:', s):
                    valid_sentences.append(s)
        
        if valid_sentences:
            # 优先选择包含判断相关词语的句子
            judgment_keywords = ["通过", "可以", "当", "表示", "说明", "显示"]
            for sentence in reversed(valid_sentences):
                if any(kw in sentence for kw in judgment_keywords):
                    # 移除可能的重复内容
                    sentence = re.sub(r'\{[\s\S]*?content:.*?\}', '', sentence, flags=re.DOTALL)
                    sentence = re.sub(r'\s+', ' ', sentence)
                    return sentence + "。"
            return valid_sentences[-1] + "。"
        
        # 6. 如果上述方法都失败，清理并返回原文
        cleaned = self._clean_text(text)
        if cleaned and len(cleaned) <= 200:
            return cleaned
        elif cleaned:
            return self._shorten_text(cleaned, 200)
        
        return "抱歉，无法生成有效答案。"

    def _verify_answer_quality(self, answer: str, query: str, context: str) -> bool:
        """验证生成答案的质量
        
        Args:
            answer: 生成的答案
            query: 原始问题
            context: 相关上下文
        
        Returns:
            bool: 答案是否满足质量要求
        """
        try:
            # 1. 检查关键词匹配
            query_keywords = set(self._extract_keywords(query))
            answer_keywords = set(self._extract_keywords(answer))
            keyword_overlap = len(query_keywords & answer_keywords)
            keyword_score = keyword_overlap / len(query_keywords) if query_keywords else 0.0

            # 2. 计算相似度得分
            similarity_score = self._calculate_similarity(answer, context, method="hybrid")
            
            # 3. 评估答案完整性
            completeness_score = min(1.0, len(answer) / 100)  # 基于理想长度100字
            
            # 4. 根据问题类型动态调整阈值
            question_type = self._get_question_type(query)
            thresholds = {
                "definition": {"keyword": 0.3, "similarity": 0.4, "completeness": 0.3},
                "how": {"keyword": 0.4, "similarity": 0.3, "completeness": 0.4},
                "what": {"keyword": 0.3, "similarity": 0.4, "completeness": 0.3}
            }
            
            # 使用默认阈值
            default_thresholds = {"keyword": 0.3, "similarity": 0.3, "completeness": 0.3}
            current_thresholds = thresholds.get(question_type, default_thresholds)
            
            # 5. 综合评分
            scores = {
                "keyword": keyword_score >= current_thresholds["keyword"],
                "similarity": similarity_score >= current_thresholds["similarity"],
                "completeness": completeness_score >= current_thresholds["completeness"]
            }
            
            # 记录详细评分
            self.logger.debug(f"答案质量评分:\n"
                             f"- 关键词匹配: {keyword_score:.2f} (阈值: {current_thresholds['keyword']})\n"
                             f"- 相似度: {similarity_score:.2f} (阈值: {current_thresholds['similarity']})\n"
                             f"- 完整性: {completeness_score:.2f} (阈值: {current_thresholds['completeness']})")
            
            # 6. 判断标准：至少满足两个条件，且总分达到阈值
            passing_criteria = sum(scores.values()) >= 2
            total_score = (keyword_score + similarity_score + completeness_score) / 3
            
            return passing_criteria and total_score >= 0.35
            
        except Exception as e:
            self.logger.error(f"验证答案质量失败: {str(e)}")
            return True  # 出错时默认通过，让后续流程继续

    def _shorten_text(self, text: str, max_len: int) -> str:
        """对文本进行智能截断"""
        if len(text) <= max_len:
            return text
        
        # 按句子截断
        sentences = re.split(r'[。！？]', text)
        result = ""
        for sent in sentences:
            if len(result) + len(sent) > max_len:
                break
            result += sent + "。"
        return result.strip()

    def _extract_keywords(self, text: str, top_k: int = 3) -> List[str]:
        """
        提取文本中的关键词，专注于动作、状态和对象
        """
        import jieba.analyse
        
        # 设置停用词
        stop_words = [
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '你', '我', '他',
            '也', '在', '已', '了', '于', '时', '中', '或', '由', '从', '到', '对', '能', '会',
            '可以', '需要', '请', '问题', '如何', '怎么', '什么', '为什么', '哪些', '这个', '那个',
            '进行', '使用', '可能', '一些', '这些', '那些', '没有', '这样', '那样', '知道', '告诉',
            '不会', '不能', '不是', '一个', '一种', '一样', '现在', '已经', '还是', '只是', '但是',
            '因为', '所以', '如果', '虽然', '并且', '或者', '不过', '然后', '开始', '一直', '一定',
            '必须', '可能', '应该', '需要', '觉得', '认为', '希望', '想要', '打算', '发现', '出现'
        ]
        # 将停用词列表写入临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            f.write('\n'.join(stop_words))
            stop_words_file = f.name
        
        try:
            # 使用临时文件设置停用词
            jieba.analyse.set_stop_words(stop_words_file)
            
            # 设置词性权重
            allowed_pos = {
                'v': 1.0,   # 动词
                'vn': 1.0,  # 动名词
                'n': 0.8,   # 名词
                'nz': 0.8,  # 专有名词
                'a': 0.6,   # 形容词
            }
            
            # 使用TextRank算法，考虑词性权重
            keywords = jieba.analyse.textrank(
                text,
                topK=top_k * 2,  # 提取更多候选词
                allowPOS=tuple(allowed_pos.keys())
            )
            
            # 对关键词进行过滤和排序
            scored_keywords = []
            stop_words_set = set(stop_words)  # 转换为set以提高查找效率
            for kw in keywords:
                # 获取词性
                words = jieba.posseg.cut(kw)
                for word, pos in words:
                    if word in stop_words_set:  # 使用set进行查找
                        continue
                    base_score = allowed_pos.get(pos, 0.1)
                    # 提升特定类型词的权重
                    if any(term in word for term in ["如何", "怎么", "是否", "能否"]):
                        base_score *= 0.5  # 降低疑问词权重
                    elif "正在" in word or "现在" in word:
                        base_score *= 1.2  # 提升状态词权重
                    scored_keywords.append((word, base_score))
            
            # 按分数排序并取top_k个
            scored_keywords.sort(key=lambda x: x[1], reverse=True)
            return [kw for kw, _ in scored_keywords[:top_k]]
            
        finally:
            # 清理临时文件
            import os
            try:
                os.unlink(stop_words_file)
            except:
                pass

    def _rewrite_query(self, query: str) -> str:
        """改写查询，提取关键信息，不包含提示词
        
        Args:
            query: 原始查询
            
        Returns:
            str: 改写后的查询
        """
        try:
            # 移除可能的提示词
            prompt_patterns = [
                r"回答请以.*?开头[。\.]?",
                r"请详细解释[：:]?",
                r"请简要回答.*?以下问题.*?[：:]?",
            ]
            clean_query = query
            for pattern in prompt_patterns:
                clean_query = re.sub(pattern, "", clean_query).strip()
            
            # 提取关键词
            keywords = list(jieba.cut(clean_query))
            keywords = [kw for kw in keywords if kw not in stopwords and len(kw) > 1]
            
            # 保持问题的完整性
            if "如何" in query or "怎么" in query:
                rewritten_query = clean_query
            else:
                rewritten_query = " ".join(keywords)
            
            return rewritten_query
            
        except Exception as e:
            self.logger.error(f"问题改写失败: {str(e)}")
            return query
            
    def retrieve_documents(self, original_query: str, rewritten_query: str, 
                         keywords: List[str], llm_answer: str) -> List[Dict]:
        """检索相关文档
        
        Args:
            original_query: 原始查询
            rewritten_query: 改写后的查询
            keywords: 关键词列表
            llm_answer: LLM生成的初步答案
            
        Returns:
            List[Dict]: 检索到的文档列表
        """
        try:
            # 1. 使用原始查询检索
            docs_from_original = self.retriever.get_relevant_documents(original_query)
            
            # 2. 使用改写后的查询检索
            if rewritten_query != original_query:
                docs_from_rewritten = self.retriever.get_relevant_documents(rewritten_query)
            else:
                docs_from_rewritten = []
            
            
            # 合并结果并去重
            all_docs = []
            seen_contents = set()
            
            for doc_list in [docs_from_original, docs_from_rewritten]:
                for doc in doc_list:
                    if isinstance(doc, dict):
                        content = doc.get("content", "")
                    else:
                        content = str(doc)
                    
                    if content and content not in seen_contents:
                        seen_contents.add(content)
                        all_docs.append(doc)
            
            logger.info(f"检索到 {len(all_docs)} 个相关文档")
            return all_docs[:10]  # 返回前10个最相关的文档
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            return []

    def _generate_response(self, prompt: str, max_prompt_length: int = 1024, max_response_length: int = 512) -> str:
        """生成回复，并确保输出格式规范"""
        try:
            # 修改系统提示
            system_prompt = """你是一个专业的汽车知识助手，请遵循以下规则：
1. 只回答与问题直接相关的车辆功能或信息
2. 不要添加任何与当前问题无关的安全提示、注意事项或背景信息
3. 使用简洁的陈述句直接回答问题
4. 严格基于参考文档中的信息回答
5. 如果问题是关于具体功能,只描述该功能的直接相关内容
6. 不要主动扩展话题或添加额外建议
7. 确保每句话都与问题核心高度相关"""
    
            formatted_prompt = f"""<system>{system_prompt}</system>
<human>{prompt}</human>
<assistant>"""
            
            # 截断prompt，只截断用户查询部分
            if len(formatted_prompt) > max_prompt_length:
                content_max_len = max_prompt_length - len(system_prompt) - len("<system></system><human></human><assistant>")
                prompt_content = self._shorten_text(prompt, content_max_len)
                formatted_prompt = f"""<system>{system_prompt}</system>
<human>{prompt_content}</human>
<assistant>"""
            
            # 使用模型生成回复
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # 准备输入数据
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True)
                    
                    # 将输入移动到正确的设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # 使用模型生成回复
                    outputs = self.model.generate(**inputs, 
                        max_new_tokens=max_response_length, 
                        temperature=0.3,  # 降低温度
                        top_p=0.85,      # 降低top_p
                        repetition_penalty=1.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.encode("</assistant>")[0],
                        bad_words_ids=[[self.tokenizer.encode(word)[0]] for word in ["[gMASK]", "sop", "[CLS]", "[SEP]"]]
                    )
                    
                    # 解码输出
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
            # 清理输出
            response = re.sub(r'<[^>]+>', '', response)  # 移除XML标记
            response = response.replace(system_prompt, "").strip()  # 移除系统提示
            response = response.replace(prompt, "").strip()  # 移除原始提示
        
            # 清理和截断响应
            response = self._clean_text(response)
            # 移除所有 JSON 格式内容和代码片段
            response = re.sub(r'\{.*?\}', '', response)
            response = re.sub(r'//.*', '', response)
            response = re.sub(r'python .*', '', response)
            response = re.sub(r'//题外话.*', '', response)
            # 确保只保留合理长度的文本
            if len(response) > max_response_length:
                response = self._shorten_text(response, max_response_length)
        
            return response
        
        except Exception as e:
            self.logger.error(f"生成回复失败: {str(e)}", exc_info=True)
            torch.cuda.empty_cache()  # 清理GPU缓存
            return ""

    def _is_better_answer(self, new_answer: str, old_answer: str, question: str) -> bool:
        """
        判断新答案是否比旧答案更好
        """
        if not old_answer:
            return True
        
        # 获取问题关键词
        question_keywords = set(self._extract_keywords(question))
        
        # 分别计算新旧答案的关键词覆盖率
        new_keywords = set(self._extract_keywords(new_answer))
        old_keywords = set(self._extract_keywords(old_answer))
        
        new_coverage = len(question_keywords & new_keywords) / len(question_keywords) if question_keywords else 0
        old_coverage = len(question_keywords & old_keywords) / len(question_keywords) if question_keywords else 0
        
        # 计算答案完整性分数
        new_completeness = min(1.0, len(new_answer) / 150)  # 假设理想长度为150
        old_completeness = min(1.0, len(old_answer) / 150)
        
        # 综合评分 (关键词覆盖率权重0.7，完整性权重0.3)
        new_score = new_coverage * 0.7 + new_completeness * 0.3
        old_score = old_coverage * 0.7 + old_completeness * 0.3
        
        return new_score > old_score

    def _evaluate_answer_quality(self, answer: str, question: str, relevant_docs: Union[List[Dict[str, Any]], List[str], str]) -> Dict[str, float]:
        """评估答案质量,使用基于规则和统计的方法
        
        Args:
            answer: 生成的答案
            question: 原始问题
            relevant_docs: 相关文档
            
        Returns:
            Dict[str, float]: 包含多个维度的质量评分
        """
        try:
            scores = {
                "completeness": 0.0,
                "relevance": 0.0,
                "clarity": 0.0,
                "factual_consistency": 0.0,
                "structure_quality": 0.0,
                "doc_similarity": 0.0
            }
            
            if not answer or len(answer) < 10:
                return scores
                
            # 1. 完整性评分 - 基于统计特征
            question_type = self._identify_question_type(question)
            completeness_score = self._evaluate_completeness(answer, question_type)
            scores["completeness"] = completeness_score
            
            # 2. 相关性评分 - 使用TF-IDF和关键词匹配
            relevance_score = self._calculate_relevance_score(answer, question, relevant_docs)
            scores["relevance"] = relevance_score
            
            # 3. 清晰度评分 - 基于语言规则
            clarity_score = self._evaluate_clarity(answer)
            scores["clarity"] = clarity_score
            
            # 4. 事实一致性 - 基于文档匹配
            factual_score = self._evaluate_factual_consistency(answer, relevant_docs)
            scores["factual_consistency"] = factual_score
            
            # 5. 结构质量 - 基于问题类型的期望结构
            structure_score = self._evaluate_structure(answer, question_type)
            scores["structure_quality"] = structure_score
            
            # 6. 文档相似度 - 使用统计方法
            doc_similarity = self._calculate_doc_similarity(answer, relevant_docs)
            scores["doc_similarity"] = doc_similarity
            
            # 根据问题类型动态调整权重
            weights = self._get_dynamic_weights(question_type)
            scores["overall_score"] = sum(score * weights[metric] for metric, score in scores.items() if metric in weights)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"评估答案质量失败: {str(e)}")
            return {k: 0.0 for k in scores.keys()}

    def _identify_question_type(self, question: str) -> str:
        """识别问题类型
        
        Args:
            question: 问题文本
            
        Returns:
            str: 问题类型
        """
        type_patterns = {
            "how": r"如何|怎么|怎样|步骤|方法",
            "what": r"是什么|什么是|定义|概念",
            "why": r"为什么|原因|为啥",
            "compare": r"区别|差异|不同|对比",
            "process": r"流程|过程|步骤",
        }
        
        for qtype, pattern in type_patterns.items():
            if re.search(pattern, question):
                return qtype
        return "other"

    def _evaluate_completeness(self, answer: str, question_type: str) -> float:
        """评估答案完整性
        
        Args:
            answer: 答案文本
            question_type: 问题类型
            
        Returns:
            float: 完整性得分
        """
        # 基于问题类型的理想长度
        ideal_lengths = {
            "how": 200,
            "what": 100,
            "why": 150,
            "compare": 180,
            "process": 250,
            "other": 150
        }
        
        target_len = ideal_lengths.get(question_type, 150)
        len_ratio = len(answer) / target_len
        
        # 检查结构完整性
        structure_patterns = {
            "how": r"^\d+[\.\、]|首先|然后|接着|最后",
            "process": r"^\d+[\.\、]|第[一二三四五六七八九十]步|首先|其次",
            "compare": r"一方面|另一方面|相比之下|不同点",
            "why": r"原因|因为|由于"
        }
        
        pattern = structure_patterns.get(question_type, "")
        has_structure = bool(pattern and re.search(pattern, answer))
        
        # 计算最终得分
        base_score = min(1.0, len_ratio if 0.5 <= len_ratio <= 1.5 else 0.0)
        return base_score * 1.2 if has_structure else base_score

    def _calculate_relevance_score(self, answer: str, question: str, docs: Union[List[Dict[str, Any]], List[str], str]) -> float:
        """计算相关性得分,使用TF-IDF和关键词匹配
        
        Args:
            answer: 答案文本
            question: 问题文本
            docs: 相关文档
            
        Returns:
            float: 相关性得分
        """
        try:
            # 1. 提取关键词
            question_keywords = set(self._extract_keywords(question))
            answer_keywords = set(self._extract_keywords(answer))
            
            # 2. 计算关键词匹配度
            if not question_keywords:
                return 0.0
                
            keyword_overlap = len(question_keywords & answer_keywords)
            keyword_score = keyword_overlap / len(question_keywords)
            
            # 3. 计算TF-IDF相似度
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # 准备文本
            texts = [question, answer]
            if isinstance(docs, str):
                texts.append(docs)
            elif isinstance(docs, list):
                texts.extend([d.get("content", "") if isinstance(d, dict) else d for d in docs])
            
            # 计算TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000)
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                similarity = 0.0
            
            # 4. 合并得分
            final_score = 0.7 * keyword_score + 0.3 * similarity
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"计算相关性得分失败: {str(e)}")
            return 0.0

    def _evaluate_clarity(self, text: str) -> float:
        """评估文本清晰度
        
        Args:
            text: 待评估文本
            
        Returns:
            float: 清晰度得分
        """
        try:
            # 1. 句子完整性检查
            sentences = [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
            if not sentences:
                return 0.0
                
            complete_sentences = len([s for s in sentences if len(s) >= 10 and not self._is_incomplete_sentence(s)])
            sentence_score = complete_sentences / len(sentences)
            
            # 2. 标点符号使用 - 降低权重
            punctuation_score = len(re.findall(r'[。！？，、：；]', text)) / len(text)
            punctuation_score = min(1.0, punctuation_score * 3)  # 降低系数从5到3
            
            # 3. 段落结构 - 增加结构评分的权重
            structure_patterns = {
                "连接词": ["因此", "所以", "但是", "而且", "如果", "当"],
                "序列词": ["首先", "其次", "然后", "最后"],
                "转折词": ["然而", "不过", "相反", "另外"],
                "总结词": ["总之", "综上", "总的来说"]
            }
            
            structure_score = 0.0
            for pattern_type, patterns in structure_patterns.items():
                if any(p in text for p in patterns):
                    structure_score += 0.25  # 每种类型加0.25分
            
            # 4. 计算最终得分 - 调整权重
            weights = {
                'sentence': 0.5,    # 提高句子完整性权重
                'punctuation': 0.2, # 降低标点符号权重
                'structure': 0.3    # 提高结构权重
            }
            
            final_score = (
                weights['sentence'] * sentence_score +
                weights['punctuation'] * punctuation_score +
                weights['structure'] * min(1.0, structure_score)
            )
            
            # 记录评分详情
            self.logger.debug(f"清晰度评分详情:\n"
                             f"- 句子完整性: {sentence_score:.2f}\n"
                             f"- 标点使用: {punctuation_score:.2f}\n"
                             f"- 结构评分: {structure_score:.2f}\n"
                             f"- 最终得分: {final_score:.2f}")
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"评估清晰度失败: {str(e)}")
            return 0.0

    def _get_dynamic_weights(self, question_type: str) -> Dict[str, float]:
        """根据问题类型获取动态权重
        
        Args:
            question_type: 问题类型
            
        Returns:
            Dict[str, float]: 权重配置
        """
        # 默认权重
        default_weights = {
            "completeness": 0.2,
            "relevance": 0.2,
            "clarity": 0.15,
            "factual_consistency": 0.2,
            "structure_quality": 0.15,
            "doc_similarity": 0.1
        }
        
        # 特定问题类型的权重调整
        type_weights = {
            "how": {
                "completeness": 0.25,
                "structure_quality": 0.25,
                "clarity": 0.2,
                "factual_consistency": 0.15,
                "relevance": 0.1,
                "doc_similarity": 0.05
            },
            "what": {
                "factual_consistency": 0.3,
                "doc_similarity": 0.2,
                "relevance": 0.2,
                "clarity": 0.15,
                "completeness": 0.1,
                "structure_quality": 0.05
            },
            "compare": {
                "structure_quality": 0.25,
                "completeness": 0.2,
                "factual_consistency": 0.2,
                "clarity": 0.15,
                "relevance": 0.1,
                "doc_similarity": 0.1
            }
        }
        
        return type_weights.get(question_type, default_weights)

    def _generate_initial_answer(self, query: str) -> str:
        """生成初步答案"""
        try:
            # 构建提示词
            prompt = f"请简要回答以下问题（100字以内）：{query}"
            
            # 生成答案
            response = self.model.chat(
                tokenizer=self.tokenizer,
                query=prompt,
                history=[],
                max_length=150,
                temperature=0.7
            )
            
            # 如果返回的是元组，只取第一个元素（答案内容）
            if isinstance(response, tuple):
                response = response[0]
            
            # 清理答案
            return self._clean_response(response)
            
        except Exception as e:
            self.logger.error(f"生成初步答案失败: {str(e)}")
            return ""

    def _optimize_with_doc(self, current_answer: str, doc_content: str, question: str) -> str:
        """使用文档内容优化当前答案
        
        Args:
            current_answer: 当前答案
            doc_content: 文档内容
            question: 用户问题
            
        Returns:
            str: 优化后的答案
        """
        try:
            # 1. 检查文档相关性
            if not self._check_semantic_relevance(doc_content, question):
                return current_answer
            
            # 2. 检查当前答案质量
            current_quality = self._evaluate_answer_quality(current_answer, question, [doc_content])
            if current_quality.get("overall_score", 0) >= 0.8:
                return current_answer
            
            # 3. 提取文档中的相关段落
            relevant_content = self._extract_relevant_content(doc_content, question)
            if not relevant_content:
                return current_answer
            
            # 4. 构建优化提示
            optimize_prompt = f"""请基于以下信息优化答案，严格遵循以下要求：

问题：{question}
当前答案：{current_answer}
相关文档内容：{relevant_content}

优化要求：
1. 必须严格基于文档内容
2. 保持与原文档的语义一致性
3. 不要添加文档未提及的信息
4. 保持专业术语的准确性
5. 如果文档内容与问题无关，保持原答案不变
6. 确保答案的连贯性和完整性

请给出优化后的答案："""
            
            # 5. 生成优化答案
            optimized_answer = self._generate_response(optimize_prompt)
            optimized_answer = self._clean_response(optimized_answer)
            
            # 6. 验证语义一致性
            if not self._verify_semantic_consistency(optimized_answer, doc_content):
                return current_answer
            
            # 7. 验证答案质量
            if not self._verify_answer_quality(optimized_answer, question, doc_content):
                return current_answer
            
            # 8. 评估优化效果
            new_quality = self._evaluate_answer_quality(optimized_answer, question, [doc_content])
            if new_quality.get("overall_score", 0) > current_quality.get("overall_score", 0):
                return optimized_answer
            
            return current_answer
            
        except Exception as e:
            self.logger.error(f"优化答案失败: {str(e)}")
            return current_answer
            
    def _verify_semantic_consistency(self, answer: str, doc_content: str) -> bool:
        """验证答案与文档的语义一致性
        
        Args:
            answer: 答案文本
            doc_content: 文档内容
            
        Returns:
            bool: 是否语义一致
        """
        try:
            # 1. 提取关键信息
            answer_keywords = set(self._extract_keywords(answer))
            doc_keywords = set(self._extract_keywords(doc_content))
            
            # 2. 计算关键词重叠度
            if not answer_keywords or not doc_keywords:
                return False
            overlap_ratio = len(answer_keywords & doc_keywords) / len(answer_keywords)
            
            # 3. 检查是否包含文档中未提及的关键词
            extra_keywords = answer_keywords - doc_keywords
            if len(extra_keywords) > len(answer_keywords) * 0.2:  # 允许20%的新增关键词
                return False
            
            # 4. 计算语义相似度
            semantic_score = self._calculate_similarity(answer, doc_content, method="semantic")
            
            # 5. 综合判断
            return overlap_ratio >= 0.7 and semantic_score >= 0.6
            
        except Exception as e:
            self.logger.error(f"验证语义一致性失败: {str(e)}")
            return False
            
    def _extract_relevant_content(self, doc_content: str, question: str) -> str:
        """提取文档中与问题相关的段落
        
        Args:
            doc_content: 文档内容
            question: 用户问题
            
        Returns:
            str: 相关段落
        """
        try:
            # 1. 分割文档为段落
            paragraphs = [p.strip() for p in doc_content.split("\n") if p.strip()]
            if not paragraphs:
                return doc_content
            
            # 2. 提取问题关键词
            question_keywords = set(self._extract_keywords(question))
            
            # 3. 计算每个段落的相关度
            scored_paragraphs = []
            for para in paragraphs:
                # 3.1 关键词匹配度
                para_keywords = set(self._extract_keywords(para))
                keyword_score = len(question_keywords & para_keywords) / len(question_keywords) if question_keywords else 0
                
                # 3.2 语义相似度
                semantic_score = self._calculate_similarity(question, para, method="semantic")
                
                # 3.3 计算综合分数
                score = 0.6 * keyword_score + 0.4 * semantic_score
                scored_paragraphs.append((para, score))
            
            # 4. 选择最相关的段落
            scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
            relevant_paras = []
            total_length = 0
            
            for para, score in scored_paragraphs:
                if score < 0.3:  # 相关度阈值
                    break
                if total_length + len(para) > 1000:  # 长度限制
                    break
                relevant_paras.append(para)
                total_length += len(para)
            
            return "\n".join(relevant_paras)
            
        except Exception as e:
            self.logger.error(f"提取相关内容失败: {str(e)}")
            return doc_content

    def _find_exact_match(self, question: str, references: List[str]) -> str:
        """在知识库文档中查找与问题高度相关的直接答案
        
        Args:
            question: 用户问题
            references: 参考文档列表
            
        Returns:
            str: 找到的最佳匹配答案
        """
        try:
            # 提取问题关键词
            question_keywords = set(jieba.cut(question)) - stopwords
            best_score = 0.0
            best_match = ""
            
            # 根据问题类型动态调整阈值
            question_type = self._identify_question_type(question)
            threshold_map = {
                "how": 0.4,    # 操作类问题需要更高准确性
                "what": 0.3,   # 定义类问题可接受稍低相关性
                "why": 0.35    # 原因类问题需要中等相关性
            }
            relevance_threshold = threshold_map.get(question_type, 0.3)
            
            for doc in references:
                # 分句
                sentences = []
                for sent in re.split(r'[。！？]', doc):
                    sent = sent.strip()
                    if sent:
                        sentences.append(sent)
                
                for i, sentence in enumerate(sentences):
                    # 计算当前句子与问题的相关性
                    sentence_relevance = self._calculate_similarity(sentence, question, method="hybrid")
                    if sentence_relevance < relevance_threshold:
                        continue
                    
                    # 构建上下文
                    context_parts = []
                    
                    # 添加上文（前一句）
                    if i > 0 and not self._is_context_boundary(sentences[i-1]):
                        prev_sent = sentences[i-1]
                        prev_relevance = self._calculate_similarity(prev_sent, question, method="hybrid")
                        if prev_relevance >= relevance_threshold:
                            context_parts.append(prev_sent)
                    
                    # 添加当前句子
                    context_parts.append(sentence)
                    
                    # 添加下文（后一句）
                    if i < len(sentences) - 1 and not self._is_context_boundary(sentences[i+1]):
                        next_sent = sentences[i+1]
                        next_relevance = self._calculate_similarity(next_sent, question, method="hybrid")
                        if next_relevance >= relevance_threshold:
                            context_parts.append(next_sent)
                    
                    # 合并上下文
                    context = "。".join(context_parts)
                    if not context.endswith("。"):
                        context += "。"
                    
                    # 计算整体匹配分数
                    match_score = self._calculate_similarity(context, question, method="hybrid")
                    
                    # 更新最佳匹配
                    if match_score > best_score:
                        best_score = match_score
                        best_match = context
        
            return best_match if best_score >= 0.5 else ""
        
        except Exception as e:
            self.logger.error(f"精确匹配查找失败: {str(e)}")
            return ""

    def _is_context_boundary(self, sentence: str) -> bool:
        """检查句子是否处于章节边界
        
        Args:
            sentence: 待检查的句子
            
        Returns:
            bool: 是否为章节边界
        """
        try:
            boundary_patterns = [
                r'^第[一二三四五六七八九十]+章',   # 章节标题
                r'^[0-9]+\.[0-9]+',            # 数字编号
                r'^【.*?】',                    # 方括号标题
                r'^（.*?）',                    # 括号标题
                r'^\d+[\s\.\、]'               # 数字序号
            ]
            return any(re.match(p, sentence.strip()) for p in boundary_patterns)
        except Exception as e:
            self.logger.error(f"边界检查失败: {str(e)}")
            return False

    def _optimize_final_answer(self, answer: str, question: str) -> str:
        """优化最终答案，使其更加简洁清晰
        
        Args:
            answer: 原始答案
            question: 用户问题
            
        Returns:
            优化后的答案
        """
        try:
            # 1. 清除所有特殊token和格式
            answer = re.sub(r'[□■◆▲▼○●]', '', answer)
            answer = re.sub(r'\{[\s\S]*?content:.*?\}', '', answer, flags=re.DOTALL)
            answer = re.sub(r'说明[：:]\s*', '', answer)
            answer = re.sub(r'\s+', ' ', answer)
            answer = answer.replace('\n', ' ').strip()
            
            # 2. 移除指代不明的表述和文档引用
            answer = re.sub(r'(这个|该|此)(按钮|功能|操作|开关|系统)', r'\2', answer)
            answer = re.sub(r'车辆功能参考手册\d*', '', answer)
            answer = re.sub(r'根据问题.*?方法[。，]', '', answer)
            
            # 3. 提取问题关键信息
            question_type = self._get_question_type(question)
            question_keywords = self._extract_keywords(question, top_k=3)
            
            # 4. 构建更精确的prompt
            optimize_prompt = f"""请针对问题提供一个简洁、清晰、完整的回答，严格遵循以下要求：

1. 问题类型：{question_type}
2. 核心关键词：{', '.join(question_keywords)}

回答要求：
- 字数限制：50-80字
- 句子结构：2-3句完整句子
- 直接回答问题，不要任何铺垫
- 完全去除与问题无关的内容

{question_type}类问题的回答结构：
- 如何类：位置说明 + 操作步骤
- 是什么类：定义说明 + 主要功能
- 在哪里类：具体位置 + 查找方法
- 什么时候类：适用场景 + 触发条件

需要删除的内容：
- 文档引用说明
- 重复内容
- 无关的系统说明
- 无关的注意事项
- 与问题无关的补充说明

问题：{question}
原始答案：{answer}

请直接给出优化后的答案："""
            
            # 5. 生成优化后的答案
            optimized = self._generate_response(optimize_prompt)
            
            # 6. 清理和格式化
            optimized = self._clean_text(optimized)
            if "答案：" in optimized:
                optimized = optimized.split("答案：")[-1]
            
            # 7. 再次清理特殊格式和重复内容
            optimized = re.sub(r'^\d+\.\s*', '', optimized)  # 移除序号
            optimized = re.sub(r'。\s*。', '。', optimized)  # 移除重复的句号
            optimized = re.sub(r'[（(].+?[)）]', '', optimized)  # 移除括号内容
            optimized = re.sub(r'车辆功能参考手册\d*', '', optimized)  # 移除文档引用
            optimized = re.sub(r'根据.*?[，。]', '', optimized)  # 移除过渡句
            optimized = re.sub(r'\{[\s\S]*?content:.*?\}', '', optimized, flags=re.DOTALL)  # 移除content标记
            optimized = re.sub(r'说明[：:]\s*', '', optimized)  # 移除"说明："
            
            # 8. 分句处理，确保每句话都与问题相关且符合逻辑顺序
            sentences = []
            seen_content = set()  # 用于去重
            
            for sent in re.split(r'[。！？]', optimized):
                sent = sent.strip()
                if not sent:
                    continue
                    
                # 规范化句子用于去重判断
                normalized = re.sub(r'\s+', '', sent)
                normalized = re.sub(r'[，,、.：:；;]', '', normalized)
                
                # 计算与问题关键词的相关性得分
                relevance_score = sum(1 for kw in question_keywords if kw in sent)
                
                # 确保句子相关且不重复
                if (normalized not in seen_content and 
                    relevance_score > 0 and  # 至少包含一个关键词
                    len(sent) >= 10 and  # 句子长度合适
                    len(sent) <= 50):  # 单句不要太长
                    sentences.append((sent, relevance_score))
                    seen_content.add(normalized)
            
            # 9. 按相关性得分排序并限制句子数量
            sentences.sort(key=lambda x: x[1], reverse=True)  # 按相关性得分排序
            selected_sentences = [sent[0] for sent in sentences[:3]]  # 最多保留3句
            
            # 10. 根据问题类型调整句子顺序
            if question_type == "how" and len(selected_sentences) >= 2:
                # 确保操作步骤在前面
                selected_sentences.sort(key=lambda x: 0 if any(w in x for w in ["点击", "按下", "选择", "操作"]) else 1)
            elif question_type == "what" and len(selected_sentences) >= 2:
                # 确保定义在前面
                selected_sentences.sort(key=lambda x: 0 if "是" in x or "指" in x or "表示" in x else 1)
            
            # 11. 合并句子并控制总长度
            optimized = "。".join(selected_sentences)
            if not optimized.endswith("。"):
                optimized += "。"
            
            # 12. 确保答案的完整性
            if len(optimized) < 10:
                return answer
                
            return optimized.strip()
            
        except Exception as e:
            self.logger.error(f"优化最终答案失败: {str(e)}")
            return answer

    def generate_answer(self, question: str, reference_docs: str, max_prompt_length: int = 1024, max_response_length: int = 512) -> str:
        """生成答案的公共方法
        
        Args:
            question: 问题
            reference_docs: 参考文档
            max_prompt_length: prompt最大长度
            max_response_length: 响应最大长度
            
        Returns:
            生成的答案
        """
        try:
            # 1. 标准化输入文档格式
            if isinstance(reference_docs, str):
                docs_list = [reference_docs]
            else:
                docs_list = reference_docs
            
            answer_prompt = f"""基于参考信息回答问题，直接给出答案，不要解释过程。请记住：只回答与问题直接相关的内容，不要添加任何安全警示或无关建议。
问题：{question}
参考：{reference_docs}
答案："""
            
            raw_answer = self._generate_response(
                answer_prompt,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length
            )
            self.logger.debug(f"[初始答案]: {raw_answer}")
            
            # 4. 提取并清理答案
            if "答案是：" in raw_answer:
                raw_answer = raw_answer.split("答案是：")[-1].strip()
            answer = self._clean_text(raw_answer)
            
            # 5. 过滤无关句子
            filtered_answer = self._filter_irrelevant_sentences(answer, question)
            self.logger.debug(f"[过滤后答案]: {filtered_answer}")
            
            # 6. 确保最终答案的质量并优化
            final_answer = self._extract_final_answer(filtered_answer)
            if not final_answer:
                final_answer = filtered_answer
            
            # 7. 最终优化
            optimized_answer = self._optimize_final_answer(final_answer, question)
            return optimized_answer
            
        except Exception as e:
            self.logger.error(f"生成答案失败: {str(e)}")
            return "抱歉，生成答案时出现错误。"
    
    def optimize_answer_iteratively(self, query: str, ranked_docs: List[str]) -> Optional[Dict[str, Any]]:
        """迭代优化答案
        
        Args:
            query: 用户问题
            ranked_docs: 排序后的文档列表
            
        Returns:
            Dict: 包含最终答案和质量评分的字典
        """
        try:
            if not ranked_docs:
                return None

            # 用于保存所有有效答案
            valid_answers = []
            current_answer = None
            
            # 1. 首先尝试从文档中直接找到答案作为基础答案
            direct_answer = self._find_direct_answer(query, ranked_docs)
            if direct_answer:
                initial_quality = self._evaluate_answer_quality(direct_answer, query, ranked_docs)
                if self._verify_answer_quality(direct_answer, query, "\n".join(ranked_docs)):
                    current_answer = direct_answer
                    valid_answers.append({
                        "answer": direct_answer,
                        "quality": initial_quality,
                        "source": "direct"
                    })
            
            # 2. 如果没有找到直接答案，使用第一个相关文档生成初始答案
            if not current_answer:
                for doc in ranked_docs:
                    if not doc.strip():
                        continue
                    
                    # 计算文档相关度
                    doc_relevance = self._calculate_similarity(doc, query, method="hybrid")
                    if doc_relevance < 0.2:
                        continue
                        
                    # 使用文档生成初始答案
                    initial_prompt = f"""请仅基于以下参考文档回答问题，不要添加任何文档中未提及的内容：
                    
                    问题：{query}
                    参考文档：{doc}
                    
                    请给出答案："""
                    
                    initial_answer = self._generate_response(initial_prompt)
                    initial_answer = self._clean_response(initial_answer)
                    
                    if self._verify_answer_quality(initial_answer, query, doc):
                        current_answer = initial_answer
                        quality = self._evaluate_answer_quality(initial_answer, query, ranked_docs)
                        valid_answers.append({
                            "answer": initial_answer,
                            "quality": quality,
                            "source": "initial"
                        })
                        break
            
            # 3. 使用其他文档逐步优化答案
            if current_answer:
                for doc in ranked_docs:
                    if not doc.strip():
                        continue
                    
                    # 计算文档相关度
                    doc_relevance = self._calculate_similarity(doc, query, method="hybrid")
                    if doc_relevance < 0.2:
                        continue
                    
                    # 构建优化prompt，强调必须基于文档内容
                    optimize_prompt = f"""请仅基于参考文档对当前答案进行优化和补充。要求：
                    1. 必须严格基于参考文档的内容
                    2. 保持专业准确的表述
                    3. 确保答案完整且清晰
                    4. 不要添加文档中未提及的内容
                    5. 去除任何无关的内容
                    6. 如果文档内容与问题无关，请保持原答案不变
                    
                    问题：{query}
                    当前答案：{current_answer}
                    参考文档：{doc}
                    
                    优化后的答案："""
                    
                    # 生成优化后的答案
                    optimized = self._generate_response(optimize_prompt)
                    optimized = self._clean_response(optimized)
                    
                    # 过滤无关句子
                    filtered = self._filter_irrelevant_sentences(optimized, query, threshold=0.2)
                    
                    # 验证优化后的答案
                    if self._verify_answer_quality(filtered, query, doc):
                        # 检查是否与原文档有足够的相似度
                        doc_similarity = self._calculate_similarity(filtered, doc, method="hybrid")
                        if doc_similarity >= 0.3:  # 确保答案来源于文档
                            answer_quality = self._evaluate_answer_quality(filtered, query, ranked_docs)
                            valid_answers.append({
                                "answer": filtered,
                                "quality": answer_quality,
                                "source": "optimized"
                            })
                            # 更新当前答案用于下一轮优化
                            current_answer = filtered
            
            # 4. 从所有有效答案中选择最佳答案
            if valid_answers:
                # 按质量分数排序
                valid_answers.sort(key=lambda x: x["quality"].get("overall_score", 0), reverse=True)
                best_result = valid_answers[0]
                
                # 确保答案不为空且长度合适
                answer_text = best_result["answer"]
                if isinstance(answer_text, str):
                    answer_text = answer_text.strip()
                    # 再次过滤无关句子
                    answer_text = self._filter_irrelevant_sentences(answer_text, query, threshold=0.3)
                    if len(answer_text) >= 10:
                        self.logger.info(f"返回最佳答案 (来源: {best_result['source']})，长度：{len(answer_text)}")
                        return {
                            "answer": answer_text,
                            "quality": best_result["quality"],
                            "source": best_result["source"]
                        }

            # 如果没有找到有效答案，返回空结果
            self.logger.warning("未找到有效答案")
            return {
                "answer": "抱歉，无法找到相关答案。",
                "quality": {"completeness": 0.0, "relevance": 0.0, "clarity": 0.0, 
                           "factual_consistency": 0.0, "overall_score": 0.0}
            }
                
        except Exception as e:
            self.logger.error(f"迭代优化答案失败: {str(e)}")
            if 'valid_answers' in locals() and valid_answers:
                # 发生错误时，尝试返回最后一个有效答案
                last_valid = valid_answers[-1]
                return {
                    "answer": last_valid["answer"].strip(),
                    "quality": last_valid["quality"],
                    "source": last_valid["source"]
                }
            return {
                "answer": "抱歉，处理过程中出现错误。",
                "quality": {"completeness": 0.0, "relevance": 0.0, "clarity": 0.0, 
                           "factual_consistency": 0.0, "overall_score": 0.0}
            }

    def _init_model(self):
        """初始化模型，优化GPU使用"""
        try:
            # 检查是否已经加载模型
            if not hasattr(self, 'model'):
                logger.warning("模型尚未加载，开始加载模型...")
                # 使用默认配置加载模型
                self.model = self._load_model()
            
            # 检测可用GPU
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.device_ids = list(range(gpu_count))
                logger.info(f"检测到 {gpu_count} 个可用GPU")
                logger.info(f"将使用 {gpu_count} 个GPU进行并行处理: {self.device_ids}")
                
                # 使用DistributedDataParallel替代DataParallel
                try:
                    import torch.nn.parallel as parallel
                    import torch.distributed as dist
                    
                    # 初始化进程组
                    if not dist.is_initialized():
                        dist.init_process_group(backend='nccl')
                    
                    self.model = parallel.DistributedDataParallel(
                        self.model.cuda(),
                        device_ids=self.device_ids,
                        output_device=self.device_ids[0]
                    )
                except Exception as e:
                    logger.warning(f"分布式并行初始化失败，将使用普通CUDA模式: {str(e)}")
                    self.model = self.model.cuda()
            else:
                logger.warning("未检测到可用GPU，将使用CPU")
                
        except Exception as e:
            logger.error(f"模型并行初始化失败: {str(e)}")
            raise

    def _load_model(self):
        """加载模型的辅助方法
        
        Returns:
            加载的模型实例
        """
        try:
            # 从配置中获取模型路径
            model_path = self.config.get('model_path', MODEL_PATHS.get('chatglm', ''))
            
            # 使用信任远程代码加载模型
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info(f"成功加载模型: {model_path}")
            return model
                
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def _find_direct_answer(self, query: str, ranked_docs: List[str]) -> str:
        """从规整后的文档中直接寻找答案
        
        Args:
            query: 用户问题
            ranked_docs: 规整后的文档列表
            
        Returns:
            str: 找到的答案，如果没找到返回空字符串
        """
        try:
            # 提取问题中的关键词和问题类型
            keywords = self._extract_keywords(query)
            is_how_question = any(word in query for word in ["如何", "怎么", "怎样"])
            is_what_question = any(word in query for word in ["什么", "是啥"])
            is_when_question = any(word in query for word in ["什么时候", "何时"])
            
            # 遍历排序后的文档
            for doc in ranked_docs:
                # 计算文档与问题的关键词匹配度
                doc_keywords = self._extract_keywords(doc)
                keyword_overlap = len(set(keywords) & set(doc_keywords))
                
                # 如果关键词匹配度高，进一步分析文档内容
                if keyword_overlap >= len(keywords) * 0.7:
                    # 根据问题类型处理文档
                    if is_how_question and ("步骤" in doc or "方法" in doc or "操作" in doc):
                        return doc
                    elif is_what_question and ("是" in doc or "指" in doc or "表示" in doc):
                        return doc
                    elif is_when_question and ("时" in doc or "当" in doc or "情况" in doc):
                        return doc
                    # 如果文档长度适中且包含完整句子，也可以直接使用
                    elif 50 <= len(doc) <= 200 and doc.endswith(("。", "！", "？")):
                        return doc
            
            return ""
            
        except Exception as e:
            self.logger.error(f"直接查找答案失败: {str(e)}")
            return ""

    def process_query(self, query: str) -> Dict[str, Any]:
        """优化后的查询处理主流程"""
        try:
            self.logger.info("开始处理查询: %s", query)
            
            # 1. 问题改写和关键词提取
            rewritten_query = self._rewrite_query(query)
            keywords = self._extract_keywords(query)
            
            # 只在这里输出一次日志
            self.logger.info("查询处理开始:")
            self.logger.info("- 原始查询: %s", query)
            self.logger.info("- 改写后的查询: %s", rewritten_query)
            self.logger.info("- 提取的关键词: %s", keywords)
            
            # 2. 生成初步答案
            initial_answer = self._generate_initial_answer(query)
            if initial_answer:
                initial_answer = self._clean_response(initial_answer)
            self.logger.info("- 初步答案: %s", initial_answer)
            
            # 3. 文档检索和重排序
            retrieved_docs = self.retrieve_documents(
                original_query=query,
                rewritten_query=rewritten_query,
                keywords=keywords,
                llm_answer=initial_answer
            )
            
            # 4. 文档重排序
            ranked_docs = self._rerank_documents(retrieved_docs, keywords)
            
            # 5. 生成最终答案
            final_result = self.optimize_answer_iteratively(query, ranked_docs)
            
            # 6. 答案质量控制
            if final_result and 'answer' in final_result:
                answer = final_result['answer']
                current_quality = final_result.get('quality', {})
                
                # 如果答案质量不满足要求，使用LLM重新生成
                if not self._verify_answer_quality(answer, query, "\n".join(ranked_docs)):
                    self.logger.info("答案质量不满足要求，尝试重新生成")
                    
                    # 提取关键词用于增强prompt
                    keywords = self._extract_keywords(query)
                    question_type = self._get_question_type(query)
                    
                    
                    
                    enhanced_prompt = f"""请根据以下要求重新生成答案：
问题：{query}
参考文档：{ranked_docs}

回答要求：
1. 必须包含以下核心要素：{', '.join(keywords)}
2. 避免使用专业术语（如必须使用需括号解释）
3. 确保答案完整且准确


请生成答案："""
                    
                    enhanced_answer = self._generate_response(enhanced_prompt)
                    enhanced_answer = self._clean_response(enhanced_answer)
                    
                    if self._verify_answer_quality(enhanced_answer, query, "\n".join(ranked_docs)):
                        # 计算新旧答案的质量分数
                        old_score = current_quality.get('overall_score', 0)
                        new_quality = self._evaluate_answer_quality(enhanced_answer, query, ranked_docs)
                        new_score = new_quality.get('overall_score', 0)
                        
                        # 记录评分对比
                        self.logger.info(f"答案质量对比:\n"
                                       f"- 原答案得分: {old_score:.2f}\n"
                                       f"- 新答案得分: {new_score:.2f}")
                        
                        # 只有新答案显著优于旧答案时才替换
                        if new_score > old_score * 1.1:  # 要求至少提升10%
                            self.logger.info("使用质量更好的新答案")
                            final_result['answer'] = enhanced_answer
                            final_result['source'] = 'llm_enhanced'
                            final_result['quality'] = new_quality
                        else:
                            self.logger.info("保留原答案（新答案未显著提升）")
                    else:
                        self.logger.info("新生成的答案质量不达标，保留原答案")
                
                final_answer = self._clean_response(self._extract_final_answer(final_result['answer']))
                self.logger.info("- 最终答案: %s", final_answer)
                final_quality = final_result.get('quality', {})
                final_source = final_result.get('source', 'unknown')
            else:
                final_answer = ""
                final_quality = {}
                final_source = 'none'
            
            return {
                "query": query,
                "rewritten_query": rewritten_query,
                "keywords": keywords,
                "initial_answer": initial_answer,
                "final_answer": final_answer,
                "doc_count": len(retrieved_docs),
                "quality_score": final_quality,
                "source": final_source
            }
                
        except Exception as e:
            self.logger.error("查询处理失败: %s", str(e))
            return {
                "query": query,
                "error": str(e),
                "initial_answer": "",
                "final_answer": "抱歉，处理您的问题时出现错误。",
                "source": 'error'
            }

    def _rerank_documents(self, documents: List[Dict], keywords: List[str] = None) -> List[str]:
        """重排序文档"""
        try:
            # 1. 提高关键词匹配的权重
            unique_docs = {}
            for doc in documents:
                content = doc.get("content", "")
                score = doc.get("score", 0)
                # 增加关键词匹配的权重
                if keywords:
                    keyword_score = sum(content.count(kw) for kw in keywords) * 0.1
                    score += keyword_score
                if content not in unique_docs or score > unique_docs[content]["score"]:
                    unique_docs[content] = doc
            
            # 记录去重后的文档数量
            self.logger.info(f"去重后文档数量: {len(unique_docs)}")
            
            # 2. 准备重排序的文档列表
            docs_to_rerank = list(unique_docs.values())
            
            # 3. 使用BGE和BCE重排序器
            query = " ".join(keywords) if keywords else ""
            try:
                bge_scores = self.reranker['bge'].rerank(query=query, documents=docs_to_rerank)
                bce_scores = self.reranker['bce'].rerank(query=query, documents=docs_to_rerank)
            except Exception as e:
                self.logger.error(f"重排序器调用失败: {str(e)}")
                return [doc.get("content", "") for doc in docs_to_rerank[:5]]
            
            # 4. 合并分数
            merged_scores = {}
            for i, doc in enumerate(docs_to_rerank):
                content = doc.get("content", "")
                bge_score = float(bge_scores[i]) if isinstance(bge_scores, (list, tuple)) else 0.0
                bce_score = float(bce_scores[i]) if isinstance(bce_scores, (list, tuple)) else 0.0
                
                merged_scores[content] = (
                    OPTIMIZED_RERANKER_WEIGHTS['bge'] * bge_score + 
                    OPTIMIZED_RERANKER_WEIGHTS['bce'] * bce_score
                )
            
            # 5. 按合并后的分数排序
            sorted_docs = sorted(
                merged_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # 6. 调整过滤策略，确保包含关键词的文档优先保留
            filtered_docs = []
            for content, score in sorted_docs:
                # 包含关键词的文档直接保留
                if keywords and any(kw in content for kw in keywords):
                    filtered_docs.append(content)
                    continue
                # 其他文档需要满足质量要求
                if len(content) >= 20 and not self._is_low_quality(content, keywords):
                    filtered_docs.append(content)
            
            return filtered_docs[:5]
            
        except Exception as e:
            self.logger.error(f"文档重排序失败: {str(e)}")
            return [doc.get("content", "") for doc in documents[:5]]

    def _is_low_quality(self, text: str, keywords: List[str] = None) -> bool:
        """判断文档是否低质量
        
        Args:
            text: 文档内容
            keywords: 从问题中提取的关键词列表，如果为None则使用默认关键词
            
        Returns:
            bool: 是否为低质量文档
        """
        try:
            # 1. 检查长度
            if len(text) < 50:
                return True
            
            # 2. 检查特殊字符比例
            special_chars = sum(1 for c in text if not c.isalnum())
            if special_chars / len(text) > 0.5:
                return True
            
            # 3. 检查是否包含关键信息
            if keywords:
                # 使用系统提取的关键词
                if not any(kw in text for kw in keywords):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"文档质量评估失败: {str(e)}")
            return True

    def _check_semantic_relevance(self, sentence: str, question: str) -> bool:
        """检查句子与问题的语义相关性
        
        Args:
            sentence: 待检查的句子
            question: 原始问题
            
        Returns:
            bool: 是否相关
        """
        try:
            # 1. 提取问题中的限定词
            question_qualifiers = {
                "前": ["前", "前部", "前面", "前挡", "前窗"],
                "后": ["后", "后部", "后面", "后挡", "后窗"],
                "侧": ["侧", "侧面", "两侧"],
            }
            
            # 2. 识别问题中的限定词
            question_scope = None
            for scope, markers in question_qualifiers.items():
                if any(marker in question for marker in markers):
                    question_scope = scope
                    break
            
            if not question_scope:
                return True  # 如果问题中没有限定词，则认为相关
            
            # 3. 检查句子中的限定词
            for scope, markers in question_qualifiers.items():
                if any(marker in sentence for marker in markers):
                    # 如果句子中的限定词与问题中的不一致，则认为不相关
                    if scope != question_scope:
                        return False
            
            # 4. 检查句子是否包含问题中的核心动词和名词
            question_words = set(jieba.cut(question))
            sentence_words = set(jieba.cut(sentence))
            
            # 提取核心动词和名词
            core_words = {word for word in question_words 
                         if word in ["启用", "打开", "开启", "使用", "除霜", "除雾", "功能"]}
            
            # 如果句子包含核心词，则认为相关
            if core_words & sentence_words:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"检查语义相关性失败: {str(e)}")
            return True  # 出错时默认相关

    def _remove_similar_sentences(self, sentences: List[str], question: str = "", similarity_threshold: float = 0.8) -> List[str]:
        """检测并删除相似度高的句子
        
        Args:
            sentences: 句子列表
            question: 原始问题，用于语义相关性检查
            similarity_threshold: 相似度阈值，超过此值视为重复
            
        Returns:
            List[str]: 去重后的句子列表
        """
        try:
            if not sentences:
                return []
                
            # 规范化句子，便于比较
            def normalize_sentence(s: str) -> str:
                # 移除标点和空白
                s = re.sub(r'[，。！？、：；]', '', s)
                s = re.sub(r'\s+', '', s)
                return s
            
            # 计算两个句子的相似度
            def sentence_similarity(s1: str, s2: str) -> float:
                if not s1 or not s2:
                    return 0.0
                    
                # 规范化处理
                s1_norm = normalize_sentence(s1)
                s2_norm = normalize_sentence(s2)
                
                # 如果完全相同
                if s1_norm == s2_norm:
                    return 1.0
                    
                # 如果一个句子包含另一个
                if s1_norm in s2_norm or s2_norm in s1_norm:
                    return 0.9
                    
                # 计算编辑距离相似度
                try:
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, s1_norm, s2_norm).ratio()
                    return similarity
                except:
                    # 如果编辑距离计算失败，使用简单的字符重叠度
                    common_chars = set(s1_norm) & set(s2_norm)
                    return len(common_chars) / max(len(set(s1_norm)), len(set(s2_norm)))
            
            # 存储不重复的句子
            unique_sentences = []
            seen_sentences = set()
            
            for i, current_sent in enumerate(sentences):
                # 首先检查语义相关性
                if question and not self._check_semantic_relevance(current_sent, question):
                    continue
                    
                current_norm = normalize_sentence(current_sent)
                
                # 如果已经完全相同的句子，跳过
                if current_norm in seen_sentences:
                    continue
                    
                # 检查与之前句子的相似度
                is_similar = False
                for prev_sent in unique_sentences:
                    similarity = sentence_similarity(current_sent, prev_sent)
                    if similarity >= similarity_threshold:
                        is_similar = True
                        break
                
                if not is_similar:
                    unique_sentences.append(current_sent)
                    seen_sentences.add(current_norm)
            
            return unique_sentences
            
        except Exception as e:
            self.logger.error(f"去除相似句子失败: {str(e)}")
            return sentences

    def _clean_response(self, response: str, question: str = "") -> str:
        """清理LLM生成的答案"""
        try:
            # 处理非字符串输入
            if isinstance(response, tuple):
                response = response[0]
            response = str(response)
            
            # 1. 移除参考章节提示
            reference_patterns = [
                r'更多.*?信息.*?请参见.*?章节.*?[。\n]',
                r'详细信息.*?请参见.*?章节.*?[。\n]',
                r'具体.*?请参见.*?章节.*?[。\n]',
                r'参考文档[：:](.*?)(?=\n|$)',
                r'\{content:.*?\}',
                r'参考[：:](.*?)(?=\n|$)',
                r'原文档[：:](.*?)(?=\n|$)',
            ]
            for pattern in reference_patterns:
                response = re.sub(pattern, '', response)
            
            # 2. 清理特殊格式和标记
            patterns = [
                r'\{[^}]*content[^}]*\}',     # 移除包含content的JSON对象
                r'\{.*?\}',                    # 移除其他JSON对象
                r'[\(\[\{].*?[\)\]\}]',       # 移除括号内容
                r'role.*?content',             # 移除角色标记
                r'metadata.*?assistant',       # 移除元数据
                r'user.*?assistant',           # 移除对话标记
                r'[\'"`]',                     # 移除引号
                r'\b\d{1,2}\.\s*',            # 移除"01." "02."等格式
                r'^\d+[\.\、]\s*',            # 移除开头的"1." "2."等格式
                r'第\d+步[：:.]?\s*',         # 移除"第1步："等格式
                r'步骤\s*\d+[：:.]?\s*',      # 移除"步骤1："等格式
                r'\([^)]*\)',                 # 移除括号内容
                r'■+\s*',                     # 移除特殊符号■
                r'●+\s*',                     # 移除特殊符号●
                r'※+\s*',                     # 移除特殊符号※
                r'\s+',                       # 合并多余空格
            ]
            
            for pattern in patterns:
                response = re.sub(pattern, '', response)
            
            # 3. 分句并去重
            sentences = []
            for sent in re.split(r'[。！？]', response):
                sent = sent.strip()
                if not sent or len(sent) < 5:  # 过滤空句和过短的句子
                    continue
                    
                # 移除数字开头的句子中的数字
                if re.match(r'^\d+', sent):
                    sent = re.sub(r'^\d+[\.、\s]*', '', sent)
                    
                if sent.strip():
                    sentences.append(sent)
            
            # 4. 使用新的去重方法，传入问题
            unique_sentences = self._remove_similar_sentences(sentences, question=question, similarity_threshold=0.8)
            
            # 5. 合并句子
            merged_response = "。".join(unique_sentences)
            
            # 6. 最终清理
            # 移除可能残留的参考文档标记
            merged_response = re.sub(r'参考文档.*?(?=。|$)', '', merged_response)
            merged_response = re.sub(r'\{content:.*?\}', '', merged_response)
            
            # 确保句子结尾有标点
            if merged_response and not merged_response.endswith(("。", "！", "？")):
                merged_response += "。"
            
            return merged_response.strip()
            
        except Exception as e:
            self.logger.error(f"清理答案失败: {str(e)}")
            return str(response)

    def _encode_query(self, query: str):
        """带缓存的查询编码"""
        if query in self._query_encoding_cache:
            return self._query_encoding_cache[query]
            
        encoding = super()._encode_query(query)
        self._query_encoding_cache[query] = encoding
        return encoding

    def _batch_encode_documents(self, documents: List[str]) -> None:
        """优化的批量文档编码方法"""
        try:
            # 过滤需要编码的文档
            docs_to_encode = []
            doc_indices = []
            
            for i, doc in enumerate(documents):
                if doc not in self._doc_encoding_cache:
                    docs_to_encode.append(doc)
                    doc_indices.append(i)
            
            if docs_to_encode:
                self.logger.info(f"批量编码 {len(docs_to_encode)} 个新文档")
                
                # 使用批处理进行编码
                batch_size = 32
                all_encodings = []
                
                for i in range(0, len(docs_to_encode), batch_size):
                    batch = docs_to_encode[i:i + batch_size]
                    batch_encodings = self.encoder.encode(batch)
                    all_encodings.extend(batch_encodings)
                
                # 更新缓存
                for doc, encoding in zip(docs_to_encode, all_encodings):
                    self._doc_encoding_cache[doc] = encoding
                
                self.logger.debug(f"文档编码完成，缓存大小: {len(self._doc_encoding_cache)}")
                
            return [self._doc_encoding_cache[doc] for doc in documents]
            
        except Exception as e:
            self.logger.error(f"批量编码文档失败: {str(e)}")
            return []

    def _check_answer_relevance(self, answer: str, question: str, relevant_docs: Union[List[Dict[str, Any]], List[str], str]) -> bool:
        """检查答案与问题的相关性
        
        Args:
            answer: 生成的答案
            question: 原始问题
            relevant_docs: 相关文档列表或字符串
        
        Returns:
            bool: 答案是否与问题相关
        """
        try:
            # 1. 标准化文档格式
            doc_text = self._normalize_doc_text(relevant_docs)
            
            # 2. 计算答案与问题的相似度
            question_similarity = self._calculate_similarity(answer, question, method="keyword")
            
            # 3. 计算答案与文档的相似度
            doc_similarity = self._calculate_similarity(answer, doc_text, method="hybrid") if doc_text else 0.0
            
            # 4. 检查问题类型匹配
            question_type = self._get_question_type(question)
            type_score = 1.0 if question_type else 0.5
            
            # 5. 计算加权得分
            final_score = (
                0.6 * question_similarity +  # 问题相关性权重
                0.3 * doc_similarity +       # 文档一致性权重
                0.1 * type_score            # 问题类型匹配权重
            )
            
            # 动态阈值：根据问题类型调整
            threshold = {
                "是什么": 0.5,
                "作用": 0.6,
                "如何": 0.5,
                "区别": 0.5,
                "原因": 0.5
            }.get(question_type, 0.5)
            
            return final_score >= threshold
            
        except Exception as e:
            self.logger.error(f"检查答案相关性失败: {str(e)}")
            return False

    def _get_question_type(self, question: str) -> str:
        """获取问题类型
        
        Args:
            question: 问题文本
            
        Returns:
            str: 问题类型，如果无法识别则返回None
        """
        type_markers = {
            "是什么": ["是什么", "什么是"],
            "作用": ["作用", "功能", "用途"],
            "如何": ["如何", "怎么", "怎样"],
            "区别": ["区别", "不同", "差异"],
            "原因": ["为什么", "原因", "导致"]
        }
        
        for qtype, markers in type_markers.items():
            if any(marker in question for marker in markers):
                return qtype
            
        return None

    def _extract_context_sentences(self, sentences: List[str], target_idx: int, window_size: int = 2) -> str:
        """提取上下文时保持语义完整性
        
        Args:
            sentences: 句子列表
            target_idx: 目标句子索引
            window_size: 上下文窗口大小
        
        Returns:
            str: 处理后的上下文
        """
        try:
            # 1. 获取初始上下文范围
            start_idx = max(0, target_idx - window_size)
            end_idx = min(len(sentences), target_idx + window_size + 1)
            
            # 2. 智能扩展上下文
            while start_idx > 0:
                prev_sent = sentences[start_idx - 1]
                # 检查是否有连续性标记
                if any(marker in prev_sent for marker in ['因此', '所以', '但是', '而且']):
                    start_idx -= 1
                else:
                    break
                
            while end_idx < len(sentences):
                next_sent = sentences[end_idx]
                if any(marker in next_sent for marker in ['因此', '所以', '但是', '而且']):
                    end_idx += 1
                else:
                    break
            
            # 3. 确保上下文的语义完整性
            context_sentences = []
            for i in range(start_idx, end_idx):
                sent = sentences[i].strip()
                if sent and not self._is_noise_sentence(sent):
                    context_sentences.append(sent)
            
            return "。".join(context_sentences) + "。"
            
        except Exception as e:
            self.logger.error(f"提取上下文失败: {str(e)}")
            return ""

    def _is_noise_sentence(self, sentence: str) -> bool:
        """检查是否为噪声句子
        
        Args:
            sentence: 待检查的句子
            
        Returns:
            bool: 是否为噪声句子
        """
        noise_patterns = [
            r'^\d+[\.\、]',  # 编号
            r'^第.*?[章节]',  # 章节标题
            r'^提示[:：]',   # 提示标记
            r'^注意[:：]',   # 注意标记
            r'^温馨提示',    # 温馨提示
            r'^说明[:：]',   # 说明标记
            r'^备注[:：]'    # 备注标记
        ]
        return any(re.match(pattern, sentence) for pattern in noise_patterns)

    def _split_sentences(self, text: str) -> List[str]:
        """智能分句,处理中文文本
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分句结果
        """
        try:
            # 1. 预处理 - 处理特殊标点和格式
            text = re.sub(r'\s+', ' ', text)  # 规范化空格
            text = re.sub(r'([。！？\?])([^"\']+)', r"\1\n\2", text)  # 断句
            text = re.sub(r'(\.{6})([^"\']+)', r"\1\n\2", text)  # 处理省略号  
            text = re.sub(r'(\…{2})([^"\']+)', r"\1\n\2", text)  # 处理中文省略号
            
            # 2. 分句 - 考虑多种情况
            sentences = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                # 处理复杂的分句情况
                if re.search(r'[。！？\?]$', line):
                    sentences.append(line)
                else:
                    # 处理未以标点结尾的句子
                    sub_sentences = re.split(r'([。！？\?])', line)
                    new_sents = []
                    for i in range(len(sub_sentences)-1):
                        if sub_sentences[i]:
                            new_sents.append(sub_sentences[i].strip() + \
                                (sub_sentences[i+1] if i+1 < len(sub_sentences) else ""))
                    if new_sents:
                        sentences.extend(new_sents)
                    elif line:
                        sentences.append(line)
            
            # 3. 后处理 - 清理和合并短句
            cleaned_sentences = []
            temp_sent = ""
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                    
                # 合并短句
                if len(sent) < 10 and temp_sent:
                    temp_sent += sent
                else:
                    if temp_sent:
                        cleaned_sentences.append(temp_sent)
                    temp_sent = sent
                    
            if temp_sent:
                cleaned_sentences.append(temp_sent)
                
            # 4. 确保句子完整性
            final_sentences = []
            for sent in cleaned_sentences:
                # 检查是否为完整句子
                if len(sent) >= 10 and not self._is_incomplete_sentence(sent):
                    final_sentences.append(sent)
                    
            return final_sentences
            
        except Exception as e:
            self.logger.error(f"分句处理失败: {str(e)}")
            return text.split("。") if text else []

    def _is_incomplete_sentence(self, sentence: str) -> bool:
        """检查是否为不完整句子
        
        Args:
            sentence: 待检查的句子
            
        Returns:
            bool: 是否不完整
        """
        # 1. 检查是否以标点符号开头
        if re.match(r'^[，,、.：:；;]', sentence):
            return True
            
        # 2. 检查关键词
        incomplete_markers = [
            '如果', '但是', '因此', '所以', '然后', '接着',
            '并且', '而且', '不过', '除此之外', '另外'
        ]
        if any(sentence.startswith(marker) for marker in incomplete_markers):
            return True
            
        # 3. 检查括号匹配
        brackets = {'(': ')', '（': '）', '[': ']', '【': '】'}
        stack = []
        for char in sentence:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return True
                if char != brackets[stack.pop()]:
                    return True
        if stack:  # 有未闭合的括号
            return True
            
        return False

    def _calculate_sentence_relevance(self, sentence: str, query: str, doc_context: str = "") -> float:
        """计算句子与查询的相关度
        
        Args:
            sentence: 候选句子
            query: 查询文本
            doc_context: 文档上下文(可选)
            
        Returns:
            float: 相关度分数(0-1)
        """
        try:
            # 1. 计算与查询的相似度
            query_score = self._calculate_similarity(sentence, query, method="keyword")
            
            # 2. 计算句子完整性分数
            completeness_score = min(1.0, len(sentence) / 50)  # 假设理想长度为50
            
            # 3. 考虑上下文相关性
            context_score = 0.0
            if doc_context:
                context_score = self._calculate_similarity(sentence, doc_context, method="keyword")
            
            # 4. 计算最终得分
            weights = {
                'query': 0.5,
                'completeness': 0.3,
                'context': 0.2
            }
            
            final_score = (
                weights['query'] * query_score +
                weights['completeness'] * completeness_score +
                weights['context'] * context_score
            )
            
            return min(1.0, final_score)
            
        except Exception as e:
            self.logger.error(f"计算句子相关度失败: {str(e)}")
            return 0.0

    def _evaluate_factual_consistency(self, answer: str, relevant_docs: Union[List[Dict[str, Any]], List[str], str]) -> float:
        """评估答案的事实一致性
        
        Args:
            answer: 生成的答案
            relevant_docs: 相关文档列表或字符串
            
        Returns:
            float: 事实一致性得分
        """
        try:
            # 1. 标准化文档格式
            doc_text = self._normalize_doc_text(relevant_docs)
            if not doc_text:
                return 0.0
            
            # 2. 计算关键词匹配度
            keyword_score = self._get_keyword_overlap_score(answer, doc_text)
            
            # 3. 计算句子级别的匹配度
            answer_sentences = self._split_sentences(answer)
            doc_sentences = self._split_sentences(doc_text)
            
            sentence_matches = 0
            for ans_sent in answer_sentences:
                if self._calculate_similarity(ans_sent, doc_text, method="keyword") > 0.5:
                    sentence_matches += 1
            
            sentence_consistency = sentence_matches / len(answer_sentences) if answer_sentences else 0.0
            
            # 4. 计算最终得分
            factual_score = 0.6 * sentence_consistency + 0.4 * keyword_score
            
            return factual_score
            
        except Exception as e:
            self.logger.error(f"评估事实一致性失败: {str(e)}")
            return 0.0

    def _evaluate_structure(self, answer: str, question_type: str) -> float:
        """评估答案的结构质量
        
        Args:
            answer: 生成的答案
            question_type: 问题类型
            
        Returns:
            float: 结构质量得分
        """
        try:
            # 1. 根据问题类型确定结构要求
            structure_patterns = {
                "how": r"^\d+[\.\、]|首先|然后|接着|最后",
                "what": r"^\d+[\.\、]|定义|概念",
                "why": r"原因|因为|由于",
                "compare": r"一方面|另一方面|相比之下|不同点",
                "process": r"^\d+[\.\、]|第[一二三四五六七八九十]步|首先|其次"
            }
            
            pattern = structure_patterns.get(question_type, "")
            has_structure = bool(pattern and re.search(pattern, answer))
            
            # 2. 计算结构完整性得分
            if has_structure:
                return 1.0
            else:
                return 0.0
            
        except Exception as e:
            self.logger.error(f"评估结构质量失败: {str(e)}")
            return 0.0

    def _calculate_doc_similarity(self, answer: str, relevant_docs: Union[List[Dict[str, Any]], List[str], str]) -> float:
        """计算答案与文档的相似度
        
        Args:
            answer: 生成的答案
            relevant_docs: 相关文档列表或字符串
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            # 1. 标准化文档格式
            doc_text = self._normalize_doc_text(relevant_docs)
            if not doc_text:
                return 0.0
            
            # 2. 使用混合方法计算相似度
            return self._calculate_similarity(answer, doc_text, method="hybrid")
            
        except Exception as e:
            self.logger.error(f"计算文档相似度失败: {str(e)}")
            return 0.0

    def _normalize_doc_text(self, docs: Union[List[Dict[str, Any]], List[str], str]) -> str:
        """标准化文档文本格式
        
        Args:
            docs: 各种格式的文档输入
            
        Returns:
            str: 标准化后的文本
        """
        try:
            if isinstance(docs, str):
                return docs
                
            if isinstance(docs, list):
                if all(isinstance(doc, dict) for doc in docs):
                    return " ".join(doc.get("content", "") for doc in docs)
                return " ".join(doc if isinstance(doc, str) else doc.get("content", "") for doc in docs)
                
            return ""
            
        except Exception as e:
            self.logger.error(f"文档格式标准化失败: {str(e)}")
            return ""

    def _get_keyword_overlap_score(self, text1: str, text2: str) -> float:
        """计算两段文本的关键词重叠度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            float: 重叠度分数(0-1)
        """
        try:
            keywords1 = set(self._extract_keywords(text1))
            keywords2 = set(self._extract_keywords(text2))
            
            if not keywords1 or not keywords2:
                return 0.0
                
            overlap = len(keywords1 & keywords2)
            return overlap / len(keywords1)
            
        except Exception as e:
            self.logger.error(f"计算关键词重叠度失败: {str(e)}")
            return 0.0

    def _calculate_similarity(self, text1: str, text2: str, method: str = "hybrid") -> float:
        """计算两段文本的相似度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            method: 相似度计算方法，可选 "hybrid"/"keyword"/"semantic"
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            if not text1 or not text2:
                return 0.0
                
            # 分词并移除停用词
            words1 = set(jieba.cut(text1)) - stopwords
            words2 = set(jieba.cut(text2)) - stopwords
            
            if method in ["hybrid", "keyword"]:
                # 计算关键词重叠度
                common_words = words1 & words2
                # 使用较小集合的长度作为分母，提高相关文本的分数
                keyword_score = len(common_words) / min(len(words1), len(words2)) if words1 and words2 else 0.0
                
                # 额外的加权：如果共同词在较短文本中占比很高，增加分数
                if keyword_score > 0:
                    coverage = len(common_words) / min(len(words1), len(words2))
                    keyword_score = keyword_score * (1 + coverage) / 2
                
                if method == "keyword":
                    return min(1.0, keyword_score)
            
            if method in ["hybrid", "semantic"]:
                # 计算语义相似度（使用编辑距离）
                try:
                    from difflib import SequenceMatcher
                    semantic_score = SequenceMatcher(None, text1, text2).ratio()
                    
                    # 对于较长的文本，调整语义相似度分数
                    avg_len = (len(text1) + len(text2)) / 2
                    if avg_len > 50:
                        semantic_score *= 1.2  # 稍微提高长文本的分数
                    
                except:
                    semantic_score = 0.0
                    
                if method == "semantic":
                    return min(1.0, semantic_score)
            
            # 混合模式：结合关键词和语义相似度
            if method == "hybrid":
                # 动态调整权重：当关键词匹配度高时，增加其权重
                if keyword_score > 0.5:
                    weight = 0.9
                else:
                    weight = 0.7
                
                final_score = weight * keyword_score + (1 - weight) * semantic_score
                return min(1.0, final_score)
                
            return 0.0
            
        except Exception as e:
            self.logger.error(f"相似度计算失败: {str(e)}")
            return 0.0

    def _filter_irrelevant_sentences(self, answer: str, question: str, threshold: float = 0.3) -> str:
        """过滤与问题无关的句子
        
        Args:
            answer: 生成的答案
            question: 原始问题
            threshold: 相关度阈值
            
        Returns:
            str: 过滤后的答案
        """
        try:
            # 1. 分句
            sentences = self._split_sentences(answer)
            if not sentences:
                return answer
                
            # 2. 提取问题关键词和问题类型
            question_keywords = set(self._extract_keywords(question))
            question_type = self._get_question_type(question)
            
            # 3. 计算每个句子的相关度并过滤
            relevant_sentences = []
            for sentence in sentences:
                # 跳过明显的无关句子
                if any(marker in sentence for marker in [
                    "您需要了解", "您还想知道", "请问您", "更多信息",
                    "如有疑问", "如需帮助", "请参考", "请注意"
                ]):
                    continue
                    
                # 计算关键词相关度
                sentence_keywords = set(self._extract_keywords(sentence))
                keyword_overlap = len(question_keywords & sentence_keywords)
                keyword_score = keyword_overlap / len(question_keywords) if question_keywords else 0
                
                # 计算向量相似度
                similarity_score = self._calculate_similarity(sentence, question, method="hybrid")
                
                # 计算句子结构得分
                structure_score = self._evaluate_sentence_structure(sentence, question_type)
                
                # 综合评分 (动态权重)
                weights = {
                    'keyword': 0.5,
                    'similarity': 0.3,
                    'structure': 0.2
                }
                
                final_score = (
                    weights['keyword'] * keyword_score +
                    weights['similarity'] * similarity_score +
                    weights['structure'] * structure_score
                )
                
                if final_score >= threshold:
                    relevant_sentences.append(sentence)
                    
            # 4. 如果过滤后没有句子,返回原文
            if not relevant_sentences:
                return answer
                
            # 5. 合并相关句子
            filtered_answer = "。".join(relevant_sentences)
            if not filtered_answer.endswith("。"):
                filtered_answer += "。"
                
            return filtered_answer
            
        except Exception as e:
            self.logger.error(f"句子过滤失败: {str(e)}")
            return answer
            
    def _evaluate_sentence_structure(self, sentence: str, question_type: str) -> float:
        """评估句子结构是否符合问题类型的要求
        
        Args:
            sentence: 待评估的句子
            question_type: 问题类型
            
        Returns:
            float: 结构评分
        """
        try:
            # 1. 基础分数
            base_score = 0.5
            
            # 2. 根据问题类型检查句子结构
            if question_type == "how":
                # 检查是否包含步骤或操作说明
                if any(word in sentence for word in ["首先", "然后", "接着", "最后", "通过", "使用", "点击", "按下"]):
                    base_score += 0.3
                    
            elif question_type == "what":
                # 检查是否包含定义或解释
                if any(word in sentence for word in ["是", "指", "表示", "包含", "包括"]):
                    base_score += 0.3
                    
            elif question_type == "why":
                # 检查是否包含原因说明
                if any(word in sentence for word in ["因为", "由于", "所以", "导致", "原因"]):
                    base_score += 0.3
                    
            # 3. 检查句子完整性
            if len(sentence) >= 10 and not self._is_incomplete_sentence(sentence):
                base_score += 0.2
                
            return min(1.0, base_score)
            
        except Exception as e:
            self.logger.error(f"评估句子结构失败: {str(e)}")
            return 0.5

    def _get_question_type(self, question: str) -> str:
        """识别问题类型
        
        Args:
            question: 输入的问题
            
        Returns:
            str: 问题类型 (definition/how/what/where/when)
        """
        # 定义类问题的关键词
        definition_keywords = ["什么是", "何为", "定义", "指的是", "概念"]
        how_keywords = ["如何", "怎样", "怎么", "步骤", "方法"]
        
        # 优先检查是否为定义类问题
        if any(keyword in question for keyword in definition_keywords):
            return "definition"
        elif any(keyword in question for keyword in how_keywords):
            return "how"
        else:
            return "what"

    def _optimize_final_answer(self, answer: str, question: str) -> str:
        """优化最终答案
        
        Args:
            answer: 初始答案
            question: 原始问题
            
        Returns:
            str: 优化后的答案
        """
        question_type = self._get_question_type(question)
        sentences = self._split_sentences(answer)
        
        # 对定义类问题特殊处理
        if question_type == "definition":
            # 识别定义句
            definition_sentences = []
            other_sentences = []
            
            for sent in sentences:
                # 检查是否为定义句
                if any(pattern in sent for pattern in ["是", "指", "称为", "表示", "意味着"]):
                    definition_sentences.append(sent)
                else:
                    other_sentences.append(sent)
            
            # 重组答案，确保定义在前
            optimized_sentences = definition_sentences + other_sentences
            
            # 如果没有找到明确的定义句，尝试构造一个
            if not definition_sentences and other_sentences:
                subject = re.sub(r'[是什么的]+$', '', question).strip()
                optimized_sentences.insert(0, f"{subject}是" + other_sentences[0])
                
            return "。".join(optimized_sentences)
        
        return answer
