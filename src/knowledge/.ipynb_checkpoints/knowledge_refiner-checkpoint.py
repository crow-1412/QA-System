from typing import List, Dict, Any, Tuple
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

logger = logging.getLogger(__name__)

stopwords = {'的', '了', '和', '是', '就', '都', '而', '及', '与'}

def evaluate_answer_quality(question: str, answer: str, relevant_docs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    评估答案质量的独立函数
    
    Args:
        question: 原始问题
        answer: 生成的答案
        relevant_docs: 相关文档列表，每个文档可以是字符串或包含 content 字段的字典
        
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
        
    # 1. 完整性评分
    ideal_length = 200
    completeness = min(len(answer) / ideal_length, 1.0)
    
    # 2. 相关性评分
    question_words = set(jieba.cut(question)) - stopwords
    answer_words = set(jieba.cut(answer)) - stopwords
    overlap = len(question_words & answer_words)
    relevance = overlap / len(question_words) if question_words else 0.0
    
    # 3. 清晰度评分
    sentences = re.split(r'[。！？]', answer)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    clarity = len(valid_sentences) / max(len(sentences), 1)
    
    # 4. 事实一致性评分
    factual_consistency = 0.0
    if relevant_docs:
        # 提取文档内容
        def get_doc_content(doc) -> str:
            if isinstance(doc, str):
                return doc
            elif isinstance(doc, dict):
                return doc.get("content", "") if "content" in doc else str(doc)
            return str(doc)
            
        # 将所有文档内容合并
        doc_text = " ".join(get_doc_content(doc) for doc in relevant_docs)
        if doc_text:
            doc_words = set(jieba.cut(doc_text)) - stopwords
            answer_fact_overlap = len(answer_words & doc_words)
            factual_consistency = answer_fact_overlap / len(answer_words) if answer_words else 0.0
    
    # 5. 综合评分
    weights = {
        "completeness": 0.25,
        "relevance": 0.3,
        "clarity": 0.2,
        "factual_consistency": 0.25
    }
    
    overall_score = (
        weights["completeness"] * completeness +
        weights["relevance"] * relevance +
        weights["clarity"] * clarity +
        weights["factual_consistency"] * factual_consistency
    )
    
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

    def retrieve_with_initial_answer(self, query: str) -> List[str]:
        """检索相关文档并生成初始答案
        Args:
            query: 查询文本
        Returns:
            检索到的相关文档列表
        """
        try:
            self.logger.info(f"混合检索查询: {query}")
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            # 确保返回的是字符串列表
            result_docs = []
            for doc in retrieved_docs:
                if isinstance(doc, dict) and "content" in doc:
                    result_docs.append(doc["content"])
                elif isinstance(doc, str):
                    result_docs.append(doc)
                else:
                    self.logger.warning(f"跳过无效文档格式: {type(doc)}")
            
            return result_docs
                    
        except Exception as e:
            self.logger.error(f"文档检索失败: {str(e)}")
            return []

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
4. 控制在50个字以内

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
            self.logger.error(f"重写问题失败: {str(e)}", exc_info=True)
            return question, []

    def refine_documents_with_llm(self, documents: List[str]) -> List[str]:
        """使用LLM规整文档内容"""
        try:
            refined_docs = []
            for doc in documents:
                prompt = f"""请对以下文档内容进行整理和优化，要求：
                1. 保持信息的完整性和准确性
                2. 使表达更清晰、连贯
                3. 去除冗余信息
                4. 保持专业术语不变
                5. 按重要性组织内容
                
                原文档：
                {doc}
                
                整理后的内容："""
                
                refined_doc = self._generate_response(prompt)
                refined_doc = self._clean_text(refined_doc)
                refined_docs.append(refined_doc)
            
            # 对整理后的文档进行摘要
            summarized_docs = self._summarize_docs(refined_docs)
            return [summarized_docs]
            
        except Exception as e:
            logger.error(f"规整文档失败: {str(e)}")
            return documents

    def _summarize_docs(self, docs: List[str], max_length: int = 2048) -> str:
        """
        对文档进行摘要，控制长度
        """
        try:
            # 1. 先尝试直接拼接
            combined = " ".join(docs)
            tokens = self.tokenizer(
                combined, 
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            if len(tokens.input_ids[0]) <= max_length:
                return combined
                
            # 2. 如果超长，对每个文档进行摘要
            summarized_docs = []
            for doc in docs:
                prompt = f"""请对以下内容进行摘要，保留关键信息：

{doc}

请生成简洁的摘要："""
                
                summary = self._generate_response(prompt)
                summarized_docs.append(summary)
                
            # 3. 合并摘要后的文档，确保不超过长度限制
            result = " ".join(summarized_docs)
            final_tokens = self.tokenizer(
                result, 
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            return self.tokenizer.decode(final_tokens.input_ids[0], skip_special_tokens=True)
            
        except Exception as e:
            self.logger.error(f"文档摘要失败: {str(e)}")
            return docs[0] if docs else ""

    def _clean_text(self, text: str) -> str:
        """清理生成的文本，移除提示词和无关内容"""
        if not text:
            return ""
            
        # 1. 移除[gMASK]和sop等特殊token
        text = re.sub(r'\[gMASK\].*?sop\s*', '', text)
        text = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]|\[MASK\]', '', text)
            
        # 2. 移除常见的提示词模式
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
        ]
        
        cleaned = text
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                
        # 3. 移除重复的问题陈述
        question_patterns = [
            r"问题[：:](.*?)[。\n]",
            r"如何.*?[。\n]",
            r".*?的判断方法.*?[。\n]",
        ]
        for pattern in question_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)
            
        # 4. 移除多余的空白字符和标点
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[。；，：]$', '。', cleaned)
        cleaned = re.sub(r'。+', '。', cleaned)
        
        # 5. 如果清理后为空，返回原文中的最后一个完整句子
        if not cleaned.strip():
            sentences = re.split(r'[。！？]', text)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) >= 10 and not any(token in s for token in ['[gMASK]', 'sop', '[CLS]', '[SEP]'])]
            if valid_sentences:
                return valid_sentences[-1] + "。"
            return ""
            
        return cleaned.strip()

    def _generate_and_optimize_answer(self, question: str, docs: List[str]) -> str:
        """生成并优化答案
        Args:
            question: 问题
            docs: 相关文档列表
        Returns:
            优化后的答案
        """
        try:
            # 1. 合并文档内容
            context = "\n".join(docs)
            
            # 2. 生成答案
            prompt = f"""请根据以下参考文档回答问题。要求：
1. 答案必须以"答案是："开头
2. 直接给出判断方法或关键信息
3. 不要解释原理
4. 使用"当...时..."或"通过...可以..."的形式
5. 答案必须简洁，不超过50个字

问题：{question}

参考文档：
{context}

请给出答案："""
            
            answer = self._generate_response(prompt)
            self.logger.debug(f"[初始答案]: {answer}")
            
            # 3. 提取并清理答案
            if "答案是：" in answer:
                answer = answer.split("答案是：")[-1].strip()
            answer = self._clean_text(answer)
            
            # 4. 如果答案太长，进行优化
            if len(answer) > 50:
                optimize_prompt = f"""请优化以下答案，使其更加简洁。要求：
1. 保持核心信息
2. 删除不必要的修饰词
3. 确保答案不超过50个字
4. 使用"当...时..."或"通过...可以..."的形式

原答案：{answer}

优化后的答案："""
                
                answer = self._generate_response(optimize_prompt)
                answer = self._clean_text(answer)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"生成答案失败: {str(e)}", exc_info=True)
            return "抱歉，生成答案时出现错误。"

    def _extract_final_answer(self, text: str) -> str:
        """
        从文本中提取最终答案，按优先级尝试不同的提取策略
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        # 1. 清理特殊token和JSON格式内容
        text = re.sub(r'\{.*?\}', '', text)  # 移除JSON格式内容
        text = re.sub(r'\[gMASK\].*?sop\s*', '', text)
        text = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]|\[MASK\]', '', text)
        
        # 2. 移除提示性话语和代码片段
        text = re.sub(r'//.*', '', text)
        text = re.sub(r'python .*', '', text)
        
        # 3. 尝试提取标记后的内容
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
                answer = re.sub(r'\{.*?\}', '', answer)
                answer = re.sub(r'\s+', ' ', answer)
                # 移除提示性话语
                answer = re.sub(r'//.*', '', answer)
                if len(answer) >= 10 and len(answer) <= 200:
                    return answer
        
        # 4. 尝试找到以特定关键词开头的句子
        sentence_markers = ["答案是：", "Final Answer:", "Answer:"]
        for marker in sentence_markers:
            if marker in text:
                answer = text.split(marker)[-1].strip()
                # 移除可能的JSON或重复内容
                answer = re.sub(r'\{.*?\}', '', answer)
                answer = re.sub(r'\s+', ' ', answer)
                return answer
        
        # 5. 提取完整的句子
        sentences = re.split(r'[。！？]', text)
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            if len(s) >= 10 and len(s) <= 200 and not any(token in s for token in ['[gMASK]', 'sop', '[CLS]', '[SEP]']):
                valid_sentences.append(s)
        
        if valid_sentences:
            # 优先选择包含判断相关词语的句子
            judgment_keywords = ["通过", "可以", "当", "表示", "说明", "显示"]
            for sentence in reversed(valid_sentences):
                if any(kw in sentence for kw in judgment_keywords):
                    # 移除可能的重复内容
                    sentence = re.sub(r'\{.*?\}', '', sentence)
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

    def _verify_answer_with_doc(self, answer: str, doc: str) -> bool:
        """
        验证答案与文档是否一致
        """
        try:
            prompt = f"""请判断下面的答案与参考信息是否一致：

答案：{answer}
参考信息：{doc}

请直接回答"一致"或"不一致"："""
            
            response = self._generate_response(prompt)
            return "一致" in response
            
        except Exception as e:
            self.logger.error(f"验证答案失败: {str(e)}")
            return True  # 出错时默认一致，避免过度修改

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
            
            # 3. 使用关键词检索
            keyword_query = " ".join(keywords[:3])  # 使用前3个关键词
            docs_from_keywords = self.retriever.get_relevant_documents(keyword_query)
            
            # 合并结果并去重
            all_docs = []
            seen_contents = set()
            
            for doc_list in [docs_from_original, docs_from_rewritten, docs_from_keywords]:
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
        """
        生成回复，并确保输出格式规范
        """
        try:
            # 添加系统提示
            system_prompt = """你是一个简洁的助手，请遵循以下规则：
1. 直接给出答案，不要解释过程
2. 使用简单的陈述句
3. 答案要具体且明确
4. 不要输出无关的内容
5. 不要包含对话历史、元数据或JSON格式内容"""
    
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
                    outputs = self.model.generate(**inputs, max_new_tokens=max_response_length, temperature=0.7, top_p=0.9, repetition_penalty=1.1, do_sample=True, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.encode("</assistant>")[0], bad_words_ids=[[self.tokenizer.encode(word)[0]] for word in ["[gMASK]", "sop", "[CLS]", "[SEP]"]])
                    
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
        简化后的判定：如果新答案更完整或更相关，则认为更好。
        """
        if not old_answer:
            return True
        new_score = self._evaluate_answer_quality(new_answer, question)
        old_score = self._evaluate_answer_quality(old_answer, question)
        return new_score > old_score

    def _evaluate_answer_quality(self, answer: str, question: str) -> float:
        """
        简化的答案质量评估方式：关键词重叠度 + 文本长度占比
        """
        if not answer:
            return 0.0
        question_keywords = set(self._extract_keywords(question))
        answer_keywords = set(self._extract_keywords(answer))
        
        overlap = len(question_keywords & answer_keywords)
        relevance = overlap / len(question_keywords) if question_keywords else 0.0
        
        # 简单看长度(假设理想答案100字符, 可根据情况调整)
        completeness = min(len(answer) / 100, 1.0)
        
        return (relevance + completeness) / 2

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
        """使用新文档优化当前答案"""
        try:
            prompt = f"""基于以下文档内容，优化当前答案。要求：
            1. 保持答案简洁明了
            2. 只在新文档提供更准确或补充信息时才修改答案
            3. 确保答案直接回答问题，不要解释原理
            4. 保持专业术语的准确性
            
            问题：{question}
            当前答案：{current_answer}
            文档内容：{doc_content}
            
            优化后的答案："""
            
            optimized_answer = self._generate_response(prompt)
            return self._clean_text(optimized_answer)
        except Exception as e:
            logger.error(f"优化答案失败: {str(e)}")
            return current_answer

    def _find_exact_match(self, question: str, references: List[str]) -> str:
        """在知识库文档中查找是否存在与问题高度相关的直接答案"""
        try:
            question_keywords = set(jieba.cut(question)) - stopwords
            best_score = 0.0
            best_match = ""
            
            for doc in references:
                sentences = re.split(r'[。！？]', doc)
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                
                    sentence_keywords = set(jieba.cut(sentence)) - stopwords
                    if not sentence_keywords:
                        continue
                        
                    keyword_overlap = len(question_keywords & sentence_keywords)
                    keyword_score = keyword_overlap / len(question_keywords)
                    
                    is_answer = bool(re.search(r'当.*时|如果.*则|表示|说明', sentence))
                    score = keyword_score * 1.5 if is_answer else keyword_score
                    
                    if score > best_score and score >= 0.6:
                        context = ""
                        if is_answer and i > 0:
                            prev = sentences[i-1].strip()
                            if prev:
                                context = prev + "。"
                        context += sentence
                        if i < len(sentences) - 1:
                            next_sent = sentences[i+1].strip()
                            if next_sent and len(context + next_sent) < 100:
                                context += "。" + next_sent
                        
                        best_score = score
                        best_match = context

            return best_match if best_score >= 0.6 else ""
                
        except Exception as e:
            self.logger.error(f"Error in exact matching: {e}")
            return ""

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
            # 先尝试在参考文档中找到直接答案
            if isinstance(reference_docs, str):
                docs_list = [reference_docs]
            else:
                docs_list = reference_docs
                
            exact_match = self._find_exact_match(question, docs_list)
            if exact_match:
                self.logger.info("找到精确匹配的答案")
                return exact_match
            
            # 如果没有找到直接答案，使用LLM生成
            self.logger.info("未找到精确匹配，使用LLM生成答案")
            answer_prompt = f"""基于参考信息回答问题，直接给出判断标准：
问题：{question}
参考：{reference_docs}
答案："""
            
            raw_answer = self._generate_response(
                answer_prompt,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length
            )
            
            # 使用_extract_final_answer提取最终答案
            return self._extract_final_answer(raw_answer)
            
        except Exception as e:
            self.logger.error(f"生成答案失败: {str(e)}")
            return ""
    
    def optimize_answer_iteratively(self, question: str, documents: List[str]) -> str:
        """迭代优化答案，每次使用一个文档来改进答案
        
        Args:
            question: 用户问题
            documents: 相关文档列表
            
        Returns:
            优化后的最终答案
        """
        try:
            if not documents:
                return ""
            
            # 生成初始答案并清理
            current_answer = self.generate_answer(question, documents[0])
            current_answer = self._clean_response(self._extract_final_answer(current_answer))
            
            # 如果找到精确匹配的答案，直接返回清理后的结果
            if self._find_direct_answer(question, [documents[0]]):
                self.logger.info("找到精确匹配答案，无需进一步优化")
                return current_answer
            
            # 尝试优化答案
            for doc in documents[1:]:
                new_answer = self._optimize_with_doc(current_answer, doc, question)
                new_answer = self._clean_response(self._extract_final_answer(new_answer))
                
                if not self._verify_answer_with_doc(new_answer, doc):
                    self.logger.info("优化后的答案与文档不一致，保持原答案")
                    break
                    
                if self._is_better_answer(new_answer, current_answer, question):
                    current_answer = new_answer
                    self.logger.info("答案已优化")
                    break
                else:
                    self.logger.info("新答案未能改进，保持原答案")
                    break
                
            return current_answer
            
        except Exception as e:
            self.logger.error(f"迭代优化答案失败: {str(e)}")
            return current_answer if 'current_answer' in locals() else ""

    def extract_keywords_from_text(self, text: str, top_k: int = 5) -> List[str]:
        """从文本中提取关键词的公共方法
        
        Args:
            text: 输入文本
            top_k: 返回的关键词数量
            
        Returns:
            关键词列表
        """
        return self._extract_keywords(text, top_k)

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
            
            # 4. 文档重排序，使用缓存优化，并传入关键词
            ranked_docs = self._rerank_documents(retrieved_docs, keywords)
            
            # 5. 生成最终答案
            final_answer = self.optimize_answer_iteratively(query, ranked_docs)
            if final_answer:
                # 确保在返回前进行最终的清理
                final_answer = self._clean_response(self._extract_final_answer(final_answer))
                self.logger.info("- 最终答案: %s", final_answer)
            
            return {
                "query": query,
                "rewritten_query": rewritten_query,
                "keywords": keywords,
                "initial_answer": initial_answer,
                "final_answer": final_answer,
                "doc_count": len(retrieved_docs)
            }
            
        except Exception as e:
            self.logger.error("查询处理失败: %s", str(e))
            return {
                "query": query,
                "error": str(e),
                "initial_answer": "",
                "final_answer": "抱歉，处理您的问题时出现错误。"
            }

    def _rerank_documents(self, documents: List[Dict], keywords: List[str] = None) -> List[str]:
        """重排序和规整文档"""
        try:
            # 1. 去重
            unique_docs = {}
            for doc in documents:
                content = doc.get("content", "")
                score = doc.get("score", 0)
                if content not in unique_docs or score > unique_docs[content]["score"]:
                    unique_docs[content] = doc
            
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
                # 确保分数是数值类型
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
            
            # 6. 过滤低质量文档，但保留包含关键词的文档
            filtered_docs = []
            for content, _ in sorted_docs:
                if keywords and any(kw in content for kw in keywords):
                    filtered_docs.append(content)
                    continue
                if len(content) >= 10 and not self._is_low_quality(content):
                    filtered_docs.append(content)
            
            # 7. 限制文档数量
            return filtered_docs[:5]
            
        except Exception as e:
            self.logger.error(f"文档重排序失败: {str(e)}")
            # 发生错误时返回原始文档的前5个
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

    def _clean_response(self, response: str) -> str:
        """清理LLM生成的答案"""
        try:
            # 处理非字符串输入
            if isinstance(response, tuple):
                response = response[0]
            response = str(response)
            
            # 1. 定义章节标题列表
            chapter_titles = {
                "前言", "用车前准备", "装载货物", "上车和下车", "驾驶前的准备",
                "仪表和灯光", "启动和驾驶", "驾驶辅助系统", "紧急情况", "保养与维护",
                "技术资料", "安全出行", "驾驶辅助", "泊车", "空调", "中央显示屏",
                "Lynk & Co App", "高压系统", "保养和维护", "OTA升级", "紧急情况下"
            }
            
            # 2. 移除开头的章节标题
            for title in chapter_titles:
                if response.startswith(title):
                    response = response[len(title):].lstrip()
                    break
            
            # 3. 清理特殊格式和标记
            patterns = [
                r'\{[^}]*content[^}]*\}',  # 移除包含content的JSON对象
                r'\{.*?\}',                 # 移除其他JSON对象
                r'[\(\[\{].*?[\)\]\}]',    # 移除括号内容
                r'role.*?content',          # 移除角色标记
                r'metadata.*?assistant',    # 移除元数据
                r'user.*?assistant',        # 移除对话标记
                r'[\'"`]',                  # 移除引号
                r'\b\d{1,2}\.\s*',         # 移除"01." "02."等格式
                r'^\d+[\.\、]\s*',         # 移除开头的"1." "2."等格式
                r'第\d+步[：:.]?\s*',      # 移除"第1步："等格式
                r'步骤\s*\d+[：:.]?\s*',   # 移除"步骤1："等格式
                r'^\d+\s*',                # 移除开头的数字
            ]
            
            for pattern in patterns:
                response = re.sub(pattern, '', response)
            
            # 4. 清理多余空白
            response = ' '.join(response.split())
            
            # 5. 确保句子完整性
            if response and not response.endswith(('。', '！', '？')):
                response += '。'
            
            return response.strip()
            
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