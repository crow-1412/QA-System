from src.retriever.tfidf_retriever import TFIDFRetriever
from src.retriever.bge_retriever import BGERetriever
from src.retriever.gte_retriever import GTERetriever
from src.retriever.bce_retriever import BCERetriever
from src.retriever.faiss_retriever import FAISSRetriever
from src.retriever.bm25_retriever import BM25
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import torch
import os
import jieba
import logging
from collections import defaultdict
import re
import faiss
from system_config import MODEL_PATHS
from src.retriever.stopwords import stopwords

# 设置CUDA内存分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logger = logging.getLogger(__name__)

# 定义同义词映射
synonym_mappings = {
    '去冰': ['除霜', 'defrost', '去霜', '除冰', '最大除霜'],
    '去雾': ['除雾', '防雾', '防止起雾'],
    '前挡风玻璃': ['前挡', '挡风玻璃', '前窗', '前风挡'],
    '空调': ['A/C', 'AC', '冷气', '制冷', '空调系统'],
    '模式': ['功能', '设置', '开关', '按键'],
    '自动': ['AUTO', 'auto', '联动', '自动模式'],
    '指示': ['指示灯', '按钮指示', '提示灯', 'LED'],
    '内循环': ['内部循环', '车内循环', '空气循环'],
    '风量': ['风速', '送风', '出风'],
    '激活': ['启动', '开启', '打开', '触发']
}

class MultiRetriever:
    """多检索器类，实现多种检索策略的组合"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化多检索器
        
        Args:
            config: 配置字典，包含documents和model_paths等
        """
        self.config = config
        self.documents = config.get("documents", [])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用system_config中的模型路径
        self.model_paths = {
            "bge": MODEL_PATHS["bge"],
            "gte": MODEL_PATHS["gte"],
            "bce": MODEL_PATHS["bce"]
        }
        
        # 如果config中有model_paths，则更新默认值
        if "model_paths" in config:
            self.model_paths.update(config["model_paths"])
            
        self.retrievers = {}
        self._init_retrievers()
        logger.info("多检索器初始化完成")
        
    def _init_retrievers(self):
        """初始化所有检索器"""
        try:
            # 检查文档列表是否为空
            if not self.documents:
                logger.warning("文档列表为空，检索器可能无法正常工作")
                return
                
            # 初始化TF-IDF检索器
            logger.info(f"初始化检索器 tfidf，文档数量: {len(self.documents)}")
            try:
                self.retrievers["tfidf"] = TFIDFRetriever(self.documents)
            except Exception as e:
                logger.error(f"初始化TF-IDF检索器失败: {str(e)}")
            
            # 初始化密集检索器
            for name in ["bge", "gte", "bce"]:
                logger.info(f"初始化检索器 {name}，类型: dense，文档数量: {len(self.documents)}")
                model_path = self.model_paths.get(name)
                if model_path:
                    try:
                        if name == "bge":
                            self.retrievers[name] = BGERetriever(self.documents, model_path)
                        elif name == "gte":
                            self.retrievers[name] = GTERetriever(self.documents, model_path)
                        elif name == "bce":
                            self.retrievers[name] = BCERetriever(self.documents, model_path)
                    except Exception as e:
                        logger.error(f"初始化检索器 {name} 失败: {str(e)}")
                else:
                    logger.warning(f"未找到{name.upper()}模型路径")
            
            # 初始化重排序器
            self.rerankers = {}
            for name, reranker_config in self.config.get("rerankers", {}).items():
                self.rerankers[name] = self._init_reranker(name, reranker_config)
            
            # 初始化FAISS索引
            self.index = None
            self.doc_store = {}
            
            if not self.retrievers:
                logger.warning("没有成功初始化任何检索器")
            
        except Exception as e:
            logger.error(f"初始化检索器时出错: {str(e)}")
            raise
        
    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """获取与查询相关的文档，并进行质量评估和重排序
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            list: 包含文档和相似度分数的列表
        """
        try:
            # 1. 提取关键词并进行同义词扩展
            keywords = [word for word in jieba.cut(query) if word not in stopwords and len(word) > 1]
            expanded_keywords = set(keywords)
            
            # 对每个关键词进行同义词扩展
            for word in keywords:
                for key, synonyms in synonym_mappings.items():
                    # 增加模糊匹配
                    if word in [key] + synonyms or any(syn in word for syn in [key] + synonyms):
                        expanded_keywords.update(synonyms)
                        expanded_keywords.add(key)
            
            logger.info(f"原始关键词: {keywords}")
            logger.info(f"扩展后的关键词: {list(expanded_keywords)}")
            
            # 2. 从每个检索器获取初始结果
            all_results = []
            for name, retriever in self.retrievers.items():
                try:
                    # 分别使用原始查询和扩展关键词进行检索
                    results = []
                    
                    # 使用原始查询
                    query_results = retriever.search(query, k=top_k)
                    results.extend(query_results)
                    
                    # 使用扩展关键词，但权重较低
                    if expanded_keywords:
                        expanded_query = " ".join(expanded_keywords)
                        keyword_results = retriever.search(expanded_query, k=top_k)
                        # 降低扩展查询结果的分数
                        for result in keyword_results:
                            result["score"] *= 0.8
                        results.extend(keyword_results)
                    
                    # 确保content字段是字符串
                    for result in results:
                        if isinstance(result.get("content"), (list, dict)):
                            result["content"] = str(result["content"])
                        elif result.get("content") is None:
                            result["content"] = ""
                    
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"检索器 {name} 检索失败: {str(e)}")
            
            # 3. 去重
            unique_results = self._dedup_and_rank_results(all_results, top_k * 3)
            
            # 4. 质量评估
            scored_results = []
            for doc in unique_results:
                # 计算相关性分数
                relevance_score = self._compute_relevance_score(query, doc)
                # 计算质量分数
                quality_score = self._compute_quality_score(doc)
                # 计算最终分数 (0.7 * 相关性 + 0.3 * 质量)
                final_score = 0.7 * relevance_score + 0.3 * quality_score
                scored_results.append((doc, final_score))
            
            # 5. 根据分数排序
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # 6. 考虑文档关系重新排序
            reranked_docs = self._rerank_with_relationships([doc for doc, _ in scored_results[:top_k*3]])
            
            # 7. 动态阈值过滤
            threshold = self._compute_dynamic_threshold(scored_results)
            filtered_results = [(doc, score) for doc, score in zip(reranked_docs, [s for _, s in scored_results[:len(reranked_docs)]]) if score >= threshold]
            
            # 根据查询复杂度动态调整返回数量
            query_words = set(jieba.cut(query)) - stopwords
            query_complexity = min(1.0, len(query_words) / 5)  # 5个关键词为基准
            min_docs = max(2, int(top_k * (1 - query_complexity)))  # 简单查询返回较少文档
            max_docs = min(8, int(top_k * (1 + query_complexity)))  # 复杂查询返回较多文档
            
            # 如果分数差异显著，返回较少文档
            if len(filtered_results) > 1 and filtered_results[0][1] - filtered_results[1][1] > 0.3:
                final_results = filtered_results[:min_docs]
            else:
                final_results = filtered_results[:max_docs]
            
            logger.info(f"查询复杂度: {query_complexity:.2f}, 返回文档数: {len(final_results)}")
            
            # 构建最终结果
            return [{
                "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                "score": float(score),
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            } for doc, score in final_results]
            
        except Exception as e:
            logger.error(f"获取相关文档失败: {str(e)}")
            return []

    def _dedup_and_rank_results(self, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """对检索结果进行去重和排序
        
        Args:
            results: 检索结果列表
            k: 返回的文档数量
            
        Returns:
            List[Dict[str, Any]]: 去重和排序后的结果
        """
        # 使用内容作为key进行去重
        unique_results = {}
        for result in results:
            content = result["content"]
            score = result["score"]
            
            # 如果已存在该内容，保留分数更高的
            if content in unique_results:
                if score > unique_results[content]["score"]:
                    unique_results[content] = result
            else:
                unique_results[content] = result
                
        # 转换回列表并按分数排序
        deduped_results = list(unique_results.values())
        sorted_results = sorted(deduped_results, key=lambda x: x["score"], reverse=True)
        
        # 只返回前k个结果
        return sorted_results[:k]
    
    def _init_reranker(self, name: str, config: Dict[str, Any]) -> Any:
        """初始化重排序器"""
        try:
            model_path = config.get("model_path", "")
            if "bge" in name:
                from src.retriever.reranker import BGEReranker
                return BGEReranker({"model_path": model_path, "device": self.device})
            elif "bce" in name:
                from src.retriever.reranker import BCEReranker
                return BCEReranker({"model_path": model_path, "device": self.device})
            else:
                logger.warning(f"未知的重排序器类型: {name}")
                return None
                
        except Exception as e:
            logger.error(f"初始化重排序器 {name} 时出错: {str(e)}")
            return None
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """添加文档到检索器
        
        Args:
            documents: 文档列表
        """
        try:
            # 为每个检索器添加文档
            for retriever in self.retrievers.values():
                try:
                    retriever.add_documents(documents)
                except Exception as e:
                    logger.warning(f"向检索器添加文档失败: {str(e)}")
            
            # 更新FAISS索引
            self._update_faiss_index(documents)
            
        except Exception as e:
            logger.error(f"添加文档时出错: {str(e)}", exc_info=True)
            raise
    
    def _update_faiss_index(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """更新FAISS索引
        
        Args:
            documents: 文档列表
        """
        try:
            # 生成文档向量
            vectors = []
            for doc in documents:
                try:
                    # 使用模型生成文档向量
                    inputs = self.tokenizer(
                        doc["content"],
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(self.model.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # 使用最后一层的[CLS]向量作为文档向量
                        vector = outputs.last_hidden_state[0, 0].cpu().numpy()
                        vectors.append(vector)
                    
                    # 存储文档
                    doc_id = len(self.doc_store)
                    self.doc_store[doc_id] = doc
                                
                except Exception as e:
                    logger.warning(f"处理文档向量失败: {str(e)}")
                    continue
                    
            if vectors:
                vectors = np.array(vectors).astype('float32')
                
                # 初始化或更新索引
                if self.index is None:
                    dimension = vectors.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                
                self.index.add(vectors)
            
        except Exception as e:
            logger.error(f"更新FAISS索引时出错: {str(e)}")
            raise
    
    def search_similar_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """使用FAISS搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相似文档列表
        """
        try:
            if self.index is None:
                return []
            
            # 生成查询向量
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                query_vector = outputs.last_hidden_state[0, 0].cpu().numpy()
            
            # 搜索相似文档
            distances, indices = self.index.search(
                query_vector.reshape(1, -1).astype('float32'),
                top_k
            )
            
            # 构建结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.doc_store):
                    doc = self.doc_store[idx].copy()
                    doc["score"] = float(1 / (1 + distances[0][i]))
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索相似文档时出错: {str(e)}", exc_info=True)
            return []
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        
        # 更新检索器配置
        if "retrievers" in new_config:
            for name, retriever_config in new_config["retrievers"].items():
                if name in self.retrievers:
                    self.retrievers[name].update_config(retriever_config)
                else:
                    self.retrievers[name] = self._init_retriever(name, retriever_config)
        
        # 更新重排序器配置
        if "rerankers" in new_config:
            for name, reranker_config in new_config["rerankers"].items():
                if name in self.rerankers:
                    self.rerankers[name].update_config(reranker_config)
                else:
                    self.rerankers[name] = self._init_reranker(name, reranker_config)
        
        self.logger.info("多检索器配置已更新")

    def _compute_relevance_score(self, query: str, doc) -> float:
        """计算文档与查询的相关性分数,考虑位置信息和章节关系
        
        Args:
            query: 查询文本
            doc: 文档对象
            
        Returns:
            float: 相关性分数
        """
        try:
            # 获取文档内容
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            
            # 1. 计算基础相关性分数
            # 1.1 关键词匹配度
            query_words = set(jieba.cut(query)) - stopwords
            doc_words = set(jieba.cut(content)) - stopwords
            
            # 计算专业术语匹配分数
            term_score = 0.0
            term_count = 0
            for key, synonyms in synonym_mappings.items():
                all_terms = set([key] + synonyms)
                query_terms = query_words & all_terms
                doc_terms = doc_words & all_terms
                if query_terms and doc_terms:
                    term_score += 1.0
                    term_count += 1
            
            term_score = term_score / max(1, term_count) if term_count > 0 else 0
            
            # 1.2 计算一般关键词匹配度
            keyword_overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
            
            # 1.3 计算语义相似度
            semantic_score = 0.0
            for reranker in self.rerankers.values():
                try:
                    semantic_score = max(semantic_score, reranker.compute_score(query, content))
                except Exception as e:
                    logger.warning(f"计算语义相似度失败: {str(e)}")
            
            # 2. 计算位置权重
            position_weight = self._get_position_weight(doc)
            
            # 3. 计算章节关系权重
            section_weight = self._get_section_weight(doc, query)
            
            # 4. 计算最终分数 (0.4 * 专业术语 + 0.2 * 一般关键词 + 0.2 * 语义 + 0.1 * 位置 + 0.1 * 章节)
            final_score = (
                0.4 * term_score +
                0.2 * keyword_overlap +
                0.2 * semantic_score +
                0.1 * position_weight +
                0.1 * section_weight
            )
            
            return float(final_score)
            
        except Exception as e:
            logger.warning(f"计算相关性分数失败: {str(e)}")
            return 0.0

    def _get_position_weight(self, doc: Union[Dict, str]) -> float:
        """计算文档的位置权重
        
        Args:
            doc: 文档对象
            
        Returns:
            float: 位置权重 (0-1)
        """
        try:
            # 获取文档元数据
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            
            # 获取文档在原文中的位置信息
            page_num = metadata.get("page", 0)
            position = metadata.get("position", 0.5)  # 默认在中间
            
            # 1. 根据页码计算权重 (假设越靠前的页面权重越大)
            page_weight = 1.0 / (1 + page_num)
            
            # 2. 根据页内位置计算权重 (U型分布,开头和结尾权重大)
            position_weight = 1.0 - 4 * (position - 0.5) ** 2
            
            # 3. 合并权重
            return (page_weight + position_weight) / 2
            
        except Exception as e:
            logger.warning(f"计算位置权重失败: {str(e)}")
            return 0.5
            
    def _get_section_weight(self, doc: Union[Dict, str], query: str) -> float:
        """计算文档的章节关系权重
        
        Args:
            doc: 文档对象
            query: 查询文本
            
        Returns:
            float: 章节权重 (0-1)
        """
        try:
            # 获取文档内容和元数据
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            
            # 1. 提取章节标题
            section_title = metadata.get("section_title", "")
            if not section_title and "：" in content:
                section_title = content.split("：")[0]
            
            if not section_title:
                return 0.5  # 默认权重
            
            # 2. 计算标题与查询的相关度
            title_words = set(jieba.cut(section_title)) - stopwords
            query_words = set(jieba.cut(query)) - stopwords
            
            # 2.1 直接匹配
            overlap_score = len(title_words & query_words) / len(query_words) if query_words else 0
            
            # 2.2 同义词匹配
            synonym_score = 0.0
            for key, synonyms in synonym_mappings.items():
                all_terms = set([key] + synonyms)
                if (title_words & all_terms) and (query_words & all_terms):
                    synonym_score += 1.0
            
            # 3. 合并分数
            return max(overlap_score, min(1.0, synonym_score / 2))
            
        except Exception as e:
            logger.warning(f"计算章节权重失败: {str(e)}")
            return 0.5

    def _compute_quality_score(self, doc) -> float:
        """计算文档质量分数"""
        try:
            # 获取文档内容
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            
            # 1. 计算文档长度分数
            length = len(content)
            length_score = min(1.0, length / 1000)  # 标准化到1000字符
            
            # 2. 计算结构完整性分数
            structure_score = 0.8  # 默认分数
            if isinstance(doc, dict):
                # 检查是否包含必要字段
                required_fields = ["content", "metadata"]
                structure_score = sum(1 for field in required_fields if field in doc) / len(required_fields)
            
            # 3. 计算内容连贯性分数
            coherence_score = 0.5  # 默认值
            sentences = re.findall(r'[^。！？]+[。！？]', content)
            if sentences:
                # 检查句子完整性
                complete_sentences = sum(1 for s in sentences if len(s) >= 10)
                coherence_score = min(1.0, complete_sentences / 3)  # 至少3个完整句子得满分
            
            # 4. 计算最终质量分数 (0.4 * 长度 + 0.3 * 结构 + 0.3 * 连贯性)
            quality_score = 0.4 * length_score + 0.3 * structure_score + 0.3 * coherence_score
            return float(quality_score)
            
        except Exception as e:
            logger.warning(f"计算质量分数失败: {str(e)}")
            return 0.0

    def _get_retriever_weight(self, retriever_name: str) -> float:
        """获取检索器权重"""
        # 可以根据经验或动态调整设置不同检索器的权重
        weights = {
            "bge": 0.35,
            "bce": 0.35,
            "keyword": 0.15,
            "hybrid": 0.15
        }
        return weights.get(retriever_name.lower(), 0.25)
        
    def _compute_dynamic_threshold(self, scored_docs: list) -> float:
        """计算动态阈值"""
        try:
            if not scored_docs:
                return 0.0
            
            # 获取所有分数
            scores = [score for _, score in scored_docs]
            
            if len(scores) < 2:
                return 0.0
                
            # 计算平均分和标准差
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # 计算最高分和次高分的差值
            top_diff = scores[0] - scores[1]
            
            # 如果最高分显著高于次高分，提高阈值
            if top_diff > 0.3:
                threshold = mean_score
            else:
                # 否则使用较低阈值
                threshold = mean_score - 0.8 * std_score
            
            # 确保阈值在合理范围内
            return float(max(0.3, min(0.8, threshold)))
            
        except Exception as e:
            logger.warning(f"计算动态阈值失败: {str(e)}")
            return 0.0
        
    def _rerank_with_relationships(self, docs: list) -> list:
        """基于文档关系重新排序
        
        Args:
            docs: 文档列表
            
        Returns:
            list: 重新排序后的文档列表
        """
        try:
            if not docs:
                return []
            
            # 1. 构建文档关系图
            doc_graph = defaultdict(list)
            for i, doc1 in enumerate(docs):
                content1 = doc1.get("content", "") if isinstance(doc1, dict) else str(doc1)
                for j, doc2 in enumerate(docs):
                    if i != j:
                        content2 = doc2.get("content", "") if isinstance(doc2, dict) else str(doc2)
                        # 计算文档相似度
                        similarity = self._compute_doc_similarity(content1, content2)
                        if similarity > 0.5:  # 相似度阈值
                            doc_graph[i].append((j, similarity))
            
            # 2. 计算PageRank分数
            scores = self._compute_pagerank(doc_graph, len(docs))
            
            # 3. 重新排序
            doc_scores = list(zip(docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, _ in doc_scores]
            
        except Exception as e:
            logger.warning(f"文档重排序失败: {str(e)}")
            return docs
            
    def _compute_doc_similarity(self, content1: str, content2: str) -> float:
        """计算两个文档的相似度
        
        Args:
            content1: 第一个文档的内容
            content2: 第二个文档的内容
            
        Returns:
            float: 相似度分数
        """
        try:
            # 使用词袋模型计算相似度
            words1 = set(jieba.cut(content1)) - stopwords
            words2 = set(jieba.cut(content2)) - stopwords
            
            if not words1 or not words2:
                return 0.0
            
            # 计算Jaccard相似度
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"计算文档相似度失败: {str(e)}")
            return 0.0

    def _compute_pagerank(self, graph: Dict[int, List[Tuple[int, float]]], n: int, 
                         d: float = 0.85, max_iter: int = 20) -> List[float]:
        """计算PageRank分数
        
        Args:
            graph: 文档关系图
            n: 文档数量
            d: 阻尼系数
            max_iter: 最大迭代次数
            
        Returns:
            List[float]: PageRank分数列表
        """
        try:
            # 初始化分数
            scores = [1.0 / n] * n
            
            # 迭代计算
            for _ in range(max_iter):
                new_scores = [(1 - d) / n] * n
                
                for i in range(n):
                    if i in graph:
                        for j, weight in graph[i]:
                            new_scores[j] += d * scores[i] * weight
                            
                # 归一化
                total = sum(new_scores)
                if total > 0:
                    scores = [s / total for s in new_scores]
                else:
                    scores = new_scores
                
            return scores
            
        except Exception as e:
            logger.warning(f"计算PageRank分数失败: {str(e)}")
            return [1.0 / n] * n 