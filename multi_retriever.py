from langchain_core.documents import Document
from tfidf_retriever import TFIDFRetriever
from bge_retriever import BGERetriever
from gte_retriever import GTERetriever
from bce_retriever import BCERetriever
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from rerank_model import reRankLLM
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

class MultiRetriever:
    def __init__(self, data, model_paths):
        self.retrievers = {
            "tfidf": TFIDFRetriever(data),
            "faiss": FaissRetriever(model_paths["m3e"], data),
            "bge": BGERetriever(model_paths["bge"], data),
            "gte": GTERetriever(model_paths["gte"], data),
            "bce": BCERetriever(model_paths["bce"], data),
            "bm25": BM25(data)
        }
        
        self.weights = {
            "tfidf": 0.6,
            "faiss": 1.2,
            "bge": 1.5,
            "gte": 1.2,
            "bce": 1.2,
            "bm25": 1.3
        }
        
    def normalize_scores(self, results, min_score=0.0, max_score=1.0):
        if not results:
            return results
        
        scores = [score for _, score in results]
        min_val = min(scores)
        max_val = max(scores)
        
        if max_val == min_val:
            return [(doc, max_score) for doc, _ in results]
            
        normalized_results = []
        for doc, score in results:
            normalized_score = (score - min_val) / (max_val - min_val)
            normalized_score = normalized_score * (max_score - min_score) + min_score
            normalized_results.append((doc, normalized_score))
        return normalized_results
        
    def get_merged_results(self, query, k=15):
        all_results = []
        
        for name, retriever in self.retrievers.items():
            try:
                if name == "bm25":
                    results = [(Document(page_content=doc.page_content), 1.0) 
                              for doc in retriever.GetBM25TopK(query, k)]
                else:
                    results = retriever.GetTopK(query, k)
                
                results = self.normalize_scores(results)
                weight = self.weights[name]
                weighted_results = [(doc, score * weight) for doc, score in results]
                all_results.extend(weighted_results)
            except Exception as e:
                print(f"Error in retriever {name}: {str(e)}")
                continue
            
        unique_results = self.dedup_results(all_results)
        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results[:k]
    
    def dedup_results(self, results):
        seen = set()
        unique_results = []
        for doc, score in results:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique_results.append((doc, score))
        return unique_results

class MultiReranker:
    def __init__(self, model_paths):
        self.rerankers = {
            "bge": reRankLLM(model_paths["bge_reranker"]),
            "bce": reRankLLM(model_paths["bce_reranker"])
        }
        
        self.weights = {
            "bge": 1.5,
            "bce": 1.2
        }
    
    def normalize_scores(self, scores, min_score=0.0, max_score=1.0):
        if not scores:
            return scores
        
        min_val = min(scores)
        max_val = max(scores)
        
        if max_val == min_val:
            return [max_score] * len(scores)
            
        normalized = []
        for score in scores:
            norm_score = (score - min_val) / (max_val - min_val)
            norm_score = norm_score * (max_score - min_score) + min_score
            normalized.append(norm_score)
        return normalized
    
    def rerank(self, query, docs, top_k=6):
        doc_scores = {}
        
        try:
            for name, reranker in self.rerankers.items():
                ranked_docs = reranker.predict(query, docs)
                n_docs = len(ranked_docs)
                for rank, doc in enumerate(ranked_docs):
                    if doc.page_content not in doc_scores:
                        doc_scores[doc.page_content] = []
                    score = np.exp(-rank / (n_docs + 1e-6))
                    length_penalty = min(1.0, 200 / (len(doc.page_content) + 1e-6))
                    weighted_score = score * self.weights[name] * length_penalty
                    doc_scores[doc.page_content].append(weighted_score)
        
        except Exception as e:
            print(f"Error in reranking: {str(e)}")
            return docs[:top_k]
        
        final_docs = []
        for doc in docs:
            if doc.page_content in doc_scores:
                scores = doc_scores[doc.page_content]
                if scores:
                    avg_score = np.exp(np.mean(np.log(np.array(scores) + 1e-6))) - 1e-6
                    final_docs.append((doc, avg_score))
        
        if final_docs:
            scores = [score for _, score in final_docs]
            normalized_scores = self.normalize_scores(scores)
            final_docs = [(doc, score) for (doc, _), score in zip(final_docs, normalized_scores)]
        
        final_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in final_docs[:top_k]] 