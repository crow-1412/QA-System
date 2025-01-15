from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

from bm25_retriever import BM25
from pdf_parse import DataProcess
from config import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "1"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# 释放gpu上没有用到的显存以及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# 加载rerank模型
class reRankLLM(object):
    def __init__(self, model_path, max_length = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()
        self.model.to(CUDA_DEVICE)
        self.max_length = max_length

    # 输入文档对，返回每一对(query, doc)的相关得分，并从大到小排序
    def predict(self, query, docs):
        """
        对输入的query和文档列表进行相关性重排序
        
        Args:
            query (str): 用户输入的查询文本
            docs (List): 待重排序的文档列表,每个文档包含page_content属性
            
        Returns:
            List: 按相关性得分从高到低排序后的文档列表
        """
        # 将query和每个文档组成(query, doc)对
        pairs = [(query, doc.page_content) for doc in docs]
        
        # 使用tokenizer对文本对进行编码,获得模型输入
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length).to(CUDA_DEVICE)
        
        # 使用模型计算相关性得分,关闭梯度计算
        with torch.no_grad():
            scores = self.model(**inputs).logits
            
        # 将得分从GPU转移到CPU并转换为numpy数组
        scores = scores.detach().cpu().clone().numpy()
        
        # 根据得分对文档进行排序,得分高的排在前面
        response = [doc for score, doc in sorted(zip(scores, docs), reverse=True, key=lambda x:x[0])]
        
        # 清理GPU显存
        torch_gc()
        
        return response

if __name__ == "__main__":
    bge_reranker_large = "./pre_train_model/bge-reranker-large"
    rerank = reRankLLM(bge_reranker_large)
