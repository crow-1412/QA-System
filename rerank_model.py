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
        print(f"\nInitializing reRankLLM from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            
            # 检查CUDA是否可用
            if torch.cuda.is_available():
                print(f"Using CUDA device: {CUDA_DEVICE}")
                try:
                    self.model.to(CUDA_DEVICE)
                    self.model.half()  # 只在成功移动到GPU后才使用半精度
                except Exception as e:
                    print(f"Error moving model to {CUDA_DEVICE}, falling back to CPU: {str(e)}")
                    self.model.to('cpu')
            else:
                print("CUDA not available, using CPU")
                self.model.to('cpu')
                
            self.max_length = max_length
            print("reRankLLM initialized successfully")
        except Exception as e:
            print(f"Error initializing reRankLLM: {str(e)}")
            raise

    def predict(self, query, docs):
        """
        对输入的query和文档列表进行相关性重排序
        
        Args:
            query (str): 用户输入的查询文本
            docs (List): 待重排序的文档列表,每个文档包含page_content属性
            
        Returns:
            List: 按相关性得分从高到低排序后的文档列表
        """
        try:
            print(f"\nReranking {len(docs)} documents...")
            
            # 将query和每个文档组成(query, doc)对
            pairs = [(query, doc.page_content) for doc in docs]
            print("Created query-document pairs")
            
            # 分批处理，避免显存溢出
            batch_size = 32
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(pairs)-1)//batch_size + 1}")
                
                # 使用tokenizer对文本对进行编码
                inputs = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=self.max_length
                )
                
                # 将输入移动到正确的设备
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # 计算得分
                with torch.no_grad():
                    batch_scores = self.model(**inputs).logits
                    all_scores.extend(batch_scores.detach().cpu().numpy())
            
            # 根据得分排序
            response = [doc for score, doc in sorted(zip(all_scores, docs), reverse=True, key=lambda x:x[0])]
            print("Reranking completed")
            
            # 清理GPU显存
            if torch.cuda.is_available():
                torch_gc()
            
            return response
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            return docs  # 如果重排序失败，返回原始文档列表

if __name__ == "__main__":
    bge_reranker_large = "./pre_train_model/bge-reranker-large"
    rerank = reRankLLM(bge_reranker_large)
