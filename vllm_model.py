import os
import torch
import time

from config import *
from vllm import LLM, SamplingParams

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 设置模型路径
MODEL_PATH = "/root/autodl-tmp/pre_train_model/Qwen-7B-Chat"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

IMEND = "<|im_end|>" # 消息结束符
ENDOFTEXT = "<|endoftext|>" # 文本结束符

# 获取stop token的id
def get_stop_words_ids(chat_format, tokenizer):
    """
    根据不同的对话格式获取对应的停止词token ID列表
    
    Args:
        chat_format (str): 对话格式,支持"raw"和"chatml"两种格式
            - raw: 原始格式,使用"Human:"作为停止词
            - chatml: ChatML格式,使用im_end和im_start作为停止词
        tokenizer (AutoTokenizer): 分词器对象,用于获取token的ID
        eod_id是文本结束符，是End of Document的缩写
        im_start_id是消息开始符，是Message Start的缩写
        im_end_id是消息结束符，是Message End的缩写
        
    Returns:
        list: 停止词token ID的嵌套列表,每个停止词对应一个子列表
            - raw格式: [["Human:"的token ID], [eod_id]]
            - chatml格式: [[im_end_id], [im_start_id]]
            
    Raises:
        NotImplementedError: 当传入未支持的chat_format时抛出异常
    """
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids

# 释放gpu显存
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class ChatLLM(object):

    def __init__(self, model_path):
        """
        初始化ChatLLM类
        
        Args:
            model_path (str): 预训练模型的路径
        """
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            pad_token='<|extra_0|>',     # 设置填充token
            eos_token='<|endoftext|>',   # 设置结束token  
            padding_side='left',         # 设置在左侧进行填充
            trust_remote_code=True       # 信任远程代码
        )
        
        # 加载生成配置
        self.generation_config = GenerationConfig.from_pretrained(
            model_path, 
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 将生成配置的结束token ID同步到tokenizer
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id
        
        # 初始化停止词ID列表
        self.stop_words_ids = []
        
        # 加载vLLM大模型
        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=2,     # 保持双卡并行
            trust_remote_code=True,
            gpu_memory_utilization=0.8, # 降低内存使用率
            dtype="bfloat16",
            max_num_seqs=4,            # 限制并行序列数
            max_num_batched_tokens=8192 # 修改为与max_model_len相同
        )
        
        # 设置停止词
        for stop_id in get_stop_words_ids(self.generation_config.chat_format, self.tokenizer):
            self.stop_words_ids.extend(stop_id)
        self.stop_words_ids.extend([self.generation_config.eos_token_id])

        # 修改采样参数
        sampling_kwargs = {
            "stop_token_ids": self.stop_words_ids,
            "top_p": 1.0,
            "top_k": 50,  # 修改为正数，避免 ValueError
            "temperature": 0.7,  # 增加一些随机性
            "max_tokens": 2000,
            "repetition_penalty": self.generation_config.repetition_penalty,
            "n": 1,
            "best_of": 1  # 确保 best_of 等于 n
        }
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def infer(self, prompts):
        """批量推理，输入一个batch，返回一个batch的答案"""
        try:
            batch_text = []
            for q in prompts:
                raw_text, _ = make_context(
                    self.tokenizer,
                    q,
                    system="You are a helpful assistant.",
                    max_window_size=self.generation_config.max_window_size,
                    chat_format=self.generation_config.chat_format,
                )
                batch_text.append(raw_text)
                
            outputs = self.model.generate(
                batch_text,
                sampling_params=self.sampling_params
            )
            
            batch_response = []
            for output in outputs:
                output_str = output.outputs[0].text
                if IMEND in output_str:
                    output_str = output_str[:-len(IMEND)]
                if ENDOFTEXT in output_str:
                    output_str = output_str[:-len(ENDOFTEXT)]
                batch_response.append(output_str)
                
            # 清理内存
            torch.cuda.empty_cache()
            return batch_response
        except Exception as e:
            print(f"Error in infer: {str(e)}")
            torch.cuda.empty_cache()
            return []

    def GetTopK(self, query, k):
        """获取top-K分数最高的文档块"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # 确保返回正确的格式
            context = []
            for doc, score in results:
                if hasattr(doc, 'page_content'):
                    context.append((doc.page_content, float(score)))
                else:
                    context.append((str(doc), float(score)))
            return context
        except Exception as e:
            print(f"Error in GetTopK: {str(e)}")
            return []

if __name__ == "__main__":
    qwen7 = "/root/autodl-tmp/pre_train_model/Qwen-7B-Chat"
    start = time.time()
    llm = ChatLLM(qwen7)
    test = ["吉利汽车座椅按摩","吉利汽车语音组手唤醒","自动驾驶功能介绍"]
    generated_text = llm.infer(test)
    print(generated_text)
    end = time.time()
    print("cost time: " + str((end-start)/60))
