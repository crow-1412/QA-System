import logging
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel

class KnowledgeBase:
    """知识处理基础类，提供共同的功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化基础类
        
        Args:
            config: 配置字典，包含模型路径等信息
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 加载模型和分词器
        self._init_model()
        
    def _init_model(self):
        """初始化模型和分词器"""
        try:
            model_path = self.config.get("model_path", "THUDM/chatglm3-6b")
            device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
            
            self.logger.info(f"{self.__class__.__name__}模型初始化完成")
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}", exc_info=True)
            raise
            
    def _generate_response(self, prompt: str) -> str:
        """使用模型生成响应
        
        Args:
            prompt: 提示文本
            
        Returns:
            模型生成的响应
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.get("max_length", 2048),
                num_return_sequences=1,
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                repetition_penalty=self.config.get("repetition_penalty", 1.1)
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"生成响应时出错: {str(e)}", exc_info=True)
            raise
            
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        self.logger.info(f"{self.__class__.__name__}配置已更新") 