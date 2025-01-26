import pytest
import re
from src.knowledge.knowledge_refiner import KnowledgeRefiner
from src.knowledge.knowledge_base import KnowledgeBase

class TestKnowledgeRefiner:
    """测试知识精炼器的上下文过滤功能"""
    
    @pytest.fixture
    def refiner(self):
        """创建一个不加载模型的KnowledgeRefiner实例"""
        class MockKnowledgeBase(KnowledgeBase):
            def __init__(self):
                self.config = {}
                self.logger = None
                self._init_model()
                
            def _init_model(self):
                pass
                
        class MockKnowledgeRefiner(KnowledgeRefiner):
            def __init__(self):
                self.config = {}
                self.logger = None
                self._init_model()
                
            def _init_model(self):
                pass
                
        return MockKnowledgeRefiner()
    
    def test_is_context_boundary(self, refiner):
        """测试边界检测功能"""
        # 测试各种边界模式
        assert refiner._is_context_boundary("第一章 空调系统")
        assert refiner._is_context_boundary("1.5 功能说明")
        assert refiner._is_context_boundary("【系统设置】")
        assert refiner._is_context_boundary("（三）操作说明")
        assert not refiner._is_context_boundary("普通的句子。")
    
    def test_calculate_similarity(self, refiner):
        """测试相似度计算功能"""
        question = "如何使用前除霜功能？"
        relevant_text = "前除霜按钮可以快速除去前挡风玻璃上的雾气"
        irrelevant_text = "后备箱开关位于驾驶员侧车门"
        
        relevant_score = refiner._calculate_similarity(relevant_text, question)
        irrelevant_score = refiner._calculate_similarity(irrelevant_text, question)
        
        assert relevant_score >= 0.2
        assert irrelevant_score < 0.2
        assert relevant_score > irrelevant_score
    
    def test_check_semantic_relevance(self, refiner):
        """测试语义相关性检查功能"""
        # 测试前除霜相关
        question = "如何使用前除霜功能？"
        relevant_text = "前除霜按钮可快速除去前挡风玻璃上的雾气"
        irrelevant_text = "后窗除霜功能可快速除去后窗玻璃上的雾气"
        
        assert refiner._check_semantic_relevance(relevant_text, question)
        assert not refiner._check_semantic_relevance(irrelevant_text, question)
        
        # 测试后除霜相关
        question = "如何使用后除霜功能？"
        relevant_text = "后除霜按钮可快速除去后挡风玻璃上的雾气"
        irrelevant_text = "前除霜功能可快速除去前挡风玻璃上的雾气"
        
        assert refiner._check_semantic_relevance(relevant_text, question)
        assert not refiner._check_semantic_relevance(irrelevant_text, question)
    
    def test_remove_similar_sentences(self, refiner):
        """测试相似句子去重功能"""
        sentences = [
            "前除霜按钮可快速除去前挡风玻璃上的雾气",
            "前除霜按钮能快速清除前挡风玻璃上的雾气",  # 相似句子
            "按下前除霜按钮时A/C会自动开启",  # 不相似句子
        ]
        
        question = "如何使用前除霜功能？"
        unique_sentences = refiner._remove_similar_sentences(sentences, question)
        
        assert len(unique_sentences) == 2  # 应该只保留两个不相似的句子
        assert any("A/C会自动开启" in sent for sent in unique_sentences)
    
    def test_clean_response(self, refiner):
        """测试答案清理功能"""
        response = """
        参考文档：前除霜功能说明
        1. 前除霜按钮可快速除去前挡风玻璃上的雾气。
        2. 按下前除霜按钮时A/C会自动开启。
        {content: 更多信息请参考使用手册}
        """
        
        question = "如何使用前除霜功能？"
        cleaned = refiner._clean_response(response, question)
        
        assert "参考文档" not in cleaned
        assert "{content:" not in cleaned
        assert "1." not in cleaned
        assert "2." not in cleaned
        assert "前除霜按钮" in cleaned
        assert "A/C会自动开启" in cleaned 