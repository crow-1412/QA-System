from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document

class TFIDFRetriever:
    def __init__(self, data):
        """初始化TF-IDF检索器
        
        Args:
            data: 包含Document对象的列表，每个Document对象应该有page_content属性
        """
        self.vectorizer = TfidfVectorizer()
        self.documents = [doc.page_content for doc in data]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.docs = data
        
    def GetTopK(self, query, k=10):
        """检索与查询最相关的k个文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            list: 包含(Document, score)元组的列表，按相关性得分降序排序
        """
        query_vec = self.vectorizer.transform([query])
        scores = (query_vec * self.tfidf_matrix.T).toarray()[0]
        top_k_indices = scores.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append((self.docs[idx], scores[idx]))
        return results 