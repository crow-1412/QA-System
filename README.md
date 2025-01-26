# QA-System

一个基于深度学习的智能问答系统。

## 环境要求

- Python 3.8+
- 所需的Python包已在 `requirements.txt` 中列出

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/crow-1412/QA-System.git
cd QA-System
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行测试

运行单个问题测试：
```bash
python tests/test_single_question.py
```

## 项目结构

```
QA-System/
├── src/                # 源代码目录
│   ├── knowledge/      # 知识库相关代码
│   └── retriever/      # 检索器相关代码
├── tests/              # 测试文件
├── data/               # 数据文件
└── requirements.txt    # 项目依赖
```

## 注意事项

- 请确保在运行代码前已安装所有必要的依赖
- 数据文件需要单独配置

# 智能问答系统

基于大语言模型的智能问答系统，支持多种检索策略和知识组织方式。

## 功能特点

- 多层次文档检索
- 智能问题分析和重写
- 分层知识组织
- 知识图谱维护
- 多路径答案生成
- 答案质量评估

## 系统要求

- Python 3.8+
- CUDA 11.6+ (如果使用GPU)
- 16GB+ RAM
- 8GB+ GPU内存 (如果使用GPU)

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/QA-System.git
cd QA-System
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载模型：
```bash
# 下载ChatGLM3模型
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('THUDM/chatglm3-6b'); AutoModel.from_pretrained('THUDM/chatglm3-6b')"

# 下载BGE模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh')"

# 下载BCE模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('maidalun1020/bce-embedding-base_v1')"
```

## 使用方法

系统提供两种使用方式：

### 1. 组件化方式（推荐）

使用 `run.py` 启动系统，这种方式提供了更灵活的组件配置和更好的性能：

```bash
python run.py --model_path /path/to/model --device cuda
```

主要特点：
- 支持多种文档解析方法
- 灵活的组件配置
- 更好的性能优化
- 实时处理反馈

### 2. 集成式方式

使用 `qa_pipeline.py` 的集成实现，主要用于测试和性能比较：

```python
from src.qa_pipeline import QAPipeline
from src.qa_config import QAConfig

# 初始化系统
config = QAConfig.get_config()
qa_system = QAPipeline(config)

# 处理问题
result = qa_system.answer_question("您的问题")
```