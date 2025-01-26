# QA-System

一个基于深度学习的智能问答系统，支持多种检索策略和知识组织方式。

## 系统特点

- 多层次文档检索：支持多种检索策略的组合
- 智能问题分析和重写：自动分析问题类型并进行问题重写
- 分层知识组织：采用多层次的知识组织结构
- 答案质量评估：多维度评估答案质量
- 答案优化迭代：通过多轮迭代优化答案质量

## 核心模块说明

### 1. 知识优化器 (knowledge_refiner.py)

核心类 `KnowledgeRefiner` 实现了以下功能：

#### 答案迭代优化机制

系统采用多阶段迭代优化策略来提升答案质量，具体包含以下核心环节：

##### 1. 直接答案提取
- 优先在文档中寻找精确匹配的答案段落
- 通过语义验证确保答案的准确性
- 避免不必要的生成开销

##### 2. 初始答案生成
- 基于相关度筛选（阈值>0.2）的高质量文档
- 使用严格约束的prompt模板
- 确保生成内容与文档一致
- 应用质量验证过滤机制

##### 3. 多轮迭代优化
- 渐进式文档优化策略
  - 按文档相关度排序逐个优化
  - 仅使用相关度高的文档段落
  - 保持答案与文档的语义一致性

- 双重验证机制
  - 语义一致性验证（verify_answer_quality）
  - 文档相似度检查（阈值>=0.3）
  - 质量评估（evaluate_answer_quality）

- 动态优化策略
  - 仅在显著提升时更新答案
  - 保留历史最优答案
  - 支持多轮优化迭代

##### 4. 质量评估体系
- 多维度评分指标
  - 完整性评分：检查必要信息覆盖
  - 相关性评分：计算问题相关度
  - 清晰度评分：评估表达质量
  - 事实一致性：验证与文档一致性
  - 文档相似度：计算参考相似度

- 动态权重调整
  - 基于问题类型的权重分配
  - 不同维度的自适应调整
  - 考虑问题特征的评分偏好

##### 5. 优化终止机制
- 质量阈值检查
  - 达到预设质量目标
  - 连续多轮无显著提升
  - 最大迭代次数限制

- 降级处理策略
  - 异常情况的优雅降级
  - 保留最近有效答案
  - 详细的错误日志记录

- 问题分析与重写
  - 问题类型识别
  - 关键词提取
  - 问题重写优化

- 文档检索与排序
  - 多策略检索
  - 文档相关性评分
  - 动态文档重排序

- 答案生成与优化
  - 基于上下文的答案生成
  - 答案质量评估
  - 迭代优化机制

- 质量评估体系
  - 完整性评估
  - 相关性评估
  - 清晰度评估
  - 事实一致性检查

### 2. 数据处理器 (data_process.py)

核心类 `DataProcess` 实现了以下功能：

- PDF文档解析
  - 多种解析策略结合
  - 智能内容区域识别
  - 字体大小分析

- 文本块处理
  - 滑动窗口处理
  - 自适应分段
  - 噪声过滤

- 文本清理
  - 目录识别与过滤
  - 页码处理
  - 特殊字符处理

## 环境要求

- Python 3.8+
- CUDA 11.6+ (如果使用GPU)
- 16GB+ RAM
- 8GB+ GPU内存 (如果使用GPU)

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
│   ├── knowledge/      # 知识处理相关代码
│   │   └── knowledge_refiner.py  # 知识优化器
│   ├── retriever/      # 检索器相关代码
│   └── data_process.py # 数据处理模块
├── tests/              # 测试文件
│   └── test_single_question.py  # 单问题测试
├── data/               # 数据文件
└── requirements.txt    # 项目依赖
```

## 注意事项

- 请确保在运行代码前已安装所有必要的依赖
- 数据文件需要单独配置
- 首次运行时会自动下载必要的模型文件

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

主要特点：
- 支持多种文档解析方法
- 灵活的组件配置
- 更好的性能优化
- 实时处理反馈
