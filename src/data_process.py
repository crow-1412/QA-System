import os
import fitz  # PyMuPDF
import re
from typing import List, Union, Dict, Tuple
import pdfplumber
from PyPDF2 import PdfReader
from collections import Counter

class DataProcess:
    """数据处理类，用于解析PDF文件和文本块
    """
    
    def __init__(self):
        """初始化数据处理器"""
        self.data = []
        self.headers = {}  # 存储页眉信息
        self.content_area = None  # 缓存内容区域检测结果
        
    def process_pdf(self, pdf_path: str, max_seq: int = 512) -> List[str]:
        """处理PDF文件，使用多种方法解析并合并结果
        
        Args:
            pdf_path: PDF文件路径
            max_seq: 最大序列长度
            
        Returns:
            List[str]: 处理后的文本块列表
        """
        try:
            self.pdf_path = pdf_path
            print(f"\n使用多种方法解析PDF: {pdf_path}")
            
            # 1. 使用PyMuPDF解析
            print("\n1. 使用PyMuPDF解析...")
            doc = fitz.open(pdf_path)
            
            # 只在第一次检测内容区域
            if self.content_area is None:
                self.content_area = self._get_content_area(doc)
            print(f"检测到内容区域边界: 左={self.content_area[0]:.2f}, 右={self.content_area[1]:.2f}, 上={self.content_area[2]:.2f}, 下={self.content_area[3]:.2f}")
            
            # 获取正文字体大小基准
            base_font_size, min_ratio, max_ratio = self._get_base_font_size(doc)
            print(f"检测到正文字体大小: {base_font_size}")
            print(f"正文字体范围: ({base_font_size * min_ratio:.4f}, {base_font_size * max_ratio:.4f})")
            
            mupdf_blocks = []
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    # 使用缓存的内容区域进行过滤
                    if not self._is_in_middle_area(block, page.rect, self.content_area):
                        continue
                        
                    if "lines" not in block:
                        continue
                        
                    block_text = ""
                    current_line_texts = []
                    
                    for line in block["lines"]:
                        line_text = ""
                        line_spans = []
                        
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text:
                                continue
                                
                            font_size = round(span["size"], 2)
                            
                            # 只过滤掉明显大于正文的文本
                            if font_size > base_font_size * max_ratio:
                                continue
                                
                            if text.strip():
                                line_spans.append((text, font_size))
                        
                        if line_spans:
                            line_text = " ".join(text for text, _ in line_spans)
                            if line_text.strip():
                                current_line_texts.append(line_text)
                    
                    if current_line_texts:
                        block_text = " ".join(current_line_texts)
                        if block_text.strip():
                            # 过滤目录内容
                            if self._is_toc_block(block_text):
                                continue
                            if len(block_text) >= 10:
                                mupdf_blocks.append(block_text.strip())
            
            doc.close()
            print(f"PyMuPDF解析得到 {len(mupdf_blocks)} 个文本块")
            
            # 2. 使用规则解析每一页
            print("\n2. 使用规则解析每一页...")
            self.data = []  # 清空之前的结果
            self.ParseOnePageWithRule(max_seq=max_seq)
            rule_blocks = [block for block in self.data if not self._is_toc_block(block)]
            print(f"规则解析得到 {len(rule_blocks)} 个文本块")
            
            # 3. 使用滑动窗口处理整个文档
            print("\n3. 使用滑动窗口处理...")
            self.data = []  # 清空之前的结果
            self.ParseAllPage(max_seq=max_seq)
            sliding_blocks = [block for block in self.data if not self._is_toc_block(block)]
            print(f"滑动窗口处理得到 {len(sliding_blocks)} 个文本块")
            
            # 合并所有结果并去重
            all_blocks = mupdf_blocks + rule_blocks + sliding_blocks
            unique_blocks = list(set(all_blocks))
            
            # 过滤和清理
            filtered_blocks = []
            for block in unique_blocks:
                # 应用基本的文本过滤规则
                filter_rules = {
                    'remove_patterns': [r'\s+', r'^\d+$'],
                    'min_length': 10,
                    'exclude_words': ['目录', '..................']
                }
                cleaned_block = self.DataFilter(block, filter_rules)
                if cleaned_block and not self._is_toc_block(cleaned_block):
                    filtered_blocks.append(cleaned_block)
            
            self.data = filtered_blocks
            print(f"\n最终得到 {len(self.data)} 个有效文本块")
            return self.data
            
        except Exception as e:
            print(f"处理PDF文件失败: {str(e)}")
            return []
        
            
    def DataFilter(self, text: str, rules: Dict[str, Union[str, List[str]]]) -> str:
        """根据规则过滤文本
        
        Args:
            text: 待过滤的文本
            rules: 过滤规则，包含：
                  - 'remove_patterns': 需要移除的正则表达式模式
                  - 'keep_patterns': 需要保留的正则表达式模式
                  - 'min_length': 最小文本长度
                  - 'exclude_words': 需要排除的词列表
                  
        Returns:
            str: 过滤后的文本
        """
        try:
            # 如果文本为空，直接返回
            if not text.strip():
                return ""
                
            filtered_text = text
            
            # 应用移除模式
            if 'remove_patterns' in rules:
                patterns = rules['remove_patterns']
                if isinstance(patterns, str):
                    patterns = [patterns]
                for pattern in patterns:
                    filtered_text = re.sub(pattern, '', filtered_text)
                    
            # 应用保留模式
            if 'keep_patterns' in rules:
                patterns = rules['keep_patterns']
                if isinstance(patterns, str):
                    patterns = [patterns]
                for pattern in patterns:
                    matches = re.finditer(pattern, filtered_text)
                    filtered_text = ' '.join(match.group() for match in matches)
                    
            # 移除排除词
            if 'exclude_words' in rules:
                for word in rules['exclude_words']:
                    filtered_text = filtered_text.replace(word, '')
                    
            # 检查最小长度
            if 'min_length' in rules and len(filtered_text) < rules['min_length']:
                return ""
                
            return filtered_text.strip()
            
        except Exception as e:
            print(f"Error filtering text: {str(e)}")
            return text
            
    def ParseBlock(self, max_seq: int = 512, font_size_threshold: float = 8.0) -> List[str]:
        """使用PyMuPDF解析PDF文件为文本块，考虑字体大小和序号处理
        
        Args:
            max_seq: 最大序列长度，默认512
            font_size_threshold: 字体大小阈值，小于此值的文本将被忽略
            
        Returns:
            List[str]: 文本块列表
        """
        try:
            print(f"解析文本块，最大序列长度={max_seq}...")
            doc = fitz.open(self.pdf_path)
            all_text = []
            
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                page_height = page.rect.height
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                        
                        
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            # 检查字体大小
                            if span["size"] >= font_size_threshold:
                                text = span["text"].strip()
                                # 过滤序号
                                text = re.sub(r'\b0?\d+[\.\)）]?\b', '', text)  # 匹配01. 01) 01） 1. 1) 1）等格式
                                text = re.sub(r'^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', text)  # 匹配圆圈数字
                                # 过滤特殊符号
                                text = re.sub(r'[□•]', '', text)
                                
                                if text.strip():
                                    line_text += text + " "
                        
                        if line_text.strip():
                            block_text += line_text
                    
                    block_text = block_text.strip()
                    if not block_text:
                        continue
                    
                    # 应用基本的文本过滤规则
                    filter_rules = {
                        'remove_patterns': [
                            r'\s+',
                            r'^\d+$',
                            r'^[警告注意说明]！$'  # 过滤独立的警告/注意/说明标记
                        ],
                        'min_length': 10,
                        'exclude_words': ['目录', '..................']
                    }
                    block_text = self.DataFilter(block_text, filter_rules)
                    
                    if not block_text:
                        continue
                    
                    # 处理超长文本
                    if len(block_text) > max_seq:
                        sentences = block_text.split('。')
                        current_block = ''
                        
                        for sent in sentences:
                            if not sent.strip():
                                continue
                                
                            if len(current_block + sent) <= max_seq:
                                current_block += sent + '。'
                            else:
                                if current_block:
                                    all_text.append(current_block)
                                current_block = sent + '。'
                                
                        if current_block:
                            all_text.append(current_block)
                    else:
                        all_text.append(block_text)
            
            doc.close()
            self.data = all_text
            print(f"找到 {len(all_text)} 个文本块")
            return all_text
            
        except Exception as e:
            print(f"解析PDF时出错: {str(e)}")
            return []
            
    def sliding_window(self, sentences: List[str], kernel: int = 512, stride: int = 1) -> None:
        """滑动窗口处理文本"""
        cur = ""  # 当前窗口文本
        fast = 0  # 快指针
        slow = 0  # 慢指针
        
        while fast < len(sentences):
            sentence = sentences[fast]
            # 如果添加当前句子会超过窗口大小
            if len(cur + sentence) > kernel and (cur + sentence) not in self.data:
                self.data.append(cur + sentence + "。")
                cur = cur[len(sentences[slow] + "。"):]
                slow += 1
            # 添加当前句子到窗口
            cur = cur + sentence + "。"
            fast += 1
            
    def ParseAllPage(self, max_seq: int = 512, min_len: int = 6) -> None:
        """处理整个PDF文件"""
        doc = fitz.open(self.pdf_path)
        all_content = []
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if not self._is_in_middle_area(block, page.rect, self.content_area):
                    continue
                    
                if "lines" not in block:
                    continue
                    
                block_text = ""
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                            
                        line_text += text + " "
                    
                    if line_text.strip():
                        block_text += line_text
                
                if block_text.strip():
                    # 按句号分割文本
                    sentences = [s.strip() + "。" for s in block_text.split("。") if s.strip()]
                    all_content.extend(sentences)
        
        doc.close()
        
        # 使用滑动窗口处理句子
        self.data = []  # 清空之前的结果
        current_block = ""
        
        for sentence in all_content:
            # 如果当前句子加入会超过最大长度
            if len(current_block + sentence) > max_seq:
                if len(current_block) >= min_len:
                    self.data.append(current_block)
                current_block = sentence
            else:
                current_block += sentence
        
        # 添加最后一个块
        if current_block and len(current_block) >= min_len:
            self.data.append(current_block)

    def ParseOnePageWithRule(self, max_seq: int = 512, min_len: int = 6) -> None:
        """按规则解析每一页"""
        doc = fitz.open(self.pdf_path)
        base_font_size, min_ratio, max_ratio = self._get_base_font_size(doc)
        
        for page in doc:
            page_content = ""
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                # 使用缓存的内容区域
                if not self._is_in_middle_area(block, page.rect, self.content_area):
                    continue
                    
                if "lines" not in block:
                    continue
                    
                block_text = ""
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                            
                        font_size = span["size"]
                        # 只过滤掉明显大于正文的文本
                        if font_size <= base_font_size * max_ratio:
                            line_text += text + " "
                            
                    if line_text.strip():
                        block_text += line_text
                        
                if block_text.strip():
                    page_content += block_text + " "
            
            # 处理页面内容
            if len(page_content) < min_len:
                continue
                
            if len(page_content) < max_seq:
                if page_content not in self.data:
                    self.data.append(page_content)
            else:
                # 按句号分割并控制长度
                sentences = page_content.split("。")
                cur = ""
                for sentence in sentences:
                    if len(cur + sentence) > max_seq and (cur + sentence) not in self.data:
                        self.data.append(cur + sentence)
                        cur = sentence
                    else:
                        cur = cur + sentence
                        
        doc.close()

    def save_blocks(self, output_path: str) -> bool:
        """保存文本块到文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for block in self.data:
                    f.write(block + '\n\n')
            return True
        except Exception as e:
            print(f"Error saving blocks: {str(e)}")
            return False

    def _get_content_area(self, doc) -> Tuple[float, float, float, float]:
        """动态识别文档的主要内容区域
        
        通过分析第一页的文本块分布来确定内容区域的边界，
        分别识别上部和下部的两个边界，中间区域为页码和小标题区域
        Returns:
            Tuple[float, float, float, float]: (正文左边界, 正文右边界, 正文上边界, 正文下边界)
        """
        # 分析多个页面以获得更准确的边界
        y_positions = []  # 所有文本块的y坐标
        
        # 收集多个页面的文本块位置
        for page_num in range(10, 20):  # 分析前10页或所有页面
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            
            # 收集所有文本块的y坐标（起始和结束）
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                # 获取块的边界
                _, y0, _, y1 = block["bbox"]
                
                # 转换为相对坐标（0-1之间）
                y_positions.append(y0 / page_height)
                y_positions.append(y1 / page_height)
        
        if not y_positions:  # 如果没有找到文本块
            return (0.1, 0.9, 0.2, 0.9)  # 返回默认值
        
        # 将y坐标排序并去重
        y_positions = sorted(list(set([round(y, 2) for y in y_positions])))
        
        # 找出上部的两个边界
        top_boundaries = []
        last_y = 0
        for y in y_positions:
            if y - last_y > 0.03:  # 如果与上一个位置间隔超过3%，认为是一个边界
                top_boundaries.append(y)
                if len(top_boundaries) == 2:  # 找到两个上边界后停止
                    break
            last_y = y
        
        # 找出下部的两个边界（从后往前）
        bottom_boundaries = []
        last_y = 1.0
        for y in reversed(y_positions):
            if last_y - y > 0.03:  # 如果与上一个位置间隔超过3%，认为是一个边界
                bottom_boundaries.append(y)
                if len(bottom_boundaries) == 2:  # 找到两个下边界后停止
                    break
            last_y = y
        
        # 如果没有找到足够的边界，使用默认值
        if len(top_boundaries) < 2:
            top_boundaries = [0.1, 0.2]
        if len(bottom_boundaries) < 2:
            bottom_boundaries = [0.8, 0.9]
        
        # 第二个上边界是正文开始的位置，第一个下边界是正文结束的位置
        content_top = top_boundaries[1]
        content_bottom = bottom_boundaries[1]
        
        print(f"检测到上部边界: {top_boundaries[0]:.2f}, {top_boundaries[1]:.2f}")
        print(f"检测到下部边界: {bottom_boundaries[1]:.2f}, {bottom_boundaries[0]:.2f}")
        print(f"正文区域: {content_top:.2f} - {content_bottom:.2f}")
        
        # 返回内容区域的边界（左右边界保持宽松）
        return (0.1, 0.9, content_top, content_bottom)

    def _is_in_middle_area(self, block: dict, page_rect, content_area: Tuple[float, float, float, float]) -> bool:
        """判断文本块是否在页面的正文区域
        
        Args:
            block: 文本块信息
            page_rect: 页面尺寸信息
            content_area: 内容区域的边界比例 (左, 右, 上, 下)
            
        Returns:
            bool: 是否在正文区域
        """
        # 获取块的边界框
        x0, y0, x1, y1 = block["bbox"]
        
        # 计算相对坐标
        page_width = page_rect.width
        page_height = page_rect.height
        
        rel_x0 = x0 / page_width
        rel_x1 = x1 / page_width
        rel_y0 = y0 / page_height
        rel_y1 = y1 / page_height
        
        # 同时检查X轴和Y轴方向
        return (rel_y0 >= content_area[2] and rel_y1 <= content_area[3] and
                rel_x0 >= content_area[0] and rel_x1 <= content_area[1])

    def _get_base_font_size(self, doc) -> Tuple[float, float, float]:
        """获取文档的基准字体大小（正文）
        
        Returns:
            Tuple[float, float, float]: (正文字体大小, 最小容差, 最大容差)
        """
        # 定义标准字体大小
        BODY_FONT_SIZE = 7.5   # 正文字号
        TITLE_FONT_SIZE = 9.5  # 标题字号
        
        # 返回固定的正文字体大小和容差范围
        return (BODY_FONT_SIZE, 0.9, 1.2)
            
    def _is_page_number(self, text: str, current_page: int) -> bool:
        """判断文本是否为页码"""
        # 清理文本
        text = text.strip()
        
        # 如果文本就是当前页码
        if text == str(current_page):
            return True
            
        # 处理带有前缀或后缀的页码，如"第1页"、"Page 1"等
        text = re.sub(r'[第页]|Page\s*', '', text, flags=re.IGNORECASE)
        
        try:
            return int(text) == current_page
        except:
            return False

    def _is_toc_block(self, text: str) -> bool:
        """判断是否为目录区块
        
        Args:
            text: 待检查的文本
            
        Returns:
            bool: 是否为目录区块
        """
        # 1. 匹配密集的点状连接符（至少3个连续点）
        if re.search(r'\.{3,}|．{3,}|…{3,}', text):
            return True
        
        # 2. 匹配"XXX..........数字"的目录格式
        if re.search(r'[\u4e00-\u9fa5]+[\.．…]{3,}\d+', text):
            return True
        
        # 3. 匹配多个短句加页码的格式（如"第一章...1 第二章...2"）
        if re.search(r'([\u4e00-\u9fa5]+[\.．…]+\d+[\s\n]*){2,}', text):
            return True
        
        # 4. 检查是否包含典型的目录关键词
        toc_keywords = ['目录', '章节', '附录', '索引']
        if any(keyword in text for keyword in toc_keywords):
            return True
        
        # 5. 检查是否为纯页码引用格式（如"141, 142, 143"）
        if re.match(r'^[\d\s,]+$', text.strip()):
            return True
        
        return False 