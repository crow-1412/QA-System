#!/usr/bin/env python
# coding: utf-8

import pdfplumber
from PyPDF2 import PdfReader


class DataProcess(object):

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = [] # 存储处理后的数据，结果列表

    # 滑动窗口功能实现，其中fast代表当前遍历句子的index，slow代表每次窗口开始滑动的起点。默认窗口直接滑动的overlap是1个句子。kernel是窗口大小，默认512个字符，stride是窗口滑动的步长，默认1个句子。
    def SlidingWindow(self, sentences, kernel = 512, stride = 1):
        """
        :param sentences: 句子列表
        :param kernel: 窗口大小
        :param stride: 窗口滑动的步长
        工作原理：使用两个指针，fast和slow，分别指向当前遍历句子的index和每次窗口开始滑动的起点。
        1. 遍历句子列表，将句子逐个添加到当前窗口中，直到窗口大小达到kernel。
        2. 如果当前窗口大小达到kernel，则将窗口中的句子添加到结果列表中，并清空窗口。
        3. 如果当前窗口大小未达到kernel，则将句子添加到窗口中。
        4. 重复上述步骤，直到遍历完所有句子。
        """
        sz = len(sentences) # 句子列表的长度
        cur = "" # 当前窗口中的句子
        fast = 0 # 当前遍历句子的index
        slow = 0 # 每次窗口开始滑动的起点
        while(fast < len(sentences)):
            sentence = sentences[fast]
            # 如果当前窗口大小达到kernel，则将窗口中的句子添加到结果列表中，并清空窗口。
            if(len(cur + sentence) > kernel and (cur + sentence) not in self.data):
                self.data.append(cur + sentence + "。")
                cur = cur[len(sentences[slow] + "。"):]
                slow = slow + 1
            # 如果当前窗口大小未达到kernel，则将句子添加到窗口中。
            cur = cur + sentence + "。"
            fast = fast + 1

    #  数据过滤，根据当前的文档内容的item划分句子，然后根据max_seq划分文档块。
    def Datafilter(self, line, header, pageid, max_seq = 1024):
        """
        对文本内容进行过滤和分段处理
        
        参数:
            line (str): 需要处理的文本行
            header (str): 文档页眉/标题
            pageid (int): 页码
            max_seq (int): 最大序列长度,默认1024字符
            
        工作流程:
            1. 检查文本长度,小于6字符的文本直接返回
            2. 如果文本长度超过max_seq:
               - 根据特殊分隔符(■, •, \t, 。)将文本分割成句子
               - 对每个子句进行处理:
                 * 去除换行符
                 * 检查长度是否在合理范围(5-max_seq)
                 * 清理标点和空白字符
                 * 如果不重复则加入结果列表
            3. 如果文本长度在合理范围内:
               - 清理标点和空白字符
               - 如果不重复则直接加入结果列表
        """
        # 获取文本长度
        sz = len(line)
        # 过滤过短的文本
        if(sz < 6):
            return

        # 处理超长文本
        if(sz > max_seq):
            # 根据不同分隔符切分文本
            if("■" in line):
                sentences = line.split("■")
            elif("•" in line):
                sentences = line.split("•")
            elif("\t" in line):
                sentences = line.split("\t")
            else:
                sentences = line.split("。")

            # 处理每个子句
            for subsentence in sentences:
                # 去除换行符
                subsentence = subsentence.replace("\n", "")

                # 检查子句长度是否合适
                if(len(subsentence) < max_seq and len(subsentence) > 5):
                    # 清理标点和空白字符
                    subsentence = subsentence.replace(",", "").replace("\n","").replace("\t","")
                    # 去重后添加到结果列表
                    if(subsentence not in self.data):
                        self.data.append(subsentence)
        # 处理正常长度的文本
        else:
            # 清理标点和空白字符
            line = line.replace("\n","").replace(",", "").replace("\t","")
            # 去重后添加到结果列表
            if(line not in self.data):
                self.data.append(line)

    # 提取页头即一级标题
    def GetHeader(self, page):
        try:
            lines = page.extract_words()[::]
        except:
            return None
        if(len(lines) > 0):
            for line in lines:
                if("目录" in line["text"] or ".........." in line["text"]):
                    return None
                if(line["top"] < 20 and line["top"] > 17):
                    return line["text"]
            return lines[0]["text"]
        return None

    # 按照每页中块提取内容,并和一级标题进行组合,配合Document 可进行意图识别
    def ParseBlock(self, max_seq = 1024):
        """
        按块解析PDF内容的主要方法
        参数:
            max_seq: 每个文本块的最大长度,默认1024
        """
        # 使用pdfplumber打开PDF文件
        with pdfplumber.open(self.pdf_path) as pdf:

            # 遍历PDF的每一页
            for i, p in enumerate(pdf.pages):
                # 获取当前页的页眉/一级标题
                header = self.GetHeader(p)

                # 如果没有获取到标题则跳过该页
                if(header == None):
                    continue

                # 提取当前页的所有文字,包含文字大小信息
                texts = p.extract_words(use_text_flow=True, extra_attrs = ["size"])[::]

                # 初始化文本序列和上一个文字大小
                squence = ""  # 用于存储当前正在处理的文本块
                lastsize = 0  # 记录上一个文字的大小

                # 遍历当前页的所有文字块
                for idx, line in enumerate(texts):
                    # 跳过第一个文字块
                    if(idx <1):
                        continue
                    # 跳过第二个文字块如果它是数字
                    if(idx == 1):
                        if(line["text"].isdigit()):
                            continue
                            
                    cursize = line["size"]  # 当前文字大小
                    text = line["text"]     # 当前文字内容
                    
                    # 跳过特殊符号
                    if(text == "□" or text == "•"):
                        continue
                    # 遇到警告/注意/说明时,处理已累积的文本并重置
                    elif(text== "警告！" or text == "注意！" or text == "说明！"):
                        if(len(squence) > 0):
                            self.Datafilter(squence, header, i, max_seq = max_seq)
                        squence = ""
                    # 如果当前文字大小与上一个相同,则拼接文本
                    elif(format(lastsize,".5f") == format(cursize,".5f")):
                        if(len(squence)>0):
                            squence = squence + text
                        else:
                            squence = text
                    # 文字大小发生变化时的处理
                    else:
                        lastsize = cursize
                        # 如果当前累积文本较短(<15),继续拼接
                        if(len(squence) < 15 and len(squence)>0):
                            squence = squence + text
                        # 否则处理已累积的文本,并开始新的文本块
                        else:
                            if(len(squence) > 0):
                                self.Datafilter(squence, header, i, max_seq = max_seq)
                            squence = text
                
                # 处理页面最后剩余的文本
                if(len(squence) > 0):
                    self.Datafilter(squence, header, i, max_seq = max_seq)

    # 按句号划分文档，然后利用最大长度划分文档块
    def ParseOnePageWithRule(self, max_seq = 512, min_len = 6):
        """
        按规则解析PDF页面内容
        Args:
            max_seq: 每个文本块的最大长度,默认512
            min_len: 文本块的最小长度,默认6
        """
        # 遍历PDF的每一页
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            # 提取当前页的文本内容
            text = page.extract_text()
            # 按换行符分割成单词列表
            words = text.split("\n")
            
            # 遍历每个单词,进行文本清洗
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                # 跳过目录和分隔符
                if("...................." in text or "目录" in text):
                    continue
                # 跳过空文本
                if(len(text) < 1):
                    continue
                # 跳过纯数字
                if(text.isdigit()):
                    continue
                # 拼接有效文本
                page_content = page_content + text
                
            # 跳过过短的页面内容    
            if(len(page_content) < min_len):
                continue
                
            # 如果页面内容小于最大长度限制,直接添加到结果中
            if(len(page_content) < max_seq):
                if(page_content not in self.data):
                    self.data.append(page_content)
            # 否则按句号分割并控制长度
            else:
                sentences = page_content.split("。")
                cur = ""
                # 遍历每个句子
                for idx, sentence in enumerate(sentences):
                    # 如果当前文本块+新句子超过最大长度,保存当前块并重新开始
                    if(len(cur + sentence) > max_seq and (cur + sentence) not in self.data):
                        self.data.append(cur + sentence)
                        cur = sentence
                    # 否则继续拼接句子
                    else:
                        cur = cur + sentence
    #  滑窗法提取段落
    #  1. 把pdf看做一个整体,作为一个字符串
    #  2. 利用句号当做分隔符,切分成一个数组
    #  3. 利用滑窗法对数组进行滑动, 此处的
    def ParseAllPage(self, max_seq = 512, min_len = 6):
        all_content = ""
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if("...................." in text or "目录" in text):
                    continue
                if(len(text) < 1):
                    continue
                if(text.isdigit()):
                    continue
                page_content = page_content + text
            if(len(page_content) < min_len):
                continue
            all_content = all_content + page_content
        sentences = all_content.split("。")
        self.SlidingWindow(sentences, kernel = max_seq)


if __name__ == "__main__":
    dp =  DataProcess(pdf_path = "./data/train_a.pdf")
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data
    out = open("all_text.txt", "w")
    for line in data:
        line = line.strip("\n")
        out.write(line)
        out.write("\n")
    out.close()
