#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

class TermCorrector:
    """
    语言纠正模型，用于识别经济学术语和股票名称。
    
    该模型能够处理以下情况：
    1. 简称或缩写 (例如: "GDP" -> "国内生产总值")
    2. 不标准称呼 (例如: "沪深300" -> "沪深300指数")
    3. 错误拼写或近似词 (例如: "CPI指数" -> "CPI(消费者价格指数)")
    4. 股票俗称/简称 (例如: "茅台" -> "贵州茅台(600519)")
    """
    
    def __init__(self):
        # 加载术语字典
        self.economic_terms = {}
        self.stock_names = {}
        self.abbreviations = {}
        
        # 向量化器用于语义相似度计算
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        
        # 相似度阈值
        self.similarity_threshold = 0.75
        
    def load_dictionaries(self, economic_terms_file=None, stock_names_file=None):
        """加载术语字典文件"""
        if economic_terms_file:
            try:
                with open(economic_terms_file, 'r', encoding='utf-8') as f:
                    self.economic_terms = json.load(f)
                print(f"成功加载经济术语: {len(self.economic_terms)}个")
            except Exception as e:
                print(f"加载经济术语文件时出错: {e}")
                
        if stock_names_file:
            try:
                with open(stock_names_file, 'r', encoding='utf-8') as f:
                    self.stock_names = json.load(f)
                print(f"成功加载股票名称: {len(self.stock_names)}个")
            except Exception as e:
                print(f"加载股票名称文件时出错: {e}")
        
        # 创建缩写和简称映射
        self._build_abbreviation_mapping()
        
        # 构建向量矩阵
        self._build_vector_matrices()
        
    def _build_abbreviation_mapping(self):
        """构建缩写和简称到完整术语的映射"""
        # 处理经济术语的简称
        for term, info in self.economic_terms.items():
            # 获取所有可能的简称和缩写
            abbrs = info.get('abbreviations', [])
            for abbr in abbrs:
                self.abbreviations[abbr] = {'type': 'economic', 'full_term': term}
            
            # 英文术语的首字母缩写
            if re.search(r'[a-zA-Z]', term):
                words = re.findall(r'[a-zA-Z]+', term)
                if len(words) > 1:
                    acronym = ''.join(word[0].upper() for word in words)
                    self.abbreviations[acronym] = {'type': 'economic', 'full_term': term}
                    
        # 处理股票名称的简称
        for code, info in self.stock_names.items():
            name = info.get('name', '')
            abbrs = info.get('abbreviations', [])
            for abbr in abbrs:
                self.abbreviations[abbr] = {'type': 'stock', 'full_term': code}
            
            # 将股票名称本身也作为一种简称
            self.abbreviations[name] = {'type': 'stock', 'full_term': code}
    
    def _build_vector_matrices(self):
        """构建术语的向量矩阵，用于相似度计算"""
        # 经济术语向量化
        economic_terms_list = list(self.economic_terms.keys())
        if economic_terms_list:
            self.economic_terms_matrix = self.vectorizer.fit_transform(economic_terms_list)
            self.economic_terms_features = economic_terms_list
        else:
            self.economic_terms_matrix = None
            self.economic_terms_features = []
            
        # 股票名称向量化
        stock_names_list = []
        for code, info in self.stock_names.items():
            name = info.get('name', '')
            if name:
                stock_names_list.append(name)
        
        if stock_names_list:
            self.stock_names_matrix = self.vectorizer.fit_transform(stock_names_list)
            self.stock_names_features = stock_names_list
        else:
            self.stock_names_matrix = None
            self.stock_names_features = []
    
    def add_economic_term(self, term, definition, abbreviations=None):
        """添加新的经济术语"""
        if abbreviations is None:
            abbreviations = []
            
        self.economic_terms[term] = {
            'definition': definition,
            'abbreviations': abbreviations
        }
        
        # 更新缩写映射
        for abbr in abbreviations:
            self.abbreviations[abbr] = {'type': 'economic', 'full_term': term}
            
        # 重建向量矩阵
        self._build_vector_matrices()
    
    def add_stock(self, code, name, abbreviations=None, industry=None):
        """添加新的股票信息"""
        if abbreviations is None:
            abbreviations = []
            
        self.stock_names[code] = {
            'name': name,
            'abbreviations': abbreviations,
            'industry': industry
        }
        
        # 更新缩写映射
        for abbr in abbreviations:
            self.abbreviations[abbr] = {'type': 'stock', 'full_term': code}
        self.abbreviations[name] = {'type': 'stock', 'full_term': code}
            
        # 重建向量矩阵
        self._build_vector_matrices()
    
    def save_dictionaries(self, economic_terms_file, stock_names_file):
        """保存术语字典到文件"""
        try:
            with open(economic_terms_file, 'w', encoding='utf-8') as f:
                json.dump(self.economic_terms, f, ensure_ascii=False, indent=2)
            print(f"经济术语已保存到 {economic_terms_file}")
        except Exception as e:
            print(f"保存经济术语时出错: {e}")
            
        try:
            with open(stock_names_file, 'w', encoding='utf-8') as f:
                json.dump(self.stock_names, f, ensure_ascii=False, indent=2)
            print(f"股票名称已保存到 {stock_names_file}")
        except Exception as e:
            print(f"保存股票名称时出错: {e}")
    
    def correct_term(self, term):
        """纠正给定的术语或股票名称"""
        # 检查是否为已知缩写或简称
        if term in self.abbreviations:
            info = self.abbreviations[term]
            if info['type'] == 'economic':
                return {
                    'corrected': self.economic_terms[info['full_term']].get('definition', info['full_term']),
                    'original': term,
                    'full_term': info['full_term'],
                    'type': 'economic',
                    'confidence': 1.0
                }
            else:  # 股票
                stock_info = self.stock_names[info['full_term']]
                return {
                    'corrected': f"{stock_info['name']}({info['full_term']})",
                    'original': term,
                    'full_term': info['full_term'],
                    'type': 'stock',
                    'confidence': 1.0
                }
        
        # 检查是否为已知经济术语或股票代码/名称
        if term in self.economic_terms:
            return {
                'corrected': self.economic_terms[term].get('definition', term),
                'original': term,
                'full_term': term,
                'type': 'economic',
                'confidence': 1.0
            }
        
        if term in self.stock_names:
            stock_info = self.stock_names[term]
            return {
                'corrected': f"{stock_info['name']}({term})",
                'original': term,
                'full_term': term,
                'type': 'stock',
                'confidence': 1.0
            }
            
        # 使用字符串相似度查找相近术语
        result = self._find_similar_term(term)
        if result:
            return result
            
        # 如果无法找到匹配项，返回原始术语
        return {
            'corrected': term,
            'original': term,
            'full_term': None,
            'type': 'unknown',
            'confidence': 0.0
        }
    
    def _find_similar_term(self, term):
        """寻找与给定术语相似的术语"""
        best_match = None
        highest_similarity = 0
        match_type = None
        
        # 检查经济术语
        if self.economic_terms_matrix is not None:
            term_vector = self.vectorizer.transform([term])
            similarities = cosine_similarity(term_vector, self.economic_terms_matrix).flatten()
            max_idx = similarities.argmax()
            
            if similarities[max_idx] > highest_similarity:
                highest_similarity = similarities[max_idx]
                best_match = self.economic_terms_features[max_idx]
                match_type = 'economic'
        
        # 检查股票名称
        if self.stock_names_matrix is not None:
            term_vector = self.vectorizer.transform([term])
            similarities = cosine_similarity(term_vector, self.stock_names_matrix).flatten()
            max_idx = similarities.argmax()
            
            if similarities[max_idx] > highest_similarity:
                highest_similarity = similarities[max_idx]
                best_match = self.stock_names_features[max_idx]
                match_type = 'stock'
        
        # 使用difflib进行更精确的字符串匹配
        close_matches = []
        
        # 经济术语匹配
        economic_matches = difflib.get_close_matches(term, self.economic_terms_features, n=1, cutoff=0.6)
        if economic_matches:
            similarity = difflib.SequenceMatcher(None, term, economic_matches[0]).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = economic_matches[0]
                match_type = 'economic'
                
        # 股票名称匹配
        stock_matches = difflib.get_close_matches(term, self.stock_names_features, n=1, cutoff=0.6)
        if stock_matches:
            similarity = difflib.SequenceMatcher(None, term, stock_matches[0]).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = stock_matches[0]
                match_type = 'stock'
        
        # 如果相似度超过阈值，返回匹配结果
        if highest_similarity >= self.similarity_threshold and best_match:
            if match_type == 'economic':
                return {
                    'corrected': self.economic_terms[best_match].get('definition', best_match),
                    'original': term,
                    'full_term': best_match,
                    'type': 'economic',
                    'confidence': highest_similarity
                }
            else:  # 股票
                for code, info in self.stock_names.items():
                    if info.get('name') == best_match:
                        return {
                            'corrected': f"{best_match}({code})",
                            'original': term,
                            'full_term': code,
                            'type': 'stock',
                            'confidence': highest_similarity
                        }
        
        return None
    
    def correct_text(self, text):
        """纠正文本中的所有术语"""
        # 简单的分词（以空格、标点等分隔）
        tokens = re.findall(r'\w+', text)
        
        corrections = {}
        for token in tokens:
            if len(token) < 2:  # 忽略单个字符
                continue
                
            result = self.correct_term(token)
            if result['type'] != 'unknown' and result['confidence'] > 0:
                corrections[token] = result
        
        # 应用纠正到原文本
        corrected_text = text
        for original, correction in corrections.items():
            # 使用正则表达式确保只替换整个词，而不是词的一部分
            pattern = r'\b' + re.escape(original) + r'\b'
            corrected_text = re.sub(pattern, correction['corrected'], corrected_text)
        
        return {
            'corrected_text': corrected_text,
            'corrections': corrections
        }

def create_sample_dictionaries():
    """创建示例字典数据"""
    # 经济术语示例
    economic_terms = {
        "GDP": {
            "definition": "国内生产总值(Gross Domestic Product)",
            "abbreviations": ["国内生产总值", "生产总值"]
        },
        "CPI": {
            "definition": "消费者价格指数(Consumer Price Index)",
            "abbreviations": ["消费者物价指数", "物价指数"]
        },
        "PPI": {
            "definition": "生产者价格指数(Producer Price Index)",
            "abbreviations": ["生产物价指数"]
        },
        "PMI": {
            "definition": "采购经理指数(Purchasing Managers' Index)",
            "abbreviations": ["采购经理人指数"]
        },
        "M2": {
            "definition": "广义货币供应量",
            "abbreviations": ["货币供应量"]
        },
        "SHIBOR": {
            "definition": "上海银行间同业拆放利率(Shanghai Interbank Offered Rate)",
            "abbreviations": ["上海银行同业拆放利率", "上海拆放利率"]
        },
        "沪深300指数": {
            "definition": "沪深300指数(CSI 300 Index)",
            "abbreviations": ["沪深300", "300指数"]
        }
    }
    
    # 股票名称示例
    stock_names = {
        "600519": {
            "name": "贵州茅台",
            "abbreviations": ["茅台", "贵茅", "茅台酒"],
            "industry": "白酒"
        },
        "601318": {
            "name": "中国平安",
            "abbreviations": ["平安", "平安保险"],
            "industry": "保险"
        },
        "000858": {
            "name": "五粮液",
            "abbreviations": ["五粮", "宜宾五粮液"],
            "industry": "白酒"
        },
        "601398": {
            "name": "工商银行",
            "abbreviations": ["工行"],
            "industry": "银行"
        },
        "000002": {
            "name": "万科A",
            "abbreviations": ["万科", "万科地产"],
            "industry": "房地产"
        }
    }
    
    return economic_terms, stock_names

def main():
    """主函数"""
    # 创建并初始化纠正模型
    corrector = TermCorrector()
    
    # 获取示例字典数据
    economic_terms, stock_names = create_sample_dictionaries()
    
    # 将示例数据添加到模型中
    for term, info in economic_terms.items():
        corrector.add_economic_term(term, info['definition'], info['abbreviations'])
    
    for code, info in stock_names.items():
        corrector.add_stock(code, info['name'], info['abbreviations'], info['industry'])
    
    # 保存字典到文件
    corrector.save_dictionaries('economic_terms.json', 'stock_names.json')
    
    # 测试纠正功能
    test_terms = [
        "GDP",
        "gdp",
        "国内生产总值",
        "CPI指数",
        "物价指数",
        "沪深300",
        "茅台",
        "贵州茅台",
        "中平",  # 错误的简称
        "五粮液",
        "工行",
        "万科"
    ]
    
    print("\n单个术语纠正测试:")
    for term in test_terms:
        result = corrector.correct_term(term)
        print(f"原始: {term} -> 纠正: {result['corrected']} (类型: {result['type']}, 置信度: {result['confidence']:.2f})")
    
    # 测试文本纠正
    test_text = """
    近期GDP增长放缓，CPI指数略有上升。投资者关注沪深300走势，茅台和五粮保持稳定增长。
    工行发布新的理财产品，而万科在房地产市场面临压力。
    """
    
    print("\n文本纠正测试:")
    result = corrector.correct_text(test_text)
    print("原文本:")
    print(test_text)
    print("\n纠正后文本:")
    print(result['corrected_text'])
    print("\n纠正详情:")
    for original, correction in result['corrections'].items():
        print(f"{original} -> {correction['corrected']}")

if __name__ == "__main__":
    main()
