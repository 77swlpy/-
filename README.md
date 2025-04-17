# 经济术语和股票名称纠正模型

这个项目实现了一个语言纠正模型，能够根据简称或不标准称呼精确识别经济学术语及股票名称，对语言进行处理来解决在问题输入过程中由于称呼简写或是名称使用不规范等问题导致的无法在数据库中检索到相关内容的问题。

## 功能特点

该模型能够处理以下情况：

1. **简称或缩写识别**：例如，将 "GDP" 识别为 "国内生产总值(Gross Domestic Product)"
2. **不标准称呼纠正**：例如，将 "沪深300" 识别为 "沪深300指数(CSI 300 Index)"
3. **错误拼写或近似词识别**：使用字符级相似度计算，识别轻微拼写错误或变体
4. **股票俗称/简称识别**：例如，将 "茅台" 识别为 "贵州茅台(600519)"

## 技术实现

1. **多级匹配策略**：
   - 直接匹配：检查输入是否为已知的术语、缩写或代码
   - 相似度匹配：使用TF-IDF向量化和余弦相似度计算字符级相似度
   - 编辑距离匹配：使用difflib库计算字符串相似度

2. **可扩展的知识库**：
   - 经济术语库：包含标准术语、定义及其可能的缩写或变体
   - 股票代码库：包含股票代码、名称、行业和常用简称
   - 支持动态添加新的术语和股票信息

3. **交互式界面**：
   - 支持单个术语纠正和整段文本纠正
   - 提供命令行交互式使用方式
   - 支持批量处理模式

## 安装与依赖

```bash
# 安装依赖
pip install -r requirements.txt
```

主要依赖：
- numpy
- scikit-learn
- difflib

## 使用方法

### 1. 命令行工具

```bash
# 交互模式
python correction_interface.py

# 术语纠正模式
python correction_interface.py --mode term --input "GDP"

# 文本纠正模式
python correction_interface.py --mode text --input "近期GDP增长放缓，沪深300波动较大"

# 初始化示例数据
python correction_interface.py --init-samples
```

### 2. 作为库导入使用

```python
from language_correction_model import TermCorrector

# 创建纠正模型
corrector = TermCorrector()

# 加载词典
corrector.load_dictionaries('economic_terms.json', 'stock_names.json')

# 术语纠正
result = corrector.correct_term("GDP")
print(result['corrected'])  # 输出: 国内生产总值(Gross Domestic Product)

# 文本纠正
text = "近期GDP增长放缓，CPI指数略有上升。茅台股价继续上涨。"
result = corrector.correct_text(text)
print(result['corrected_text'])
```

### 3. 添加新术语

通过交互界面：
```
# 添加经济术语
add economic ROE 净资产收益率 资产收益率

# 添加股票信息
add stock 601988 中国银行 中行 银行
```

通过代码：
```python
# 添加经济术语
corrector.add_economic_term("ROE", "净资产收益率(Return On Equity)", ["资产收益率"])

# 添加股票信息
corrector.add_stock("601988", "中国银行", ["中行"], "银行")
```

## 示例

```
原文本:
近期GDP增长放缓，CPI指数略有上升。投资者关注沪深300走势，茅台和五粮保持稳定增长。
工行发布新的理财产品，而万科在房地产市场面临压力。

纠正后文本:
近期国内生产总值(Gross Domestic Product)增长放缓，消费者价格指数(Consumer Price Index)略有上升。投资者关注沪深300指数(CSI 300 Index)走势，贵州茅台(600519)和五粮液(000858)保持稳定增长。
工商银行(601398)发布新的理财产品，而万科A(000002)在房地产市场面临压力。
```

## 扩展与优化方向

1. **增强分词功能**：集成专业的中文分词工具，提高复杂文本的处理能力
2. **增加上下文理解**：考虑术语的上下文来提高纠正精度
3. **扩充知识库**：添加更多经济术语和上市公司信息
4. **添加模糊匹配机制**：支持更复杂的模糊匹配和纠错算法
5. **集成数据库**：使用SQL数据库存储大规模术语库
