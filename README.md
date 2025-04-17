# 投资风险预测混合专家模型 (MoE)

这个项目实现了一个混合专家模型(Mixture of Experts, MoE)用于投资风险预测。MoE模型结合了多个"专家"神经网络的预测结果，通过门控网络动态分配权重，以实现更精确的风险评估。

## 模型架构

该MoE模型由以下组件构成：

1. **多个专家网络**：每个专家网络专注于捕捉不同类型的投资风险模式。
2. **门控网络**：根据输入特征动态决定每个专家网络的权重。
3. **集成机制**：将各专家网络的输出按权重合并，生成最终的风险预测。

## 特点

- 能够捕捉复杂的非线性风险因素和交互效应
- 通过专家分工提高模型在不同市场环境下的适应性
- 可视化专家网络的贡献权重，提供模型解释性

## 环境配置

```bash
pip install -r requirements.txt
```

## 运行示例

直接运行主脚本:

```bash
python investment_risk_moe.py
```

## 数据说明

当前实现使用合成数据进行演示，包含以下类型的特征:

- 市场指标（市场回报率、波动率等）
- 公司财务指标（PE比率、收益增长、负债比率等）
- 宏观经济指标（GDP增长、通胀率、失业率等）
- 行业和技术指标

## 使用自己的数据

要使用自己的投资数据，请按以下步骤操作:

1. 准备包含投资特征的数据集 (X) 和相应的风险标签 (y)
2. 修改 `main()` 函数中的数据加载部分，替换合成数据生成:

```python
# 替换
X, y, feature_names = generate_synthetic_investment_data(n_samples=1000)

# 为
X = pd.read_csv('your_data.csv')  # 加载您的特征数据
y = X.pop('risk_label')  # 假设风险标签是数据集中的一列
feature_names = X.columns.tolist()
X = X.values  # 转换为NumPy数组
```

## 输出结果

运行完成后，会生成两个可视化文件:

1. `risk_prediction_results.png` - 显示模型预测风险与实际风险的对比
2. `expert_contributions.png` - 展示各专家网络的平均贡献权重

## 调优参数

在`train_investment_risk_model`函数中可以调整以下参数:

- `num_experts`: 专家网络的数量
- `epochs`: 训练轮数
- `batch_size`: 批处理大小
- `learning_rate`: 学习率
