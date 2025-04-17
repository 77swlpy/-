#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class InvestmentRiskMoE(Model):
    """
    混合专家模型(Mixture of Experts)用于投资风险预测
    
    该模型包含多个"专家"网络和一个"门控"网络：
    - 专家网络：每个专家专注于不同类型的投资风险模式
    - 门控网络：学习如何根据输入数据为每个专家分配权重
    """
    
    def __init__(self, num_experts=5, expert_units=[64, 32], gate_units=[32], output_dim=1):
        super(InvestmentRiskMoE, self).__init__()
        
        self.num_experts = num_experts
        
        # 创建门控网络
        self.gate = tf.keras.Sequential()
        for units in gate_units:
            self.gate.add(layers.Dense(units, activation='relu'))
        self.gate.add(layers.Dense(num_experts, activation='softmax'))
        
        # 创建专家网络
        self.experts = []
        for _ in range(num_experts):
            expert = tf.keras.Sequential()
            for units in expert_units:
                expert.add(layers.Dense(units, activation='relu'))
            expert.add(layers.Dense(output_dim))
            self.experts.append(expert)
    
    def call(self, inputs, training=None):
        # 获取每个专家的权重
        gate_outputs = self.gate(inputs)
        
        # 计算每个专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(inputs))
        
        # 将专家输出堆叠成形状为 [batch_size, num_experts, output_dim] 的张量
        stacked_expert_outputs = tf.stack(expert_outputs, axis=1)
        
        # 将门控输出扩展为 [batch_size, num_experts, 1]，以便与专家输出相乘
        expanded_gate_outputs = tf.expand_dims(gate_outputs, -1)
        
        # 计算加权和
        final_output = tf.reduce_sum(stacked_expert_outputs * expanded_gate_outputs, axis=1)
        
        return final_output

def prepare_financial_data(X, y, test_size=0.2, random_state=42):
    """准备金融数据集"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_investment_risk_model(X_train, y_train, X_test, y_test, 
                               num_experts=5, 
                               epochs=50, 
                               batch_size=32,
                               learning_rate=0.001):
    """训练投资风险MoE模型"""
    
    # 获取输入特征维度
    input_dim = X_train.shape[1]
    
    # 创建模型
    model = InvestmentRiskMoE(num_experts=num_experts)
    
    # 编译模型
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # 均方误差用于回归问题
        metrics=['mae']  # 平均绝对误差
    )
    
    # 构建模型输入形状
    model.build(input_shape=(None, input_dim))
    model.summary()
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    # 在测试集上评估模型
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集MAE: {test_mae:.4f}")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算误差
    errors = np.abs(y_pred.flatten() - y_test)
    
    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('实际风险')
    plt.ylabel('预测风险')
    plt.title('MoE模型: 实际风险 vs. 预测风险')
    plt.grid(True)
    plt.savefig('risk_prediction_results.png')
    plt.close()
    
    # 分析各专家网络的贡献
    analyze_experts_contribution(model, X_test)
    
    return y_pred, errors

def analyze_experts_contribution(model, X_test):
    """分析各专家网络的贡献"""
    # 获取门控网络输出（专家权重）
    gate_outputs = model.gate(X_test).numpy()
    
    # 计算每个专家的平均权重
    expert_weights = np.mean(gate_outputs, axis=0)
    
    # 可视化专家权重
    plt.figure(figsize=(10, 6))
    plt.bar(range(model.num_experts), expert_weights)
    plt.xlabel('专家编号')
    plt.ylabel('平均权重')
    plt.title('MoE模型: 各专家网络的平均贡献')
    plt.xticks(range(model.num_experts))
    plt.grid(True, axis='y')
    plt.savefig('expert_contributions.png')
    plt.close()
    
    print("各专家网络的平均贡献:")
    for i, weight in enumerate(expert_weights):
        print(f"专家 {i+1}: {weight:.4f}")

def generate_synthetic_investment_data(n_samples=1000, n_features=15, random_state=42):
    """
    生成合成投资数据用于演示
    
    特征包括:
    - 市场指标 (如市场回报率、波动率)
    - 公司财务指标 (如PE比率、收益增长)
    - 宏观经济指标 (如GDP增长、通胀率)
    - 行业特定指标
    - 技术指标
    """
    np.random.seed(random_state)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 为特征赋予不同的权重来模拟它们对风险的影响
    feature_weights = np.random.uniform(-1, 1, n_features)
    
    # 基本风险得分基于特征的线性组合
    base_risk = np.dot(X, feature_weights)
    
    # 添加非线性影响，模拟特定市场条件下风险的突然变化
    market_condition = X[:, 0]  # 使用第一个特征作为市场条件指标
    non_linear_effect = np.where(market_condition < -1.0, 
                               np.square(market_condition), 
                               0.2 * market_condition)
    
    # 添加相互作用效应，模拟某些特征组合导致的风险放大
    interaction_effect = X[:, 1] * X[:, 2] * 0.5
    
    # 最终风险得分 = 基本风险 + 非线性效应 + 相互作用效应 + 随机噪声
    y = base_risk + non_linear_effect + interaction_effect + np.random.normal(0, 0.5, n_samples)
    
    # 将风险标准化到[0, 1]区间内，其中1表示最高风险
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # 创建特征名称
    feature_names = [
        "市场回报率", "市场波动率", "利率水平", 
        "PE比率", "收益增长", "负债比率", "现金流", 
        "GDP增长", "通胀率", "失业率",
        "行业增长", "行业竞争度", 
        "技术趋势1", "技术趋势2", "技术趋势3"
    ]
    
    return X, y, feature_names

def main():
    print("生成投资数据...")
    X, y, feature_names = generate_synthetic_investment_data(n_samples=1000)
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print("特征集包含:", ", ".join(feature_names))
    
    print("\n准备数据...")
    X_train, X_test, y_train, y_test, scaler = prepare_financial_data(X, y)
    
    print("\n训练投资风险MoE模型...")
    model, history = train_investment_risk_model(
        X_train, y_train, X_test, y_test,
        num_experts=5,
        epochs=50,
        batch_size=32
    )
    
    print("\n评估模型性能...")
    y_pred, errors = evaluate_model(model, X_test, y_test)
    
    print("\n完成! 模型已成功训练并评估。")
    print("可以在当前目录下查看生成的图表：risk_prediction_results.png 和 expert_contributions.png")

if __name__ == "__main__":
    main()
