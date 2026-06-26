#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HMM（隐马尔可夫模型）市场状态识别模块
独立测试入口，用于训练并可视化市场状态划分。
状态含义（根据波动率水平划分）：
- State 0：低波动（市场平静，蓄势期）
- State 1：中波动（趋势运行，方向明确）
- State 2：高波动（剧烈震荡，风险期）
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 将项目根目录添加到 sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.features.data_loader import DataLoader

# 尝试导入 hmmlearn，若未安装则给出提示并退出
try:
    from hmmlearn import hmm
except ImportError:
    print("❌ 请先安装 hmmlearn：pip install hmmlearn")
    sys.exit(1)


class HMModel:
    """
    隐马尔可夫模型封装
    使用高斯HMM对市场状态进行无监督学习
    """
    def __init__(self, n_states=3, n_mix=1, random_state=42):
        """
        :param n_states: 隐状态数量，默认3
        :param n_mix: 混合高斯成分数，默认1（单高斯）
        :param random_state: 随机种子
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=random_state,
            tol=1e-4
        )
        self.fitted = False

    def fit(self, X):
        """
        训练模型
        :param X: 特征矩阵 (n_samples, n_features)
        """
        self.model.fit(X)
        self.fitted = True
        return self

    def predict(self, X):
        """
        预测隐状态序列
        :param X: 特征矩阵
        :return: 状态数组 (n_samples,)
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用 fit()")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        预测状态概率
        :param X: 特征矩阵
        :return: 概率矩阵 (n_samples, n_states)
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用 fit()")
        return self.model.predict_proba(X)


def prepare_features(df, lookback=20):
    """
    从原始数据中提取 HMM 所需的特征
    :param df: 包含 'close', 'volume', 'high', 'low' 的 DataFrame（升序）
    :param lookback: 用于计算波动率的窗口
    :return: 特征 DataFrame（去除缺失值）
    """
    data = df.copy()
    # 对数收益率
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    # 波动率（滚动标准差）
    data['volatility'] = data['log_ret'].rolling(lookback).std()
    # 成交量变化率
    data['volume_change'] = data['volume'].pct_change()
    # 价格振幅 (high-low)/close
    data['amplitude'] = (data['high'] - data['low']) / data['close']
    # 相对强弱（短期涨跌幅）
    data['ret_5'] = data['close'].pct_change(5)
    data['ret_10'] = data['close'].pct_change(10)

    # 选择特征列
    feature_cols = ['log_ret', 'volatility', 'volume_change', 'amplitude', 'ret_5', 'ret_10']
    features = data[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    return features


def plot_states(df, states, title="HMM 市场状态识别"):
    """
    绘制价格曲线并标注隐状态（3种状态）
    状态颜色：
      0: #0055FF (亮蓝)  - 低波动
      1: #00CC44 (翠绿)  - 中波动
      2: #D62728 (亮红)  - 高波动
    """
    # 对齐数据
    df_aligned = df.loc[states.index].copy()
    df_aligned['state'] = states.values

    # 定义状态颜色和名称
    state_colors = {
        0: '#0055FF',   # 亮蓝
        1: '#00CC44',   # 翠绿
        2: '#D62728'    # 亮红
    }
    state_names = {
        0: '低波动 (0)',
        1: '中波动 (1)',
        2: '高波动 (2)'
    }

    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.9, 0.1],
        vertical_spacing=0.05,
        subplot_titles=("价格走势", "市场状态 (颜色带)")
    )

    # ----- 第一行：价格曲线 -----
    fig.add_trace(go.Scatter(
        x=df_aligned['date'],
        y=df_aligned['close'],
        mode='lines',
        name='收盘价',
        line=dict(color='black', width=1.5)
    ), row=1, col=1)

    # ----- 第二行：状态热力图（使用单行 y=[0]，并压缩 y 轴范围）-----
    state_values = df_aligned['state'].values.reshape(1, -1)
    colorscale = [
        [0.0, state_colors[0]],
        [0.5, state_colors[1]],
        [1.0, state_colors[2]]
    ]
    fig.add_trace(go.Heatmap(
        z=state_values,
        x=df_aligned['date'],
        y=[0],
        colorscale=colorscale,
        zmin=0,
        zmax=2,
        showscale=False,
        hoverinfo='x+y+text',
        text=[[state_names[s] for s in state_values.flatten()]],
        hovertemplate='日期: %{x}<br>状态: %{text}<extra></extra>'
    ), row=2, col=1)

    # 手动添加图例（用散点图伪造）
    for state, color in state_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=state_names[state]
        ), row=2, col=1)

    # 更新布局：压缩第二行 y 轴范围，使色带变细
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(
        title_text="",
        row=2, col=1,
        showticklabels=False,
        range=[-0.2, 0.2]
    )

    fig.update_layout(
        title=title,
        template='plotly_white',
        hovermode='x unified',
        height=700,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


# ========================== 独立测试入口 ==========================
if __name__ == "__main__":
    # ---------- 配置 ----------
    TICKER = "MSFT"
    START_DATE = "2007-01-01"
    END_DATE = "2026-12-31"
    N_STATES = 3
    OUTPUT_DIR = f"outputs/figures/{TICKER}"

    print(f"📊 加载 {TICKER} 数据（{START_DATE} ~ {END_DATE}）...")
    loader = DataLoader(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        data_root="data"
    )
    try:
        df = loader.load_processed_data()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 请先运行数据流水线：python src/features/data_loader.py")
        sys.exit(1)

    if df.empty:
        print("⚠️ 数据为空，请检查日期范围。")
        sys.exit(1)

    # 转为升序
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    # 准备特征
    print("🧮 提取特征...")
    features = prepare_features(df)
    print(f"特征矩阵形状：{features.shape}")

    # 训练 HMM
    print("🔄 训练 HMM 模型...")
    hmm_model = HMModel(n_states=N_STATES)
    hmm_model.fit(features.values)

    # 预测状态
    states = hmm_model.predict(features.values)
    state_series = pd.Series(states, index=features.index, name='state')

    # 统计状态分布
    print("\n📊 HMM 状态分布：")
    state_counts = state_series.value_counts().sort_index()
    for s, count in state_counts.items():
        print(f"  State {s}: {count} 天 ({count/len(state_series)*100:.1f}%)")

    # 绘制状态图
    print(f"\n📈 生成状态可视化图表...")
    fig = plot_states(df, state_series)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.write_html(os.path.join(OUTPUT_DIR, "hmm_states.html"))
    print(f"✅ 图表已保存至 {OUTPUT_DIR}/hmm_states.html")

    # 展示近期的状态转移（可选）
    print("\n🔍 最近 30 天的状态：")
    recent = state_series.tail(30)
    print(recent.tolist())

    print("\n🎯 HMM 测试完成！")
    print("👉 请用浏览器打开 outputs/figures/hmm_states.html 查看状态划分。")