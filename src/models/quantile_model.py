#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantile（Z-Score）模型 - 价格偏离度评估模块
独立测试入口，用于计算并可视化价格相对于均线的标准差偏离度（Z-Score）。
用途：衡量当前价格偏离均线的波动率倍数，辅助仓位管理。
- Z-Score > +2.0：极度超买（价格严重偏离），风险区，应减仓或拒绝开仓。
- Z-Score < -2.0：极度超卖（价格严重低估），机会区，可加仓。
- -1.0 ~ +1.5：正常健康趋势，策略信号正常执行。
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


class QuantileModel:
    """
    Z-Score 模型封装
    计算价格相对均线的标准差偏离度（Z-Score），衡量超买/超卖程度。
    """
    def __init__(self, lookback=20):
        """
        :param lookback: 计算滚动标准差（波动率）的窗口，默认20天
        """
        self.lookback = lookback

    def get_zscore_series(self, df, ma_column='MA_20'):
        """
        计算整个DataFrame中每日的 Z-Score 序列
        :param df: 包含 'close' 和指定均线列的 DataFrame（升序）
        :param ma_column: 均线列名，如 'MA_20', 'MA_50'
        :return: Series，索引与df一致，值为Z-Score
        """
        if ma_column not in df.columns:
            raise ValueError(f"列 {ma_column} 不存在于DataFrame中")
        
        # 价格相对于均线的偏离（绝对值）
        diff = df['close'] - df[ma_column]
        # 价格滚动标准差（衡量波动率）
        price_vol = df['close'].rolling(self.lookback).std()
        # Z-Score = 偏离 / 波动率
        z_score = diff / price_vol
        
        # 处理无穷大或NaN（例如波动率为0时）
        z_score = z_score.replace([np.inf, -np.inf], np.nan)
        
        return pd.Series(z_score, index=df.index, name='z_score')


def plot_zscore(df, zscore_series, ma_column='MA_20', title="价格偏离度 (Z-Score)"):
    """
    绘制价格、均线、Z-Score 曲线（带超买/超卖参考线）
    """
    # 删除缺失值
    z_series = zscore_series.dropna()
    df_aligned = df.loc[z_series.index].copy()
    df_aligned['zscore'] = z_series

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.05,
        subplot_titles=("价格与均线", f"Z-Score (基于{ma_column})")
    )

    # 上子图：价格 + 均线
    fig.add_trace(go.Scatter(
        x=df_aligned['date'],
        y=df_aligned['close'],
        mode='lines',
        name='收盘价',
        line=dict(color='black', width=1.5)
    ), row=1, col=1)

    if ma_column in df_aligned.columns:
        fig.add_trace(go.Scatter(
            x=df_aligned['date'],
            y=df_aligned[ma_column],
            mode='lines',
            name=ma_column,
            line=dict(color='blue', width=1, dash='dash')
        ), row=1, col=1)

    # 下子图：Z-Score
    fig.add_trace(go.Scatter(
        x=df_aligned['date'],
        y=df_aligned['zscore'],
        mode='lines',
        name='Z-Score',
        line=dict(color='purple', width=2)
    ), row=2, col=1)

    # 参考线：超买 +2σ，超卖 -2σ，中性 0
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1,
                  annotation_text="超买 (+2σ)", annotation_position="bottom right")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="green", row=2, col=1,
                  annotation_text="超卖 (-2σ)", annotation_position="bottom right")
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=2, col=1,
                  annotation_text="中性 (0)", annotation_position="bottom right")

    # 背景填充超买/超卖区域
    fig.add_hrect(y0=2.0, y1=5.0, line_width=0, fillcolor="red", opacity=0.1, row=2, col=1)
    fig.add_hrect(y0=-5.0, y1=-2.0, line_width=0, fillcolor="green", opacity=0.1, row=2, col=1)

    fig.update_layout(
        title=title,
        template='plotly_white',
        hovermode='x unified',
        height=700,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1, range=[-3.5, 3.5])

    return fig


# ========================== 独立测试入口 ==========================
if __name__ == "__main__":
    # ---------- 配置 ----------
    TICKER = "MSFT"
    START_DATE = "2007-01-01"
    END_DATE = "2026-12-31"
    MA_COLUMN = "MA_20"           # 可选择 'MA_20', 'MA_50', 'MA_70'
    LOOKBACK = 20                 # 标准差窗口
    OUTPUT_DIR = "outputs/figures"

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

    # 检查均线列是否存在
    if MA_COLUMN not in df.columns:
        print(f"❌ 均线列 {MA_COLUMN} 不存在，请检查 processed 数据是否包含该列。")
        print("   可用的均线列：", [c for c in df.columns if c.startswith('MA_')])
        sys.exit(1)

    print(f"🧮 计算 {MA_COLUMN} 的 Z-Score（窗口={LOOKBACK}天）...")
    model = QuantileModel(lookback=LOOKBACK)
    zscore_series = model.get_zscore_series(df, ma_column=MA_COLUMN)

    # 统计 Z-Score 分布
    z_valid = zscore_series.dropna()
    print(f"\n📊 Z-Score 统计（有效天数：{len(z_valid)}）：")
    print(f"  平均值: {z_valid.mean():.3f}")
    print(f"  标准差: {z_valid.std():.3f}")
    print(f"  最小值: {z_valid.min():.3f}")
    print(f"  最大值: {z_valid.max():.3f}")
    # 超买/超卖天数占比
    overbought = (z_valid > 2.0).sum()
    oversold = (z_valid < -2.0).sum()
    print(f"  超买天数 (>+2σ): {overbought} ({overbought/len(z_valid)*100:.1f}%)")
    print(f"  超卖天数 (<-2σ): {oversold} ({oversold/len(z_valid)*100:.1f}%)")

    # 绘制图表
    print(f"\n📈 生成 Z-Score 可视化图表...")
    fig = plot_zscore(df, zscore_series, ma_column=MA_COLUMN,
                      title=f"{TICKER} {MA_COLUMN} 价格偏离度 (Z-Score)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.write_html(os.path.join(OUTPUT_DIR, "quantile_analysis.html"))
    print(f"✅ 图表已保存至 {OUTPUT_DIR}/quantile_analysis.html")

    # 展示最近30天的 Z-Score
    print("\n🔍 最近 30 天的 Z-Score：")
    recent = z_valid.tail(30)
    print(recent.round(3).tolist())

    print("\n🎯 Z-Score 模型测试完成！")
    print("👉 请用浏览器打开 outputs/figures/quantile_analysis.html 查看 Z-Score 走势。")