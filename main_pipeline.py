#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_pipeline.py
整合数据加载、海龟策略信号生成、HMM 状态识别，
并在三行子图中展示：价格+信号、MA100斜率柱状图、HMM 市场状态（3种状态颜色）。
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 将项目根目录添加到 sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from src.features.data_loader import DataLoader, run_data_pipeline
from src.strategies.turtle_strategy import TurtleStrategy, load_strategy_config
from src.models.hmm_model import HMModel, prepare_features


def plot_combined(df_price, signals_df, state_series, title="海龟策略信号与HMM市场状态"):
    """
    绘制三行组合图表：
    上子图：价格 + 均线 + 买卖信号
    中子图：MA100 斜率柱状图（正浅绿，负浅红）
    下子图：HMM 市场状态色带（3种状态）
    """
    # ---- 处理 HMM 状态（原始3种状态） ----
    date_index = df_price.iloc[state_series.index]['date'].values
    state_by_date = pd.Series(state_series.values, index=date_index)

    # ---- 计算 MA100 斜率（5日变化率）并准备柱状图颜色 ----
    if 'M100' not in df_price.columns:
        df_price['M100'] = df_price['close'].rolling(window=100).mean()
    df_price['M100_slope'] = (df_price['M100'] / df_price['M100'].shift(5) - 1) * 100
    slope_data = df_price[['date', 'M100_slope']].dropna()
    # 柱状图颜色：正浅绿，负浅红
    colors = ["#02FF02" if val >= 0 else "#FF0228" for val in slope_data['M100_slope']]

    # ---- 定义颜色（3种状态） ----
    state_colors = {
        0: '#0055FF',   # 亮蓝 - 低波动
        1: '#00CC44',   # 翠绿 - 中波动
        2: '#D62728'    # 亮红 - 高波动
    }
    state_names = {
        0: '低波动 (0)',
        1: '中波动 (1)',
        2: '高波动 (2)'
    }

    # ---- 创建子图（三行） ----
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.2, 0.1],
        vertical_spacing=0.05,
        subplot_titles=("价格与信号", "MA100 斜率 (5日变化%)", "HMM 市场状态 (3种状态)")
    )

    # ===== 第一行：价格 + 均线 + 信号 =====
    # 收盘价
    fig.add_trace(go.Scatter(
        x=df_price['date'],
        y=df_price['close'],
        mode='lines',
        name='收盘价',
        line=dict(color='black', width=1.5)
    ), row=1, col=1)

    # 均线
    ma_colors = {'M20': 'blue', 'M50': 'orange', 'M70': 'green', 'M350': 'red'}
    for ma in ['M20', 'M50', 'M70', 'M350']:
        if ma in df_price.columns:
            fig.add_trace(go.Scatter(
                x=df_price['date'],
                y=df_price[ma],
                mode='lines',
                name=ma,
                line=dict(color=ma_colors.get(ma, 'gray'), width=1, dash='dash')
            ), row=1, col=1)

    # 买入信号
    if not signals_df.empty:
        setup_styles = {
            '1_Consolidation': {'color': 'blue', 'symbol': 'circle', 'name': '下跌/横盘'},
            '2_Pullback':      {'color': 'green', 'symbol': 'triangle-up', 'name': '上涨回撤'},
            '3_Surging':       {'color': 'red', 'symbol': 'star', 'name': '大幅上涨'}
        }
        for setup, group in signals_df.groupby('setup'):
            style = setup_styles.get(setup, {'color': 'gray', 'symbol': 'circle', 'name': setup})
            fig.add_trace(go.Scatter(
                x=group['date'],
                y=group['entry_price'],
                mode='markers',
                name=f'买入 - {style["name"]}',
                marker=dict(
                    symbol=style['symbol'],
                    size=12,
                    color=style['color'],
                    line=dict(width=1, color='white')
                ),
                text=group['note'],
                hovertemplate='<b>%{text}</b><br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

    # ===== 第二行：MA100 斜率柱状图 =====
    fig.add_trace(go.Bar(
        x=slope_data['date'],
        y=slope_data['M100_slope'],
        name='MA100斜率',
        marker_color=colors,
        opacity=0.8
    ), row=2, col=1)

    # 添加零线参考线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1,
                  annotation_text="0", annotation_position="bottom right")

    # ===== 第三行：HMM 状态色带（3种状态） =====
    state_values = state_by_date.values.reshape(1, -1)
    state_dates = state_by_date.index

    colorscale = [
        [0.0, state_colors[0]],
        [0.5, state_colors[1]],
        [1.0, state_colors[2]]
    ]
    fig.add_trace(go.Heatmap(
        z=state_values,
        x=state_dates,
        y=['状态'],
        colorscale=colorscale,
        zmin=0,
        zmax=2,
        showscale=False,
        hoverinfo='x+y+text',
        text=[[state_names[s] for s in state_values.flatten()]],
        hovertemplate='日期: %{x}<br>状态: %{text}<extra></extra>'
    ), row=3, col=1)

    # 添加 HMM 图例（3种）
    for state, color in state_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=state_names[state]
        ), row=3, col=1)

    # ---- 布局设置 ----
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="斜率 (%)", row=2, col=1)
    fig.update_yaxes(
        title_text="",
        row=3, col=1,
        showticklabels=False,
        range=[-0.2, 0.2]
    )

    fig.update_layout(
        title=title,
        template='plotly_white',
        hovermode='x unified',
        height=900,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


if __name__ == "__main__":
    # ---------- 配置 ----------
    TICKER = "MSFT"  # 可改为 "AAPL", "SH600036" 等

    # 加载策略配置
    try:
        stock_config = load_strategy_config(TICKER)   # 返回股票配置字典
    except Exception as e:
        print(f"⚠️ 加载配置失败：{e}，使用默认参数")
        stock_config = {}

    # 从配置中读取参数，若不存在则使用默认值
    data_cfg = stock_config.get('data', {})
    strategy_cfg = stock_config.get('strategy', {})
    START_DATE = data_cfg.get('start_date', "2007-01-01")
    END_DATE = data_cfg.get('end_date', "2026-12-31")
    OUTPUT_DIR = data_cfg.get('output_dir', f"outputs/figures/{TICKER}")
    ATR_PERIOD = strategy_cfg.get('atr_period', 14)
    N_STATES = 3

    # ---------- 新增：检查 processed 文件是否存在，若不存在则运行数据流水线 ----------
    processed_file = os.path.join("data", TICKER, f"{TICKER}_processed.xlsx")
    if not os.path.exists(processed_file):
        print(f"⚠️ 未找到 {TICKER} 的 processed 数据，正在运行数据流水线...")
        try:
            run_data_pipeline(tickers=[TICKER])   # 仅更新当前 TICKER 的数据
        except Exception as e:
            print(f"❌ 数据流水线运行失败：{e}")
            sys.exit(1)

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

    # 提取基础 OHLCV 用于策略
    base_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in base_cols):
        print("❌ 缺少必要的 OHLCV 列，请检查 processed 数据格式。")
        sys.exit(1)
    df_base = df[base_cols].copy()

    # ---------- 1. 运行海龟策略（传入配置） ----------
    print(f"\n🐢 初始化海龟策略（ATR周期={ATR_PERIOD}）...")
    strategy = TurtleStrategy(df_base, config=stock_config, atr_period=ATR_PERIOD)
    print("🔍 扫描买入信号...")
    signals_df = strategy.scan(earnings_soon=False, stop_method='atr')

    # ---------- 2. 运行 HMM ----------
    print("🧮 准备 HMM 特征...")
    features = prepare_features(df)
    print(f"特征矩阵形状：{features.shape}")

    print("🔄 训练 HMM 模型...")
    hmm_model = HMModel(n_states=N_STATES)
    hmm_model.fit(features.values)
    states = hmm_model.predict(features.values)
    state_series = pd.Series(states, index=features.index, name='state')

    # 统计状态分布
    print("\n📊 HMM 状态分布：")
    state_counts = state_series.value_counts().sort_index()
    for s, count in state_counts.items():
        print(f"  State {s}: {count} 天 ({count/len(state_series)*100:.1f}%)")

    # ---------- 3. 合并绘图 ----------
    print(f"\n📈 生成组合图表（三行，含 MA100 斜率柱状图）...")
    fig = plot_combined(
        df_price=strategy.df,
        signals_df=signals_df,
        state_series=state_series,
        title=f"{TICKER} 海龟策略信号 + MA100斜率 + HMM 市场状态 (3种状态)"
    )

    # 确保输出目录存在，并保存为 {TICKER}_pipeline.html
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{TICKER}_pipeline.html")
    fig.write_html(output_path)
    print(f"✅ 组合图表已保存至 {output_path}")

    # ---------- 4. 打印信号摘要 ----------
    setup_names = {
        '1_Consolidation': '场景1：下跌后或横盘中',
        '2_Pullback':      '场景2：上涨回撤中',
        '3_Surging':       '场景3：大幅上涨中'
    }
    print("\n" + "="*60)
    print("策略买入信号 (标准止损 2倍ATR，非财报期)")
    print("="*60)
    if signals_df.empty:
        print("⚠️ 未发现任何符合条件的买入信号。")
    else:
        print(f"✅ 共发现 {len(signals_df)} 个买入信号。")
        grouped = signals_df.groupby('setup')
        for setup_key, group in grouped:
            setup_name = setup_names.get(setup_key, setup_key)
            print(f"\n--- {setup_name} ---")
            print(f"信号数量: {len(group)}")
            print(group.drop(columns=['setup']).head(5).to_string(index=False))
            print("-" * 80)

    print("\n🎯 Pipeline 运行完成！")
    print(f"👉 请用浏览器打开 {output_path} 查看综合图表。")