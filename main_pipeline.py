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
import re
from datetime import datetime
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


def plot_combined(df_price, signals_df, state_series, title="海龟策略信号与HMM市场状态", exit_signals=None):
    """
    绘制三行组合图表：
    上子图：价格 + 均线 + 买卖信号（买入+卖出）
    中子图：MA100 斜率柱状图（正浅绿，负浅红）
    下子图：HMM 市场状态色带（3种状态）

    新增参数：
        exit_signals: DataFrame 包含 'date', 'close', 'exit_reason'，用于显示卖出信号
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

    # ---- 新增：卖出信号（如果提供了 exit_signals） ----
    if exit_signals is not None and not exit_signals.empty:
        fig.add_trace(go.Scatter(
            x=exit_signals['date'],
            y=exit_signals['close'],
            mode='markers',
            name='卖出信号 (均线缠绕向下)',
            marker=dict(
                symbol='triangle-down',
                size=14,
                color='red',
                line=dict(width=1, color='darkred')
            ),
            text=exit_signals['exit_reason'],
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


def normalize_tickers(tickers_input):
    """标准化 ticker 输入，支持 list/tuple 或逗号分隔字符串。"""
    if tickers_input is None:
        return ["AAPL"]

    if isinstance(tickers_input, str):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        return tickers or ["AAPL"]

    if isinstance(tickers_input, (list, tuple, set)):
        tickers = [str(t).strip().upper() for t in tickers_input if str(t).strip()]
        return tickers or ["AAPL"]

    raise TypeError("tickers_input 必须是 list/tuple/set 或逗号分隔字符串")


def should_refresh_processed_data(processed_file, end_date):
    """
    判断是否需要更新 processed 数据。
    规则：若 processed 文件最新日期 < min(end_date, today)，则需要更新。
    """
    today = pd.Timestamp.today().normalize()
    end_dt = pd.to_datetime(end_date, errors='coerce')
    target_dt = min(end_dt.normalize(), today) if pd.notna(end_dt) else today

    if not os.path.exists(processed_file):
        return True, None, target_dt, "未找到 processed 文件"

    try:
        df_date = pd.read_excel(processed_file, usecols=['date'])
        if df_date.empty or 'date' not in df_date.columns:
            return True, None, target_dt, "processed 文件缺少有效 date 列"

        latest_dt = pd.to_datetime(df_date['date'], errors='coerce').dropna().max()
        if pd.isna(latest_dt):
            return True, None, target_dt, "processed 文件日期列无法解析"

        latest_dt = latest_dt.normalize()
        needs_update = latest_dt < target_dt
        reason = f"最新日期 {latest_dt.date()}，目标日期 {target_dt.date()}"
        return needs_update, latest_dt, target_dt, reason
    except Exception as e:
        return True, None, target_dt, f"读取 processed 文件失败: {e}"


def run_one_ticker_pipeline(ticker, n_states=3):
        """运行单个 ticker 的完整流程，返回图表与摘要信息。"""
        print("\n" + "=" * 80)
        print(f"🚀 开始处理 {ticker}")
        print("=" * 80)

        # 加载策略配置
        try:
                stock_config = load_strategy_config(ticker)
        except Exception as e:
                print(f"⚠️ 加载 {ticker} 配置失败：{e}，使用默认参数")
                stock_config = {}

        data_cfg = stock_config.get('data', {})
        strategy_cfg = stock_config.get('strategy', {})
        start_date = data_cfg.get('start_date', "2007-01-01")
        end_date = data_cfg.get('end_date', "2026-12-31")
        output_dir = data_cfg.get('output_dir', f"outputs/figures/{ticker}")
        atr_period = strategy_cfg.get('atr_period', 14)

        # 按 end_date、今天日期、processed 最新日期决定是否更新数据
        processed_file = os.path.join("data", ticker, f"{ticker}_processed.xlsx")
        need_update, latest_dt, target_dt, update_reason = should_refresh_processed_data(processed_file, end_date)
        if need_update:
            print(f"⚠️ {ticker} 需要更新数据（{update_reason}），正在运行数据流水线...")
            try:
                run_data_pipeline(tickers=[ticker])
            except Exception as e:
                raise RuntimeError(f"数据流水线运行失败：{e}") from e
        else:
            print(f"✅ {ticker} 数据为最新（{update_reason}），跳过更新")

        print(f"📊 加载 {ticker} 数据（{start_date} ~ {end_date}）...")
        loader = DataLoader(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                data_root="data"
        )
        df = loader.load_processed_data()
        if df.empty:
                raise ValueError(f"{ticker} 在所选日期范围内数据为空")

        # 转为升序
        df = df.sort_values("date", ascending=True).reset_index(drop=True)

        # 提取基础 OHLCV 用于策略
        base_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in base_cols):
                raise ValueError(f"{ticker} 缺少必要的 OHLCV 列")
        df_base = df[base_cols].copy()

        # 1) 运行海龟策略
        print(f"🐢 初始化海龟策略（ATR周期={atr_period}）...")
        strategy = TurtleStrategy(df_base, config=stock_config, atr_period=atr_period)
        print("🔍 扫描买入信号...")
        signals_df = strategy.scan(earnings_soon=False, stop_method='atr')

        # 生成卖出信号（规则三）
        print("🔻 生成卖出信号（规则三：均线缠绕向下，反抽MA20）...")
        strategy._stop_setup_1()
        df_with_exit = strategy.df
        exit_signals = df_with_exit[df_with_exit['exit_signal'] == True]
        print(f"✅ 共发现 {len(exit_signals)} 个卖出信号点。")

        # 2) 运行 HMM
        print("🧮 准备 HMM 特征...")
        features = prepare_features(df)
        print(f"特征矩阵形状：{features.shape}")

        print("🔄 训练 HMM 模型...")
        hmm_model = HMModel(n_states=n_states)
        hmm_model.fit(features.values)
        states = hmm_model.predict(features.values)
        state_series = pd.Series(states, index=features.index, name='state')

        # 统计状态分布
        print("📊 HMM 状态分布：")
        state_counts = state_series.value_counts().sort_index()
        for s, count in state_counts.items():
                print(f"  State {s}: {count} 天 ({count / len(state_series) * 100:.1f}%)")

        # 3) 绘图
        print("📈 生成组合图表（三行，含 MA100 斜率柱状图，并显示卖出信号）...")
        fig = plot_combined(
                df_price=strategy.df,
                signals_df=signals_df,
                state_series=state_series,
                title=f"{ticker} 海龟策略信号 + MA100斜率 + HMM 市场状态 (3种状态)",
                exit_signals=exit_signals
        )

        return {
                "ticker": ticker,
                "figure": fig,
                "signals_df": signals_df,
                "output_dir": output_dir
        }


def save_integrated_dashboard(results, output_path):
        """保存整合模式 HTML，含 ticker 下拉菜单切换。"""
        tickers = [item["ticker"] for item in results]
        option_html = "\n".join([
                f'<option value="{t}">{t}</option>' for t in tickers
        ])

        plot_blocks = []
        for idx, item in enumerate(results):
                ticker = item["ticker"]
                fig = item["figure"]
                safe_id = "plot_" + re.sub(r'[^A-Za-z0-9_]', '_', ticker)
                plot_html = fig.to_html(
                        full_html=False,
                        include_plotlyjs=(idx == 0),
                        div_id=safe_id,
                        config={"responsive": True}
                )
                display = "block" if idx == 0 else "none"
                wrapped = f'<div class="ticker-panel" data-ticker="{ticker}" style="display:{display};">{plot_html}</div>'
                plot_blocks.append(wrapped)

        html = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>FinTrade Integrated Dashboard</title>
    <style>
        body {{
            margin: 0;
            padding: 16px;
            font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
            background: #f6f8fb;
            color: #222;
        }}
        .toolbar {{
            position: sticky;
            top: 0;
            z-index: 10;
            background: #ffffff;
            border: 1px solid #e6e9f0;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .toolbar label {{
            font-weight: 600;
        }}
        .toolbar select {{
            border: 1px solid #c8d0e0;
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 14px;
            background: #fff;
        }}
    </style>
</head>
<body>
    <div class=\"toolbar\">
        <label for=\"tickerSelect\">选择Ticker:</label>
        <select id=\"tickerSelect\">{option_html}</select>
    </div>

    {''.join(plot_blocks)}

    <script>
        const selectEl = document.getElementById('tickerSelect');
        const panels = document.querySelectorAll('.ticker-panel');

        function switchTicker(ticker) {{
            panels.forEach((panel) => {{
                panel.style.display = panel.dataset.ticker === ticker ? 'block' : 'none';
            }});
            window.dispatchEvent(new Event('resize'));
        }}

        selectEl.addEventListener('change', (e) => switchTicker(e.target.value));
        switchTicker(selectEl.value);
    </script>
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)


def run_multi_ticker_pipeline(tickers_input, mode="standalone", n_states=3):
    """运行多 ticker 主流程。"""
    tickers = normalize_tickers(tickers_input)
    mode = str(mode).strip().lower()
    if mode not in {"standalone", "integrated"}:
        raise ValueError("mode 必须是 'standalone' 或 'integrated'")

    print("\n📌 本次运行参数：")
    print(f"- tickers: {tickers}")
    print(f"- mode: {mode}")
    print(f"- n_states: {n_states}")

    results = []
    failed = []

    for ticker in tickers:
        try:
            result = run_one_ticker_pipeline(ticker=ticker, n_states=n_states)
            results.append(result)
        except Exception as e:
            print(f"❌ {ticker} 处理失败：{e}")
            failed.append(ticker)

    if not results:
        print("\n❌ 所有 ticker 均处理失败，程序结束。")
        sys.exit(1)

    if mode == "standalone":
        for item in results:
            ticker = item["ticker"]
            fig = item["figure"]
            output_dir = item["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{ticker}_pipeline.html")
            fig.write_html(output_path)
            item["output_path"] = output_path
            print(f"✅ [{ticker}] 组合图表已保存至 {output_path}")
    else:
        integrated_dir = os.path.join("outputs", "figures")
        os.makedirs(integrated_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        integrated_name = f"integrated_pipeline_{timestamp}.html"
        integrated_path = os.path.join(integrated_dir, integrated_name)
        save_integrated_dashboard(results, integrated_path)
        print(f"✅ 整合图表已保存至 {integrated_path}")

    # ---------- 打印信号摘要 ----------
    setup_names = {
        '1_Consolidation': '场景1：下跌后或横盘中',
        '2_Pullback':      '场景2：上涨回撤中',
        '3_Surging':       '场景3：大幅上涨中'
    }
    for item in results:
        ticker = item["ticker"]
        signals_df = item["signals_df"]
        print("\n" + "=" * 60)
        print(f"[{ticker}] 策略买入信号 (标准止损 2倍ATR，非财报期)")
        print("=" * 60)
        if signals_df.empty:
            print("⚠️ 未发现任何符合条件的买入信号。")
            continue
        print(f"✅ 共发现 {len(signals_df)} 个买入信号。")
        grouped = signals_df.groupby('setup')
        for setup_key, group in grouped:
            setup_name = setup_names.get(setup_key, setup_key)
            print(f"\n--- {setup_name} ---")
            print(f"信号数量: {len(group)}")
            print(group.drop(columns=['setup']).head(5).to_string(index=False))
            print("-" * 80)

    if failed:
        print(f"\n⚠️ 以下 ticker 处理失败：{failed}")

    print("\n🎯 Pipeline 运行完成！")
    if mode == "standalone":
        print("👉 已按独立模式输出每个 ticker 的 HTML。")
    else:
        print("👉 已输出整合模式 HTML（含 ticker 菜单切换）。")


if __name__ == "__main__":
    # ---------- 输入区（按需修改） ----------
    TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "AMD", "NFLX", "V", "GOOGL", "META", "TSM", "META"]  # 可修改为任意股票代码列表
    MODE = "integrated"   # 可选: "standalone" 或 "integrated"
    N_STATES = 3

    run_multi_ticker_pipeline(
        tickers_input=TICKERS,
        mode=MODE,
        n_states=N_STATES
    )