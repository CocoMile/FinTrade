#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交互式图表绘制器（基于 Plotly）
用于展示海龟策略的买入信号、价格、均线及辅助线
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class StrategyPlotter:
    """
    生成交互式 HTML 图表
    """

    def __init__(self, df, signals_df, output_dir='outputs/figures/'):
        self.df = df.copy()
        self.signals = signals_df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_price_with_signals(self, filename='price_signals.html'):
        fig = go.Figure()

        # 收盘价
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=self.df['close'],
            mode='lines',
            name='收盘价',
            line=dict(color='black', width=1.5)
        ))

        # 均线
        ma_colors = {'M20': 'blue', 'M50': 'orange', 'M70': 'green', 'M350': 'red'}
        for ma in ['M20', 'M50', 'M70', 'M350']:
            if ma in self.df.columns:
                fig.add_trace(go.Scatter(
                    x=self.df['date'],
                    y=self.df[ma],
                    mode='lines',
                    name=ma,
                    line=dict(color=ma_colors.get(ma, 'gray'), width=1, dash='dash')
                ))

        # 买入信号
        if not self.signals.empty:
            setup_styles = {
                '1_Consolidation': {'color': 'blue', 'symbol': 'circle', 'name': '下跌/横盘'},
                '2_Pullback':      {'color': 'green', 'symbol': 'triangle-up', 'name': '上涨回撤'},
                '3_Surging':       {'color': 'red', 'symbol': 'star', 'name': '大幅上涨'}
            }
            for setup, group in self.signals.groupby('setup'):
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
                ))

        # 辅助线：最新 M20 水平线
        last_m20 = self.df['M20'].iloc[-1]
        fig.add_hline(y=last_m20, line_dash="dot", line_color="blue", 
                      annotation_text=f"最新 M20 = {last_m20:.2f}", annotation_position="bottom right")

        fig.update_layout(
            title='价格走势与海龟策略买入信号',
            xaxis_title='日期',
            yaxis_title='价格',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=600
        )

        full_path = os.path.join(self.output_dir, filename)
        fig.write_html(full_path)
        print(f"✅ 价格信号图已保存至 {full_path}")
        return fig

    def plot_equity_curve(self, equity_series, benchmark_series=None, filename='equity_curve.html'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode='lines',
            name='策略净值',
            line=dict(color='darkblue', width=2)
        ))
        if benchmark_series is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_series.index,
                y=benchmark_series.values,
                mode='lines',
                name='买入持有基准',
                line=dict(color='gray', width=1, dash='dot')
            ))
        fig.update_layout(
            title='策略净值 vs 基准',
            xaxis_title='日期',
            yaxis_title='净值 (起始=1)',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        full_path = os.path.join(self.output_dir, filename)
        fig.write_html(full_path)
        print(f"✅ 权益曲线图已保存至 {full_path}")
        return fig

    def plot_scenario_returns(self, trades_df, filename='scenario_returns.html'):
        if trades_df.empty:
            print("无交易数据，跳过箱线图。")
            return None
        fig = go.Figure()
        setups = trades_df['setup'].unique()
        for st in setups:
            data = trades_df[trades_df['setup'] == st]['return_pct']
            fig.add_trace(go.Box(
                y=data,
                name=st,
                boxmean='sd',
                marker_color='lightblue'
            ))
        fig.update_layout(
            title='各场景交易收益率分布',
            xaxis_title='场景',
            yaxis_title='收益率 (%)',
            template='plotly_white',
            height=450
        )
        full_path = os.path.join(self.output_dir, filename)
        fig.write_html(full_path)
        print(f"✅ 场景收益率箱线图已保存至 {full_path}")
        return fig

    def plot_all(self, equity_series=None, benchmark_series=None, trades_df=None):
        self.plot_price_with_signals()
        if equity_series is not None:
            self.plot_equity_curve(equity_series, benchmark_series)
        if trades_df is not None and not trades_df.empty:
            self.plot_scenario_returns(trades_df)