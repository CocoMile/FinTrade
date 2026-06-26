#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 将项目根目录添加到 sys.path，以便导入 src 下的其他模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))  # 回到项目根目录
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.features.data_loader import DataLoader, run_data_pipeline
# from src.visualization.plotter import StrategyPlotter  # 已注释，不再生成 price_signals.html

def load_strategy_config(ticker):
    """加载指定股票的策略配置"""
    config_path=f'config/{ticker}_settings.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    stock_cfg = config['stocks'].get(ticker)
    return stock_cfg


# ========================== 海龟策略（买入头寸版） ==========================
class TurtleStrategy:
    """
    基于三种形态的买入头寸策略
    1. 下跌/横盘中的买入
    2. 上涨回撤中的买入
    3. 大幅上涨中的买入
    """

    def __init__(self, df, config, atr_period=14):
        """
        初始化策略
        :param df: 包含 'open', 'high', 'low', 'close' 的 DataFrame（日期升序）
        :param atr_period: ATR 计算周期，默认14
        """
        self.df = df.copy()
        self.config = config
        self.atr_period = atr_period
        self._calculate_indicators()

    def _calculate_indicators(self):
        """计算所有需要的技术指标"""
        df = self.df
        
        # 1. 均线系统
        for period in [10, 20, 50, 70, 350]:
            df[f'M{period}'] = df['close'].rolling(window=period).mean()
        
        # 2. ATR (平均真实波幅)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['ATR'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # 3. 5日斜率 (百分比变化)
        for period in [20, 50, 70]:
            df[f'M{period}_slope'] = (df[f'M{period}'] / df[f'M{period}'].shift(5) - 1) * 100
        
        # 4. M20 连续3日向上 (严格连续)
        df['M20_rising_3d'] = (
            (df['M20'] > df['M20'].shift(1)) &
            (df['M20'].shift(1) > df['M20'].shift(2)) &
            (df['M20'].shift(2) > df['M20'].shift(3))
        )
        
        # 5. M20与M350的差距 (百分比)
        df['gap_20_350'] = (df['M350'] - df['M20']) / df['M350'] * 100
        
        # 6. 缠绕程度 (M20, M50, M70 三者之间的最大距离百分比)
        df['entanglement'] = df[['M20', 'M50', 'M70']].max(axis=1) / df[['M20', 'M50', 'M70']].min(axis=1) - 1
        df['entanglement'] *= 100  # 转为百分比
        
        # 7. 20日高低点 (用于突破/破位)
        df['high_20d_prev'] = df['high'].rolling(window=20).max().shift(1)
        df['low_20d'] = df['low'].rolling(window=20).min()
        
        # ---------- 用于场景3的斜率差 ----------
        df['M20_slope_diff_50'] = df['M20_slope'] - df['M50_slope']
        df['M20_slope_diff_70'] = df['M20_slope'] - df['M70_slope']
        df['M20_slope_diff_50_prev'] = df['M20_slope_diff_50'].shift(1)
        df['M20_slope_diff_70_prev'] = df['M20_slope_diff_70'].shift(1)
        
        # 清理临时列
        df.drop('tr', axis=1, inplace=True)

    def _apply_stop_loss(self, entry_price, atr, earnings_soon=False, stop_method='atr', low_20d=None):
        """
        统一计算止损价
        :param earnings_soon: 是否财报前，True则小于2倍ATR (取1.5倍)
        :param stop_method: 'atr' 或 'low_break' (20日向下破位)
        """
        multiplier = 1.5 if earnings_soon else 2.0
        
        if stop_method == 'low_break' and low_20d is not None:
            # 20日向下破位法：跌破20日最低点止损
            return low_20d * 0.995  # 留一点缓冲
        else:
            # 标准ATR止损
            return entry_price - multiplier * atr

    def scan(self, earnings_soon=False, stop_method='atr'):
        """
        扫描全市场数据，生成买入信号
        :param earnings_soon: 是否在财报发布前，True则调低止损倍数
        :param stop_method: 止损方式 ('atr' 或 'low_break')
        :return: DataFrame 包含所有买入信号
        """
        df = self.df
        all_signals = []
        
        # 从第350根K线开始，确保所有均线有效
        for i in range(350, len(df)):
            row = df.iloc[i]
            
            # 跳过ATR无效的时期
            if pd.isna(row['ATR']):
                continue
            
            # ---- 检查三种买入场景 ----
            signals_1 = self._check_setup_1(i, row, earnings_soon, stop_method)
            signals_2 = self._check_setup_2(i, row, earnings_soon, stop_method)
            signals_3 = self._check_setup_3(i, row, earnings_soon, stop_method)
            
            # 合并信号
            for sig in (signals_1 or []) + (signals_2 or []) + (signals_3 or []):
                sig['date'] = row['date']
                all_signals.append(sig)
        
        if not all_signals:
            return pd.DataFrame()
        
        return pd.DataFrame(all_signals)

    # ---------- 场景 1: 下跌后或者横盘中 ----------
    def _check_setup_1(self, i, row, earnings_soon, stop_method):
        cfg = self.config.get('scenario1', {})
        # 读取阈值，若不存在则使用原有硬编码值
        m20_slope_min = cfg.get('m20_slope_min', 0)
        m350_slope_min = cfg.get('m350_slope_min', -0.5)
        gap_max = cfg.get('gap_20_350_max', 10)
        m50_slope_min = cfg.get('m50_slope_min', -1)
        m70_slope_min = cfg.get('m70_slope_min', -1)
        entanglement_max = cfg.get('entanglement_max', 5)  
        
        # a. M20至少水平或向上 (斜率 >= 0)
        if row['M20_slope'] < m20_slope_min:
            return None
        
        # b. 如果M20在M350下方 (必须连续3日向上，且差距小于5%)
        #    注：原规则为10%，但后续要求<5%更严，直接采用<5% (即为有效条件，过滤掉10%的冗余)
        m350_slope = (row['M350'] / self.df['M350'].shift(5).iloc[i] - 1) * 100
        if m350_slope < m350_slope_min:
            return None
        if not row['M20_rising_3d'] or row['gap_20_350'] >= gap_max:
            return None
        
        # c. M50, M70 向下斜率不大 (5日跌幅 < 2% 即 > -2)
        if row['M50_slope'] < m50_slope_min or row['M70_slope'] < m70_slope_min:
            return None
        
        # 缠绕最好 (三者最大距离 < 6%) - 作为加分项或过滤项，这里设为可选过滤
        # 如果不强制，可以注释掉下行；这里我们作为硬性条件（体现“最好”的意图）
        if row['entanglement'] >= entanglement_max:
            return None
        
        # 计算入场价 (以收盘价建仓)
        entry = row['close']
        stop = self._apply_stop_loss(entry, row['ATR'], earnings_soon, stop_method, row['low_20d'])
        
        return [{
            'setup': '1_Consolidation',
            'entry_price': round(entry, 2),
            'stop_loss': round(stop, 2),
            'position_type': 'First',
            'note': f'M20斜率:{row["M20_slope"]:.1f}%, 缠绕:{row["entanglement"]:.1f}%'
        }]

    # ---------- 场景 2: 上涨回撤中 ----------
    def _check_setup_2(self, i, row, earnings_soon, stop_method):
        signals = []
        cfg = self.config.get('scenario2', {})
        m20_slope_max = cfg.get('m20_slope_max', 0)
        m50_slope_min = cfg.get('m50_slope_min', 0)
        m70_slope_min = cfg.get('m70_slope_min', 0)
        near_m20_threshold = cfg.get('near_m20_threshold', 0.01)
        near_m50_threshold = cfg.get('near_m50_threshold', 0.01)
        near_m70_threshold = cfg.get('near_m70_threshold', 0.01)

        # a. M20接近水平或向下 (斜率 <= 0)，但 M50, M70 向上 (斜率 > 0)
        if row['M20_slope'] > m20_slope_max:
            return None
        if row['M50_slope'] <= m50_slope_min or row['M70_slope'] <= m70_slope_min:
            return None

        # b. 首个头寸：在 M20 均线点位买入
        price = row['close']
        if abs(price - row['M20']) / row['M20'] < near_m20_threshold:
            entry_first = row['close']
            stop_first = self._apply_stop_loss(entry_first, row['ATR'], earnings_soon, stop_method, row['low_20d'])
            signals.append({
                'setup': '2_Pullback',
                'entry_price': round(entry_first, 2),
                'stop_loss': round(stop_first, 2),
                'position_type': 'First',
                'note': '首次入场于 M20 均线'
            })

        # c. 后续头寸：如果当前价格贴近 M50 或 M70
        if abs(price - row['M50']) / row['M50'] < near_m50_threshold:
            stop_add = self._apply_stop_loss(price, row['ATR'], earnings_soon, stop_method, row['low_20d'])
            signals.append({
                'setup': '2_Pullback',
                'entry_price': round(price, 2),
                'stop_loss': round(stop_add, 2),
                'position_type': 'Add-on',
                'note': '加仓于 M50 均线'
            })
        if abs(price - row['M70']) / row['M70'] < near_m70_threshold:
            stop_add = self._apply_stop_loss(price, row['ATR'], earnings_soon, stop_method, row['low_20d'])
            signals.append({
                'setup': '2_Pullback',
                'entry_price': round(price, 2),
                'stop_loss': round(stop_add, 2),
                'position_type': 'Add-on',
                'note': '加仓于 M70 均线'
            })

        return signals if signals else None

    # ---------- 场景 3: 大幅上涨中（修复：增加均线方向向上约束） ----------
    def _check_setup_3(self, i, row, earnings_soon, stop_method):
        signals = []
        cfg = self.config.get('scenario3', {})
        slope_min = cfg.get('slope_min', 0)
        obvious_gap = cfg.get('obvious_gap', 0.5)
        volume_ratio = cfg.get('volume_ratio', 1.2)
        near_m10_threshold = cfg.get('near_m10_threshold', 0.02)
        near_m20_threshold = cfg.get('near_m20_threshold', 0.02)

        # 均线必须全部向上
        if row['M20_slope'] <= slope_min or row['M50_slope'] <= slope_min or row['M70_slope'] <= slope_min:
            return None

        # M20 斜率条件 (二选一)
        cond1_obvious = (
            row['M20_slope'] > row['M50_slope'] + obvious_gap and
            row['M20_slope'] > row['M70_slope'] + obvious_gap
        )
        cond2_expanding = (
            row['M20_slope_diff_50'] > row['M20_slope_diff_50_prev'] and
            row['M20_slope_diff_70'] > row['M20_slope_diff_70_prev']
        )
        if not (cond1_obvious or cond2_expanding):
            return None

        # 首个头寸：带量突破
        volume_ma20 = self.df['volume'].rolling(20).mean().iloc[i]
        is_volume_confirmed = not pd.isna(volume_ma20) and row['volume'] > volume_ma20 * volume_ratio
        if row['close'] > row['high_20d_prev'] and is_volume_confirmed:
            entry_first = row['close']
            note = f'20日带量突破 (量比{row["volume"]/volume_ma20:.1f}倍)'
            stop_first = self._apply_stop_loss(entry_first, row['ATR'], earnings_soon, stop_method, row['low_20d'])
            signals.append({
                'setup': '3_Surging',
                'entry_price': round(entry_first, 2),
                'stop_loss': round(stop_first, 2),
                'position_type': 'First',
                'note': note
            })

        # 后续加仓
        price = row['close']
        if abs(price - row['M10']) / row['M10'] < near_m10_threshold:
            stop_add = self._apply_stop_loss(price, row['ATR'], earnings_soon, stop_method, row['low_20d'])
            signals.append({
                'setup': '3_Surging',
                'entry_price': round(price, 2),
                'stop_loss': round(stop_add, 2),
                'position_type': 'Add-on',
                'note': '加仓于 M10 均线附近'
            })
        if abs(price - row['M20']) / row['M20'] < near_m20_threshold:
            stop_add = self._apply_stop_loss(price, row['ATR'], earnings_soon, stop_method, row['low_20d'])
            signals.append({
                'setup': '3_Surging',
                'entry_price': round(price, 2),
                'stop_loss': round(stop_add, 2),
                'position_type': 'Add-on',
                'note': '加仓于 M20 均线附近'
            })

        return signals if signals else None

    # ---------- 新增：卖出信号（规则三） ，更名为 _stop_setup_1 ----------
    def _stop_setup_1(self):
        """
        根据规则三生成卖出信号点：
        - M20、M50、M70 缠绕（entanglement < 阈值）
        - 三条均线斜率均为负
        - 股价贴近 M20（偏离度 < 阈值）
        满足以上条件时，标记该 K 线为卖出信号点。
        """
        df = self.df
        # 从配置中读取退出参数（若无则使用默认值）
        exit_cfg = self.config.get('exit', {})
        confluence_threshold = exit_cfg.get('confluence_threshold', 2.0)   # 缠绕阈值（%）
        near_ma20_threshold = exit_cfg.get('near_ma20_threshold', 1.5)    # 贴近M20阈值（%）

        # 初始化列
        df['exit_signal'] = False
        df['exit_reason'] = ''

        # 从足够长的位置开始遍历（确保所有均线有效）
        for i in range(350, len(df)):
            row = df.iloc[i]
            # 跳过任何指标计算不完整的行
            if pd.isna(row['entanglement']) or pd.isna(row['M20_slope']) or pd.isna(row['M50_slope']) or pd.isna(row['M70_slope']):
                continue
            # 条件1：缠绕
            if row['entanglement'] >= confluence_threshold:
                continue
            # 条件2：三条均线斜率均为负
            if row['M20_slope'] >= 0 or row['M50_slope'] >= 0 or row['M70_slope'] >= 0:
                continue
            # 条件3：股价贴近 M20（偏离度小于阈值）
            if abs(row['close'] - row['M20']) / row['M20'] * 100 >= near_ma20_threshold:
                continue

            # 所有条件满足 → 标记卖出信号
            df.at[df.index[i], 'exit_signal'] = True
            df.at[df.index[i], 'exit_reason'] = '均线缠绕向下_反抽MA20'

        return df


# ========================== 独立测试入口 ==========================
if __name__ == "__main__":
    """
    使用真实数据（通过 DataLoader）测试策略，并生成交互式图表。
    可自定义股票代码和日期范围。
    """
    # ---------- 配置 ----------
    TICKER = "MSFT"  # 可以从 settings.json 读取或硬编码
    # 加载策略配置
    try:
        stock_config = load_strategy_config(TICKER)   # 返回股票配置字典
    except Exception as e:
        print(f"⚠️ 加载配置失败：{e}，使用默认参数")
        stock_config = {}
    
    # 从配置中读取数据范围（如果策略配置有 data 部分）
    data_cfg = stock_config.get('data', {})
    START_DATE = data_cfg.get('start_date', "2007-01-01")
    END_DATE = data_cfg.get('end_date', "2026-12-31")
    OUTPUT_DIR = data_cfg.get('output_dir', f"outputs/figures/{TICKER}")
    ATR_PERIOD = 14

    print(f"📊 正在加载 {TICKER} 数据（{START_DATE} ~ {END_DATE}）...")
    
    # 若 processed 文件不存在，自动运行数据流水线
    processed_file = os.path.join("data", TICKER, f"{TICKER}_processed.xlsx")
    if not os.path.exists(processed_file):
        print(f"⚠️ 未找到 {TICKER} 的 processed 数据，正在运行数据流水线...")
        try:
            run_data_pipeline(tickers=[TICKER])
        except Exception as e:
            print(f"❌ 数据流水线运行失败：{e}")
            sys.exit(1)

    # 加载数据（使用 DataLoader）
    loader = DataLoader(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        data_root="data"            # 数据目录相对于项目根目录
    )
    
    try:
        df_processed = loader.load_processed_data()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 请先运行数据流水线：python src/features/data_loader.py")
        sys.exit(1)

    if df_processed.empty:
        print("⚠️ 数据为空，请检查日期范围。")
        sys.exit(1)

    # 策略需要升序数据（从旧到新）
    df_processed = df_processed.sort_values("date", ascending=True).reset_index(drop=True)
    print(f"✅ 数据加载完成，共 {len(df_processed)} 行")

    # 提取策略需要的列（OHLCV）
    base_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df_processed.columns for col in base_cols):
        print("❌ 缺少必要的 OHLCV 列，请检查 processed 数据格式。")
        sys.exit(1)
    df_base = df_processed[base_cols].copy()

    # ---------- 运行策略 ----------
    print(f"\n🐢 初始化海龟策略（ATR周期={ATR_PERIOD}）...")
    strategy = TurtleStrategy(df_base, stock_config, atr_period=ATR_PERIOD)

    print("🔍 扫描买入信号...")
    signals_df = strategy.scan(earnings_soon=False, stop_method='atr')

    # ---------- 新增：生成卖出信号（规则三），调用 _stop_setup_1 ----------
    print("🔻 生成卖出信号（规则三：均线缠绕向下，反抽MA20）...")
    strategy._stop_setup_1()          # 更新 strategy.df
    df = strategy.df                  # 获取更新后的数据（包含 exit_signal 列）
    exit_signals = df[df['exit_signal'] == True]
    print(f"✅ 共发现 {len(exit_signals)} 个卖出信号点。")

    # ---------- 控制台输出 ----------
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

    # ========== 已移除生成 price_signals.html 的代码 ==========
    # 不再调用 StrategyPlotter，因此不再生成 price_signals.html

    # ---------- 新增：组合图（价格+信号 + MA100斜率柱状图） ----------
    print(f"\n📈 生成组合图（价格+信号 + MA100斜率柱状图）...")
    # 注意：df 已经包含 exit_signal 列（从 strategy.df 复制而来）
    # 但为保险，我们重新从 strategy.df 复制一份，以确保包含最新列
    df = strategy.df.copy()
    
    # 计算 MA100 和斜率
    df['M100'] = df['close'].rolling(window=100).mean()
    df['M100_slope'] = (df['M100'] / df['M100'].shift(5) - 1) * 100
    slope_data = df[['date', 'M100_slope']].dropna()
    
    # 为柱状图设置颜色：正值浅绿，负值浅红
    colors = ["#08FD08" if val >= 0 else "#F80126" for val in slope_data['M100_slope']]

    # 创建组合图：2行，高度比 0.8 / 0.2
    fig_combined = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.05,
        subplot_titles=("价格与信号", "MA100 斜率 (5日变化%)")
    )

    # 上子图：价格 + 均线 + 信号（与 StrategyPlotter 类似，但需要独立复制）
    # 收盘价
    fig_combined.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='收盘价',
        line=dict(color='black', width=1.5)
    ), row=1, col=1)

    # 均线
    ma_colors = {'M20': 'blue', 'M50': 'orange', 'M70': 'green', 'M350': 'red'}
    for ma in ['M20', 'M50', 'M70', 'M350']:
        if ma in df.columns:
            fig_combined.add_trace(go.Scatter(
                x=df['date'],
                y=df[ma],
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
            fig_combined.add_trace(go.Scatter(
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

    # ---------- 新增：卖出信号（规则三） ----------
    if not exit_signals.empty:
        fig_combined.add_trace(go.Scatter(
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

    # 下子图：MA100 斜率柱状图
    fig_combined.add_trace(go.Bar(
        x=slope_data['date'],
        y=slope_data['M100_slope'],
        name='MA100斜率',
        marker_color=colors,
        opacity=0.8
    ), row=2, col=1)

    # 添加零线参考线
    fig_combined.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1,
                           annotation_text="0", annotation_position="bottom right")

    # 布局设置
    fig_combined.update_yaxes(title_text="价格", row=1, col=1)
    fig_combined.update_yaxes(title_text="斜率 (%)", row=2, col=1)

    fig_combined.update_layout(
        title=f"{TICKER} 海龟策略信号与 MA100 斜率",
        template='plotly_white',
        hovermode='x unified',
        height=800,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # 确保输出目录存在，并保存为 {TICKER}_turtle_strategy.html
    combined_path = os.path.join(OUTPUT_DIR, f"{TICKER}_turtle_strategy.html")
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    fig_combined.write_html(combined_path)
    print(f"✅ 组合图已保存至 {combined_path}")

    print("\n🎯 测试完成！")
    print(f"👉 请用浏览器打开 {combined_path} 查看组合图（含MA100斜率柱状图）。")