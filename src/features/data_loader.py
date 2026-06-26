#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader.py
数据统一加载 & 特征工程（ATR、突破价、波动率等）

职责：
- 读取 config/settings.json 配置
- 下载/更新原始行情数据 → 保存至 data/{ticker}/{ticker}_raw.xlsx（日期升序，A列自动列宽）
- 计算所有特征 → 保存至 data/{ticker}/{ticker}_processed.xlsx（日期降序，A列自动列宽）
- 提供 DataLoader 类供其他模块调用（加载已处理的数据）
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import akshare as ak
import yfinance as yf
from openpyxl.utils import get_column_letter

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)


def load_strategy_config(ticker):
    """加载指定股票配置；若不存在则回退到通用配置。"""
    stock_path = os.path.join(project_root, "config", f"{ticker}_settings.json")
    general_path = os.path.join(project_root, "config", "general_setting.json")
    config_path = stock_path if os.path.exists(stock_path) else general_path

    with open(config_path, 'r', encoding='utf-8') as f:
        stocks = json.load(f).get('stocks', {})

    return stocks.get(ticker) or stocks.get('DEFAULT', {})


# ==================== 数据获取 ====================
def fetch_a_stock(symbol: str):
    """使用 AkShare 获取 A 股数据，日期列为字符串 'YYYY-MM-DD'"""
    print(f"📈 正在下载 A 股数据：{symbol}")
    try:
        code = symbol[2:] if symbol.startswith(("sh", "sz")) else symbol
        df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
        df.reset_index(inplace=True)
        df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception as e:
        print(f"❌ 下载 A 股 {symbol} 失败：{e}")
        return None


def fetch_us_stock(symbol: str):
    """
    使用 yfinance 获取美股数据，使用 period="max" 获取全部历史
    日期列为字符串 'YYYY/MM/DD' 格式，避免时区问题
    """
    print(f"📈 正在下载美股数据：{symbol}")
    try:
        time.sleep(0.5)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max")  # 获取尽可能长的历史
        if df is not None and not df.empty:
            df.reset_index(inplace=True)
            df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y/%m/%d")
            df.rename(columns={
                "Open": "open",
                "Close": "close",
                "High": "high",
                "Low": "low",
                "Volume": "volume"
            }, inplace=True)
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("date").reset_index(drop=True)
            return df
        else:
            print(f"⚠️ 未获取到数据 (symbol={symbol})")
            return None
    except Exception as e:
        print(f"❌ 下载美股 {symbol} 失败：{e}")
        return None


# ==================== 保存函数（自动调整A列宽度） ====================
def save_to_excel(df: pd.DataFrame, file_path: str, sheet_name: str = "Sheet1"):
    if df is None or df.empty:
        print(f"⚠️ 数据为空，跳过保存 {file_path}")
        return False
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]
        date_strs = df["date"].astype(str)
        max_len = max(date_strs.map(len).max(), len("date")) + 2
        worksheet.column_dimensions[get_column_letter(1)].width = max_len
    return True


def save_raw_data(df: pd.DataFrame, ticker: str, data_root: str):
    if df is None or df.empty:
        print(f"⚠️ {ticker} 原始数据为空，不保存")
        return False
    folder = os.path.join(data_root, ticker)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{ticker}_raw.xlsx")
    return save_to_excel(df, file_path)


def save_processed_data(df: pd.DataFrame, ticker: str, data_root: str):
    if df is None or df.empty:
        print(f"⚠️ {ticker} 处理后的数据为空，不保存")
        return False
    folder = os.path.join(data_root, ticker)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{ticker}_processed.xlsx")
    return save_to_excel(df, file_path)


# ==================== 特征工程（修改部分） ====================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加全部特征，输入升序，输出降序，所有数值列保留4位小数"""
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["close_prev"] = df["close"].shift(1)
    df["return_t"] = (df["close"] - df["close_prev"]) / df["close_prev"]
    df["high_lower_t"] = (df["high"] - df["close_prev"]) / df["close_prev"]
    df["low_lower_t"] = (df["low"] - df["close_prev"]) / df["close_prev"]
    df["open_gap_t"] = (df["open"] - df["close_prev"]) / df["close_prev"]

    df["vol_ma20"] = df["volume"].shift(1).rolling(window=20, min_periods=20).mean()
    df["volume_ratio_t"] = df["volume"] / df["vol_ma20"]

    high_low = df["high"] - df["low"]
    high_prev_close = abs(df["high"] - df["close"].shift(1))
    low_prev_close = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14, min_periods=14).mean()

    # ======== 删除以下计算 ========
    # df["high_breakout"] = ...
    # df["low_breakout"] = ...
    # df["exit_breakout"] = ...
    # df["normalized_return"] = ...

    # 波动率（保留）
    df["volatility_20d"] = df["return_t"].rolling(window=20, min_periods=20).std()

    # 均线特征（新参数：MA20, MA50, MA70, MA350）
    df["MA_20"] = df["close"].rolling(window=20, min_periods=20).mean()
    df["MA_50"] = df["close"].rolling(window=50, min_periods=50).mean()
    df["MA_70"] = df["close"].rolling(window=70, min_periods=70).mean()
    df["MA_350"] = df["close"].rolling(window=350, min_periods=350).mean()

    # ======== 新增特征 ========
    # 1. MA20/MA350 比率
    df["ma20_ma350_ratio"] = df["MA_20"] / df["MA_350"]

    # 2. MA20 与 MA350 的差距（MA350 - MA20）/ MA350，正值表示MA20在下
    df["ma20_ma350_gap"] = (df["MA_350"] - df["MA_20"]) / df["MA_350"]

    # 3. MA50 与 MA70 的相对差值 (MA50 - MA70) / MA70
    df["ma50_ma70_diff_pct"] = (df["MA_50"] - df["MA_70"]) / df["MA_70"]

    # 4. MA20, MA50, MA70 三者最大距离 / MA70
    df["ma20_ma50_ma70_max_distance"] = (
        df[["MA_20", "MA_50", "MA_70"]].max(axis=1) -
        df[["MA_20", "MA_50", "MA_70"]].min(axis=1)
    ) / df["MA_70"]

    # 5. MA20 连续上升天数（自定义滚动函数）
    def count_consecutive_up(series):
        """计算从当前日期往前数，MA20连续上升的天数（至少连续1天）"""
        count = 0
        for i in range(len(series)-1, -1, -1):
            if i == 0:
                break
            if series.iloc[i] > series.iloc[i-1]:
                count += 1
            else:
                break
        return count

    # 使用滚动窗口应用（注意性能，数据量不大可接受）
    df["ma_uptrend_days"] = df["MA_20"].rolling(window=20, min_periods=1).apply(
        lambda x: count_consecutive_up(x) if len(x) > 1 else 0, raw=False
    ).fillna(0).astype(int)

    # 价格加速度（保留）
    df["Price_Acceleration"] = df["return_t"].diff()

    # 移除旧的 MA60 相关列（如果有）
    for col in ["MA_60", "MA_diff", "price_vs_MA60"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df.drop(columns=["close_prev"], inplace=True, errors="ignore")

    # 将日期列恢复为字符串
    df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")
    df.drop(columns=["date_dt"], inplace=True)

    # 转为降序（最新在前）
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    cols = ["date"] + [c for c in df.columns if c != "date"]
    df = df[cols]

    # ========== 所有数值列保留 4 位小数 ==========
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(4)

    return df


# ==================== DataLoader 类 ====================
class DataLoader:
    def __init__(self, ticker: str, start_date=None, end_date=None, data_root: str = None):
        self.ticker = ticker
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        if data_root is None:
            data_root = os.path.join(project_root, "data")
        self.data_root = data_root

    def load_processed_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.data_root, self.ticker, f"{self.ticker}_processed.xlsx")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"处理后的数据不存在：{file_path}\n请先运行数据流水线：python src/features/data_loader.py")
        df = pd.read_excel(file_path)
        df["date"] = pd.to_datetime(df["date"])
        if self.start_date and self.end_date:
            mask = (df["date"] >= self.start_date) & (df["date"] <= self.end_date)
            df = df.loc[mask].copy()
        if not df.empty:
            df = df.sort_values("date", ascending=False).reset_index(drop=True)
            cols = ["date"] + [c for c in df.columns if c != "date"]
            df = df[cols]
            print(f"✅ 加载处理后的数据 {self.ticker}：{len(df)} 行（降序）")
        else:
            print(f"⚠️ 在日期范围内无数据")
        return df

    def get_clean_dataframe(self):
        return self.load_processed_data()


# ==================== 主流程 ====================
def run_data_pipeline(tickers=None):
    import glob
    config_dir = os.path.join(project_root, "config")
    data_root = os.path.join(project_root, "data")

    # 自动发现所有 {ticker}_settings.json 中的股票
    if tickers is None:
        config_files = glob.glob(os.path.join(config_dir, "*_settings.json"))
        tickers = [os.path.basename(f).replace("_settings.json", "") for f in config_files]

    print(f"📁 数据根目录：{data_root}")
    print(f"📊 股票列表：{tickers}")

    if not tickers:
        print("⚠️ 股票列表为空。")
        return

    for ticker in tickers:
        try:
            stock_cfg = load_strategy_config(ticker)
        except Exception as e:
            print(f"⚠️ 加载 {ticker} 配置失败：{e}，跳过")
            continue
        data_cfg = stock_cfg.get('data', {}) if stock_cfg else {}
        start_date = data_cfg.get('start_date', None)
        end_date = data_cfg.get('end_date', None)
        market = data_cfg.get('market', 'US')

        print(f"📅 {ticker} 日期范围：{start_date if start_date else '全部'} ~ {end_date if end_date else '全部'}")
        raw_file = os.path.join(data_root, ticker, f"{ticker}_raw.xlsx")
        need_download = False
        df_raw = None

        if os.path.exists(raw_file):
            try:
                df_raw = pd.read_excel(raw_file)
                if df_raw["date"].dtype != "object":
                    df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.strftime("%Y-%m-%d")
                print(f"📂 原始数据已存在：{raw_file}，共 {len(df_raw)} 行")
                if start_date and end_date:
                    dates = pd.to_datetime(df_raw["date"])
                    if dates.min() > pd.to_datetime(start_date) or dates.max() < pd.to_datetime(end_date):
                        print(f"⚠️ 现有数据日期范围 ({dates.min().date()} ~ {dates.max().date()}) 不满足配置，重新下载")
                        need_download = True
                        df_raw = None
                else:
                    if df_raw.empty:
                        print("⚠️ 现有数据为空，重新下载")
                        need_download = True
                        df_raw = None
            except Exception as e:
                print(f"⚠️ 读取 {raw_file} 失败 ({e})，重新下载")
                need_download = True
                df_raw = None
        else:
            need_download = True

        if need_download:
            print(f"⬇️ 开始下载 {ticker} 数据...")
            if market.upper() == "A":
                df_raw = fetch_a_stock(ticker)
            else:
                df_raw = fetch_us_stock(ticker)

            if df_raw is not None and not df_raw.empty:
                if save_raw_data(df_raw, ticker, data_root):
                    print(f"✅ 原始数据下载并保存成功：{ticker}")
                else:
                    print(f"⚠️ 保存原始数据失败，跳过 {ticker}")
                    continue
            else:
                print(f"⚠️ 下载 {ticker} 失败，跳过")
                if os.path.exists(raw_file):
                    os.remove(raw_file)
                    print(f"   🗑️ 已删除空文件 {raw_file}")
                continue

        if df_raw is None or df_raw.empty:
            print(f"❌ {ticker} 无有效原始数据，跳过")
            continue

        # 按日期过滤
        if start_date and end_date:
            df_raw["date_dt"] = pd.to_datetime(df_raw["date"])
            mask = (df_raw["date_dt"] >= start_date) & (df_raw["date_dt"] <= end_date)
            df_sub = df_raw.loc[mask].copy()
            df_sub.drop(columns=["date_dt"], inplace=True)
        else:
            df_sub = df_raw.copy()

        if len(df_sub) == 0:
            print(f"⚠️ {ticker} 在指定日期范围内无数据，跳过")
            continue

        # 特征工程
        df_feat = add_features(df_sub)

        # 删除最前面的 350 行（即最早的数据，因为保存降序时位于末尾）
        if len(df_feat) > 350:
            df_feat = df_feat.iloc[:-350].reset_index(drop=True)
        else:
            print(f"⚠️ {ticker} 数据量不足 350 行，可能无法计算 MA350，保留全部")

        if save_processed_data(df_feat, ticker, data_root):
            print(f"✅ {ticker} 处理完成。")
        else:
            print(f"❌ {ticker} 保存处理数据失败。")

    print("🎯 全部数据处理完成！")


if __name__ == "__main__":
    run_data_pipeline()

    print("\n" + "="*50)
    print("测试 DataLoader 加载...")
    try:
        import glob
        config_dir = os.path.join(project_root, "config")
        config_files = glob.glob(os.path.join(config_dir, "*_settings.json"))
        if not config_files:
            raise FileNotFoundError("未找到任何 *_settings.json 配置文件")
        ticker = os.path.basename(config_files[0]).replace("_settings.json", "")
        stock_cfg = load_strategy_config(ticker)
        data_cfg = stock_cfg.get('data', {}) if stock_cfg else {}
        start = data_cfg.get('start_date', None)
        end = data_cfg.get('end_date', None)
        loader = DataLoader(ticker=ticker, start_date=start, end_date=end)
        df_test = loader.get_clean_dataframe()
        if not df_test.empty:
            print(f"✅ DataLoader 测试通过，共加载 {len(df_test)} 行数据。")
        else:
            print("⚠️ DataLoader 加载结果为空。")
    except Exception as e:
        print(f"❌ DataLoader 测试失败：{e}")