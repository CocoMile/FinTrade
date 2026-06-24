# FinTrade – 多因子海龟策略交易系统

> 基于海龟交易法则，融合 HMM 状态识别与 Quantile 回归模型，实现智能买卖点决策与回测可视化。

---

## 📌 项目概述

FinTrade 是一个模块化的量化交易系统，核心策略是**海龟交易法则**（Turtle Trading System），同时引入**隐马尔可夫模型（HMM）**和**分位数回归（Quantile）**模型作为辅助因子，提升入场/出场信号的可靠性。系统提供数据预处理、策略执行、回测评估和交互式可视化全套工具链。

---

## 🏗️ 整体架构
┌─────────────────────────────────────────────────────────────┐
│ main.py │
│ （总控制器：调度数据、策略、模型、回测、绘图） │
└───────────────┬─────────────────┬─────────────────────────┘
│ │
▼ ▼
┌─────────────────┐ ┌──────────────────┐
│ data_loader.py │ │ hmm_model.py │
│ (数据获取与预处理) │ │ quantile_model.py │
└─────────────────┘ │ (提供额外辅助信号) │
│ └──────────────────┘
▼ │
┌─────────────────────────┘
▼
┌─────────────────────────────────────────┐
│ turtle_strategy.py │
│ (核心策略：生成买卖信号，基于均线、ATR等) │
└──────────────────┬──────────────────────┘
▼
┌─────────────────────────────────────────┐
│ backtest_runner.py │
│ (回测引擎：模拟持仓、计算收益曲线、统计指标) │
└──────────────────┬──────────────────────┘
▼
┌─────────────────────────────────────────┐
│ plotter.py │
│ (可视化：绘制价格、信号、权益曲线、箱线图) │
└─────────────────────────────────────────┘

text

---

## 📂 项目目录结构
FinTrade/
├── main.py # 主入口，串联所有模块
├── README.md # 项目说明文档
├── requirements.txt # 依赖清单
├── config/
│ └── settings.yaml # 全局配置（如数据源、模型参数）
├── data/ # 原始数据存放目录
├── outputs/ # 输出目录
│ ├── figures/ # 图表（HTML / PNG）
│ └── reports/ # 回测报告（CSV / JSON）
├── src/
│ ├── init.py
│ ├── data_loader.py # 数据加载与预处理
│ ├── strategies/
│ │ ├── init.py
│ │ └── turtle_strategy.py # 海龟策略实现
│ ├── models/
│ │ ├── init.py
│ │ ├── hmm_model.py # HMM 状态识别模型
│ │ └── quantile_model.py # 分位数回归模型
│ ├── pipeline/
│ │ ├── init.py
│ │ └── backtest_runner.py # 回测执行器
│ ├── visualization/
│ │ ├── init.py
│ │ └── plotter.py # 交互式图表绘制
│ └── utils/ # 通用工具函数
└── .venv/ # 虚拟环境（可选）

text

---

## 🧩 模块职责详解

### 1. `data_loader.py` – 数据获取与预处理
- 从本地或外部数据源（如 CSV、数据库、API）读取 OHLCV 数据。
- 执行数据清洗（缺失值处理、除权除息调整）、时间序列对齐、技术指标预计算（可选）。
- 输出标准化的 `DataFrame`，供后续策略和模型使用。

### 2. `hmm_model.py` 与 `quantile_model.py` – 辅助模型
- **HMM 模型**：对市场状态（如趋势/震荡/高波动）进行隐状态识别，输出状态标签或概率，用于过滤策略信号。
- **Quantile 模型**：对价格未来区间进行分位数预测，辅助设定动态止损/止盈位，或评估当前价格是否处于极端分位。
- 两者均以 `data_loader` 的输出为输入，输出额外的特征或决策因子，供策略在运行时参考。

### 3. `turtle_strategy.py` – 核心策略
- 实现经典海龟交易法则的入场/出场逻辑（基于突破、均线、ATR 等）。
- 支持三种买入场景（下跌/横盘、上涨回撤、大幅上涨）的精细化判定。
- 接受 `data_loader` 提供的价格数据，并可选地接收 HMM/Quantile 模型的辅助信号，生成最终的买卖点信号（`signals` DataFrame）。
- **策略本身不涉及绘图或回测**，专注于信号生成。

### 4. `backtest_runner.py` – 回测引擎
- 接收策略信号、原始价格数据以及可选的模型辅助信号。
- 模拟真实交易：包括开仓、加仓、止损（ATR 或破位法）、平仓、滑点、手续费等。
- 计算每日权益曲线、最大回撤、夏普比率、胜率、盈亏比等关键绩效指标。
- 输出交易明细（`trades`）和净值序列（`equity_curve`），供后续分析和可视化。

### 5. `plotter.py` – 可视化模块
- 基于 **Plotly** 生成交互式 HTML 图表，支持缩放、悬停查看细节、图例开关。
- 核心功能：
  - **价格+均线+买卖点图**：按场景（下跌/横盘、回撤、暴涨）区分颜色，标注信号备注。
  - **权益曲线图**：对比策略净值 vs 基准（如买入持有）。
  - **场景收益箱线图**：统计不同场景下每笔交易的收益率分布，直观评估策略优劣。
- 完全独立于策略和回测，仅在 `main.py` 中被调用，用于结果展示。

### 6. `main.py` – 系统总控制器
- 初始化配置（`settings.yaml`）。
- 调用 `data_loader` 获取所有目标股票的数据。
- 可选地训练/加载 HMM 和 Quantile 模型，获得辅助因子。
- 实例化 `TurtleStrategy`，传入数据和辅助因子，执行 `scan()` 获得买卖信号。
- 将信号和价格数据送入 `backtest_runner` 进行回测，得到净值曲线和交易明细。
- 最后调用 `plotter` 绘制所有图表，并输出回测报告。

---

## 🔄 数据流与执行流程

1. **数据准备**  
   `main.py` → `data_loader.load_data()` → 返回 `df_raw`（包含 OHLCV）

2. **模型辅助（可选）**  
   `main.py` → 初始化 `HMModel` / `QuantileModel` → 从 `df_raw` 计算特征 → 输出 `aux_signals`

3. **策略信号生成**  
   `main.py` → `TurtleStrategy(df_raw, aux_signals).scan()` → 返回 `signals_df`

4. **回测执行**  
   `main.py` → `BacktestRunner(df_raw, signals_df).run()` → 返回 `equity_curve`, `trades_df`

5. **可视化输出**  
   `main.py` → `StrategyPlotter(df_raw, signals_df, equity_curve, trades_df).plot_all()` → 生成 HTML 图表保存至 `outputs/figures/`

6. **报告生成（可选）**  
   将回测指标汇总并保存为 JSON/CSV 至 `outputs/reports/`

---

## 🚀 快速开始

### 环境要求
- Python 3.9+
- 推荐使用虚拟环境（venv）

### 安装依赖
```bash
pip install -r requirements.txt