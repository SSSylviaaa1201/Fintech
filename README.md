# NLP-Driven Reinforcement Learning Trading Platform

End-to-end intelligent trading platform: **Raw News → NLP Sentiment → RL Trading Decision**

Pipeline: Data Ingestion → NLP (3-method) → DQN Training → Ablation Study → Dashboard

## Architecture

| Module | Responsibility | Status |
|--------|---------------|--------|
| ① Data Ingestion | Yahoo Direct + Finnhub news + RSS; automated scheduler | ✅ |
| ② NLP Pipeline | VADER + Logistic Regression + FinBERT (3 methods) | ✅ |
| ③ Data Storage | SQLite (market_data, news, sentiment_signals, trading_logs) | ✅ |
| ④ RL Trading Engine | Custom Gym env + Double DQN (from scratch, PyTorch) | ✅ |
| ⑤ Front-End Dashboard | Streamlit: sentiment trends, agent decisions, portfolio, health | ✅ |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys (optional — defaults to Yahoo/Finnhub free tier)
cp .env.example .env

# 3. Full pipeline: collect → NLP → DQN train → ablation → paper trade
python main.py --ablate

# 4. Launch dashboard
streamlit run dashboard/app.py

# 5. (Optional) Start automated scheduler for live trading
python data_ingestion/scheduler.py
```

## Key Features

- **3 sentiment methods**: VADER (lexicon), Logistic Regression, FinBERT (transformer)
- **F1 score reporting**: per-method F1 vs FinBERT as pseudo-ground-truth
- **Double DQN**: decoupled action selection & evaluation; LR decay; gradient clipping
- **Multi-seed ablation**: quantify DQN training variance (set `ABLATION_MULTI_SEED=True`)
- **3-layer validation**: (1) NLP ablation Δ, (2) DQN vs Buy&Hold, (3) Paper trading forward test
- **Risk controls**: max drawdown termination (20%), position concentration limit, trade frequency penalty
- **Walk-forward**: chronological 60/20/20 split; market data aligned to news coverage window

## State Vector (11-dim)

`[price_ratio, MA50_ratio, MA200_ratio, RSI_norm, MACD_ratio, position_pct, cash_pct, sentiment, sentiment_ma5, sentiment_trend, sentiment_vol]`

Base 8 features meet course requirements; +3 sentiment momentum features (EMA, trend, volatility) enhance signal quality.

## Evaluation Metrics

- Sharpe Ratio, Maximum Drawdown (MDD)
- Buy-and-Hold benchmark comparison
- Walk-Forward Validation (no look-ahead bias)
- NLP ablation: with-NLP vs without-NLP Sharpe Δ
- Per-sector performance breakdown (6 sectors, 28 stocks)

## Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| Yahoo Direct API | OHLCV prices | 2020-01 to 2024-12 |
| Finnhub | Financial news | ~12 months (free tier) |
| RSS (Yahoo Finance, Google News, etc.) | Supplemental headlines | Real-time |

## Report

See `run_report_2026-05-09.md` for the latest full-run results.
See `ablation_report.md` for detailed NLP contribution analysis.
See `config.py` for all tunable parameters.
