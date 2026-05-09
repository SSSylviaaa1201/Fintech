"""Main entry point: NLP-RL Trading Platform — full pipeline.

Pipeline:
  raw news + market data  →  NLP sentiment signal  →  RL trading agent  →  dashboard

Usage:
  python main.py              # Run full pipeline (collect → analyze → train → eval)
  python main.py --skip-collect  # Skip data collection, use cached DB data
  python main.py --ablate     # Run ablation study only
"""

import argparse
import logging
import sys

import pandas as pd

import numpy as np
import config
from data_storage.db_manager import DatabaseManager
from utils.indicators import compute_indicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


# ──────────────────────────────────────────────────────────────────────
# STEP 1: Data Ingestion
# ──────────────────────────────────────────────────────────────────────

def step_ingest(db: DatabaseManager) -> None:
    """Fetch market data and news, store in DB."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion")
    logger.info("=" * 60)

    from data_ingestion.market_data import fetch_all_tickers
    from data_ingestion.news_fetcher import fetch_news_for_all_tickers, fetch_sample_news

    # Market data
    logger.info("Fetching market data...")
    market_data = fetch_all_tickers()
    for ticker, df in market_data.items():
        if not df.empty:
            db.insert_market_data(ticker, df)
            logger.info("  %s: %d rows", ticker, len(df))
        else:
            logger.warning("  %s: no data", ticker)

    # News
    logger.info("Fetching news...")
    news_records, failed_tickers = fetch_news_for_all_tickers()
    if news_records:
        count = db.insert_news(news_records)
        logger.info("  %d news articles inserted", count)
    if failed_tickers:
        logger.warning("NewsAPI failed for %s, using sample data.", failed_tickers)
        for ticker in failed_tickers:
            sample_df = fetch_sample_news(ticker)
            db.insert_news(sample_df.to_dict("records"))
        logger.info("  Sample news inserted for %d tickers", len(failed_tickers))


# ──────────────────────────────────────────────────────────────────────
# STEP 2: NLP Pipeline
# ──────────────────────────────────────────────────────────────────────

def step_nlp(db: DatabaseManager) -> pd.DataFrame:
    """Run all 4 sentiment methods on stored news, write sentiment to DB."""
    logger.info("=" * 60)
    logger.info("STEP 2: NLP Sentiment Pipeline")
    logger.info("=" * 60)

    from nlp_pipeline.preprocessor import preprocess_news_df
    from nlp_pipeline.sentiment_lexicon import vader_sentiment_batch
    from nlp_pipeline.sentiment_lr import lr_sentiment_batch
    from nlp_pipeline.sentiment_finbert import finbert_sentiment_batch
    from nlp_pipeline.sentiment_llm import llm_sentiment_batch
    from nlp_pipeline.aggregator import get_merged_sentiment
    from vector_store.chroma_store import index_news

    all_aggregated = []

    for ticker in config.TICKERS:
        # Skip if sentiment already computed for this ticker
        existing = db.get_sentiment(ticker)
        if not existing.empty and existing["method"].nunique() >= 3:
            logger.info("  %s: sentiment already exists (%d methods), skipping", ticker, existing["method"].nunique())
            continue

        news_df = db.get_news(ticker, limit=2000)  # process all available news, not just 200
        if news_df.empty:
            logger.warning("  %s: no news, skipping", ticker)
            continue

        logger.info("  %s: preprocessing %d articles", ticker, len(news_df))
        news_df = preprocess_news_df(news_df)

        # 1) VADER
        logger.info("    - VADER sentiment...")
        df_vader = vader_sentiment_batch(news_df)

        # 2) Logistic Regression
        logger.info("    - LR sentiment...")
        df_lr = lr_sentiment_batch(news_df)

        # 3) FinBERT
        logger.info("    - FinBERT sentiment...")
        df_finbert = finbert_sentiment_batch(news_df)

        # 4) LLM (Volcano Engine)
        logger.info("    - LLM sentiment...")
        df_llm = llm_sentiment_batch(news_df)

        # Merge and aggregate to daily
        result = get_merged_sentiment(df_vader, df_lr, df_finbert, df_llm)
        aggregated = result["aggregated"]
        all_aggregated.append(aggregated)

        # Log agreement metrics
        if result.get("agreement"):
            ag = result["agreement"]
            logger.info("    Agreement: kappa=%s, level=%s", ag.get("kappa"), ag.get("agreement_level"))

        # Store in DB
        records = aggregated.to_dict("records")
        db.upsert_sentiment(records)
        logger.info("    Stored %d daily sentiment records for %s", len(records), ticker)

        # Index in vector store for RAG
        try:
            index_news(news_df)
        except Exception:
            logger.debug("Vector indexing skipped for %s", ticker)

    if all_aggregated:
        return pd.concat(all_aggregated, ignore_index=True)
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────
# STEP 3: Build feature DataFrames for RL
# ──────────────────────────────────────────────────────────────────────

def _process_sentiment_signal(
    sent_df: pd.DataFrame,
    ema_span: int = config.SENTIMENT_EMA_SPAN,
    neutral_threshold: float = config.SENTIMENT_NEUTRAL_THRESHOLD,
) -> pd.Series:
    """
    Convert raw per-method sentiment records into a clean daily signal.

    1. Consensus: weight FinBERT/LLM 2x when inter-method correlation < 0.6
    2. EMA smoothing to reduce day-to-day noise
    3. Neutral gating: zero out weak signals
    """
    if sent_df.empty:
        return pd.Series(dtype=float)

    # Pivot: dates × method
    pivot = sent_df.pivot_table(
        index="date", columns="method",
        values="sentiment_score", aggfunc="mean",
    )

    if pivot.shape[1] >= 2:
        # Inter-method correlation determines weighting strategy
        corr = pivot.corr()
        vals = corr.values
        triu_idx = np.triu_indices_from(vals, k=1)
        mean_corr = float(vals[triu_idx].mean()) if len(triu_idx[0]) > 0 else 1.0

        weights = {}
        for col in pivot.columns:
            m = str(col).lower()
            if m in ("finbert", "llm"):
                weights[col] = 2.0 if mean_corr < 0.6 else 1.0
            elif m in ("lr", "logistic_regression"):
                weights[col] = 1.5 if mean_corr < 0.6 else 1.0
            else:
                weights[col] = 1.0

        total_w = sum(weights.get(c, 1.0) for c in pivot.columns)
        consensus = sum(
            pivot[c].fillna(0.0) * weights.get(c, 1.0) for c in pivot.columns
        ) / total_w
    else:
        consensus = pivot.iloc[:, 0]

    # EMA smoothing
    smoothed = consensus.ewm(span=ema_span, min_periods=1).mean()

    # Neutral gating
    smoothed = smoothed.where(smoothed.abs() >= neutral_threshold, 0.0)

    return smoothed


def build_rl_features(db: DatabaseManager, with_sentiment: bool = True) -> dict[str, pd.DataFrame]:
    """
    Build per-ticker DataFrames with columns:
    [close, MA50, MA200, RSI, MACD, sentiment_score]
    """
    logger.info("Building RL feature DataFrames (with_sentiment=%s)...", with_sentiment)

    feature_dfs = {}

    for ticker in config.TICKERS:
        market_df = db.get_market_data(ticker)
        if market_df.empty:
            logger.warning("  %s: no market data, skipping", ticker)
            continue

        # Compute technical indicators on FULL market data first (MA200 needs 200 rows)
        df = compute_indicators(market_df)

        # Then align to news period
        sent_df = db.get_sentiment(ticker)
        df = _align_market_to_news(df, sent_df)
        df = df.reset_index(drop=True)  # ensure clean index, date as column only

        # Attach sentiment if requested (consensus → EMA → gate)
        if with_sentiment:
            sent_df = db.get_sentiment(ticker)
            if not sent_df.empty:
                signal = _process_sentiment_signal(sent_df)
                if not signal.empty:
                    signal_df = signal.reset_index()
                    signal_df.columns = ["date", "sentiment_score"]
                    signal_df["date"] = pd.to_datetime(signal_df["date"]).dt.date
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    df = df.merge(signal_df, on="date", how="left")
                    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
                else:
                    df["sentiment_score"] = 0.0
            else:
                df["sentiment_score"] = 0.0
        else:
            df["sentiment_score"] = 0.0

        # Sentiment momentum: short-term avg and trend direction
        df["sentiment_ma5"] = df["sentiment_score"].rolling(window=5, min_periods=1).mean()
        df["sentiment_ma20"] = df["sentiment_score"].rolling(window=20, min_periods=1).mean()
        df["sentiment_trend"] = df["sentiment_ma5"] - df["sentiment_ma20"]
        # Sentiment volatility: signal reliability indicator (higher → noisier)
        df["sentiment_vol"] = df["sentiment_score"].rolling(window=10, min_periods=1).std()

        # Forward-fill missing indicator values
        df = df.ffill().fillna(0.0)
        feature_dfs[ticker] = df

    logger.info("  Feature DataFrames built for %d tickers", len(feature_dfs))
    return feature_dfs


# ──────────────────────────────────────────────────────────────────────
# STEP 4: Train & Evaluate
# ──────────────────────────────────────────────────────────────────────

def step_train_evaluate(
    feature_dfs: dict[str, pd.DataFrame],
    episodes: int = config.EPISODES,
    label: str = "",
) -> dict:
    """Train DQN and backtest for each ticker."""
    logger.info("=" * 60)
    logger.info("STEP 3: RL Training & Evaluation %s", label)
    logger.info("=" * 60)

    from rl_engine.evaluation import backtest, walk_forward_split
    from rl_engine.train import train_dqn

    results = {}

    for ticker, df in feature_dfs.items():
        logger.info("--- %s ---", ticker)

        train_df, val_df, test_df = walk_forward_split(df)
        logger.info("  Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

        # Train
        logger.info("  Training DQN (%d episodes)...", episodes)
        agent = train_dqn(train_df, val_df, episodes=episodes)

        # Backtest
        logger.info("  Backtesting...")
        metrics = backtest(agent, test_df)
        metrics["ticker"] = ticker
        results[ticker] = metrics

        logger.info("  %s: Sharpe=%.4f, MDD=%.4f, Return=%.4f, BH Return=%.4f",
                    ticker, metrics["sharpe_ratio"], metrics["max_drawdown"],
                    metrics["total_return"], metrics["buy_and_hold_return"])

    return results


# ──────────────────────────────────────────────────────────────────────
# STEP 5: Ablation Study
# ──────────────────────────────────────────────────────────────────────

def _align_market_to_news(
    market_df: pd.DataFrame, sent_df: pd.DataFrame, lookback_padding: int = config.LOOKBACK_WINDOW
) -> pd.DataFrame:
    """
    Truncate market data to match news coverage period + padding for indicators.
    Fixes the core sparsity bug: market data 2020-2026 but news only 2025-2026.
    """
    if sent_df.empty:
        return market_df

    sent_dates = pd.to_datetime(sent_df["date"].unique())
    first_news = sent_dates.min().date()
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.date
    # Start padding days before first news to allow MA/indicator warmup
    cutoff = first_news - pd.Timedelta(days=lookback_padding)
    truncated = market_df[market_df["date"] >= cutoff]
    if len(truncated) < 100:
        logger.warning("Market data after news-alignment too short (%d rows), using full range", len(truncated))
        return market_df
    logger.info("  Market data aligned: %d rows (was %d), density %.1f%%",
                len(truncated), len(market_df),
                100 * sent_df["date"].nunique() / max(len(truncated), 1))
    return truncated


def step_ablation(db: DatabaseManager) -> dict:
    """Run the ablation study: RL with vs. without NLP sentiment signal."""
    logger.info("=" * 60)
    logger.info("STEP 4: Ablation Study — NLP vs. No-NLP")
    logger.info("=" * 60)

    from rl_engine.evaluation import run_ablation_study

    ablation_results = {}

    for ticker in config.TICKERS:
        market_df = db.get_market_data(ticker)
        if market_df.empty:
            continue

        # Compute technical indicators on FULL market data first (MA200 needs 200 rows)
        df_base = compute_indicators(market_df).reset_index(drop=True)

        # Then align to news period
        sent_df = db.get_sentiment(ticker)
        df_base = _align_market_to_news(df_base, sent_df)
        df_base["date"] = pd.to_datetime(df_base["date"]).dt.date

        # With NLP sentiment (consensus → EMA → gate)
        if not sent_df.empty:
            signal = _process_sentiment_signal(sent_df)
            if not signal.empty:
                signal_df = signal.reset_index()
                signal_df.columns = ["date", "sentiment_score"]
                signal_df["date"] = pd.to_datetime(signal_df["date"]).dt.date
                df_with = df_base.merge(signal_df, on="date", how="left")
                df_with["sentiment_score"] = df_with["sentiment_score"].fillna(0.0)
            else:
                df_with = df_base.copy()
                df_with["sentiment_score"] = 0.0
        else:
            df_with = df_base.copy()
            df_with["sentiment_score"] = 0.0
        df_with = df_with.ffill().fillna(0.0)
        df_with["sentiment_ma5"] = df_with["sentiment_score"].rolling(window=5, min_periods=1).mean()
        df_with["sentiment_ma20"] = df_with["sentiment_score"].rolling(window=20, min_periods=1).mean()
        df_with["sentiment_trend"] = df_with["sentiment_ma5"] - df_with["sentiment_ma20"]
        df_with["sentiment_vol"] = df_with["sentiment_score"].rolling(window=10, min_periods=1).std()

        # Without NLP (sentiment always 0 → derived features also 0)
        df_without = df_base.copy()
        df_without["sentiment_score"] = 0.0
        df_without["sentiment_ma5"] = 0.0
        df_without["sentiment_ma20"] = 0.0
        df_without["sentiment_trend"] = 0.0
        df_without["sentiment_vol"] = 0.0
        df_without = df_without.ffill().fillna(0.0)

        logger.info("Running ablation for %s...", ticker)
        result = run_ablation_study(df_with, df_without)
        ablation_results[ticker] = result

    # ── Layer 1: Ablation Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("LAYER 1: NLP Contribution (Ablation Δ)")
    logger.info("=" * 60)

    nlp_positive = 0
    nlp_neutral = 0
    nlp_negative = 0
    for ticker, result in ablation_results.items():
        s = result["summary"]
        logger.info("%s: Sharpe Δ=%+.4f, Return Δ=%+.4f, NLP helps=%s",
                    ticker, s["sharpe_delta"], s["return_delta"], s["nlp_improves_sharpe"])
        if s["sharpe_delta"] > 0.01:
            nlp_positive += 1
        elif s["sharpe_delta"] < -0.01:
            nlp_negative += 1
        else:
            nlp_neutral += 1

    logger.info("NLP Positive: %d/%d (%.1f%%) | Neutral: %d | Negative: %d",
                nlp_positive, len(ablation_results),
                nlp_positive / len(ablation_results) * 100 if ablation_results else 0,
                nlp_neutral, nlp_negative)

    # ── Layer 2: DQN vs Buy&Hold ──
    logger.info("\n" + "=" * 60)
    logger.info("LAYER 2: DQN+NLP vs Buy&Hold (Absolute Performance)")
    logger.info("=" * 60)

    import config as cfg
    sector_map = dict(zip(cfg.TICKERS, [
        "Tech"] * 7 + ["Finance"] * 5 + ["Healthcare"] * 4 +
        ["Consumer"] * 5 + ["Energy/Industrial"] * 5 + ["Comm/Utility"] * 2))

    bh_wins_sharpe = 0
    bh_wins_return = 0
    sector_dqn_vs_bh = {}
    layer2_rows = []

    for ticker, result in ablation_results.items():
        w = result["with_nlp"]
        dqn_sharpe = w["sharpe_ratio"]
        bh_sharpe = w["buy_and_hold_sharpe"]
        dqn_ret = w["total_return"]
        bh_ret = w["buy_and_hold_return"]
        excess_sharpe = dqn_sharpe - bh_sharpe
        excess_ret = dqn_ret - bh_ret

        if excess_sharpe > 0:
            bh_wins_sharpe += 1
        if excess_ret > 0:
            bh_wins_return += 1

        sector = sector_map.get(ticker, "Other")
        if sector not in sector_dqn_vs_bh:
            sector_dqn_vs_bh[sector] = {"dqn_sharpe": [], "bh_sharpe": [], "wins": 0, "total": 0}
        sector_dqn_vs_bh[sector]["dqn_sharpe"].append(dqn_sharpe)
        sector_dqn_vs_bh[sector]["bh_sharpe"].append(bh_sharpe)
        sector_dqn_vs_bh[sector]["total"] += 1
        if excess_sharpe > 0:
            sector_dqn_vs_bh[sector]["wins"] += 1

        layer2_rows.append({
            "ticker": ticker, "sector": sector,
            "dqn_sharpe": dqn_sharpe, "bh_sharpe": bh_sharpe,
            "dqn_return": dqn_ret, "bh_return": bh_ret,
            "excess_sharpe": excess_sharpe, "excess_return": excess_ret,
            "nlp_sharpe_delta": result["summary"]["sharpe_delta"],
        })

    logger.info("DQN beats BH on Sharpe: %d/%d (%.1f%%)",
                bh_wins_sharpe, len(ablation_results),
                bh_wins_sharpe / len(ablation_results) * 100 if ablation_results else 0)
    logger.info("DQN beats BH on Return: %d/%d (%.1f%%)",
                bh_wins_return, len(ablation_results),
                bh_wins_return / len(ablation_results) * 100 if ablation_results else 0)

    logger.info("\nBy Sector (DQN vs BH Sharpe):")
    for sector, data in sorted(sector_dqn_vs_bh.items()):
        avg_dqn = np.mean(data["dqn_sharpe"]) if data["dqn_sharpe"] else 0
        avg_bh = np.mean(data["bh_sharpe"]) if data["bh_sharpe"] else 0
        logger.info("  %s: DQN Sharpe=%.3f, BH Sharpe=%.3f, Win=%d/%d",
                    sector, avg_dqn, avg_bh, data["wins"], data["total"])

    # ── Layer 3: Paper Trading Forward Validation ──
    logger.info("\n" + "=" * 60)
    logger.info("LAYER 3: Paper Trading Forward Validation")
    logger.info("=" * 60)

    from paper_trader import run_paper_validation
    paper_results = run_paper_validation(db, ablation_results)

    # ── Save structured results ──
    import json, os
    from datetime import datetime
    results_file = os.path.join(os.path.dirname(__file__), "data", "ablation_results.json")
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "lookback_window": cfg.LOOKBACK_WINDOW,
            "ema_span": cfg.SENTIMENT_EMA_SPAN,
            "neutral_threshold": cfg.SENTIMENT_NEUTRAL_THRESHOLD,
            "episodes": cfg.EPISODES,
        },
        "layer1_ablation": {
            "nlp_positive": nlp_positive,
            "nlp_neutral": nlp_neutral,
            "nlp_negative": nlp_negative,
            "positive_rate": round(nlp_positive / len(ablation_results) * 100, 1) if ablation_results else 0,
            "tickers": {t: r["summary"] for t, r in ablation_results.items()},
        },
        "layer2_dqn_vs_bh": {
            "sharpe_win_rate": round(bh_wins_sharpe / len(ablation_results) * 100, 1) if ablation_results else 0,
            "return_win_rate": round(bh_wins_return / len(ablation_results) * 100, 1) if ablation_results else 0,
            "by_sector": {s: {"avg_dqn_sharpe": round(np.mean(d["dqn_sharpe"]), 3), "avg_bh_sharpe": round(np.mean(d["bh_sharpe"]), 3), "wins": d["wins"], "total": d["total"]} for s, d in sector_dqn_vs_bh.items()},
            "tickers": layer2_rows,
        },
        "layer3_paper_trading": paper_results,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", results_file)

    return ablation_results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NLP-RL Trading Platform")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip data collection, use cached data")
    parser.add_argument("--skip-nlp", action="store_true",
                        help="Skip NLP sentiment, use cached sentiment")
    parser.add_argument("--ablate", action="store_true",
                        help="Run ablation study (with vs. without NLP)")
    parser.add_argument("--episodes", type=int, default=config.EPISODES,
                        help="Number of training episodes")
    parser.add_argument("--tickers", nargs="+", default=config.TICKERS,
                        help="Ticker symbols to trade")
    args = parser.parse_args()

    # Override config
    config.TICKERS = args.tickers
    config.EPISODES = args.episodes

    db = DatabaseManager()
    logger.info("Platform initialized. DB: %s", config.DB_PATH)

    # ── Pipeline ──────────────────────────────────────────────────

    if not args.skip_collect:
        step_ingest(db)

    if not args.skip_nlp:
        step_nlp(db)

    # Ablation study (NLP vs. No-NLP) — skips normal train/eval when enabled
    if args.ablate:
        step_ablation(db)
    else:
        # Build features (with sentiment)
        feature_dfs = build_rl_features(db, with_sentiment=True)

        # Train & evaluate
        results = step_train_evaluate(feature_dfs, episodes=config.EPISODES, label="[with NLP]")

        # Print final results
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        for ticker, m in results.items():
            logger.info(
                f"{ticker}: Sharpe={m['sharpe_ratio']:.4f}, MDD={m['max_drawdown']:.4f}, "
                f"Return={m['total_return']:.4f}, BH Return={m['buy_and_hold_return']:.4f}"
            )

    logger.info("\n=== Pipeline complete. Launch dashboard: streamlit run dashboard/app.py ===")


if __name__ == "__main__":
    main()
