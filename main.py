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

        news_df = db.get_news(ticker, limit=200)
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

        # Compute technical indicators
        df = compute_indicators(market_df)
        df = df.reset_index(drop=True)  # ensure clean index, date as column only

        # Attach sentiment if requested
        if with_sentiment:
            sent_df = db.get_sentiment(ticker)  # all methods
            if not sent_df.empty:
                # Average across methods per date
                daily_sent = sent_df.groupby("date")["sentiment_score"].mean().reset_index()
                daily_sent["date"] = pd.to_datetime(daily_sent["date"])
                df["date"] = pd.to_datetime(df["date"]).dt.date
                daily_sent["date"] = daily_sent["date"].dt.date
                df = df.merge(daily_sent, on="date", how="left")
            df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
        else:
            df["sentiment_score"] = 0.0

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

        df_base = compute_indicators(market_df).reset_index(drop=True)
        df_base["date"] = pd.to_datetime(df_base["date"]).dt.date

        # With NLP sentiment
        sent_df = db.get_sentiment(ticker)
        if not sent_df.empty:
            daily_sent = sent_df.groupby("date")["sentiment_score"].mean().reset_index()
            daily_sent["date"] = pd.to_datetime(daily_sent["date"]).dt.date
            df_with = df_base.merge(daily_sent, on="date", how="left")
        else:
            df_with = df_base.copy()
            df_with["sentiment_score"] = 0.0
        df_with["sentiment_score"] = df_with["sentiment_score"].fillna(0.0)
        df_with = df_with.ffill().fillna(0.0)

        # Without NLP (sentiment always 0)
        df_without = df_base.copy()
        df_without["sentiment_score"] = 0.0
        df_without = df_without.ffill().fillna(0.0)

        logger.info("Running ablation for %s...", ticker)
        result = run_ablation_study(df_with, df_without)
        ablation_results[ticker] = result

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 60)
    for ticker, result in ablation_results.items():
        s = result["summary"]
        logger.info("%s: Sharpe Δ=%+.4f, Return Δ=%+.4f, NLP helps=%s",
                    ticker, s["sharpe_delta"], s["return_delta"], s["nlp_improves_sharpe"])

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

    # Ablation study
    if args.ablate:
        step_ablation(db)

    logger.info("\n=== Pipeline complete. Launch dashboard: streamlit run dashboard/app.py ===")


if __name__ == "__main__":
    main()
