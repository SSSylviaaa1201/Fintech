"""Parameter sensitivity analysis for key custom weights.

Tests the three most impactful parameters that lack literature support:
  1. SENTIMENT_ALIGNMENT_SCALE  — reward shaping coefficient
  2. REWARD_TURNOVER_PENALTY     — trading disincentive magnitude
  3. Consensus weighting scheme  — equal vs. FinBERT-weighted sentiment

Runs on 5 representative stocks (one per sector) with single seed + 100 episodes
for speed (~2 min per run). Reports Sharpe ratio delta for each combination.

Usage:  python utils/sensitivity.py
"""

import itertools
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from data_storage.db_manager import DatabaseManager
from main import build_rl_features, _process_sentiment_signal, compute_indicators
from rl_engine.dqn import DQNAgent
from rl_engine.env import FinancialTradingEnv, STATE_DIM
from rl_engine.train import train_dqn, walk_forward_split, run_episode
from utils.metrics import sharpe_ratio, max_drawdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("sensitivity")

# Representative stocks: one per sector, already have ablation data
SENSITIVITY_TICKERS = ["AAPL", "JPM", "JNJ", "KO", "XOM"]
SENSITIVITY_EPISODES = 60     # reduced for speed (relative comparison only)
SENSITIVITY_SEED = 42         # single seed for controlled comparison

# Parameter grids
ALIGNMENT_SCALES = [0, 0.0001, 0.0005, 0.001]
TURNOVER_PENALTIES = [0, 0.0001, 0.0005]
CONSENSUS_SCHEMES = ["equal", "finbert_weighted"]


def _build_single_ticker_features(db, ticker, scheme="finbert_weighted"):
    """Build features for ONE ticker only (not all 60) with configurable consensus."""
    import config as cfg
    from main import _align_market_to_news, _process_sentiment_signal

    market_df = db.get_market_data(ticker)
    if market_df.empty:
        return None

    df = compute_indicators(market_df)
    price_baseline = float(df["close"].iloc[0]) if len(df) > 0 else 1.0

    sent_df = db.get_sentiment(ticker)
    df = _align_market_to_news(df, sent_df)
    df = df.reset_index(drop=True)
    df["price_baseline"] = price_baseline

    if not sent_df.empty:
        if scheme == "equal":
            # Equal-weight consensus across methods
            pivot = sent_df.pivot_table(
                index="date", columns="method",
                values="sentiment_score", aggfunc="mean",
            )
            if pivot.shape[1] >= 1:
                consensus = pivot.mean(axis=1)
            else:
                consensus = pivot.iloc[:, 0] if pivot.shape[1] == 1 else None
        else:
            # FinBERT-weighted consensus (default)
            signal = _process_sentiment_signal(sent_df)
            consensus = signal

        if consensus is not None and not consensus.empty:
            signal_df = consensus.reset_index()
            signal_df.columns = ["date", "sentiment_score"]
            signal_df["date"] = pd.to_datetime(signal_df["date"]).dt.date
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.merge(signal_df, on="date", how="left")
            df["sentiment_mask"] = df["sentiment_score"].notna().astype(float)
            df["sentiment_score"] = df["sentiment_score"].ffill().fillna(0.0)
        else:
            df["sentiment_score"] = 0.0
            df["sentiment_mask"] = 0.0
    else:
        df["sentiment_score"] = 0.0
        df["sentiment_mask"] = 0.0

    # Sentiment momentum features
    df["sentiment_ma5"] = df["sentiment_score"].rolling(5, min_periods=1).mean()
    df["sentiment_ma20"] = df["sentiment_score"].rolling(20, min_periods=1).mean()
    df["sentiment_trend"] = df["sentiment_ma5"] - df["sentiment_ma20"]
    df["sentiment_vol"] = df["sentiment_score"].rolling(10, min_periods=1).std()
    df = df.ffill().fillna(0.0)
    return df


def run_single_config(ticker, df, alignment_scale, turnover_penalty, seed,
                      episodes=100, initial_capital=100_000.0):
    """Train + evaluate DQN with given reward parameters. Returns Sharpe, MDD, Return."""
    train_df, val_df, test_df = walk_forward_split(df)
    agent = DQNAgent(seed=seed)
    agent = train_dqn(train_df, val_df, episodes=episodes,
                     initial_capital=initial_capital, agent=agent,
                     seed=seed, ticker=f"{ticker}_sens",
                     sentiment_bonus_enabled=True,
                     alignment_scale=alignment_scale,
                     turnover_penalty=turnover_penalty)

    # Evaluate on test set
    env = FinancialTradingEnv(test_df, initial_capital=initial_capital,
                              sentiment_bonus_enabled=True,
                              alignment_scale=alignment_scale,
                              turnover_penalty=turnover_penalty)
    total_reward, logs = run_episode(env, agent, train=False, episode=0, ticker=ticker)
    log_df = pd.DataFrame(logs)
    log_df["returns"] = log_df["portfolio_value"].pct_change().fillna(0)
    equity = log_df["portfolio_value"]

    sharpe = sharpe_ratio(log_df["returns"])
    mdd = max_drawdown(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)

    return {"sharpe": round(sharpe, 4), "mdd": round(mdd, 4),
            "return": round(total_return, 4), "n_trades": int((log_df["action"] != 0).sum())}


def run_sensitivity(alignment_scales=None, turnover_penalties=None,
                    consensus_schemes=None, tickers=None,
                    episodes=100, seed=42):
    """Full sensitivity grid search. Returns results dict."""
    alignment_scales = alignment_scales or ALIGNMENT_SCALES
    turnover_penalties = turnover_penalties or TURNOVER_PENALTIES
    consensus_schemes = consensus_schemes or CONSENSUS_SCHEMES
    tickers = tickers or SENSITIVITY_TICKERS

    db = DatabaseManager()
    results = {}

    total_runs = len(tickers) * len(alignment_scales) * len(turnover_penalties) * len(consensus_schemes)
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSIS: %d tickers × %d align × %d turnover × %d consensus = %d runs",
                len(tickers), len(alignment_scales), len(turnover_penalties),
                len(consensus_schemes), total_runs)
    logger.info("=" * 60)

    run_idx = 0
    t0 = time.time()

    for ticker in tickers:
        for scheme in consensus_schemes:
            df = _build_single_ticker_features(db, ticker, scheme)
            if df is None:
                logger.warning("Skipping %s: no features", ticker)
                continue

            for align in alignment_scales:
                for turnover in turnover_penalties:
                    run_idx += 1
                    elapsed = time.time() - t0
                    eta = (elapsed / run_idx) * (total_runs - run_idx) if run_idx > 0 else 0

                    logger.info("[%d/%d] %s | scheme=%s | align=%.5f | turnover=%.5f | ETA %.0f min",
                                run_idx, total_runs, ticker, scheme, align, turnover, eta / 60)

                    metrics = run_single_config(ticker, df, align, turnover, seed, episodes)

                    key = f"{ticker}|{scheme}|align={align}|turnover={turnover}"
                    results[key] = metrics
                    logger.info("  → Sharpe=%.3f MDD=%.3f Return=%.3f Trades=%d",
                                metrics["sharpe"], metrics["mdd"],
                                metrics["return"], metrics["n_trades"])

    # Summary: find best parameters per ticker and globally
    logger.info("\n" + "=" * 60)
    logger.info("SENSITIVITY RESULTS SUMMARY")
    logger.info("=" * 60)

    # Aggregate: mean Sharpe per parameter value
    def agg_by_param(results, param_name):
        buckets = {}
        for key, metrics in results.items():
            # Parse param value from key
            for part in key.split("|"):
                if part.startswith(param_name + "="):
                    val = part.split("=")[1]
                    if val not in buckets:
                        buckets[val] = []
                    buckets[val].append(metrics["sharpe"])
        return {v: round(float(np.mean(s)), 4) for v, s in buckets.items()}

    align_summary = agg_by_param(results, "align")
    turnover_summary = agg_by_param(results, "turnover")
    scheme_summary = agg_by_param(results, "scheme")

    # Best overall configuration (highest mean Sharpe)
    best_key = max(results, key=lambda k: results[k]["sharpe"])
    best_metrics = results[best_key]

    summary = {
        "total_runs": total_runs,
        "runtime_seconds": round(time.time() - t0, 1),
        "tickers_tested": tickers,
        "best_config": best_key,
        "best_sharpe": best_metrics["sharpe"],
        "alignment_scale_summary": align_summary,
        "turnover_penalty_summary": turnover_summary,
        "consensus_scheme_summary": scheme_summary,
        "recommendation": {
            "alignment_scale": max(align_summary, key=align_summary.get),
            "turnover_penalty": max(turnover_summary, key=turnover_summary.get),
            "consensus_scheme": max(scheme_summary, key=scheme_summary.get),
        },
        "all_results": results,
    }

    for k, v in align_summary.items():
        logger.info("  Align=%-7s → mean Sharpe %.4f", k, v)
    for k, v in turnover_summary.items():
        logger.info("  Turnover=%-7s → mean Sharpe %.4f", k, v)

    rec = summary["recommendation"]
    logger.info("\nRecommended: align=%s turnover=%s scheme=%s",
                rec["alignment_scale"], rec["turnover_penalty"], rec["consensus_scheme"])

    # Save
    output_path = Path(config.DATA_DIR) / "sensitivity_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved to %s", output_path)

    return summary


if __name__ == "__main__":
    run_sensitivity()
