"""Evaluation: backtesting, metrics, and ablation study runner."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL
from rl_engine.dqn import DQNAgent
from rl_engine.train import evaluate_agent, walk_forward_split, train_dqn
from utils.metrics import cumulative_returns, max_drawdown, sharpe_ratio

logger = logging.getLogger(__name__)


def backtest(
    agent: DQNAgent,
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """Run a full backtest and return performance metrics."""
    log_df, total_reward = evaluate_agent(agent, df, initial_capital)

    equity = log_df["portfolio_value"]
    returns = log_df["returns"]

    # Buy-and-hold benchmark
    bh_shares = int(initial_capital / df["close"].iloc[0])
    bh_equity = df["close"] * bh_shares

    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "final_portfolio_value": float(equity.iloc[-1]),
        "buy_and_hold_return": float(bh_equity.iloc[-1] / bh_equity.iloc[0] - 1),
        "buy_and_hold_sharpe": sharpe_ratio(bh_equity.pct_change().dropna()),
        "buy_and_hold_mdd": max_drawdown(bh_equity),
        "n_trades": int((log_df["action"] != 0).sum()),
        "log_df": log_df,
    }


def run_ablation_study(
    df_with_sentiment: pd.DataFrame,
    df_without_sentiment: pd.DataFrame,
    episodes: int = 200,
    initial_capital: float = INITIAL_CAPITAL,
    seeds: Optional[list] = None,
) -> dict:
    """
    Compare RL performance with vs. without NLP sentiment signal.

    When seeds is provided (e.g. [42, 123, 456]), trains with each seed and
    returns seed-averaged metrics + per-seed details for variance quantification.
    When seeds is None, uses the default DQN_SEED from config (fast single-run).

    Returns dict with metrics for both conditions + summary.
    """
    seed_list = seeds if seeds else [None]  # None → use config default seed
    metric_keys = ["sharpe_ratio", "total_return", "max_drawdown",
                   "buy_and_hold_return", "buy_and_hold_sharpe", "buy_and_hold_mdd",
                   "final_portfolio_value", "n_trades"]

    all_with = []
    all_without = []

    for seed in seed_list:
        seed_label = f"seed={seed}" if seed is not None else "default"
        for label, df in [("with_nlp", df_with_sentiment), ("without_nlp", df_without_sentiment)]:
            logger.info("=== Ablation: %s [%s] ===", label, seed_label)
            train_df, val_df, test_df = walk_forward_split(df)

            agent = DQNAgent(seed=seed)
            agent = train_dqn(train_df, val_df, episodes=episodes,
                            initial_capital=initial_capital, agent=agent, seed=seed)
            metrics = backtest(agent, test_df, initial_capital=initial_capital)
            metrics["seed"] = seed

            if label == "with_nlp":
                all_with.append(metrics)
            else:
                all_without.append(metrics)

            logger.info("%s [%s]: Sharpe=%.4f, MDD=%.4f, Return=%.4f",
                        label, seed_label, metrics["sharpe_ratio"],
                        metrics["max_drawdown"], metrics["total_return"])

    # Aggregate across seeds
    def _aggregate(seed_metrics: list[dict]) -> dict:
        avg = {}
        std = {}
        for k in metric_keys:
            values = [m[k] for m in seed_metrics if k in m]
            avg[k] = float(np.mean(values)) if values else 0.0
            std[k] = float(np.std(values)) if len(values) > 1 else 0.0
        avg["seed_count"] = len(seed_metrics)
        avg["seed_std"] = std
        avg["seed_details"] = seed_metrics
        return avg

    results = {
        "with_nlp": _aggregate(all_with),
        "without_nlp": _aggregate(all_without),
    }

    # Comparison summary (use averaged values)
    delta_sharpe = results["with_nlp"]["sharpe_ratio"] - results["without_nlp"]["sharpe_ratio"]
    delta_return = results["with_nlp"]["total_return"] - results["without_nlp"]["total_return"]
    sharpe_std = np.sqrt(
        results["with_nlp"]["seed_std"].get("sharpe_ratio", 0) ** 2 +
        results["without_nlp"]["seed_std"].get("sharpe_ratio", 0) ** 2
    )
    logger.info("Ablation delta: Sharpe=%+.4f±%.4f, Return=%+.4f", delta_sharpe, sharpe_std, delta_return)

    results["summary"] = {
        "sharpe_delta": delta_sharpe,
        "return_delta": delta_return,
        "sharpe_delta_std": sharpe_std,
        "nlp_improves_sharpe": delta_sharpe > 0,
        "nlp_improves_return": delta_return > 0,
        "n_seeds": len(seed_list),
    }

    return results
