"""Evaluation: backtesting, metrics, and ablation study runner."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL, TRANSACTION_COST_PCT
from rl_engine.dqn import DQNAgent
from rl_engine.train import evaluate_agent, walk_forward_split, train_dqn
from utils.metrics import cumulative_returns, max_drawdown, sharpe_ratio

logger = logging.getLogger(__name__)


def backtest(
    agent: DQNAgent,
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    episode: int = 0,
    ticker: str = "",
    sentiment_bonus_enabled: bool = True,
) -> dict:
    """Run a full backtest and return performance metrics."""
    log_df, total_reward = evaluate_agent(agent, df, initial_capital,
                                          episode=episode, ticker=ticker,
                                          sentiment_bonus_enabled=sentiment_bonus_enabled)

    equity = log_df["portfolio_value"]
    returns = log_df["returns"]

    # Buy-and-hold benchmark: same starting conditions as DQN agent
    # DQN pays transaction_cost_pct per trade; B&H also pays it on initial purchase
    test_start_price = float(df["close"].iloc[0])
    tc = TRANSACTION_COST_PCT
    gross_shares = initial_capital / test_start_price
    bh_shares = int(gross_shares)
    bh_cost = test_start_price * bh_shares * (1 + tc)
    bh_cash = initial_capital - bh_cost  # unspent cash due to integer rounding
    if bh_cash < 0:  # rounding edge case
        bh_shares = int(initial_capital / (test_start_price * (1 + tc)))
        bh_cost = test_start_price * bh_shares * (1 + tc)
        bh_cash = initial_capital - bh_cost
    bh_equity = df["close"] * bh_shares + bh_cash

    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "final_portfolio_value": float(equity.iloc[-1]),
        "buy_and_hold_return": float((bh_equity.iloc[-1] / initial_capital) - 1),
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
    ticker: str = "",
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

    seed_idx = 0
    for seed in seed_list:
        seed_label = f"seed={seed}" if seed is not None else "default"
        for label, df in [("with_nlp", df_with_sentiment), ("without_nlp", df_without_sentiment)]:
            logger.info("=== Ablation: %s [%s] ===", label, seed_label)
            train_df, val_df, test_df = walk_forward_split(df)

            agent = DQNAgent(seed=seed)
            # Unique ticker tag so convergence curves don't overwrite across seeds/conditions
            curve_tag = f"{ticker}_seed{seed}_{label}"
            agent = train_dqn(train_df, val_df, episodes=episodes,
                            initial_capital=initial_capital, agent=agent, seed=seed,
                            ticker=curve_tag, sentiment_bonus_enabled=False)
            metrics = backtest(agent, test_df, initial_capital=initial_capital,
                             episode=seed_idx + 1, ticker=ticker,
                             sentiment_bonus_enabled=False)
            metrics["seed"] = seed

            if label == "with_nlp":
                all_with.append(metrics)
            else:
                all_without.append(metrics)

            logger.info("%s [%s]: Sharpe=%.4f, MDD=%.4f, Return=%.4f",
                        label, seed_label, metrics["sharpe_ratio"],
                        metrics["max_drawdown"], metrics["total_return"])
        seed_idx += 1

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
    delta_mdd = results["with_nlp"]["max_drawdown"] - results["without_nlp"]["max_drawdown"]
    sharpe_std = np.sqrt(
        results["with_nlp"]["seed_std"].get("sharpe_ratio", 0) ** 2 +
        results["without_nlp"]["seed_std"].get("sharpe_ratio", 0) ** 2
    )
    logger.info("Ablation delta: Sharpe=%+.4f±%.4f, Return=%+.4f, MDD=%+.4f",
                delta_sharpe, sharpe_std, delta_return, delta_mdd)

    results["summary"] = {
        "sharpe_delta": delta_sharpe,
        "return_delta": delta_return,
        "mdd_delta": delta_mdd,
        "sharpe_delta_std": sharpe_std,
        "nlp_improves_sharpe": delta_sharpe > 0,
        "nlp_improves_return": delta_return > 0,
        "nlp_improves_mdd": delta_mdd > 0,
        "n_seeds": len(seed_list),
    }

    return results
