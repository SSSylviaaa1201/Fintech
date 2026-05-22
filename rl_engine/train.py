"""Training loop with walk-forward validation and convergence analysis."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import EPISODES, TRAIN_SPLIT, VAL_SPLIT, INITIAL_CAPITAL, DATA_DIR
from rl_engine.dqn import DQNAgent
from rl_engine.env import FinancialTradingEnv

logger = logging.getLogger(__name__)


def walk_forward_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically: train / val / test. No look-ahead bias."""
    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def run_episode(
    env: FinancialTradingEnv,
    agent: DQNAgent,
    train: bool = True,
    episode: int = 0,
    ticker: str = "",
) -> tuple[float, list[dict]]:
    """Run a single episode. Returns (total_reward, log_records)."""
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    logs = []

    while not done:
        action = agent.select_action(state, evaluate=not train)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if train:
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

        logs.append({
            "episode": episode,
            "ticker": ticker,
            "step": env.current_step,
            "action": action,
            "price": info["price"],
            "portfolio_value": info["portfolio_value"],
            "cash": info["cash"],
            "shares": info["shares"],
            "reward": reward,
            "sentiment_score": info.get("sentiment_score", 0.0),
            "date": info.get("date", None),
        })

        state = next_state
        total_reward += reward

    return total_reward, logs


def train_dqn(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    episodes: int = EPISODES,
    initial_capital: float = INITIAL_CAPITAL,
    agent: Optional[DQNAgent] = None,
    seed: Optional[int] = None,
    ticker: str = "",
    sentiment_bonus_enabled: bool = True,
    alignment_scale: Optional[float] = None,
    turnover_penalty: Optional[float] = None,
) -> DQNAgent:
    """Train DQN agent on training data with optional validation.

    alignment_scale / turnover_penalty: override config defaults for sensitivity analysis.
    """

    if agent is None:
        agent = DQNAgent(seed=seed)

    train_env = FinancialTradingEnv(train_df, initial_capital=initial_capital,
                                    sentiment_bonus_enabled=sentiment_bonus_enabled,
                                    alignment_scale=alignment_scale,
                                    turnover_penalty=turnover_penalty)
    val_env = (FinancialTradingEnv(val_df, initial_capital=initial_capital,
                                   sentiment_bonus_enabled=sentiment_bonus_enabled,
                                   alignment_scale=alignment_scale,
                                   turnover_penalty=turnover_penalty)
               if val_df is not None else None)

    best_val_return = -np.inf
    episode_rewards = []
    val_rewards_log = []  # track validation rewards for overfitting detection

    for ep in range(1, episodes + 1):
        train_reward, _ = run_episode(train_env, agent, train=True, episode=ep, ticker=ticker)
        episode_rewards.append(train_reward)
        agent.decay_epsilon()

        if (val_env is not None) and (ep % 10 == 0):
            val_reward, _ = run_episode(val_env, agent, train=False, episode=ep, ticker=ticker)
            val_rewards_log.append((ep, val_reward))
            if val_reward > best_val_return:
                best_val_return = val_reward
                agent.save()

        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            logger.info("Episode %d/%d | avg_reward=%.2f | epsilon=%.4f",
                        ep, episodes, avg_reward, agent.epsilon)

    # ── Convergence analysis ──
    rewards = np.array(episode_rewards)
    n_eps = len(rewards)
    first_half = np.mean(rewards[:n_eps // 2]) if n_eps >= 4 else 0.0
    second_half = np.mean(rewards[n_eps // 2:]) if n_eps >= 2 else 0.0
    conv_ratio = second_half / max(abs(first_half), 1e-8)

    # Linear trend slope (positive = still improving)
    x = np.arange(len(rewards))
    slope = np.polyfit(x, rewards, 1)[0] if len(rewards) >= 2 else 0.0
    # Coefficient of variation in last 25% (lower = more stable = converged)
    tail_n = max(10, n_eps // 4)
    tail_cv = float(np.std(rewards[-tail_n:]) / max(abs(np.mean(rewards[-tail_n:])), 1e-8))
    plateau_converged = tail_cv < 0.3 and abs(slope) < 1e-5

    # Overfitting check: train vs val divergence in later episodes
    overfit_warning = False
    if len(val_rewards_log) >= 2:
        val_episodes, val_rewards = zip(*val_rewards_log)
        # Check if train improves while val degrades in last 40% of val evals
        split_idx = len(val_episodes) * 3 // 5
        early_val = np.mean(val_rewards[:split_idx])
        late_val = np.mean(val_rewards[split_idx:])
        early_train = np.mean(rewards[:n_eps // 2])
        late_train = np.mean(rewards[n_eps // 2:])
        if late_train > early_train and late_val < early_val:
            overfit_warning = True
            logger.warning("Overfitting: train reward ↑ (%.4f→%.4f) but val reward ↓ (%.4f→%.4f)",
                          early_train, late_train, early_val, late_val)

    # Compute convergence interpretation
    if plateau_converged:
        conv_interpretation = (
            "Policy has converged (plateau detected). "
            "The agent has found a stable strategy — further training yields diminishing returns. "
            "However, convergence on training data does NOT guarantee out-of-sample performance; "
            "check test-set metrics for generalization."
        )
    elif slope > 0.001:
        conv_interpretation = (
            "Policy still improving (positive trend, slope={:.5f}). "
            "More episodes may yield better performance but risk overfitting.".format(slope)
        )
    elif overfit_warning:
        conv_interpretation = (
            "Overfitting detected: train reward rising while validation reward falling. "
            "The agent is memorizing training patterns rather than learning generalizable signals. "
            "Recommend: reduce episodes, increase regularization, or simplify state space."
        )
    else:
        conv_interpretation = (
            "Policy shows no clear convergence or divergence. "
            "The agent may be exploring a noisy reward landscape; multi-seed analysis is recommended "
            "to distinguish signal from DQN training variance."
        )

    logger.info("Training complete. Best val return: %.2f", best_val_return)
    logger.info("Convergence: first_half=%.4f, second_half=%.4f, ratio=%.2f, slope=%.6f",
                first_half, second_half, conv_ratio, slope)
    logger.info("  Tail CV=%.3f (%.4f±%.4f), Plateau=%s, Overfitting=%s",
                tail_cv, np.mean(rewards[-tail_n:]), np.std(rewards[-tail_n:]),
                plateau_converged, overfit_warning)
    logger.info("  Interpretation: %s", conv_interpretation)

    # Save training curve with diagnostics
    try:
        curve_path = Path(DATA_DIR) / "training_curves"
        curve_path.mkdir(parents=True, exist_ok=True)
        safe_name = ticker.replace("/", "_") if ticker else "latest"
        np.savez_compressed(curve_path / f"{safe_name}_rewards.npz",
                            rewards=rewards, conv_ratio=conv_ratio, slope=slope,
                            tail_cv=tail_cv, plateau_converged=plateau_converged,
                            overfit_warning=overfit_warning,
                            val_episodes=np.array([v[0] for v in val_rewards_log]) if val_rewards_log else np.array([]),
                            val_rewards=np.array([v[1] for v in val_rewards_log]) if val_rewards_log else np.array([]))
    except Exception:
        pass

    # Load best checkpoint (not the final potentially-overfit model)
    if best_val_return > -np.inf:
        agent.load()
    return agent


def evaluate_agent(
    agent: DQNAgent,
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    episode: int = 0,
    ticker: str = "",
    sentiment_bonus_enabled: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Run agent on data and return log DataFrame + total return."""
    env = FinancialTradingEnv(df, initial_capital=initial_capital,
                              sentiment_bonus_enabled=sentiment_bonus_enabled)
    total_reward, logs = run_episode(env, agent, train=False, episode=episode, ticker=ticker)

    log_df = pd.DataFrame(logs)
    log_df["returns"] = log_df["portfolio_value"].pct_change().fillna(0)
    return log_df, total_reward
