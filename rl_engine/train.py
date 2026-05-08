"""Training loop with walk-forward validation."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import EPISODES, TRAIN_SPLIT, VAL_SPLIT, INITIAL_CAPITAL
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


def run_episode(env: FinancialTradingEnv, agent: DQNAgent, train: bool = True) -> tuple[float, list[dict]]:
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
            "step": env.current_step,
            "action": action,
            "price": info["price"],
            "portfolio_value": info["portfolio_value"],
            "cash": info["cash"],
            "shares": info["shares"],
            "reward": reward,
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
) -> DQNAgent:
    """Train DQN agent on training data with optional validation."""

    if agent is None:
        agent = DQNAgent()

    train_env = FinancialTradingEnv(train_df, initial_capital=initial_capital)
    val_env = FinancialTradingEnv(val_df, initial_capital=initial_capital) if val_df is not None else None

    best_val_return = -np.inf
    episode_rewards = []

    for ep in range(1, episodes + 1):
        train_reward, _ = run_episode(train_env, agent, train=True)
        episode_rewards.append(train_reward)
        agent.decay_epsilon()

        if (val_env is not None) and (ep % 10 == 0):
            val_reward, _ = run_episode(val_env, agent, train=False)
            if val_reward > best_val_return:
                best_val_return = val_reward
                agent.save()

        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            logger.info("Episode %d/%d | avg_reward=%.2f | epsilon=%.4f",
                        ep, episodes, avg_reward, agent.epsilon)

    logger.info("Training complete. Best val return: %.2f", best_val_return)
    # Load best checkpoint (not the final potentially-overfit model)
    if best_val_return > -np.inf:
        agent.load()
    return agent


def evaluate_agent(
    agent: DQNAgent,
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple[pd.DataFrame, float]:
    """Run agent on data and return log DataFrame + total return."""
    env = FinancialTradingEnv(df, initial_capital=initial_capital)
    total_reward, logs = run_episode(env, agent, train=False)

    log_df = pd.DataFrame(logs)
    log_df["returns"] = log_df["portfolio_value"].pct_change().fillna(0)
    return log_df, total_reward
