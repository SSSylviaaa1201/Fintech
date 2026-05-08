"""Custom Gym environment for financial trading with NLP sentiment signal.

State vector (all normalized to roughly [-1, 1] or [0, 1]):
  [price_ratio, MA50_ratio, MA200_ratio, RSI_norm, MACD_ratio, position_pct, cash_pct, sentiment]

Actions: 0=Hold, 1=Buy (25% of capital), 2=Sell (25% of position)
"""

from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import SENTIMENT_ALIGNMENT_BONUS

STATE_DIM = 11  # 8 base + sentiment_score + sentiment_ma5 + sentiment_trend + sentiment_vol
N_ACTIONS = 3


class FinancialTradingEnv(gym.Env):
    """A trading environment that steps through market data day by day."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        trade_fraction: float = 0.25,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.trade_fraction = trade_fraction  # fraction of cash/position per trade

        # Ensure required columns exist
        required = ["close", "MA50", "MA200", "RSI", "MACD", "sentiment_score"]
        for col in required:
            if col not in self.df.columns:
                self.df[col] = 0.0
        for col in ["sentiment_ma5", "sentiment_trend", "sentiment_vol"]:
            if col not in self.df.columns:
                self.df[col] = 0.0

        # Fill NaN values and precompute normalization constants
        self.df = self.df.ffill().fillna(0.0)
        self._price_0 = float(self.df["close"].iloc[0]) if len(self.df) > 0 else 1.0

        self.n_steps = len(self.df)
        self.current_step = 0
        self.render_mode = render_mode

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(N_ACTIONS)

        # State space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32,
        )

        # Episode state
        self.cash = initial_capital
        self.shares = 0
        self.portfolio_value = initial_capital
        self.prev_portfolio_value = initial_capital

    def _norm_state(self) -> np.ndarray:
        """Build normalized state vector where all features are similar scale."""
        row = self.df.iloc[self.current_step]
        price = float(row["close"])
        if price <= 0:
            price = self._price_0

        # Price indicators: scale to initial price
        price_ratio = price / self._price_0
        ma50_ratio = float(row["MA50"]) / price if price > 0 else 1.0
        ma200_ratio = float(row["MA200"]) / price if price > 0 else 1.0

        # RSI: [0, 100] → [0, 1]
        rsi_norm = float(row["RSI"]) / 100.0

        # MACD: scale by price
        macd_ratio = float(row["MACD"]) / price if price > 0 else 0.0

        # Position: fraction of portfolio in stock
        position_value = self.shares * price
        portfolio = self.cash + position_value
        position_pct = position_value / portfolio if portfolio > 0 else 0.0

        # Cash: fraction of portfolio
        cash_pct = self.cash / portfolio if portfolio > 0 else 1.0

        # Sentiment: already in [-1, 1]
        sentiment = float(row.get("sentiment_score", 0.0))
        sentiment_ma5 = float(row.get("sentiment_ma5", sentiment))
        sentiment_trend = float(row.get("sentiment_trend", 0.0))
        sentiment_vol = float(row.get("sentiment_vol", 0.0))

        return np.array([
            price_ratio,
            ma50_ratio,
            ma200_ratio,
            rsi_norm,
            macd_ratio,
            position_pct,
            cash_pct,
            sentiment,
            sentiment_ma5,
            sentiment_trend,
            sentiment_vol,
        ], dtype=np.float32)

    def _get_price(self) -> float:
        return float(self.df.iloc[self.current_step]["close"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        return self._norm_state(), {}

    def step(self, action: int):
        price = self._get_price()
        if price <= 0:
            price = self._price_0

        self.prev_portfolio_value = self.portfolio_value
        trade_cost = 0.0

        if action == 1:  # Buy: use trade_fraction of cash
            cash_to_use = self.cash * self.trade_fraction
            if cash_to_use >= price:  # can buy at least 1 share
                gross_shares = int(cash_to_use / price)
                cost = price * gross_shares * (1 + self.transaction_cost_pct)
                if cost <= self.cash:
                    self.shares += gross_shares
                    self.cash -= cost
                    trade_cost = price * gross_shares * self.transaction_cost_pct

        elif action == 2:  # Sell: sell trade_fraction of current shares
            if self.shares > 0:
                shares_to_sell = max(1, int(self.shares * self.trade_fraction))
                revenue = price * shares_to_sell * (1 - self.transaction_cost_pct)
                self.shares -= shares_to_sell
                self.cash += revenue
                trade_cost = price * shares_to_sell * self.transaction_cost_pct

        # Update portfolio value
        self.portfolio_value = self.cash + self.shares * price

        # Reward: percentage return (scale-invariant)
        if self.prev_portfolio_value > 0:
            pct_return = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        else:
            pct_return = 0.0

        # Cash holding penalty: small cost for idle cash (~2.5% annual → per trading day)
        cash_penalty = -0.0001 if self.shares == 0 else 0.0

        # Sentiment-position alignment bonus (reward shaping)
        sentiment = float(self.df.iloc[self.current_step].get("sentiment_score", 0.0))
        sentiment_bonus = 0.0
        if sentiment > 0.3 and self.shares > 0:
            sentiment_bonus = SENTIMENT_ALIGNMENT_BONUS
        elif sentiment < -0.3 and self.shares > 0:
            sentiment_bonus = -SENTIMENT_ALIGNMENT_BONUS
        elif sentiment < -0.3 and self.shares == 0:
            sentiment_bonus = SENTIMENT_ALIGNMENT_BONUS

        reward = pct_return + cash_penalty + sentiment_bonus

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "price": price,
            "trade_cost": trade_cost,
            "pct_return": pct_return,
            "sentiment_bonus": sentiment_bonus,
        }

        obs = self._norm_state() if not terminated else np.zeros(STATE_DIM, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.current_step}: price={self._get_price():.2f}, "
                  f"shares={self.shares}, cash={self.cash:.2f}, "
                  f"portfolio={self.portfolio_value:.2f}")
