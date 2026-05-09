"""Paper trading module: DQN inference → trade signals → position tracking → P&L.

Runs on top of the trained model and live data without real money.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from data_storage.db_manager import DatabaseManager
from main import build_rl_features
from rl_engine.dqn import QNetwork
from rl_engine.env import STATE_DIM, N_ACTIONS

logger = logging.getLogger(__name__)

MODEL_PATH = config.MODEL_DIR / "dqn_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_LABELS = {0: "HOLD", 1: "BUY", 2: "SELL"}


@dataclass
class Position:
    ticker: str
    shares: int = 0
    avg_cost: float = 0.0
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    @property
    def pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        return (self.pnl / self.cost_basis * 100) if self.cost_basis > 0 else 0.0


@dataclass
class PaperPortfolio:
    cash: float = 100_000.0
    positions: dict[str, Position] = field(default_factory=dict)
    trade_history: list[dict] = field(default_factory=list)

    @property
    def total_equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.total_equity - 100_000.0

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / 100_000.0 * 100


class PaperTrader:
    """Loads trained DQN model, generates daily signals, tracks simulated positions."""

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or DatabaseManager()
        self.portfolio = PaperPortfolio()
        self._model: QNetwork | None = None
        self._last_signal_date: dict[str, date] = {}

    def load_model(self) -> QNetwork:
        """Load the best checkpoint. Cache for repeated use."""
        if self._model is not None:
            return self._model

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Run training first: python main.py"
            )

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self._model = QNetwork(STATE_DIM, N_ACTIONS).to(DEVICE)

        # Handle both raw state_dict and full training checkpoint
        if isinstance(checkpoint, dict) and "q_network" in checkpoint:
            self._model.load_state_dict(checkpoint["q_network"])
            logger.info("Model loaded from checkpoint (ep=%d, epsilon=%.4f)",
                        checkpoint.get("episode_count", 0), checkpoint.get("epsilon", 0.0))
        else:
            self._model.load_state_dict(checkpoint)

        self._model.eval()
        logger.info("Model loaded from %s (%s)", MODEL_PATH, DEVICE)
        return self._model

    def generate_signal(self, ticker: str) -> dict:
        """Run DQN inference on the latest data point and return a trade signal."""
        model = self.load_model()

        # Build features using the same pipeline as training
        feature_dfs = build_rl_features(self.db, with_sentiment=True)
        if ticker not in feature_dfs:
            return {"ticker": ticker, "action": 0, "label": "HOLD",
                    "reason": "no_features", "confidence": 0.0}

        df = feature_dfs[ticker]
        if df.empty:
            return {"ticker": ticker, "action": 0, "label": "HOLD",
                    "reason": "empty_df", "confidence": 0.0}

        # Use the last row (most recent day) as state
        row = df.iloc[-1]
        price = float(row["close"])

        # Build state vector matching _norm_state
        price_0 = float(df["close"].iloc[0]) if len(df) > 0 else price
        if price_0 <= 0:
            price_0 = 1.0

        state = np.array([
            price / price_0,
            float(row["MA50"]) / price if price > 0 else 1.0,
            float(row["MA200"]) / price if price > 0 else 1.0,
            float(row["RSI"]) / 100.0,
            float(row["MACD"]) / price if price > 0 else 0.0,
            0.0,  # position_pct (no current position assumed for signal generation)
            1.0,  # cash_pct (assume full cash for clean signal)
            float(row.get("sentiment_score", 0.0)),
            float(row.get("sentiment_ma5", 0.0)),
            float(row.get("sentiment_trend", 0.0)),
            float(row.get("sentiment_vol", 0.0)),
        ], dtype=np.float32)

        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        # Inference
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q_values = model(state_t).cpu().numpy().flatten()
            action = int(np.argmax(q_values))
            q_max = float(q_values[action])

        confidence = 1.0 / (1.0 + np.exp(-abs(q_max)))  # sigmoid scaling for interpretability

        signal_date = pd.to_datetime(df["date"].iloc[-1]).date()

        return {
            "ticker": ticker,
            "date": signal_date,
            "action": action,
            "label": ACTION_LABELS.get(action, "HOLD"),
            "confidence": round(confidence, 4),
            "price": price,
            "q_values": q_values.tolist(),
            "sentiment_score": float(row.get("sentiment_score", 0.0)),
        }

    def execute_signal(self, signal: dict, price: float | None = None) -> dict | None:
        """Execute a paper trade based on the signal. Returns trade record or None."""
        ticker = signal["ticker"]
        action = signal["action"]
        exec_price = price or signal.get("price", 0.0)

        if exec_price <= 0:
            return None

        # Lazy Position init
        if ticker not in self.portfolio.positions:
            self.portfolio.positions[ticker] = Position(ticker=ticker)

        pos = self.portfolio.positions[ticker]
        pos.last_price = exec_price

        trade_record = None
        trade_fraction = 0.25

        if action == 1:  # BUY
            cash_to_use = self.portfolio.cash * trade_fraction
            if cash_to_use >= exec_price:
                shares = int(cash_to_use / exec_price)
                cost = exec_price * shares * 1.001  # transaction cost
                if cost <= self.portfolio.cash:
                    self.portfolio.cash -= cost
                    total_cost = pos.shares * pos.avg_cost + cost
                    pos.shares += shares
                    pos.avg_cost = total_cost / pos.shares if pos.shares > 0 else 0.0
                    trade_record = {
                        "ticker": ticker,
                        "date": str(signal.get("date", date.today())),
                        "action": "BUY",
                        "shares": shares,
                        "price": exec_price,
                        "cost": cost,
                        "portfolio_value_after": self.portfolio.total_equity,
                    }
                    self.portfolio.trade_history.append(trade_record)

        elif action == 2:  # SELL
            if pos.shares > 0:
                shares_to_sell = max(1, int(pos.shares * trade_fraction))
                revenue = exec_price * shares_to_sell * 0.999  # transaction cost
                self.portfolio.cash += revenue
                pos.shares -= shares_to_sell
                if pos.shares == 0:
                    pos.avg_cost = 0.0
                trade_record = {
                    "ticker": ticker,
                    "date": str(signal.get("date", date.today())),
                    "action": "SELL",
                    "shares": shares_to_sell,
                    "price": exec_price,
                    "cost": revenue,
                    "portfolio_value_after": self.portfolio.total_equity,
                }
                self.portfolio.trade_history.append(trade_record)

        if trade_record:
            self.db.insert_trade_order(trade_record)

        self._last_signal_date[ticker] = signal.get("date", date.today())
        return trade_record

    def generate_and_execute_all(self, tickers: list[str] | None = None) -> list[dict]:
        """Run signals for all tickers and execute trades. Returns executed trades."""
        tickers = tickers or config.TICKERS
        trades = []
        for ticker in tickers:
            try:
                signal = self.generate_signal(ticker)
                trade = self.execute_signal(signal)
                if trade:
                    trades.append(trade)
                    logger.info("  %s: %s %d shares @ $%.2f",
                                ticker, trade["action"], trade["shares"], trade["price"])
                else:
                    logger.info("  %s: %s (no trade)", ticker, signal["label"])
            except Exception:
                logger.exception("  %s: signal generation failed", ticker)
        return trades

    def get_portfolio_summary(self) -> dict:
        """Return current portfolio state for dashboard display."""
        positions_data = []
        for ticker, pos in self.portfolio.positions.items():
            if pos.shares > 0:
                positions_data.append({
                    "ticker": ticker,
                    "shares": pos.shares,
                    "avg_cost": round(pos.avg_cost, 2),
                    "last_price": round(pos.last_price, 2),
                    "market_value": round(pos.market_value, 2),
                    "pnl": round(pos.pnl, 2),
                    "pnl_pct": round(pos.pnl_pct, 2),
                })

        return {
            "cash": round(self.portfolio.cash, 2),
            "total_equity": round(self.portfolio.total_equity, 2),
            "total_pnl": round(self.portfolio.total_pnl, 2),
            "total_pnl_pct": round(self.portfolio.total_pnl_pct, 2),
            "positions": positions_data,
            "trade_count": len(self.portfolio.trade_history),
        }


def run_paper_trading_cycle(db: DatabaseManager | None = None):
    """One full paper trading cycle: data → NLP → RL inference → execute."""
    from main import step_ingest, step_nlp

    db = db or DatabaseManager()

    logger.info("=" * 50)
    logger.info("Paper Trading Cycle: %s", datetime.now().isoformat())

    # 1. Collect latest data
    logger.info("Step 1/4: Data collection...")
    step_ingest(db)

    # 2. NLP sentiment
    logger.info("Step 2/4: NLP sentiment...")
    step_nlp(db)

    # 3. RL inference + execute
    logger.info("Step 3/4: RL inference & execution...")
    trader = PaperTrader(db)
    trades = trader.generate_and_execute_all()

    # 4. Summary
    logger.info("Step 4/4: Portfolio summary...")
    summary = trader.get_portfolio_summary()
    logger.info("  Equity: $%.2f | P&L: $%.2f (%.2f%%) | Trades today: %d",
                summary["total_equity"], summary["total_pnl"],
                summary["total_pnl_pct"], len(trades))
    logger.info("=" * 50)

    return summary


def run_paper_validation(db: DatabaseManager, ablation_results: dict) -> dict:
    """Post-ablation paper trading validation.

    Generates forward-looking signals for all tickers using the trained model
    and logs them as paper trade orders. Complements the backtest BH comparison
    with an operational readiness check.
    """
    logger.info("Generating paper trading signals from trained model...")

    buy_signals = []
    sell_signals = []
    hold_signals = []
    errors = []

    try:
        trader = PaperTrader(db)
        _ = trader.load_model()  # verify model loads
    except FileNotFoundError:
        logger.warning("No trained model found — skipping paper validation")
        return {"status": "skipped", "reason": "no_model"}

    for ticker in config.TICKERS:
        try:
            signal = trader.generate_signal(ticker)
            trader.execute_signal(signal)

            if signal["action"] == 1:
                buy_signals.append(ticker)
            elif signal["action"] == 2:
                sell_signals.append(ticker)
            else:
                hold_signals.append(ticker)

            logger.info("  %s: %s (conf=%.2f, sentiment=%.3f)",
                        ticker, signal["label"], signal["confidence"], signal["sentiment_score"])
        except Exception as e:
            logger.warning("  %s: signal failed — %s", ticker, e)
            errors.append(str(ticker))

    summary = trader.get_portfolio_summary()

    logger.info("Paper validation complete:")
    logger.info("  BUY: %d (%s)", len(buy_signals), ", ".join(buy_signals[:5]) if buy_signals else "none")
    logger.info("  SELL: %d (%s)", len(sell_signals), ", ".join(sell_signals[:5]) if sell_signals else "none")
    logger.info("  HOLD: %d", len(hold_signals))
    logger.info("  Portfolio: $%.2f (%+.2f%%)", summary["total_equity"], summary["total_pnl_pct"])

    return {
        "status": "completed",
        "buy_count": len(buy_signals),
        "sell_count": len(sell_signals),
        "hold_count": len(hold_signals),
        "errors": len(errors),
        "buy_tickers": buy_signals,
        "sell_tickers": sell_signals,
        "portfolio_summary": summary,
    }
