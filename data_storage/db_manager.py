"""Database manager: connection management and CRUD operations."""

from __future__ import annotations

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import DB_PATH
from data_storage.schema import ALL_TABLES


class DatabaseManager:
    """SQLite database manager for the trading platform."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for ddl in ALL_TABLES:
                conn.execute(ddl)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── News ──────────────────────────────────────────────────────────

    def insert_news(self, records: list[dict]) -> int:
        """Insert news articles. Returns count inserted (ignores duplicates)."""
        with self._connect() as conn:
            cur = conn.executemany(
                "INSERT OR IGNORE INTO news (ticker, source, title, content, url, published_at) "
                "VALUES (:ticker, :source, :title, :content, :url, :published_at)",
                records,
            )
            conn.commit()
            return cur.rowcount

    def get_news(self, ticker: str, limit: int = 100) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM news WHERE ticker=? ORDER BY published_at DESC LIMIT ?",
                conn,
                params=(ticker, limit),
            )

    def get_news_stratified(self, ticker: str, per_year: int = 50) -> pd.DataFrame:
        """Sample news evenly across years to avoid recency bias.

        Uses SQLite window functions to select up to per_year articles per year,
        providing balanced coverage across the full date range.
        """
        with self._connect() as conn:
            return pd.read_sql_query(
                """SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY strftime('%Y', published_at) ORDER BY published_at DESC
                    ) as rn
                    FROM news WHERE ticker=?
                ) WHERE rn <= ?
                ORDER BY published_at DESC""",
                conn,
                params=(ticker, per_year),
            )

    # ── Market Data ───────────────────────────────────────────────────

    def insert_market_data(self, ticker: str, df: pd.DataFrame) -> int:
        """Insert OHLCV rows. Expects df with columns: date, open, high, low, close, volume."""
        if df.empty:
            return 0
        records = df.copy()
        records["ticker"] = ticker
        records["date"] = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
        records = records.where(pd.notnull(records), None)
        with self._connect() as conn:
            cur = conn.executemany(
                "INSERT OR REPLACE INTO market_data (ticker, date, open, high, low, close, volume) "
                "VALUES (:ticker, :date, :open, :high, :low, :close, :volume)",
                records.to_dict("records"),
            )
            conn.commit()
            return cur.rowcount

    def get_market_data(self, ticker: str) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM market_data "
                "WHERE ticker=? ORDER BY date ASC",
                conn,
                params=(ticker,),
            )
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date", drop=False)

    def get_all_tickers(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT ticker FROM market_data").fetchall()
            return [r["ticker"] for r in rows]

    # ── Sentiment ─────────────────────────────────────────────────────

    def upsert_sentiment(self, records: list[dict]):
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO sentiment_signals (ticker, date, method, sentiment_score, confidence, label) "
                "VALUES (:ticker, :date, :method, :sentiment_score, :confidence, :label)",
                records,
            )
            conn.commit()

    def get_sentiment(self, ticker: str, method: str | None = None) -> pd.DataFrame:
        with self._connect() as conn:
            if method:
                df = pd.read_sql_query(
                    "SELECT ticker, date, method, sentiment_score FROM sentiment_signals "
                    "WHERE ticker=? AND method=? ORDER BY date ASC",
                    conn,
                    params=(ticker, method),
                )
            else:
                df = pd.read_sql_query(
                    "SELECT ticker, date, method, sentiment_score FROM sentiment_signals "
                    "WHERE ticker=? ORDER BY date ASC",
                    conn,
                    params=(ticker,),
                )
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── Trading Logs ──────────────────────────────────────────────────

    def insert_trading_log(self, records: list[dict]):
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO trading_logs (episode, step, ticker, date, action, price, "
                "position, cash, portfolio_value, sentiment_score, reward) "
                "VALUES (:episode, :step, :ticker, :date, :action, :price, "
                ":position, :cash, :portfolio_value, :sentiment_score, :reward)",
                records,
            )
            conn.commit()

    def get_trading_logs(self, episode: int | None = None) -> pd.DataFrame:
        with self._connect() as conn:
            if episode is not None:
                return pd.read_sql_query(
                    "SELECT * FROM trading_logs WHERE episode=? ORDER BY step ASC",
                    conn,
                    params=(episode,),
                )
            return pd.read_sql_query("SELECT * FROM trading_logs ORDER BY episode, step ASC", conn)

    def clear_trading_logs(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM trading_logs")
            conn.commit()

    # ── Trade Orders ──────────────────────────────────────────────────

    def insert_trade_order(self, record: dict):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trade_orders (ticker, date, action, shares, price, cost, portfolio_value_after) "
                "VALUES (:ticker, :date, :action, :shares, :price, :cost, :portfolio_value_after)",
                record,
            )
            conn.commit()

    def get_trade_orders(self, ticker: str | None = None) -> pd.DataFrame:
        with self._connect() as conn:
            if ticker:
                return pd.read_sql_query(
                    "SELECT * FROM trade_orders WHERE ticker=? ORDER BY date ASC",
                    conn,
                    params=(ticker,),
                )
            return pd.read_sql_query("SELECT * FROM trade_orders ORDER BY date ASC", conn)
