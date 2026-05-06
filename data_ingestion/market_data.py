"""Fetch OHLCV market data: yfinance → Alpha Vantage → synthetic fallback."""

import logging
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import TICKERS, START_DATE, END_DATE

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds
TICKER_DELAY = 3  # seconds between tickers

TICKER_PROFILES = {
    "AAPL": {"base_price": 180, "volatility": 0.015, "drift": 0.0004},
    "MSFT": {"base_price": 380, "volatility": 0.014, "drift": 0.0005},
    "GOOGL": {"base_price": 150, "volatility": 0.016, "drift": 0.0003},
    "AMZN": {"base_price": 180, "volatility": 0.018, "drift": 0.0004},
    "TSLA": {"base_price": 240, "volatility": 0.025, "drift": 0.0002},
}


def _generate_synthetic_ohlcv(
    ticker: str,
    start: str = START_DATE,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for a ticker."""
    end = end or datetime.now().strftime("%Y-%m-%d")
    dates = pd.date_range(start=start, end=end, freq="B")

    profile = TICKER_PROFILES.get(ticker, {"base_price": 100, "volatility": 0.015, "drift": 0.0003})
    base = profile["base_price"]
    vol = profile["volatility"]
    drift = profile["drift"]

    np.random.seed(hash(ticker) % 2**31)
    returns = np.random.normal(drift, vol, len(dates))
    prices = base * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"date": dates, "close": prices})
    df["open"] = df["close"].shift(1).fillna(base) * np.random.uniform(0.995, 1.005, len(df))
    df["high"] = df[["open", "close"]].max(axis=1) * np.random.uniform(1.001, 1.02, len(df))
    df["low"] = df[["open", "close"]].min(axis=1) * np.random.uniform(0.98, 0.999, len(df))
    df["volume"] = np.random.randint(5_000_000, 80_000_000, len(df))
    df["adjusted_close"] = df["close"]
    return df


def fetch_ohlcv_alpha_vantage(ticker: str, start: str = "2020-01-01", end: Optional[str] = None, outputsize: str = "compact"):
    """Fetch OHLCV from Alpha Vantage. Returns DataFrame or None.

    Args:
        outputsize: "compact" (~100 rows, last ~5 months, free tier) or "full" (full 20-year history, premium only).
                    Free tier: 25 requests/day.
    """
    try:
        from alpha_vantage.timeseries import TimeSeries
    except ImportError:
        logger.warning("alpha-vantage not installed")
        return None

    key = os.getenv("ALPHA_VANTAGE_KEY", "")
    if not key:
        logger.info("No ALPHA_VANTAGE_KEY, skipping Alpha Vantage")
        return None

    end = end or datetime.now().strftime("%Y-%m-%d")

    for av_attempt in range(1, 4):
        try:
            ts = TimeSeries(key=key, output_format="pandas")
            df, meta = ts.get_daily(symbol=ticker, outputsize=outputsize)

            df = df.reset_index()
            # Normalize column names: "1. open" → "1_open" → "open"
            df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]

            col_map = {
                "1__open": "open", "2__high": "high",
                "3__low": "low", "4__close": "close", "5__volume": "volume",
                "1_open": "open", "2_high": "high",
                "3_low": "low", "4_close": "close", "5_volume": "volume",
                "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            if "date" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "date"})
            if "date" not in df.columns:
                date_col = df.columns[0]
                df = df.rename(columns={date_col: "date"})

            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            df["adjusted_close"] = df.get("close", df.iloc[:, -1])

            keep_cols = ["date", "open", "high", "low", "close", "volume", "adjusted_close"]
            available = [c for c in keep_cols if c in df.columns]
            if available:
                logger.info("Alpha Vantage: %d rows for %s", len(df), ticker)
                return df[available]
        except Exception as e:
            if av_attempt < 3:
                logger.warning("Alpha Vantage attempt %d/3 for %s: %s", av_attempt, ticker, e)
                time.sleep(2)
            else:
                logger.warning("Alpha Vantage failed for %s after 3 attempts", ticker)

    return None


def _try_yfinance(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Try fetching from yfinance with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("yfinance %s (attempt %d/%d)", ticker, attempt, MAX_RETRIES)
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
            if df.empty:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            if "adj close" in df.columns:
                df = df.rename(columns={"adj close": "adjusted_close"})
            return df
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning("Retrying yfinance (%s): %s", ticker, e)
                time.sleep(RETRY_BASE_DELAY * attempt)
            else:
                logger.warning("yfinance failed for %s: %s", ticker, e)
    return None


def fetch_ohlcv(
    ticker: str,
    start: str = START_DATE,
    end: Optional[str] = None,
    prefer: str = "alpha_vantage",
) -> pd.DataFrame:
    """Fetch OHLCV data with fallback chain.

    Args:
        prefer: "alpha_vantage" (stable, API key required) or "yfinance" (unstable)
    """
    end = end or datetime.now().strftime("%Y-%m-%d")

    sources = []
    if prefer == "alpha_vantage":
        sources = [("Alpha Vantage", lambda: fetch_ohlcv_alpha_vantage(ticker, start, end)),
                   ("yfinance", lambda: _try_yfinance(ticker, start, end))]
    else:
        sources = [("yfinance", lambda: _try_yfinance(ticker, start, end)),
                   ("Alpha Vantage", lambda: fetch_ohlcv_alpha_vantage(ticker, start, end))]

    for name, fetcher in sources:
        try:
            df = fetcher()
            if df is not None and not df.empty:
                logger.info("Got %d rows from %s for %s", len(df), name, ticker)
                return df
        except Exception as e:
            logger.warning("%s failed for %s: %s", name, ticker, e)

    logger.warning("All sources failed for %s, using synthetic data.", ticker)
    return _generate_synthetic_ohlcv(ticker, start, end)


def fetch_incremental(ticker: str, since_date: str) -> Optional[pd.DataFrame]:
    """Fetch only data newer than since_date. Returns None if DB has latest."""
    end = datetime.now().strftime("%Y-%m-%d")
    if since_date >= end:
        return None  # already up to date
    logger.info("Incremental fetch for %s since %s", ticker, since_date)
    return fetch_ohlcv(ticker, start=since_date, end=end)


def fetch_all_tickers(
    tickers: Optional[list[str]] = None,
    start: str = START_DATE,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for all tickers with delays between requests."""
    tickers = tickers or TICKERS
    result = {}
    for i, t in enumerate(tickers):
        if i > 0:
            time.sleep(TICKER_DELAY)
        df = fetch_ohlcv(t, start=start)
        if not df.empty:
            result[t] = df
    return result
