"""Standalone data collection — fetch once, train many times.

Usage:
  python collect_data.py              # Fetch all tickers, full date range
  python collect_data.py --tickers AAPL MSFT  # Specific tickers only
  python collect_data.py --market-only       # Skip news, only market data
  python collect_data.py --news-only         # Skip market, only news
"""

import argparse
import logging
import sys

import config
from data_storage.db_manager import DatabaseManager
from data_ingestion.market_data import fetch_ohlcv, fetch_ohlcv_alpha_vantage
from data_ingestion.news_fetcher import fetch_news_for_all_tickers, fetch_news_rss, fetch_sample_news

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("collect")

DB = DatabaseManager()


def collect_market(tickers: list[str]):
    """Fetch market data for given tickers."""
    for ticker in tickers:
        logger.info("Fetching market data for %s...", ticker)

        # Try Alpha Vantage first (more stable)
        df = fetch_ohlcv_alpha_vantage(ticker)
        if df is None or df.empty:
            logger.info("Alpha Vantage failed, trying yfinance...")
            df = fetch_ohlcv(ticker, prefer="yfinance")

        if df is not None and not df.empty:
            new = DB.insert_market_data(ticker, df)
            logger.info("  %s: %d rows in DB (%d new/updated)", ticker, len(df), new)
        else:
            logger.warning("  %s: all sources failed", ticker)


def collect_news(tickers: list[str]):
    """Fetch news for given tickers."""
    logger.info("Fetching news...")
    records, failed = fetch_news_for_all_tickers(tickers)

    if records:
        count = DB.insert_news(records)
        logger.info("  %d news articles inserted", count)

    for ticker in failed:
        logger.info("  Trying RSS for %s...", ticker)
        rss_records = fetch_news_rss(ticker)
        if rss_records:
            DB.insert_news(rss_records)
            logger.info("  RSS: %d articles for %s", len(rss_records), ticker)
        else:
            logger.info("  Using sample news for %s", ticker)
            sample = fetch_sample_news(ticker)
            DB.insert_news(sample.to_dict("records"))


def show_db_stats():
    """Print current DB state."""
    import sqlite3
    conn = sqlite3.connect(str(DB.db_path))
    for table in ["market_data", "news", "sentiment_signals", "trading_logs"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        latest = conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()[0] if count > 0 else "N/A"
        print(f"  {table}: {count} rows, latest: {latest}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Standalone data collection for NLP-RL Platform")
    parser.add_argument("--tickers", nargs="+", default=config.TICKERS)
    parser.add_argument("--market-only", action="store_true")
    parser.add_argument("--news-only", action="store_true")
    parser.add_argument("--stats", action="store_true", help="Show DB stats only, no fetch")
    args = parser.parse_args()

    if args.stats:
        show_db_stats()
        return

    if not args.news_only:
        collect_market(args.tickers)

    if not args.market_only:
        collect_news(args.tickers)

    print()
    show_db_stats()
    logger.info("Data collection complete.")


if __name__ == "__main__":
    main()
