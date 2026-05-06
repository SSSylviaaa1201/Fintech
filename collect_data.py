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
from data_ingestion.market_data import fetch_ohlcv
from data_ingestion.news_fetcher import fetch_news_for_all_tickers, fetch_sample_news
from data_ingestion.rss_fetcher import fetch_news_rss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("collect")

DB = DatabaseManager()


def collect_market(tickers: list[str]):
    """Fetch market data for given tickers. Respects DATA_SOURCE_PRIORITY from config."""
    from config import DATA_SOURCE_PRIORITY

    for i, ticker in enumerate(tickers):
        # Respect TICKER_DELAY between tickers to avoid rate limiting
        if i > 0:
            import time
            time.sleep(config.TICKER_DELAY)

        logger.info("Fetching market data for %s (priority: %s)...", ticker, DATA_SOURCE_PRIORITY)
        df = fetch_ohlcv(ticker, prefer=DATA_SOURCE_PRIORITY[0])

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
    date_cols = {
        "market_data": "date",
        "news": "published_at",
        "sentiment_signals": "date",
        "trading_logs": "date",
    }
    for table in ["market_data", "news", "sentiment_signals", "trading_logs"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count > 0 and table in date_cols:
            latest = conn.execute(f"SELECT MAX({date_cols[table]}) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} rows, latest: {latest}")
        else:
            print(f"  {table}: {count} rows")
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
