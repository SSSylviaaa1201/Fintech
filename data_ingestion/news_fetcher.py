"""Fetch financial news: Finnhub → Alpha Vantage News → NewsAPI → RSS → sample fallback."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from config import TICKERS, NEWS_LOOKBACK_DAYS, NEWSAPI_KEY, ALPHA_VANTAGE_KEY, FINNHUB_API_KEY

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def fetch_news_newsapi(
    ticker: str,
    api_key: Optional[str] = None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
) -> list[dict]:
    """Fetch news articles from NewsAPI for a given ticker."""
    key = api_key or NEWSAPI_KEY
    if not key:
        logger.info("No NewsAPI key, will fall back to RSS")
        return []

    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    params = {
        "q": f"{ticker} stock",
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 50,
        "apiKey": key,
    }
    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
    except requests.RequestException as e:
        logger.warning("NewsAPI request failed for %s: %s", ticker, e)
        return []

    if resp.status_code != 200:
        logger.warning("NewsAPI returned %d: %s", resp.status_code, resp.text[:200])
        return []

    articles = resp.json().get("articles", [])
    records = []
    for a in articles:
        records.append({
            "ticker": ticker,
            "source": a.get("source", {}).get("name", ""),
            "title": a.get("title", ""),
            "content": a.get("content") or a.get("description", ""),
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
        })
    return records


def _fetch_rss_fallback(ticker: str) -> list[dict]:
    """Fetch news via rss_fetcher module (4 sources) as fallback."""
    from data_ingestion.rss_fetcher import fetch_news_rss
    return fetch_news_rss(ticker, max_per_source=30)


def fetch_news_finnhub(
    ticker: str,
    api_key: Optional[str] = None,
    lookback_months: int = 12,
) -> list[dict]:
    """Fetch news from Finnhub (free tier: 60 calls/min, ~1yr lookback)."""
    key = api_key or FINNHUB_API_KEY
    if not key:
        logger.info("No Finnhub API key, skipping")
        return []

    today = datetime.now().date()
    all_records = []
    seen_urls = set()

    # Fetch in 2-month chunks to maximize per-call density
    chunk_months = 2
    for offset in range(0, lookback_months, chunk_months):
        chunk_end = today - timedelta(days=offset * 30)
        chunk_start = today - timedelta(days=(offset + chunk_months) * 30)
        try:
            r = requests.get(
                FINNHUB_NEWS_URL,
                params={
                    "symbol": ticker,
                    "from": chunk_start.isoformat(),
                    "to": chunk_end.isoformat(),
                    "token": key,
                },
                timeout=15,
            )
        except requests.RequestException as e:
            logger.warning("Finnhub request failed for %s: %s", ticker, e)
            continue

        if r.status_code != 200:
            logger.warning("Finnhub returned %d: %s", r.status_code, r.text[:200])
            continue

        articles = r.json() if isinstance(r.json(), list) else []
        if not articles:
            break  # no more historical data

        for a in articles:
            url = a.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            ts = a.get("datetime")
            pub_date = datetime.fromtimestamp(ts).isoformat() if ts else ""

            all_records.append({
                "ticker": ticker,
                "source": a.get("source", "Finnhub"),
                "title": a.get("headline", ""),
                "content": a.get("summary", "") or a.get("headline", ""),
                "url": url,
                "published_at": pub_date,
            })

        if offset == 0:
            logger.info("Finnhub chunk %s~%s: %d articles",
                        chunk_start.isoformat(), chunk_end.isoformat(), len(articles))

    logger.info("Finnhub: %d unique articles for %s (%d months)", len(all_records), ticker, lookback_months)
    return all_records


def fetch_news_alphavantage(
    ticker: str,
    api_key: Optional[str] = None,
    lookback_months: int = 24,
) -> list[dict]:
    """Fetch news from Alpha Vantage NEWS_SENTIMENT (free: 25 calls/day, deeper history)."""
    key = api_key or ALPHA_VANTAGE_KEY
    if not key:
        logger.info("No Alpha Vantage key, skipping")
        return []

    today = datetime.now().strftime("%Y%m%dT%H%M")
    all_records = []
    seen_urls = set()

    # Fetch in 3-month chunks
    chunk_months = 3
    for offset in range(0, lookback_months, chunk_months):
        chunk_end = datetime.now() - timedelta(days=offset * 30)
        chunk_start = datetime.now() - timedelta(days=(offset + chunk_months) * 30)
        time_from = chunk_start.strftime("%Y%m%dT%H%M")
        time_to = chunk_end.strftime("%Y%m%dT%H%M")

        try:
            r = requests.get(
                ALPHA_VANTAGE_URL,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "time_from": time_from,
                    "time_to": time_to,
                    "limit": 1000,
                    "apikey": key,
                },
                timeout=15,
            )
        except requests.RequestException as e:
            logger.warning("AV news request failed for %s: %s", ticker, e)
            continue

        if r.status_code != 200:
            continue

        data = r.json()
        articles = data.get("feed", [])
        if not articles:
            continue

        for a in articles:
            url = a.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            all_records.append({
                "ticker": ticker,
                "source": a.get("source", "Alpha Vantage"),
                "title": a.get("title", ""),
                "content": a.get("summary", "") or a.get("title", ""),
                "url": url,
                "published_at": a.get("time_published", ""),
            })

        if offset == 0:
            logger.info("AV chunk %s~%s: %d articles", time_from[:8], time_to[:8], len(articles))

        # Rate limit: 5 calls/min for free tier
        time.sleep(0.5)

    logger.info("AlphaVantage: %d unique articles for %s", len(all_records), ticker)
    return all_records


def fetch_news_for_all_tickers(
    tickers: Optional[list[str]] = None,
) -> tuple[list[dict], list[str]]:
    """Fetch news for all tickers. Fallback chain: Finnhub → Alpha Vantage → NewsAPI → RSS → sample."""
    tickers = tickers or TICKERS
    all_records = []
    failed = []

    for t in tickers:
        # 1) Finnhub (fast, 60 calls/min, ~12 months lookback)
        records = fetch_news_finnhub(t)
        if records:
            all_records.extend(records)
            logger.info("Finnhub: %d articles for %s", len(records), t)
            continue

        # 2) Alpha Vantage News (25 calls/day, deeper history)
        logger.info("Finnhub exhausted, trying Alpha Vantage news for %s...", t)
        records = fetch_news_alphavantage(t)
        if records:
            all_records.extend(records)
            logger.info("Alpha Vantage: %d articles for %s", len(records), t)
            continue

        # 3) NewsAPI
        records = fetch_news_newsapi(t)
        if records:
            all_records.extend(records)
            logger.info("NewsAPI: %d articles for %s", len(records), t)
            continue

        # 4) RSS fallback
        logger.info("NewsAPI exhausted, trying RSS feeds for %s...", t)
        records = _fetch_rss_fallback(t)
        if records:
            all_records.extend(records)
            logger.info("RSS: %d articles for %s", len(records), t)
        else:
            failed.append(t)
            logger.warning("All sources failed for %s", t)

    return all_records, failed


def fetch_sample_news(ticker: str = "AAPL") -> pd.DataFrame:
    """
    Generate synthetic sample news when NewsAPI is unavailable.
    Used for development/testing.
    """
    dates = pd.date_range(end=datetime.now(), periods=NEWS_LOOKBACK_DAYS, freq="D")
    headlines = [
        f"{ticker} reports strong quarterly earnings",
        f"{ticker} announces new product line",
        f"{ticker} faces regulatory scrutiny",
        f"Analysts upgrade {ticker} rating to Buy",
        f"{ticker} stock dips amid market uncertainty",
        f"{ticker} expands into new markets",
        f"Supply chain issues impact {ticker} production",
        f"{ticker} beats revenue expectations",
        f"Concerns grow over {ticker} valuation",
        f"{ticker} announces partnership deal",
    ]
    import random
    random.seed(42)
    records = []
    for d in dates:
        records.append({
            "ticker": ticker,
            "source": "sample",
            "title": random.choice(headlines),
            "content": random.choice(headlines),
            "url": f"sample://{ticker}/{d.date()}",
            "published_at": d.isoformat(),
        })
    return pd.DataFrame(records)
