"""GDELT Project news fetcher: global news with pre-computed sentiment (GCAM tone).

GDELT 2.0 API provides free access to global news articles from 2015-present,
with GCAM (Global Content Analysis Measures) tone scores ranging from -100 to +100.

This module:
  1. Fetches news articles by ticker/company name via GDELT Article List API
  2. Extracts title, URL, publication date, and GCAM tone score
  3. Converts GCAM tone (-100/+100) to our sentiment format (-1/+1)

Rate limits: GDELT API is free but throttled. Use 1 request/second.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GDELT_ARTICLE_SEARCH = "https://api.gdeltproject.org/api/v2/doc/doc"

# Company name mapping for GDELT queries (searches full text, not just ticker)
TICKER_COMPANY_NAMES = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "ADBE": "Adobe",
    "INTC": "Intel",
    "CRM": "Salesforce",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "V": "Visa Inc",
    "MA": "Mastercard",
    "GS": "Goldman Sachs",
    "BLK": "BlackRock",
    "AXP": "American Express",
    "MS": "Morgan Stanley",
    "WFC": "Wells Fargo",
    "C": "Citigroup",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth",
    "PFE": "Pfizer",
    "ABBV": "AbbVie",
    "MRK": "Merck",
    "TMO": "Thermo Fisher",
    "ABT": "Abbott",
    "BMY": "Bristol Myers Squibb",
    "GILD": "Gilead Sciences",
    "LLY": "Eli Lilly",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "WMT": "Walmart",
    "COST": "Costco",
    "NKE": "Nike",
    "HD": "Home Depot",
    "MCD": "McDonalds",
    "PG": "Procter Gamble",
    "SBUX": "Starbucks",
    "LOW": "Lowes",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "CAT": "Caterpillar",
    "BA": "Boeing",
    "GE": "General Electric",
    "COP": "ConocoPhillips",
    "DE": "Deere",
    "UPS": "United Parcel Service",
    "LMT": "Lockheed Martin",
    "RTX": "Raytheon Technologies",
    "DIS": "Disney",
    "NFLX": "Netflix",
    "NEE": "NextEra Energy",
    "T": "AT&T",
    "VZ": "Verizon",
    "CMCSA": "Comcast",
    "TMUS": "T-Mobile US",
    "SO": "Southern Company",
    "DUK": "Duke Energy",
    "CHTR": "Charter Communications",
}


def _gdelt_tone_to_sentiment(tone: float) -> dict:
    """Convert GDELT GCAM tone (-100 to +100) to our sentiment format (-1 to +1)."""
    score = max(-1.0, min(1.0, tone / 100.0))
    label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
    confidence = min(1.0, abs(tone) / 50.0)  # higher absolute tone → higher confidence
    return {"sentiment_score": round(score, 4), "confidence": round(confidence, 4), "label": label}


def fetch_gdelt_articles(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_articles: int = 100,
    mode: str = "ArtList",
) -> list[dict]:
    """Fetch articles from GDELT 2.0 API by keyword query.

    Args:
        query: Search query (company name or ticker)
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format
        max_articles: Max articles to return (GDELT caps at 250 per call)
        mode: GDELT mode — 'ArtList' for article list, 'ToneChart' for aggregated tone
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d%H%M%S")
    else:
        start_date = start_date.replace("-", "") + "000000"

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        end_date = end_date.replace("-", "") + "235959"

    params = {
        "query": quote(query),
        "mode": mode,
        "format": "json",
        "maxrecords": min(max_articles, 250),
        "startdatetime": start_date,
        "enddatetime": end_date,
        "sort": "DateDesc",
    }

    try:
        resp = requests.get(GDELT_ARTICLE_SEARCH, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("GDELT API request failed for '%s': %s", query, e)
        return []
    except ValueError:
        logger.warning("GDELT API returned non-JSON for '%s'", query)
        return []

    articles = data.get("articles", [])
    records = []
    for a in articles:
        tone = float(a.get("tone", 0))
        sentiment = _gdelt_tone_to_sentiment(tone)
        records.append({
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("domain", "GDELT"),
            "published_at": a.get("seendate", ""),
            "content": a.get("title", ""),  # GDELT free tier doesn't return full text
            "gdelt_tone": tone,
            "sentiment_score": sentiment["sentiment_score"],
            "confidence": sentiment["confidence"],
            "label": sentiment["label"],
            "language": a.get("language", ""),
        })

    return records


def fetch_gdelt_for_ticker(
    ticker: str,
    lookback_days: int = 365,
    max_articles: int = 200,
) -> list[dict]:
    """Fetch GDELT articles for a single ticker using company name + ticker search.

    Tries ticker first, then falls back to company name for better coverage.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    company = TICKER_COMPANY_NAMES.get(ticker, ticker)

    all_records = []
    seen_urls = set()

    # Query 1: ticker symbol + "stock"
    for query in [f'"{ticker}" stock', f'"{company}"']:
        if len(all_records) >= max_articles:
            break
        remaining = max_articles - len(all_records)
        try:
            articles = fetch_gdelt_articles(
                query, start_date, end_date, max_articles=remaining,
            )
            for a in articles:
                if a["url"] not in seen_urls:
                    a["ticker"] = ticker
                    a["query_used"] = query
                    seen_urls.add(a["url"])
                    all_records.append(a)
            time.sleep(1.0)  # rate limit: 1 req/sec
        except Exception as e:
            logger.warning("GDELT query '%s' failed: %s", query, e)

    logger.info("GDELT: %d articles for %s (%d days)",
                len(all_records), ticker, lookback_days)
    return all_records


def fetch_gdelt_for_all_tickers(
    tickers: Optional[list[str]] = None,
    lookback_days: int = 365,
    max_per_ticker: int = 200,
) -> tuple[list[dict], list[str]]:
    """Fetch GDELT articles for all tickers. Returns (records, failed_tickers)."""
    tickers = tickers or list(TICKER_COMPANY_NAMES.keys())
    all_records = []
    failed = []

    for i, ticker in enumerate(tickers):
        if ticker not in TICKER_COMPANY_NAMES:
            logger.debug("Skipping %s: no company name mapping", ticker)
            continue

        try:
            records = fetch_gdelt_for_ticker(ticker, lookback_days, max_per_ticker)
            if records:
                all_records.extend(records)
            else:
                failed.append(ticker)
        except Exception as e:
            logger.warning("GDELT failed for %s: %s", ticker, e)
            failed.append(ticker)

        # Rate limiting between tickers
        if i < len(tickers) - 1:
            time.sleep(1.5)

    logger.info("GDELT total: %d articles for %d/%d tickers (%d failed)",
                len(all_records), len(tickers) - len(failed), len(tickers), len(failed))
    return all_records, failed
