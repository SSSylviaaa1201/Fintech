"""FNSPID dataset loader: stream + filter NASDAQ news for our ticker universe.

FNSPID (Financial News Sentiment and Price Influence Dataset) is a 21.6GB CSV
hosted on HuggingFace with full-text NASDAQ news articles from ~2022-2024.

Usage:
  python data_ingestion/fnspid_loader.py              # stream download + filter
  python data_ingestion/fnspid_loader.py --stats      # show loaded stats
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests

from config import TICKERS, ROOT_DIR

# FNSPID has full article text — fields can exceed 128KB default limit
# Use 1GB limit (max safe value for C long on 64-bit Windows)
csv.field_size_limit(2**30)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("fnspid")

FNSPID_URL = "https://hf-mirror.com/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
OUTPUT_PATH = Path(ROOT_DIR) / "data" / "fnspid" / "filtered_news.parquet"
CHUNK_SIZE = 10_000  # rows per chunk (streaming)
TICKER_SET = set(TICKERS)


def _download_file(url: str, dest: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    """Download a file with progress logging. Returns file size in bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing partial download
    existing_size = dest.stat().st_size if dest.exists() else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        logger.info("Resuming from byte %d (%.1f MB)", existing_size, existing_size / (1024**2))

    response = requests.get(url, stream=True, timeout=(30, 3600), headers=headers)
    total_size = int(response.headers.get("content-length", 0)) + existing_size
    logger.info("Total size: %.1f GB", total_size / (1024**3) if total_size else 0)

    mode = "ab" if existing_size > 0 else "wb"
    downloaded = existing_size
    t0 = time.time()
    last_log = t0

    with open(dest, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                # Log progress every 30 seconds
                now = time.time()
                if now - last_log > 30:
                    elapsed = now - t0
                    rate = (downloaded - existing_size) / elapsed / (1024**2) if elapsed > 0 else 0
                    pct = downloaded / total_size * 100 if total_size else 0
                    eta = (total_size - downloaded) / (rate * 1024 * 1024) if rate > 0 else 0
                    logger.info("Download: %.1f%% (%.1f/%.1f GB) %.1f MB/s ETA %.0f min",
                                pct, downloaded / (1024**3), total_size / (1024**3),
                                rate, eta / 60)
                    last_log = now

    elapsed = time.time() - t0
    logger.info("Download complete: %.1f GB in %.0f sec (%.1f MB/s)",
                downloaded / (1024**3), elapsed,
                (downloaded - existing_size) / elapsed / (1024**2) if elapsed > 0 else 0)
    return downloaded


def stream_and_filter(url: str, tickers: set, output: Path, chunk_size: int = 50_000):
    """Download CSV, then filter to tickers and save as parquet."""
    raw_path = output.parent / "nasdaq_exteral_data.csv"

    # Step 1: Download
    if raw_path.exists() and raw_path.stat().st_size > 20_000_000_000:
        logger.info("File already downloaded: %.1f GB", raw_path.stat().st_size / (1024**3))
    else:
        _download_file(url, raw_path)

    # Step 2: Filter
    logger.info("Filtering to %d tickers...", len(tickers))
    t0 = time.time()
    all_chunks = []
    total_rows = 0
    matched_rows = 0

    reader = pd.read_csv(
        raw_path,
        chunksize=chunk_size,
        engine="python",  # tolerant of malformed EOF (incomplete last line)
        on_bad_lines="skip",
        encoding="utf-8",
    )

    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.upper().str.strip()
        filtered = chunk[chunk["Stock_symbol"].isin(tickers)]

        if not filtered.empty:
            all_chunks.append(filtered)
            matched_rows += len(filtered)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            logger.info("Scanned %d rows, matched %d (%.2f%%), %.0f rows/s",
                        total_rows, matched_rows,
                        matched_rows / total_rows * 100 if total_rows else 0,
                        total_rows / elapsed if elapsed else 0)

    # Save
    if all_chunks:
        result = pd.concat(all_chunks, ignore_index=True)
        result.to_parquet(output, index=False, compression="snappy")
        logger.info("Saved %d matched rows to %s (%.1f MB)",
                     len(result), output, output.stat().st_size / (1024**2))
    else:
        logger.warning("No matching rows found")

    elapsed = time.time() - t0
    logger.info("Done: %d rows scanned in %.0f sec, %d matched (%.2f%%)",
                total_rows, elapsed, matched_rows,
                matched_rows / total_rows * 100 if total_rows else 0)
    return matched_rows


def show_stats(output: Path):
    """Display summary statistics of the filtered dataset."""
    if not output.exists():
        logger.error("No data at %s", output)
        return

    df = pd.read_parquet(output)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    logger.info("=== FNSPID FILTERED DATASET ===")
    logger.info("Total rows: %d", len(df))
    logger.info("Tickers: %d", df["Stock_symbol"].nunique())
    logger.info("Date range: %s → %s", df["Date"].min().date(), df["Date"].max().date())

    # Per-ticker counts
    counts = df.groupby("Stock_symbol").size().sort_values(ascending=False)
    logger.info("Top 10 tickers by article count:")
    for ticker, count in counts.head(10).items():
        logger.info("  %s: %d", ticker, count)
    logger.info("Bottom 5:")
    for ticker, count in counts.tail(5).items():
        logger.info("  %s: %d", ticker, count)

    # Year distribution
    df["year"] = df["Date"].dt.year
    logger.info("Articles per year:")
    for year, count in df.groupby("year").size().items():
        logger.info("  %d: %d", int(year), count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNSPID dataset loader")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    args = parser.parse_args()

    if args.stats:
        show_stats(OUTPUT_PATH)
    else:
        stream_and_filter(FNSPID_URL, TICKER_SET, OUTPUT_PATH)
