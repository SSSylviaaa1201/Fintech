"""
Backfill VADER sentiment - per-ticker streaming with flush.
"""
import sqlite3, sys, time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DB = 'data/trading.db'
analyzer = SentimentIntensityAnalyzer()

def log(msg):
    print(msg)
    sys.stdout.flush()

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("SELECT DISTINCT ticker FROM news ORDER BY ticker")
tickers = [r[0] for r in cur.fetchall()]
log(f"Processing {len(tickers)} tickers...")

total_missing = 0
total_inserted = 0
t_start = time.time()

for ti, ticker in enumerate(tickers):
    t0 = time.time()

    # Existing VADER dates
    cur.execute("SELECT date FROM sentiment_signals WHERE ticker=? AND method='vader'", (ticker,))
    existing_dates = set(r[0] for r in cur.fetchall())

    # All news for this ticker
    cur.execute(
        "SELECT DATE(published_at), title, content FROM news WHERE ticker=? ORDER BY DATE(published_at)",
        (ticker,)
    )
    rows = cur.fetchall()
    if not rows:
        continue

    # Group by date
    date_texts = {}
    for date, title, content in rows:
        text = f"{title or ''} {content or ''}"
        if date not in date_texts:
            date_texts[date] = []
        date_texts[date].append(text)

    # Missing dates
    missing_dates = [d for d in date_texts if d not in existing_dates]
    if not missing_dates:
        log(f"  [{ti+1}/{len(tickers)}] {ticker}: skip ({len(date_texts)} dates all have VADER)")
        continue

    # VADER
    batch = []
    for date in missing_dates:
        combined = ' '.join(date_texts[date])
        score = analyzer.polarity_scores(combined)['compound']
        batch.append((ticker, date, 'vader', score, 1.0, '', ''))

    # Insert
    cur.executemany(
        "INSERT INTO sentiment_signals (ticker, date, method, sentiment_score, confidence, label, reasoning) VALUES (?,?,?,?,?,?,?)",
        batch
    )
    conn.commit()

    total_missing += len(missing_dates)
    total_inserted += len(batch)
    elapsed = time.time() - t_start
    rate = (ti + 1) / elapsed if elapsed > 0 else 0
    eta = (len(tickers) - ti - 1) / rate if rate > 0 else 0

    log(f"  [{ti+1}/{len(tickers)}] {ticker}: +{len(missing_dates)} ({len(date_texts)} dates, {len(date_texts)-len(missing_dates)} had) | total {total_inserted:,} | {elapsed:.0f}s | ETA {eta:.0f}s")

log(f"\n=== DONE in {time.time()-t_start:.0f}s ===")
log(f"Inserted {total_inserted:,} VADER records across {len(tickers)} tickers")

# Verify
cur.execute("SELECT COUNT(*) FROM sentiment_signals WHERE method='vader'")
total_vader = cur.fetchone()[0]
cur.execute("SELECT COUNT(DISTINCT ticker || '_' || DATE(published_at)) FROM news")
total_news = cur.fetchone()[0]
log(f"VADER coverage: {total_vader:,}/{total_news:,} = {total_vader/total_news*100:.1f}%")
conn.close()
