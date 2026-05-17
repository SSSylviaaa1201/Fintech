"""QuantumTrade v2.0 Dashboard — Professional Trading Terminal UI.

Data sources (4 DB tables):
  market_data      → K-line, indicators, ticker tape, heatmap
  sentiment_signals → 3-method overlay, sentiment summary
  trading_logs     → equity curve, drawdown, best episode
  trade_orders     → paper trading portfolio, P&L, order history
"""

import sys
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH, TICKERS, INITIAL_CAPITAL, REFRESH_INTERVAL_SECONDS, DATA_DIR
from data_storage.db_manager import DatabaseManager
from dashboard.components.charts import (
    COLORS, DARK_TEMPLATE,
    create_candlestick_chart, create_sentiment_quad, create_heatmap,
    create_rsi_chart, create_macd_chart, create_equity_curve,
    create_convergence_chart,
)

# ═══════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="QuantumTrade v2 · NLP-RL Terminal",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: radial-gradient(ellipse at top, #0f172a 0%, #0a0f1a 60%, #060c14 100%); }

/* ── Ticker tape ── */
@keyframes ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
.ticker-wrap {
    overflow: hidden; white-space: nowrap;
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(17,25,46,0.7));
    border: 1px solid rgba(56,189,248,0.12); border-radius: 10px;
    padding: 10px 0; margin-bottom: 20px;
    backdrop-filter: blur(10px);
}
.ticker-track { display: inline-block; animation: ticker 50s linear infinite; }
.ticker-item { display: inline-block; padding: 0 28px; font-size: 13px; font-weight: 500; }
.ticker-item strong { color: #e2e8f0; margin-right: 6px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f1a 0%, #0e1320 100%);
    border-right: 1px solid rgba(56,189,248,0.06);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(17,25,46,0.5));
    border: 1px solid rgba(56,189,248,0.1); border-radius: 14px;
    padding: 18px !important; transition: border-color 0.3s;
    backdrop-filter: blur(8px);
}
[data-testid="stMetric"]:hover { border-color: rgba(56,189,248,0.3); }
[data-testid="stMetric"] label {
    color: #94a3b8 !important; font-size: 0.68rem !important;
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important; font-size: 1.6rem !important; font-weight: 700 !important;
}

/* ── Section headers ── */
.section-header {
    color: #e2e8f0; font-size: 1.05rem; font-weight: 600;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 2px solid rgba(56,189,248,0.15);
    letter-spacing: 0.02em;
}

/* ── Glass card ── */
.glass-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(17,25,46,0.5));
    border: 1px solid rgba(56,189,248,0.08); border-radius: 12px;
    padding: 16px; margin-bottom: 10px;
    backdrop-filter: blur(8px);
}
.glass-card .label { color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
.glass-card .value { color: #e2e8f0; font-size: 18px; font-weight: 600; }
.glass-card .value-up { color: #ef4444; }
.glass-card .value-down { color: #22c55e; }

/* ── News cards ── */
.news-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(17,25,46,0.5));
    border: 1px solid rgba(56,189,248,0.06); border-radius: 10px;
    padding: 10px 14px; margin-bottom: 6px; font-size: 12px;
    transition: all 0.2s;
}
.news-card:hover { border-color: rgba(56,189,248,0.3); transform: translateY(-1px); }
.news-source { color: #38bdf8; font-size: 10px; font-weight: 600; letter-spacing: 0.03em; }
.news-title { color: #e2e8f0; margin: 4px 0; line-height: 1.4; }
.news-date { color: #64748b; font-size: 10px; }

/* ── Status pulse ── */
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.status-pulse { animation: pulse 2s ease-in-out infinite; }
.status-live { color: #22c55e; }
.status-idle { color: #fbbf24; }
.status-error { color: #ef4444; }

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    border: 1px solid rgba(56,189,248,0.2) !important; color: #38bdf8 !important;
    border-radius: 8px !important; font-weight: 500 !important; font-size: 0.8rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    border-color: #38bdf8 !important; box-shadow: 0 0 12px rgba(56,189,248,0.12);
}

/* ── Chat ── */
.stChatMessage { background: rgba(15,23,42,0.6) !important; border-radius: 10px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: transparent !important; border: 1px solid rgba(56,189,248,0.08) !important;
    border-radius: 10px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { font-size: 11px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0e17; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }

/* ── Tooltip ── */
.freshness-badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 10px; font-weight: 500;
    background: rgba(56,189,248,0.1); color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.2);
}

/* ── Select box ── */
[data-testid="stSelectbox"] label, [data-testid="stMultiselect"] label {
    color: #94a3b8 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DB + Caching
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_db():
    return DatabaseManager(DB_PATH)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_market_data(_db, ticker):
    return _db.get_market_data(ticker)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_watchlist_data(_db, tickers):
    """Batch-load market data for multiple tickers at once."""
    result = {}
    for t in tickers:
        df = _db.get_market_data(t)
        if not df.empty:
            result[t] = df
    return result


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_sentiment_data(_db, ticker):
    return _db.get_sentiment(ticker)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_trading_logs(_db):
    return _db.get_trading_logs()


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_all_sentiment(_db):
    """Load all sentiment data for ablation summary."""
    return _db.get_sentiment()


@st.cache_data(ttl=300)
def get_data_freshness(_db):
    """Return the latest data date across market_data and sentiment."""
    mkt = _db.get_market_data("AAPL")
    last_mkt = mkt["date"].max() if not mkt.empty else "N/A"
    sent = _db.get_sentiment("AAPL")
    last_sent = sent["date"].max() if not sent.empty else "N/A"
    return str(last_mkt), str(last_sent)


db = get_db()

# ═══════════════════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════════════════

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(TICKERS[:8])
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = "AAPL"
if "rss_cache" not in st.session_state:
    st.session_state.rss_cache = {}

# Handle URL query param: ?ticker=XXX for watchlist dot clicks
query_params = st.query_params
if "ticker" in query_params:
    url_ticker = query_params["ticker"].upper()
    if url_ticker in TICKERS:
        st.session_state.current_ticker = url_ticker
        # Clear query params to avoid sticky selection
        st.query_params.clear()

ticker = st.session_state.current_ticker

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Brand ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#38bdf8,#818cf8);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    font-weight:800;color:#0a0e17;font-size:16px;">Q</div>
        <div>
            <div style="font-weight:700;font-size:16px;color:#e2e8f0;letter-spacing:-0.01em;">QuantumTrade</div>
            <div style="font-size:11px;color:#64748b;letter-spacing:0.04em;">NLP-RL TERMINAL v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data Freshness Badge ──
    try:
        last_mkt, last_sent = get_data_freshness(db)
        st.markdown(f"""
        <div style="display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap;">
            <span class="freshness-badge">MKT {last_mkt}</span>
            <span class="freshness-badge">NLP {last_sent}</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    # ── Search ──
    st.markdown('<p style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Ticker Search</p>', unsafe_allow_html=True)
    from data_ingestion.ticker_lookup import search_tickers

    search_query = st.text_input(
        "Enter ticker", placeholder="e.g. AAPL, NVDA...",
        label_visibility="collapsed", key="ticker_search",
    )
    if search_query:
        matches = search_tickers(search_query)
        if matches:
            cols = st.columns(min(len(matches), 3))
            for i, m in enumerate(matches[:9]):
                with cols[i % 3]:
                    if st.button(m["symbol"], key=f"srch_{m['symbol']}", use_container_width=True):
                        st.session_state.current_ticker = m["symbol"]
                        st.cache_data.clear()
                        st.rerun()
        else:
            if st.button(f"Lookup '{search_query.upper()}'", use_container_width=True):
                from data_ingestion.ticker_lookup import lookup_ticker
                result = lookup_ticker(search_query.upper(), db)
                if result["has_data"]:
                    st.session_state.current_ticker = search_query.upper()
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning(result.get("error", "Not found"))

    st.markdown("---")

    # ── Watchlist ──
    st.markdown('<p style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Watchlist</p>', unsafe_allow_html=True)
    watchlist = st.multiselect(
        "Select tickers", TICKERS,
        default=st.session_state.watchlist,
        label_visibility="collapsed",
    )
    if watchlist:
        st.session_state.watchlist = watchlist

    # Quick ticker buttons — HTML styled with colored dots
    cols = st.columns(min(len(watchlist[:12]), 3) if len(watchlist[:12]) > 1 else 1)
    for i, t in enumerate(watchlist[:12]):
        is_active = t == ticker
        dot_color = "#38bdf8" if is_active else "#475569"
        dot_glow = "0 0 8px rgba(56,189,248,0.5)" if is_active else "none"
        with cols[i % len(cols)]:
            st.markdown(f"""
            <a href="?ticker={t}" style="text-decoration:none;display:block;">
            <div style="
                display:flex;align-items:center;gap:6px;
                padding:8px 10px;border-radius:8px;margin-bottom:6px;
                border:1px solid {'rgba(56,189,248,0.4)' if is_active else 'rgba(71,85,105,0.3)'};
                background:{'rgba(56,189,248,0.1)' if is_active else 'rgba(15,23,42,0.6)'};
                transition:all 0.2s;
            ">
                <span style="
                    width:8px;height:8px;border-radius:50%;
                    background:{dot_color};box-shadow:{dot_glow};
                    flex-shrink:0;
                "></span>
                <span style="color:{'#e2e8f0' if is_active else '#94a3b8'};font-size:13px;font-weight:{'600' if is_active else '400'};">{t}</span>
            </div>
            </a>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── News Feed (cached) ──
    st.markdown('<p style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Live News</p>', unsafe_allow_html=True)

    cache_key = f"rss_{ticker}"
    if cache_key not in st.session_state.rss_cache:
        try:
            from data_ingestion.rss_fetcher import fetch_news_rss
            with st.spinner("Fetching headlines..."):
                st.session_state.rss_cache[cache_key] = fetch_news_rss(ticker, max_per_source=3)
        except Exception:
            st.session_state.rss_cache[cache_key] = []

    rss_news = st.session_state.rss_cache.get(cache_key, [])
    if rss_news:
        for article in rss_news[:8]:
            pub_date = str(article.get("published_at", ""))[:10]
            st.markdown(f"""
            <div class="news-card">
                <div class="news-source">{article['source']}</div>
                <div class="news-title">{article['title'][:90]}</div>
                <div class="news-date">{pub_date}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No headlines available")

    if st.button("Refresh News", use_container_width=True, key="refresh_rss"):
        st.session_state.rss_cache.pop(cache_key, None)
        st.rerun()

    st.markdown("---")

    # ── AI Chat ──
    st.markdown('<p style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">AI Assistant</p>', unsafe_allow_html=True)
    with st.expander("Ask about " + ticker, expanded=False):
        for msg in st.session_state.chat_messages[-6:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if chat_input := st.chat_input(f"Ask about {ticker}...", key="chat_input"):
            st.session_state.chat_messages.append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.markdown(chat_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from agents.research_agent import ask
                        response = ask(question=chat_input, ticker=ticker)
                    except Exception as e:
                        # Fallback: semantic search without LLM
                        try:
                            from vector_store.chroma_store import search
                            results = search(chat_input, top_k=3, ticker_filter=ticker)
                            if results:
                                lines = ["*Semantic search results (LLM unavailable):*"]
                                for r in results:
                                    lines.append(f"- {r['metadata'].get('title','?')[:80]} [{r['relevance_score']:.2f}]")
                                response = "\n".join(lines)
                            else:
                                response = f"*No relevant articles found for '{ticker}'.*"
                        except Exception:
                            response = "*AI assistant unavailable. Try again after training a model.*"
                st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # ── System Status ──
    st.markdown("---")
    db_path = Path(DB_PATH)
    db_size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
    st.caption(f"DB: {db_size_mb:.1f} MB  ·  Capital: ${INITIAL_CAPITAL:,.0f}  ·  {datetime.now().strftime('%H:%M')}")

# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

market_df = load_market_data(db, ticker)
sentiment_df = load_sentiment_data(db, ticker)
logs_df = load_trading_logs(db)

# Compute technical indicators on the raw OHLCV data
if not market_df.empty:
    from utils.indicators import compute_indicators
    market_df = compute_indicators(market_df)

has_market = not market_df.empty
has_sentiment = not sentiment_df.empty
has_logs = not logs_df.empty

# ═══════════════════════════════════════════════════════════════════════════
# Header: Ticker Tape
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
    <div style="display:flex;align-items:center;gap:16px;">
        <h1 style="margin:0;font-size:1.8rem;font-weight:700;letter-spacing:-0.02em;">◆ QuantumTrade</h1>
        <span style="color:#64748b;font-size:13px;font-weight:400;">v2.0</span>
    </div>
    <div style="display:flex;gap:24px;font-size:12px;color:#64748b;align-items:center;">
        <span>◆ DQN + FinBERT (3-method NLP)</span>
        <span class="status-pulse" style="color:{'#22c55e' if has_logs else '#fbbf24'};">
            {'● Live' if has_logs else '○ Standby'}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Ticker tape (batched)
watchlist_data = load_watchlist_data(db, st.session_state.watchlist[:12])
tape_parts = []
for t in st.session_state.watchlist[:12]:
    mkt = watchlist_data.get(t)
    if mkt is not None and len(mkt) > 1:
        latest = mkt["close"].iloc[-1]
        prev = mkt["close"].iloc[-2]
        chg = ((latest - prev) / prev * 100) if prev != 0 else 0
        clr = COLORS["up"] if chg >= 0 else COLORS["down"]
        arrow = "▲" if chg >= 0 else "▼"
        tape_parts.append(
            f'<span class="ticker-item"><strong>{t}</strong> ${latest:.2f} '
            f'<span style="color:{clr};">{arrow} {abs(chg):.2f}%</span></span>'
        )

if tape_parts:
    tape_html = "".join(tape_parts)
    st.markdown(f"""
    <div class="ticker-wrap">
        <div class="ticker-track">{tape_html}{tape_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Date Range Filter
# ═══════════════════════════════════════════════════════════════════════════

if "date_range" not in st.session_state:
    st.session_state.date_range = (None, None)

# Determine available date range from market data
date_series = None
if has_market:
    date_series = pd.to_datetime(market_df["date"])
elif has_logs and "date" in logs_df.columns:
    date_series = pd.to_datetime(logs_df["date"])

if date_series is not None and len(date_series) > 1:
    data_min = date_series.min().date()
    data_max = date_series.max().date()

    with st.expander("📅 Filter Date Range", expanded=False):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            start = st.date_input("From", value=st.session_state.date_range[0] or data_min,
                                  min_value=data_min, max_value=data_max,
                                  key="date_start")
        with c2:
            end = st.date_input("To", value=st.session_state.date_range[1] or data_max,
                                min_value=data_min, max_value=data_max,
                                key="date_end")
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Reset", use_container_width=True):
                st.session_state.date_range = (None, None)
                st.rerun()

        if start and end and (start > data_min or end < data_max):
            st.session_state.date_range = (start, end)
            # Filter market data
            if has_market:
                market_dates = pd.to_datetime(market_df["date"])
                mask = (market_dates >= pd.Timestamp(start)) & (market_dates <= pd.Timestamp(end))
                market_df = market_df[mask].reset_index(drop=True)
                has_market = not market_df.empty
            # Filter sentiment data
            if has_sentiment:
                sent_dates = pd.to_datetime(sentiment_df["date"])
                mask = (sent_dates >= pd.Timestamp(start)) & (sent_dates <= pd.Timestamp(end))
                sentiment_df = sentiment_df[mask].reset_index(drop=True)
                has_sentiment = not sentiment_df.empty
            # Filter trading logs
            if has_logs and "date" in logs_df.columns:
                log_dates = pd.to_datetime(logs_df["date"])
                mask = (log_dates >= pd.Timestamp(start)) & (log_dates <= pd.Timestamp(end))
                logs_df = logs_df[mask].reset_index(drop=True)
                has_logs = not logs_df.empty
        else:
            st.session_state.date_range = (None, None)

# ═══════════════════════════════════════════════════════════════════════════
# KPI Row
# ═══════════════════════════════════════════════════════════════════════════

if has_market:
    latest_price = market_df["close"].iloc[-1]
    prev_price = market_df["close"].iloc[-2] if len(market_df) > 1 else latest_price
    day_chg = (latest_price - prev_price) / prev_price * 100 if prev_price != 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(f"{ticker} Price", f"${latest_price:.2f}", delta=f"{day_chg:+.2f}%")
    with c2:
        st.metric("Market Data", f"{len(market_df):,} rows")
    with c3:
        st.metric("Sentiment", f"{len(sentiment_df):,} days" if has_sentiment else "—")
    with c4:
        st.metric("Training Logs", f"{len(logs_df):,}" if has_logs else "—")
    with c5:
        if has_logs:
            best = logs_df.groupby("episode")["portfolio_value"].last().idxmax()
            ep_logs = logs_df[logs_df["episode"] == best]
            final_val = ep_logs["portfolio_value"].iloc[-1]
            pnl_pct = (final_val / INITIAL_CAPITAL - 1) * 100
            st.metric(f"Best Ep #{best}", f"${final_val:,.0f}", delta=f"{pnl_pct:+.1f}%")
        else:
            st.metric("Status", "Ready")

# ═══════════════════════════════════════════════════════════════════════════
# Main Area
# ═══════════════════════════════════════════════════════════════════════════

if not has_market:
    st.info("No market data yet. Run `python main.py` or `python collect_data.py` first.")
    st.stop()

main_col, side_col = st.columns([0.68, 0.32], gap="medium")

with main_col:
    # Row 1: Candlestick
    st.markdown('<p class="section-header">Candlestick Chart</p>', unsafe_allow_html=True)
    trades = None
    if has_logs:
        best_ep = logs_df.groupby("episode")["portfolio_value"].last().idxmax()
        ep_logs = logs_df[logs_df["episode"] == best_ep]
        trades = ep_logs[ep_logs["action"] != 0][["step", "action", "price"]].copy()

    try:
        fig = create_candlestick_chart(market_df.tail(200), trades)
        st.plotly_chart(fig, key="candlestick", use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.error(f"Chart error: {e}")

    # Row 2: RSI | MACD
    c_ta1, c_ta2 = st.columns(2)
    with c_ta1:
        st.markdown('<p class="section-header">RSI (14)</p>', unsafe_allow_html=True)
        try:
            fig_rsi = create_rsi_chart(market_df.tail(200))
            st.plotly_chart(fig_rsi, key="rsi", use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.caption("RSI unavailable")
    with c_ta2:
        st.markdown('<p class="section-header">MACD</p>', unsafe_allow_html=True)
        try:
            fig_macd = create_macd_chart(market_df.tail(200))
            st.plotly_chart(fig_macd, key="macd", use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.caption("MACD unavailable")

    # Row 3: Sentiment
    st.markdown('<p class="section-header">NLP Sentiment — 3 Methods</p>', unsafe_allow_html=True)
    if has_sentiment:
        try:
            fig_sent = create_sentiment_quad(sentiment_df)
            st.plotly_chart(fig_sent, key="sentiment", use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.caption("Sentiment chart unavailable")
    else:
        st.info("Run `python main.py --skip-collect` to generate sentiment data.")

    # Row 4: Portfolio
    if has_logs:
        st.markdown('<p class="section-header">Portfolio Performance</p>', unsafe_allow_html=True)
        # Filter to current ticker's best episode
        tk_logs = logs_df[logs_df["ticker"] == ticker] if "ticker" in logs_df.columns else logs_df
        if not tk_logs.empty:
            best_ep = tk_logs.groupby("episode")["portfolio_value"].last().idxmax()
            ep_logs = tk_logs[tk_logs["episode"] == best_ep]
        else:
            ep_logs = tk_logs

        c_perf1, c_perf2 = st.columns(2)
        with c_perf1:
            try:
                fig_eq = create_equity_curve(ep_logs, INITIAL_CAPITAL)
                st.plotly_chart(fig_eq, key="equity", use_container_width=True, config={"displayModeBar": False})
            except Exception:
                st.caption("Equity curve unavailable")
        with c_perf2:
            try:
                cummax = ep_logs["portfolio_value"].cummax()
                dd = (ep_logs["portfolio_value"] - cummax) / cummax * 100
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=ep_logs["step"], y=dd, mode="lines", name="Drawdown",
                    line=dict(color=COLORS["down"], width=1.5),
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
                ))
                fig_dd.update_layout(
                    template=DARK_TEMPLATE, height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(title="Drawdown (%)", gridcolor="rgba(255,255,255,0.05)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_dd, key="drawdown", use_container_width=True, config={"displayModeBar": False})
            except Exception:
                st.caption("Drawdown chart unavailable")

    # Row 5: Convergence plot
    curves_dir = Path(DATA_DIR) / "training_curves"
    # Prefer multi-seed (ablation) data: {ticker}_seed*_with_nlp_rewards.npz
    seed_files = sorted(curves_dir.glob(f"{ticker}_seed*_with_nlp_rewards.npz")) if curves_dir.exists() else []
    single_file = curves_dir / f"{ticker}_rewards.npz" if curves_dir.exists() else None
    legacy_file = curves_dir / "rewards.npz" if curves_dir.exists() else None

    has_multi = len(seed_files) >= 2
    has_single = single_file.exists() if single_file else False
    has_legacy = legacy_file.exists() if legacy_file else False

    if has_multi or has_single or has_legacy:
        st.markdown('<p class="section-header">Convergence</p>', unsafe_allow_html=True)
        try:
            if has_multi:
                # Load all seeds and stack into matrix
                seed_rewards = []
                for sf in seed_files:
                    d = np.load(sf)
                    if "rewards" in d:
                        seed_rewards.append(d["rewards"])
                if len(seed_rewards) >= 2:
                    min_len = min(len(r) for r in seed_rewards)
                    seeds_matrix = np.array([r[:min_len] for r in seed_rewards])
                    fig_conv = create_convergence_chart(
                        seeds_matrix=seeds_matrix, ticker=f"{ticker} (with NLP)")
                    st.plotly_chart(fig_conv, key="conv_multi",
                                    use_container_width=True, config={"displayModeBar": False})
            elif has_single:
                conv_data = np.load(single_file)
                fig_conv = create_convergence_chart(
                    rewards=conv_data["rewards"],
                    conv_ratio=float(conv_data.get("conv_ratio", 0.0)),
                    slope=float(conv_data.get("slope", 0.0)),
                    ticker=ticker)
                st.plotly_chart(fig_conv, key="conv_single",
                                use_container_width=True, config={"displayModeBar": False})
            elif has_legacy:
                conv_data = np.load(legacy_file)
                fig_conv = create_convergence_chart(
                    rewards=conv_data["rewards"],
                    conv_ratio=float(conv_data.get("conv_ratio", 0.0)),
                    slope=float(conv_data.get("slope", 0.0)),
                    ticker=ticker)
                st.plotly_chart(fig_conv, key="conv_legacy",
                                use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.caption("Convergence plot unavailable")

    # Row 6: Market returns heatmap (tickers × days)
    st.markdown('<p class="section-header">Market Heatmap</p>', unsafe_allow_html=True)
    returns_matrix = {}
    for t in st.session_state.watchlist[:15]:
        mkt = watchlist_data.get(t)
        if mkt is not None and len(mkt) > 1:
            rets = mkt["close"].pct_change().dropna().tail(20).values
            if len(rets) >= 3:
                returns_matrix[t] = rets
    if returns_matrix:
        try:
            fig_hm = create_heatmap(returns_matrix)
            st.plotly_chart(fig_hm, key="heatmap", use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.caption("Heatmap unavailable")

# ═══════════════════════════════════════════════════════════════════════════
# Side Panel
# ═══════════════════════════════════════════════════════════════════════════

with side_col:
    # ── Quick Stats ──
    st.markdown('<p class="section-header">Quick Stats</p>', unsafe_allow_html=True)
    if has_market:
        high_52 = market_df["high"].tail(252).max() if len(market_df) >= 252 else market_df["high"].max()
        low_52 = market_df["low"].tail(252).min() if len(market_df) >= 252 else market_df["low"].min()
        vol_avg = market_df["volume"].tail(20).mean()

        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span class="label">52-Week High</span>
                <span style="color:#ef4444;font-weight:600;">${high_52:.2f}</span></div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span class="label">52-Week Low</span>
                <span style="color:#22c55e;font-weight:600;">${low_52:.2f}</span></div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span class="label">Avg Vol (20d)</span>
                <span style="color:#38bdf8;font-weight:600;">{vol_avg/1e6:.1f}M</span></div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span class="label">Data Points</span>
                <span style="color:#e2e8f0;font-weight:600;">{len(market_df):,}</span></div>
            <div style="display:flex;justify-content:space-between;">
                <span class="label">NLP Methods</span>
                <span style="color:#f472b6;font-weight:600;">{sentiment_df['method'].nunique() if has_sentiment else 0}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Sentiment Summary ──
    if has_sentiment:
        st.markdown('<p class="section-header">Sentiment Summary</p>', unsafe_allow_html=True)
        for m in sorted(sentiment_df["method"].unique()):
            sub = sentiment_df[sentiment_df["method"] == m]
            if not sub.empty:
                avg = sub["sentiment_score"].mean()
                if avg > 0.05:
                    emoji, clr = "🟢", COLORS["up"]
                elif avg < -0.05:
                    emoji, clr = "🔴", COLORS["down"]
                else:
                    emoji, clr = "🟡", COLORS["lr"]
                st.markdown(f"{emoji} **{m.upper()}**: <span style='color:{clr};'>{avg:+.4f}</span>", unsafe_allow_html=True)

    # ── Paper Trading ──
    st.markdown('<p class="section-header">Paper Trading</p>', unsafe_allow_html=True)
    try:
        orders = db.get_trade_orders()
        if not orders.empty:
            orders_sorted = orders.sort_values("date")
            last_order = orders_sorted.iloc[-1]
            final_value = float(last_order.get("portfolio_value_after", INITIAL_CAPITAL))
            pnl = final_value - INITIAL_CAPITAL
            pnl_pct = pnl / INITIAL_CAPITAL * 100
            pnl_color = COLORS["up"] if pnl >= 0 else COLORS["down"]
            pnl_sign = "+" if pnl >= 0 else ""

            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                    <span class="label">Portfolio Value</span>
                    <span style="color:#e2e8f0;font-weight:700;">${final_value:,.2f}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                    <span class="label">P&amp;L</span>
                    <span style="color:{pnl_color};font-weight:700;">{pnl_sign}${pnl:,.2f} ({pnl_pct:+.2f}%)</span></div>
                <div style="display:flex;justify-content:space-between;">
                    <span class="label">Total Trades</span>
                    <span style="color:#e2e8f0;font-weight:600;">{len(orders_sorted)}</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.caption("Recent Orders")
            recent = orders_sorted.tail(5)[["ticker", "date", "action", "shares", "price"]]
            st.dataframe(recent, hide_index=True, use_container_width=True)
        else:
            st.caption("No paper trades yet. Train a model first.")
            if st.button("Run Paper Trading Cycle", use_container_width=True):
                with st.spinner("Running..."):
                    try:
                        from paper_trader import run_paper_trading_cycle
                        summary = run_paper_trading_cycle(db)
                        st.success(f"Done! Equity: ${summary['total_equity']:,.2f}")
                    except FileNotFoundError:
                        st.error("No trained model found. Run `python main.py` first.")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
                st.rerun()
    except Exception as exc:
        st.caption(f"Paper trading unavailable: {exc}")

    # ── Ablation Summary ──
    st.markdown('<p class="section-header">Ablation Results</p>', unsafe_allow_html=True)

    import json
    ablation_json = Path(__file__).parent.parent / "data" / "ablation_results.json"
    if ablation_json.exists():
        try:
            ablation_data = json.loads(ablation_json.read_text(encoding="utf-8"))
            l1 = ablation_data.get("layer1_ablation", {})
            pos = l1.get("nlp_positive", 0)
            neg = l1.get("nlp_negative", 0)
            neu = l1.get("nlp_neutral", 0)
            rate = l1.get("positive_rate", 0)
            mdd_imp = l1.get("nlp_mdd_improved", 0)

            # Top performer by Sharpe delta
            tickers = l1.get("tickers", {})
            top_ticker, top_delta, top_mdd_delta = None, -999, 0
            for t, s in tickers.items():
                d = s.get("sharpe_delta", 0)
                if d > top_delta:
                    top_delta, top_ticker, top_mdd_delta = d, t, s.get("mdd_delta", 0)

            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span class="label">NLP Positive (Sharpe)</span>
                    <span style="color:#22c55e;font-weight:600;">{pos}/28 ({rate}%)</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span class="label">NLP Neutral</span>
                    <span style="color:#fbbf24;font-weight:600;">{neu}/28 ({round(neu/28*100,1)}%)</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span class="label">NLP Negative</span>
                    <span style="color:#ef4444;font-weight:600;">{neg}/28 ({round(neg/28*100,1)}%)</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span class="label">MDD Improved</span>
                    <span style="color:#38bdf8;font-weight:600;">{mdd_imp}/28 ({round(mdd_imp/28*100,1)}%)</span></div>
                <div style="display:flex;justify-content:space-between;">
                    <span class="label">Top Performer</span>
                    <span style="color:#38bdf8;font-weight:600;">{top_ticker or 'N/A'} Δ{top_delta:+.3f}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Source: data/ablation_results.json  ·  {ablation_data.get('timestamp','')[:16]}")
        except Exception:
            st.caption("Could not read ablation results.")
    elif has_logs:
        st.info(f"{len(logs_df):,} training log entries available.")
    else:
        st.caption("No ablation data yet. Run: `python main.py --ablate`")

# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════

last_mkt_str, last_sent_str = ("?", "?")
try:
    last_mkt_str, last_sent_str = get_data_freshness(db)
except Exception:
    pass

st.markdown(f"""
<div style="text-align:center;padding:20px 0 4px 0;border-top:1px solid rgba(56,189,248,0.06);margin-top:24px;">
    <span style="color:#475569;font-size:10px;letter-spacing:0.03em;">
        QuantumTrade v2.0 · NLP-Driven RL Trading Platform · Fintech Group 2026 · Data through {last_mkt_str}
    </span>
</div>
""", unsafe_allow_html=True)
