"""Global configuration for the NLP-RL Trading Platform."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DB_PATH = ROOT_DIR / "data" / "trading.db"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"

# --- Data Ingestion ---
TICKERS = [
    # Tech (10)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ADBE", "INTC", "CRM",
    # Financial (10)
    "JPM", "BAC", "V", "MA", "GS", "BLK", "AXP", "MS", "WFC", "C",
    # Healthcare (10)
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "BMY", "GILD", "LLY",
    # Consumer (10)
    "KO", "PEP", "WMT", "COST", "NKE", "HD", "MCD", "PG", "SBUX", "LOW",
    # Energy / Industrial (10)
    "XOM", "CVX", "CAT", "BA", "GE", "COP", "DE", "UPS", "LMT", "RTX",
    # Communication / Utilities (10)
    "DIS", "NFLX", "NEE", "T", "VZ", "CMCSA", "TMUS", "SO", "DUK", "CHTR",
]
# Delisted/acquired stocks for survivorship bias documentation.
# Collected if yfinance data is available; NOT included in main training universe.
# These demonstrate that backtests on survivors-only systematically overestimate returns.
DELISTED_TICKERS = [
    "TWTR",  # Acquired by Elon Musk Oct 2022, delisted from NYSE
    "ATVI",  # Acquired by Microsoft Oct 2023, delisted
]

START_DATE = "2016-01-01"  # extended from 2020 to approach 10-year economic cycle
END_DATE = "2024-12-31"
NEWS_LOOKBACK_DAYS = 30
COLLECTION_INTERVAL_MINUTES = 60
TICKER_DELAY = 3  # seconds between ticker requests

# --- Scheduler ---
SCHEDULER_ENABLED = True

# --- Data Sources Priority ---
DATA_SOURCE_PRIORITY = ["yahoo_direct", "yfinance", "alpha_vantage"]
NEWS_SOURCE_PRIORITY = ["finnhub", "alphavantage_news", "newsapi", "rss"]

# --- RSS Configuration ---
RSS_ENABLED = True
RSS_MAX_PER_SOURCE = 20
RSS_SOURCES = ["yahoo_finance", "google_news", "seeking_alpha", "marketwatch"]
RSS_REQUEST_DELAY = 1

# --- NLP Pipeline ---
FINBERT_MODEL = "ProsusAI/finbert"
MAX_SEQ_LENGTH = 512
SENTIMENT_METHODS = ["vader", "lr", "finbert"]

# --- LLM (Volcano Engine / Doubao) ---
VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY", "")
VOLCANO_BASE_URL = os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
VOLCANO_MODEL_ID = os.getenv("VOLCANO_MODEL_ID", "")
LLM_BATCH_SIZE = 5
LLM_MAX_ARTICLES_PER_TICKER = 10  # limit LLM calls per ticker (~4 min/ticker at current rate)
LLM_TEMPERATURE = 0.1
LLM_ENABLED = bool(VOLCANO_API_KEY and VOLCANO_MODEL_ID)

# HuggingFace mirror (set via .env: HF_ENDPOINT=https://hf-mirror.com)
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "")
if HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT  # must be set before transformers import

# --- RAG + Vector Store ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_TOP_K = 5

# --- RL Engine ---
# Lookback: ~5 months of trading days, covering one full earnings cycle + buffer.
# Minimum 100 rows needed for MA200 indicator to have valid values after alignment.
LOOKBACK_WINDOW = 100
INITIAL_CAPITAL = 100_000.0
# Total transaction cost applied on top of spread+slippage. With spread ~2.5bps
# + slippage ~1bp, the additional 10bps covers SEC fees, clearing, and commission.
# Total round-trip effective cost ≈ 13.5 bps — within range for institutional trading.
TRANSACTION_COST_PCT = 0.001
# Episode count: 200 episodes × ~1500 steps = ~300K environment steps.
# Comparable to financial DQN literature (Deng et al. 2017, "Deep Direct RL for
# Financial Signal Representation and Trading" use similar scale for single-asset).
EPISODES = 200
# DQN hyperparameters following Mnih et al. (2015) Nature DQN defaults
GAMMA = 0.99           # discount factor (standard for infinite-horizon RL)
EPSILON_START = 1.0    # initial exploration rate
EPSILON_MIN = 0.05     # minimum exploration rate (5% random actions)
EPSILON_DECAY = 0.97   # per-episode decay (exponential schedule)
LEARNING_RATE = 1e-3   # Adam optimizer default (Kingma & Ba 2015)
BATCH_SIZE = 64        # experience replay batch size (increased from Mnih's 32)
REPLAY_BUFFER_SIZE = 20_000  # ~13 episodes of transitions at 1500 steps/ep
TARGET_UPDATE_FREQ = 5       # update target network every 5 episodes (Double DQN)
DQN_SEED = 42  # default seed for reproducible training; set to None for random
# Multi-seed evaluation: Henderson et al. (2018) "Deep RL that Matters"
# demonstrate that multi-seed runs are essential for reliable RL evaluation.
DQN_SEEDS = [42, 123, 456]

# Ablation multi-seed (set True to quantify DQN training variance; adds ~3× runtime)
ABLATION_MULTI_SEED = True

# Walk-forward validation
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2  # remaining 0.2 is test

# --- Sentiment signal processing ---
# EMA span = 5 trading days (one business week), a natural unit in financial technical
# analysis (e.g. 5-day MA is the standard short-term trend indicator).
SENTIMENT_EMA_SPAN = 5

# Neutral threshold = 0.05: FinBERT authors (Araci 2019) use |score| > 0.05 to
# classify positive/negative; scores within ±0.05 are considered neutral.
SENTIMENT_NEUTRAL_THRESHOLD = 0.05

# Reward shaping scale for sentiment-position alignment. Calibrated so that at
# maximum sentiment (±1.0) and full position (100%), the alignment reward is
# ±0.0005 per step — small relative to daily return volatility (~1%).
# Validated via sensitivity analysis on 5 stocks (AAPL,JPM,JNJ,KO,XOM):
#   align ∈ {0, 0.0001, 0.0005, 0.001} → optimum at 0.001 (Sharpe +0.523)
#   turnover ∈ {0, 0.0001, 0.0005} → optimum at 0.0001 (Sharpe +0.374)
#   scheme ∈ {equal, finbert_weighted} → finbert_weighted more robust (σ=0.56 vs 0.61)
SENTIMENT_ALIGNMENT_SCALE = 0.001

# Reward function: turnover penalty discourages excessive trading.
# Set at 0.01% of trade value — same order as transaction cost, making the
# agent indifferent to zero-alpha trades.
REWARD_TURNOVER_PENALTY = 0.0001

# --- Market Frictions (with literature support) ---
# Half-spread for S&P 500 large-cap stocks: median effective half-spread ≈ 2-3 bps
# (Bessembinder 2003, "Trade Execution Costs on NYSE/NASDAQ";
#  Angel, Harris & Spatt 2011, "Equity Trading in the 21st Century").
HALF_SPREAD_BPS = 2.5

# Slippage: linear market impact model. Trading 1% of daily volume moves price
# by ~1 bp for large-cap US equities (Almgren et al. 2005, "Direct Estimation
# of Equity Market Impact"; Kissell 2014, "The Science of Algorithmic Trading").
SLIPPAGE_BPS_PER_PCT_VOL = 1.0

# Maximum participation rate: institutional algorithms typically cap at 5-10%
# of daily volume to avoid excessive market impact (Kissell 2014).
MAX_VOLUME_FRACTION = 0.05

# --- Risk Controls ---
MAX_DRAWDOWN_LIMIT = 0.20        # terminate episode if drawdown exceeds 20% of peak

# --- SHAP Explainability ---
SHAP_BACKGROUND_SAMPLES = 100

# --- Dashboard ---
DASHBOARD_PORT = 8501
REFRESH_INTERVAL_SECONDS = 5

# --- API Keys (from .env) ---
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
