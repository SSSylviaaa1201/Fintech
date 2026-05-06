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
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Financial
    "JPM", "BAC", "V", "MA", "GS",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Consumer
    "KO", "PEP", "WMT", "COST", "NKE",
    # Energy / Industrial
    "XOM", "CVX", "CAT", "BA",
    # Communication / Utilities
    "DIS", "NFLX", "NEE",
]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
NEWS_LOOKBACK_DAYS = 30
COLLECTION_INTERVAL_MINUTES = 60

# --- Scheduler ---
SCHEDULER_ENABLED = True

# --- Data Sources Priority ---
DATA_SOURCE_PRIORITY = ["yfinance", "alpha_vantage"]
NEWS_SOURCE_PRIORITY = ["newsapi", "rss"]

# --- RSS Configuration ---
RSS_ENABLED = True
RSS_MAX_PER_SOURCE = 20
RSS_SOURCES = ["yahoo_finance", "google_news", "seeking_alpha", "marketwatch"]
RSS_REQUEST_DELAY = 1

# --- NLP Pipeline ---
FINBERT_MODEL = "ProsusAI/finbert"
MAX_SEQ_LENGTH = 512
SENTIMENT_METHODS = ["vader", "logistic_regression", "finbert", "llm"]

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
LOOKBACK_WINDOW = 200
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_PCT = 0.001
EPISODES = 200
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.97
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 20_000
TARGET_UPDATE_FREQ = 10

# Walk-forward validation
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2  # remaining 0.2 is test

# --- SHAP Explainability ---
SHAP_BACKGROUND_SAMPLES = 100

# --- Dashboard ---
DASHBOARD_PORT = 8501
REFRESH_INTERVAL_SECONDS = 5

# --- API Keys (from .env) ---
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
