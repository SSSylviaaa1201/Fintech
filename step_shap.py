"""Post-hoc SHAP analysis: explain DQN decisions for top/bottom NLP stocks.

Usage:
  python step_shap.py --ticker AAPL          # single stock
  python step_shap.py --all                   # all 28 stocks
  python step_shap.py --top 5  --output-dir data/shap_reports
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from data_storage.db_manager import DatabaseManager
from main import build_rl_features, compute_indicators, _process_sentiment_signal, _align_market_to_news
from rl_engine.dqn import DQNAgent
from rl_engine.env import FinancialTradingEnv
from rl_engine.train import train_dqn, walk_forward_split
from rl_engine.explainer import TradingExplainer, STATE_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("shap_analysis")


def train_agent_for_ticker(db: DatabaseManager, ticker: str) -> tuple[DQNAgent, pd.DataFrame]:
    """Train a DQN agent for a single ticker and return (agent, test_df)."""
    logger.info("Training DQN for %s...", ticker)

    market_df = db.get_market_data(ticker)
    df_base = compute_indicators(market_df).reset_index(drop=True)
    sent_df = db.get_sentiment(ticker)
    df_base = _align_market_to_news(df_base, sent_df)
    df_base["date"] = pd.to_datetime(df_base["date"]).dt.date

    if not sent_df.empty:
        signal = _process_sentiment_signal(sent_df)
        if not signal.empty:
            signal_df = signal.reset_index()
            signal_df.columns = ["date", "sentiment_score"]
            signal_df["date"] = pd.to_datetime(signal_df["date"]).dt.date
            df_with = df_base.merge(signal_df, on="date", how="left")
            df_with["sentiment_score"] = df_with["sentiment_score"].fillna(0.0)
        else:
            df_with = df_base.copy()
            df_with["sentiment_score"] = 0.0
    else:
        df_with = df_base.copy()
        df_with["sentiment_score"] = 0.0
    df_with = df_with.ffill().fillna(0.0)

    # Add derived sentiment features
    df_with["sentiment_ma5"] = df_with["sentiment_score"].rolling(window=5, min_periods=1).mean()
    df_with["sentiment_ma20"] = df_with["sentiment_score"].rolling(window=20, min_periods=1).mean()
    df_with["sentiment_trend"] = df_with["sentiment_ma5"] - df_with["sentiment_ma20"]
    df_with["sentiment_vol"] = df_with["sentiment_score"].rolling(window=10, min_periods=1).std()

    train_df, val_df, test_df = walk_forward_split(df_with)
    agent = DQNAgent()
    agent = train_dqn(train_df, val_df, episodes=config.EPISODES, ticker=ticker)
    return agent, test_df


def run_shap_analysis(agent: DQNAgent, df: pd.DataFrame, ticker: str,
                      n_background: int = 100, n_samples: int = 50) -> dict:
    """Run SHAP analysis on a trained agent and return feature importance."""
    logger.info("Running SHAP for %s...", ticker)

    env = FinancialTradingEnv(df)
    explainer = TradingExplainer(agent)
    explainer.fit(env, n_background=n_background)

    global_importance = explainer.global_feature_importance(df, n_samples=n_samples)

    # Also explain a few specific states
    state_explanations = []
    state, _ = env.reset()
    for step in range(min(5, len(df) - 5)):
        try:
            exp = explainer.explain_state(state)
            state_explanations.append({
                "step": step,
                "action": exp["action_name"],
                "top_drivers": exp["feature_importance"][:5],
                "explanation": exp["explanation_text"],
            })
        except Exception:
            break
        action = env.action_space.sample()
        state, _, term, _, _ = env.step(action)
        if term:
            break

    return {
        "ticker": ticker,
        "global_importance": global_importance.to_dict() if not global_importance.empty else {},
        "state_explanations": state_explanations,
    }


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for DQN trading agent")
    parser.add_argument("--ticker", type=str, help="Single ticker to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all 28 tickers")
    parser.add_argument("--top", type=int, default=0, help="Analyze top N Sharpe-delta stocks from ablation")
    parser.add_argument("--episodes", type=int, default=config.EPISODES)
    parser.add_argument("--output-dir", type=str, default="data/shap_reports")
    args = parser.parse_args()

    config.EPISODES = args.episodes
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = DatabaseManager()

    if args.ticker:
        tickers = [args.ticker]
    elif args.all:
        tickers = list(config.TICKERS)
    elif args.top > 0:
        # Read ablation results to find top Sharpe-delta stocks
        ablation_path = Path("data/ablation_results.json")
        if not ablation_path.exists():
            logger.error("ablation_results.json not found. Run ablation first or use --ticker/--all.")
            sys.exit(1)
        with open(ablation_path) as f:
            ablation = json.load(f)
        ticker_deltas = []
        for t, r in ablation["layer1_ablation"]["tickers"].items():
            ticker_deltas.append((t, r["sharpe_delta"]))
        ticker_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
        tickers = [t for t, _ in ticker_deltas[:args.top]]
        logger.info("Top %d by |Sharpe delta|: %s", args.top, tickers)
    else:
        parser.print_help()
        return

    all_results = {}
    for ticker in tickers:
        try:
            agent, test_df = train_agent_for_ticker(db, ticker)
            result = run_shap_analysis(agent, test_df, ticker)
            all_results[ticker] = result

            # Print summary
            imp = result["global_importance"]
            if imp:
                top3 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info("%s SHAP top features: %s", ticker,
                            ", ".join([f"{f}={v:.3f}" for f, v in top3]))
        except Exception as e:
            logger.error("%s SHAP failed: %s", ticker, e)

    # Save results
    out_path = out_dir / "shap_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("SHAP results saved to %s", out_path)


if __name__ == "__main__":
    main()
