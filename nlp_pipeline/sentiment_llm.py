"""LLM-based sentiment analysis using Volcano Engine (Doubao) via OpenAI-compatible API.

Few-shot prompt with scoring rubric for consistent, calibrated financial sentiment.
"""

import json
import logging

import numpy as np
import pandas as pd
from openai import OpenAI

from config import (
    VOLCANO_API_KEY, VOLCANO_BASE_URL, VOLCANO_MODEL_ID,
    LLM_BATCH_SIZE, LLM_MAX_ARTICLES_PER_TICKER, LLM_TEMPERATURE, LLM_ENABLED,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial sentiment analysis expert. Analyze the given news headline and content. Return a JSON object with exactly these fields:
- "score": a float from -1.0 (very bearish) to +1.0 (very bullish), 0.0 is neutral
- "label": one of "positive", "negative", or "neutral"
- "confidence": a float from 0.0 to 1.0 indicating your certainty
- "reasoning": a brief one-sentence explanation of your judgment

SCORING RUBRIC (numerical boundaries):
| Score Range | Label    | When to Use                                              |
|-------------|----------|----------------------------------------------------------|
| +0.8 ~ +1.0 | positive | Earnings beat >20%, major product launch, FDA approval   |
| +0.3 ~ +0.7 | positive | Analyst upgrade, partnership, moderate growth            |
| -0.2 ~ +0.2 | neutral  | Routine news, minor management change, rebalancing       |
| -0.7 ~ -0.3 | negative | Earnings miss, supply chain disruption, downgrade        |
| -1.0 ~ -0.8 | negative | Fraud/scandal, bankruptcy risk, major recall, fine >$100M|

FEW-SHOT EXAMPLES:

Example 1:
Title: "Apple Reports Record Q3 Revenue of $89.5B, Beating Estimates by 12%"
Analysis: {"score": 0.85, "label": "positive", "confidence": 0.95, "reasoning": "Strong earnings beat of 12% with record revenue signals robust business performance and positive market momentum."}

Example 2:
Title: "Microsoft Announces New Azure AI Partnership with OpenAI"
Analysis: {"score": 0.65, "label": "positive", "confidence": 0.80, "reasoning": "Strategic AI partnership expands cloud revenue potential though financial impact is not yet quantified."}

Example 3:
Title: "Tesla Recalls 2 Million Vehicles Over Autopilot Safety Concerns"
Analysis: {"score": -0.75, "label": "negative", "confidence": 0.90, "reasoning": "Massive recall affecting nearly all vehicles raises significant safety and regulatory concerns with potential brand damage."}

Example 4:
Title: "Google CFO Announces Retirement After 8 Years, Successor Named"
Analysis: {"score": -0.05, "label": "neutral", "confidence": 0.70, "reasoning": "Planned CFO transition is routine corporate governance; successor already named reduces uncertainty."}

Example 5:
Title: "Amazon Warehouse Workers Vote to Unionize in Historic First"
Analysis: {"score": -0.30, "label": "negative", "confidence": 0.65, "reasoning": "Unionization could increase labor costs but impact limited to single location; broader implications uncertain."

ANTI-MANIPULATION RULES:
- Do NOT give high scores to vague hype ("could revolutionize", "game-changing") without quantified evidence
- Consider source credibility implicitly (earnings reports > analyst speculation > social media rumors)
- Distinguish between company-specific news and macro/industry news
- When in doubt, err toward neutral (0.0) with lower confidence

Respond ONLY with valid JSON, no markdown, no extra text."""


def _call_llm(articles: list[dict]) -> list[dict]:
    """Send batch of articles to LLM, return sentiment results."""
    if not LLM_ENABLED:
        return [_neutral_fallback(a) for a in articles]

    client = OpenAI(api_key=VOLCANO_API_KEY, base_url=VOLCANO_BASE_URL)
    results = []

    for article in articles:
        user_content = f"Title: {article.get('title', '')}\nContent: {article.get('content', '')[:500]}"
        try:
            resp = client.chat.completions.create(
                model=VOLCANO_MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=150,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(text)
            results.append({
                "sentiment_score": np.clip(float(result.get("score", 0.0)), -1.0, 1.0),
                "label": result.get("label", "neutral"),
                "confidence": np.clip(float(result.get("confidence", 0.5)), 0.0, 1.0),
                "reasoning": result.get("reasoning", ""),
            })
        except Exception:
            logger.debug("LLM analysis failed for: %.60s...", article.get("title", ""))
            results.append(_neutral_fallback(article))

    return results


def _neutral_fallback(article: dict) -> dict:
    return {"sentiment_score": 0.0, "label": "neutral", "confidence": 0.0, "reasoning": "Analysis unavailable"}


def llm_sentiment_batch(news_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze sentiment using LLM (Volcano Engine Doubao)."""
    if news_df.empty:
        return pd.DataFrame(columns=["date", "sentiment_score", "confidence", "label", "method", "reasoning"])

    if not LLM_ENABLED:
        logger.info("LLM disabled (VOLCANO_API_KEY or VOLCANO_MODEL_ID not set). Skipping LLM sentiment.")
        return pd.DataFrame(columns=["date", "sentiment_score", "confidence", "label", "method", "reasoning"])

    articles = news_df[["title", "content", "published_at"]].to_dict("records")

    # Limit LLM calls per ticker to avoid hours-long runs
    if len(articles) > LLM_MAX_ARTICLES_PER_TICKER:
        logger.info("Limiting LLM to %d/%d articles per ticker", LLM_MAX_ARTICLES_PER_TICKER, len(articles))
        articles = articles[:LLM_MAX_ARTICLES_PER_TICKER]

    all_results = []

    for i in range(0, len(articles), LLM_BATCH_SIZE):
        batch = articles[i:i + LLM_BATCH_SIZE]
        logger.info("LLM analyzing articles %d-%d/%d...", i + 1, min(i + LLM_BATCH_SIZE, len(articles)), len(articles))
        batch_results = _call_llm(batch)
        all_results.extend(batch_results)

    records = []
    for idx, r in enumerate(all_results):
        article = articles[idx] if idx < len(articles) else {}
        pub_date = article.get("published_at", "")
        try:
            dt = pd.to_datetime(pub_date).date()
        except Exception:
            dt = None
        records.append({
            "date": dt,
            "sentiment_score": np.clip(r["sentiment_score"], -1.0, 1.0),
            "confidence": np.clip(r["confidence"], 0.0, 1.0),
            "label": r["label"],
            "method": "llm",
            "reasoning": r["reasoning"],
        })

    df = pd.DataFrame(records)
    logger.info("LLM sentiment complete: %d articles analyzed", len(df))
    return df
