"""Aggregate daily sentiment scores per ticker from up to 4 methods with agreement metrics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_col: str = "published_at",
    ticker_col: str = "ticker",
    method_col: str = "method",
    score_col: str = "sentiment_score",
    confidence_col: str = "confidence",
    label_col: str = "label",
) -> pd.DataFrame:
    """Aggregate article-level sentiment to daily ticker-level per method."""
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", "method", "sentiment_score", "confidence", "label"])

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format="mixed", utc=True)
    df = df.dropna(subset=[date_col])
    df["date"] = df[date_col].dt.date

    def agg_group(group):
        total_conf = group[confidence_col].sum()
        if total_conf == 0:
            avg_score = group[score_col].mean()
        else:
            avg_score = (group[score_col] * group[confidence_col]).sum() / total_conf
        avg_conf = group[confidence_col].mean()
        mode_label = group[label_col].mode()
        majority_label = mode_label.iloc[0] if not mode_label.empty else "neutral"
        reasoning = group.get("reasoning")
        sample_reasoning = reasoning.iloc[0] if reasoning is not None and not reasoning.empty else ""
        result = {
            "sentiment_score": avg_score,
            "confidence": avg_conf,
            "label": majority_label,
        }
        if sample_reasoning:
            result["reasoning"] = sample_reasoning
        return pd.Series(result)

    result = df.groupby([ticker_col, "date", method_col], as_index=False).apply(agg_group)
    return result.reset_index(drop=True)


def compute_agreement_metrics(merged: pd.DataFrame) -> dict:
    """
    Compute inter-method agreement metrics.
    Returns dict with:
      - 'kappa': Fleiss' Kappa per date (mean across all dates)
      - 'correlation': pairwise Pearson r matrix between methods
      - 'agreement_level': 'high' / 'moderate' / 'low'
    """
    methods = merged["method"].unique()
    if len(methods) < 2:
        return {"kappa": None, "correlation": None, "agreement_level": "insufficient_data"}

    # Pivot: dates x methods with sentiment_score
    pivot = merged.pivot_table(
        index=["ticker", "date"], columns="method",
        values="sentiment_score", aggfunc="mean",
    ).dropna()

    if pivot.empty or pivot.shape[1] < 2:
        return {"kappa": None, "correlation": None, "agreement_level": "insufficient_data"}

    # Pairwise Pearson correlations
    corr_matrix = pivot.corr()

    # Fleiss' Kappa approximation: convert scores to 3-category labels
    def to_category(x):
        if x > 0.05:
            return 2  # positive
        elif x < -0.05:
            return 0  # negative
        return 1      # neutral

    cat_pivot = pivot.map(to_category)
    kappa_values = []
    for date_idx in cat_pivot.index:
        row = cat_pivot.loc[date_idx].values
        if len(row) < 2:
            continue
        # Build contingency: each rater (method) × category counts
        n_methods = len(row)
        n_categories = 3
        counts = np.zeros((n_methods, n_categories))
        for m, cat in enumerate(row):
            counts[m, int(cat)] = 1  # one rating per method per day
        try:
            k = _fleiss_kappa(counts)
            kappa_values.append(k)
        except Exception:
            continue

    avg_kappa = np.mean(kappa_values) if kappa_values else None

    if avg_kappa is not None:
        if avg_kappa > 0.6:
            level = "high"
        elif avg_kappa > 0.3:
            level = "moderate"
        else:
            level = "low"
    else:
        level = "insufficient_data"

    return {
        "kappa": round(avg_kappa, 4) if avg_kappa is not None else None,
        "correlation": corr_matrix,
        "agreement_level": level,
    }


def _fleiss_kappa(counts: np.ndarray) -> float:
    """Compute Fleiss' Kappa for inter-rater agreement.

    counts: shape (n_raters, n_categories) where each row is one rater's category assignments.
    """
    n_raters, n_categories = counts.shape
    n_items = n_raters  # Each rater is one "item" (method)
    n_per_item = np.sum(counts, axis=1)

    if np.any(n_per_item == 0):
        return 0.0

    # Proportion of all assignments to each category
    p_j = np.sum(counts, axis=0) / np.sum(counts)

    # Agreement per item
    P_i = (np.sum(counts**2, axis=1) - n_per_item) / (n_per_item * (n_per_item - 1) + 1e-10)
    P_bar = np.mean(P_i)

    # Expected agreement
    P_e = np.sum(p_j**2)

    if abs(1 - P_e) < 1e-10:
        return 1.0

    return (P_bar - P_e) / (1 - P_e)


def get_merged_sentiment(
    df_vader: pd.DataFrame,
    df_lr: pd.DataFrame,
    df_finbert: pd.DataFrame,
    df_llm: pd.DataFrame | None = None,
    compute_kappa: bool = True,
) -> dict:
    """Merge all sentiment DataFrames and optionally compute agreement metrics.

    Returns dict with:
      - 'aggregated': combined daily DataFrame
      - 'agreement': agreement metrics dict (or None if skipped)
    """
    dfs = [df_vader, df_lr, df_finbert]
    if df_llm is not None and not df_llm.empty:
        dfs.append(df_llm)

    combined = pd.concat(dfs, ignore_index=True)
    aggregated = aggregate_daily_sentiment(combined)

    result = {"aggregated": aggregated}

    if compute_kappa:
        try:
            agreement = compute_agreement_metrics(aggregated)
            result["agreement"] = agreement
            if agreement["kappa"] is not None:
                logger.info("Fleiss' Kappa: %.4f (%s agreement)", agreement["kappa"], agreement["agreement_level"])
        except Exception as e:
            logger.warning("Failed to compute agreement metrics: %s", e)
            result["agreement"] = None

    return result


def compute_consensus_score(aggregated: pd.DataFrame, agreement: dict | None = None) -> pd.DataFrame:
    """
    Compute weighted consensus score across methods.
    If high agreement: use simple mean.
    If low agreement: weight FinBERT more heavily.
    """
    methods = aggregated["method"].unique()
    if len(methods) <= 1:
        return aggregated.copy()

    pivot = aggregated.pivot_table(
        index=["ticker", "date"], columns="method",
        values="sentiment_score", aggfunc="mean",
    )

    kappa = agreement["kappa"] if agreement else None
    if kappa and kappa > 0.6:
        # High agreement: simple mean
        weights = {m: 1.0 for m in pivot.columns}
    else:
        # Low/moderate: weight sophisticated methods higher
        weights = {}
        for m in pivot.columns:
            if m == "finbert":
                weights[m] = 2.0
            elif m == "lr":
                weights[m] = 1.5
            else:
                weights[m] = 1.0

    total_w = sum(weights.get(m, 1.0) for m in pivot.columns)
    consensus = sum(pivot[m] * weights.get(m, 1.0) for m in pivot.columns) / total_w

    result = pd.DataFrame({"consensus_score": consensus}).reset_index()
    result["agreement_level"] = agreement.get("agreement_level", "unknown") if agreement else "unknown"
    return result


def compute_inter_method_agreement(
    df_vader: pd.DataFrame,
    df_lr: pd.DataFrame,
    df_finbert: pd.DataFrame,
) -> dict:
    """Compute pairwise directional agreement between all sentiment methods.

    This is an inter-method consistency measure, NOT an accuracy evaluation.
    None of the methods is treated as ground truth. Instead, we compute:
      - Pairwise F1 (directional agreement) for all method pairs
      - Overall agreement rate (including neutrals)

    Returns a symmetric agreement matrix + per-pair metrics.
    """
    dfs = {"vader": df_vader, "lr": df_lr, "finbert": df_finbert}

    # Merge all methods on date
    merged = None
    for method, df in dfs.items():
        if df.empty:
            continue
        sub = df[["date", "sentiment_score"]].copy()
        sub["date"] = pd.to_datetime(sub["date"]).dt.date
        sub = sub.groupby("date")["sentiment_score"].mean().reset_index()
        sub.columns = ["date", f"score_{method}"]
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on="date", how="inner")

    if merged is None or len(merged) < 10:
        return {"error": "insufficient_data", "n_samples": len(merged) if merged is not None else 0}

    def to_binary(series: pd.Series, threshold: float = 0.05) -> pd.Series:
        """Convert scores to: 2=positive, 0=negative, 1=neutral."""
        result = pd.Series(1, index=series.index)
        result[series > threshold] = 2
        result[series < -threshold] = 0
        return result

    def compute_pairwise_f1(yt: pd.Series, yp: pd.Series) -> dict:
        """Compute macro F1 between two binary series (excluding neutral)."""
        mask = (yt != 1) & (yp != 1)
        if mask.sum() < 5:
            return {"error": "too_few_directional_samples", "n": int(mask.sum())}
        yt_m, yp_m = yt[mask], yp[mask]
        tp_pos = int(((yt_m == 2) & (yp_m == 2)).sum())
        fp_pos = int(((yt_m != 2) & (yp_m == 2)).sum())
        fn_pos = int(((yt_m == 2) & (yp_m != 2)).sum())
        prec_pos = tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0.0
        rec_pos = tp_pos / (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else 0.0
        f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos) if (prec_pos + rec_pos) > 0 else 0.0
        tp_neg = int(((yt_m == 0) & (yp_m == 0)).sum())
        fp_neg = int(((yt_m != 0) & (yp_m == 0)).sum())
        fn_neg = int(((yt_m == 0) & (yp_m != 0)).sum())
        prec_neg = tp_neg / (tp_neg + fp_neg) if (tp_neg + fp_neg) > 0 else 0.0
        rec_neg = tp_neg / (tp_neg + fn_neg) if (tp_neg + fn_neg) > 0 else 0.0
        f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0
        return {
            "f1_macro": round((f1_pos + f1_neg) / 2, 4),
            "f1_positive": round(f1_pos, 4),
            "f1_negative": round(f1_neg, 4),
            "n_samples": int(mask.sum()),
        }

    # Compute all pairwise F1
    methods = ["vader", "lr", "finbert"]
    pairwise = {}
    matrix = {}
    for m1 in methods:
        row = {}
        for m2 in methods:
            if m1 == m2:
                row[m2] = {"f1_macro": 1.0}
            elif f"{m2}_{m1}" in pairwise:
                row[m2] = pairwise[f"{m2}_{m1}"]
            else:
                col1 = f"score_{m1}"
                col2 = f"score_{m2}"
                if col1 not in merged.columns or col2 not in merged.columns:
                    row[m2] = {"error": "no_data"}
                else:
                    y1 = to_binary(merged[col1])
                    y2 = to_binary(merged[col2])
                    result = compute_pairwise_f1(y1, y2)
                    pairwise[f"{m1}_{m2}"] = result
                    row[m2] = result
        matrix[m1] = row

    # Overall agreement rates (including neutrals)
    total = len(merged)
    agreement_rates = {}
    for m1 in methods:
        for m2 in methods:
            if m1 >= m2:
                continue
            col1, col2 = f"score_{m1}", f"score_{m2}"
            if col1 in merged.columns and col2 in merged.columns:
                y1 = to_binary(merged[col1])
                y2 = to_binary(merged[col2])
                ag = int((y1 == y2).sum()) / total if total > 0 else 0.0
                agreement_rates[f"{m1}_vs_{m2}"] = round(ag, 4)

    # Average pairwise F1 (excluding self-comparisons)
    f1_values = [v["f1_macro"] for k, v in pairwise.items() if "error" not in v]
    avg_pairwise_f1 = round(float(np.mean(f1_values)), 4) if f1_values else None

    return {
        "pairwise_f1_matrix": matrix,
        "pairwise_agreement_rates": agreement_rates,
        "average_pairwise_f1": avg_pairwise_f1,
        "n_dates": len(merged),
        "note": "All metrics measure inter-method directional agreement, NOT accuracy. "
                "No method is treated as ground truth.",
    }


def compute_news_quality(df: pd.DataFrame, ticker_col: str = "ticker",
                         content_col: str = "cleaned_text",
                         title_col: str = "title") -> dict:
    """Detect template/duplicate news content and compute quality metrics per ticker.

    Returns dict with per-ticker:
      - uniqueness_ratio: fraction of articles with no near-duplicate (Jaccard > 0.7)
      - template_ratio: fraction of articles matching common financial news templates
      - short_content_ratio: fraction with <50 chars of content
      - n_articles: total article count
    """
    if df.empty:
        return {"per_ticker": {}, "global": {"uniqueness_ratio": 0.0, "n_articles": 0}}

    # Common template patterns in financial news
    TEMPLATE_PATTERNS = [
        r'\{TICKER\}\s+(reports|announces|posts|beats|misses)',
        r'(shares|stock)\s+(of\s+)?\{TICKER\}\s+(rose|fell|dipped|surged|plunged)',
        r'(analysts|wall\s+street)\s+(upgrade|downgrade|rate)',
        r'\{TICKER\}\s+(to\s+)?(buy|sell|hold)',
    ]

    import re as _re
    per_ticker = {}

    for ticker, group in df.groupby(ticker_col):
        n = len(group)
        if n < 2:
            per_ticker[ticker] = {
                "uniqueness_ratio": 1.0, "template_ratio": 0.0,
                "short_content_ratio": 0.0, "n_articles": n,
            }
            continue

        texts = group[content_col].fillna("").tolist()
        titles = group[title_col].fillna("").tolist()

        # Near-duplicate detection via Jaccard on word bigrams
        def _bigrams(text: str) -> set:
            words = str(text).lower().split()
            return {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}

        bigram_sets = [_bigrams(t) for t in texts]
        dup_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = bigram_sets[i], bigram_sets[j]
                union = len(bi | bj)
                if union > 0:
                    jaccard = len(bi & bj) / union
                    if jaccard > 0.7:
                        dup_count += 1
                        break  # article i has at least one near-duplicate

        uniqueness = 1.0 - dup_count / n if n > 0 else 1.0

        # Template detection
        template_count = 0
        for t in titles + texts:
            t_lower = str(t).lower()
            for pattern in TEMPLATE_PATTERNS:
                if _re.search(pattern.replace('{TICKER}', ticker.lower()), t_lower):
                    template_count += 1
                    break

        template_ratio = min(template_count / (n * 2), 1.0)  # cap at 1.0

        # Short content ratio
        short_count = sum(1 for t in texts if len(str(t)) < 50)

        per_ticker[ticker] = {
            "uniqueness_ratio": round(uniqueness, 3),
            "template_ratio": round(template_ratio, 3),
            "short_content_ratio": round(short_count / n, 3) if n > 0 else 0.0,
            "n_articles": n,
        }

    # Global summary
    total_articles = sum(p["n_articles"] for p in per_ticker.values())
    avg_uniqueness = (sum(p["uniqueness_ratio"] * p["n_articles"] for p in per_ticker.values())
                      / max(total_articles, 1))
    avg_template = (sum(p["template_ratio"] * p["n_articles"] for p in per_ticker.values())
                    / max(total_articles, 1))
    low_quality_tickers = [t for t, p in per_ticker.items()
                           if p["uniqueness_ratio"] < 0.5 and p["n_articles"] >= 5]

    return {
        "per_ticker": per_ticker,
        "global": {
            "uniqueness_ratio": round(avg_uniqueness, 3),
            "template_ratio": round(avg_template, 3),
            "n_articles": total_articles,
            "low_quality_tickers": low_quality_tickers,
            "quality_warning": (
                f"Low news quality for {len(low_quality_tickers)} tickers "
                f"(uniqueness < 50%): {', '.join(low_quality_tickers[:5])}"
                if low_quality_tickers else "News quality acceptable"
            ),
        },
    }
