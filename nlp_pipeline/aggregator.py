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


def compute_f1_against_finbert(
    df_vader: pd.DataFrame,
    df_lr: pd.DataFrame,
    df_finbert: pd.DataFrame,
) -> dict:
    """
    Compute per-method F1 scores using FinBERT as pseudo-ground truth.

    Converts continuous scores to binary (|score| > 0.05 → positive/negative),
    then computes precision, recall, F1 for each method vs. FinBERT.
    Returns dict with per-method metrics suitable for NLP quality reporting.
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
        """Convert scores to: 2=positive, 0=negative, 1=neutral (excluded later)."""
        result = pd.Series(1, index=series.index)  # neutral
        result[series > threshold] = 2              # positive
        result[series < -threshold] = 0             # negative
        return result

    y_true = to_binary(merged["score_finbert"])

    metrics = {}
    for method in ["vader", "lr"]:
        col = f"score_{method}"
        if col not in merged.columns:
            metrics[method] = {"error": "no_data"}
            continue
        y_pred = to_binary(merged[col])

        # Exclude neutral from F1 (only evaluate directional agreement)
        mask = (y_true != 1) & (y_pred != 1)
        if mask.sum() < 5:
            metrics[method] = {"error": "too_few_directional_samples", "n": int(mask.sum())}
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        # Positive class (2) metrics
        tp_pos = int(((yt == 2) & (yp == 2)).sum())
        fp_pos = int(((yt != 2) & (yp == 2)).sum())
        fn_pos = int(((yt == 2) & (yp != 2)).sum())

        prec_pos = tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0.0
        rec_pos = tp_pos / (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else 0.0
        f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos) if (prec_pos + rec_pos) > 0 else 0.0

        # Negative class (0) metrics
        tp_neg = int(((yt == 0) & (yp == 0)).sum())
        fp_neg = int(((yt != 0) & (yp == 0)).sum())
        fn_neg = int(((yt == 0) & (yp != 0)).sum())

        prec_neg = tp_neg / (tp_neg + fp_neg) if (tp_neg + fp_neg) > 0 else 0.0
        rec_neg = tp_neg / (tp_neg + fn_neg) if (tp_neg + fn_neg) > 0 else 0.0
        f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0

        # Macro F1
        f1_macro = (f1_pos + f1_neg) / 2.0

        metrics[method] = {
            "f1_macro": round(f1_macro, 4),
            "f1_positive": round(f1_pos, 4),
            "f1_negative": round(f1_neg, 4),
            "precision_macro": round((prec_pos + prec_neg) / 2, 4),
            "recall_macro": round((rec_pos + rec_neg) / 2, 4),
            "n_samples": int(mask.sum()),
        }

    # Overall agreement rate (including neutrals)
    total = len(merged)
    if total > 0:
        for method in ["vader", "lr"]:
            col = f"score_{method}"
            if col in merged.columns and method in metrics and "error" not in metrics[method]:
                yp_all = to_binary(merged[col])
                agreement = int((y_true == yp_all).sum())
                metrics[method]["agreement_rate"] = round(agreement / total, 4)

    return metrics
