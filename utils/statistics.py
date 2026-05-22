"""Statistical significance tests for DQN vs BH comparison.

Addresses the key academic concern: a 14/28 win rate with a simple count
is indistinguishable from a coin flip. This module provides:
  - Binomial test (win/loss count significance)
  - Paired t-test (Sharpe difference significance)
  - Bootstrap confidence intervals (effect size + uncertainty)
  - Multiple comparison correction (Bonferroni + FDR)
"""

import numpy as np
from scipy import stats


def binomial_test(n_wins: int, n_total: int, p_null: float = 0.5) -> dict:
    """Two-sided binomial test: is the observed win rate significantly different from p_null?

    H0: win probability = p_null (coin flip)
    H1: win probability != p_null
    """
    if n_total == 0:
        return {"p_value": 1.0, "significant_05": False, "significant_01": False,
                "n_wins": 0, "n_total": 0, "win_rate": 0.0, "ci_95": (0.0, 0.0)}
    p_value = stats.binomtest(n_wins, n_total, p=p_null, alternative="two-sided").pvalue
    win_rate = n_wins / n_total

    # Wilson score CI for proportion
    z = 1.96
    denom = 1 + z**2 / n_total
    center = (win_rate + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((win_rate * (1 - win_rate) + z**2 / (4 * n_total)) / n_total) / denom
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)

    return {
        "p_value": round(float(p_value), 6),
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "n_wins": n_wins,
        "n_total": n_total,
        "win_rate": round(win_rate, 4),
        "ci_95": (round(ci_lower, 4), round(ci_upper, 4)),
    }


def paired_t_test(dqn_values: list[float], bh_values: list[float]) -> dict:
    """Paired t-test on Sharpe ratios: DQN vs BH across tickers.

    H0: mean(DQN_sharpe - BH_sharpe) = 0
    H1: mean(DQN_sharpe - BH_sharpe) != 0

    Also computes Cohen's d effect size.
    """
    dqn = np.array(dqn_values)
    bh = np.array(bh_values)
    diffs = dqn - bh
    n = len(diffs)

    if n < 3:
        return {"p_value": 1.0, "significant_05": False, "mean_diff": 0.0,
                "cohens_d": 0.0, "n": n}

    t_stat, p_value = stats.ttest_rel(dqn, bh)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # Cohen's d interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    return {
        "p_value": round(float(p_value), 6),
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "mean_diff": round(mean_diff, 6),
        "std_diff": round(std_diff, 6),
        "cohens_d": round(cohens_d, 4),
        "effect_size": effect_size,
        "n": n,
        "ci_95": (
            round(mean_diff - 1.96 * std_diff / np.sqrt(n), 6),
            round(mean_diff + 1.96 * std_diff / np.sqrt(n), 6),
        ),
    }


def bootstrap_ci(
    dqn_values: list[float], bh_values: list[float],
    n_bootstrap: int = 10_000, alpha: float = 0.05, seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for mean Sharpe difference.

    Non-parametric bootstrap of the paired difference distribution.
    Returns empirical CI and bootstrap distribution statistics.
    """
    rng = np.random.default_rng(seed)
    dqn = np.array(dqn_values)
    bh = np.array(bh_values)
    n = len(dqn)

    if n < 3:
        return {"ci_95": (0.0, 0.0), "mean": 0.0, "std": 0.0, "n_bootstrap": 0}

    diffs = dqn - bh
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diffs[idx])

    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    boot_mean = float(np.mean(boot_means))
    boot_std = float(np.std(boot_means))

    # Is zero inside the CI?
    zero_in_ci = lower <= 0 <= upper

    return {
        "ci_95": (round(float(lower), 6), round(float(upper), 6)),
        "ci_99": (
            round(float(np.percentile(boot_means, 0.5)), 6),
            round(float(np.percentile(boot_means, 99.5)), 6),
        ),
        "mean": round(boot_mean, 6),
        "std": round(boot_std, 6),
        "zero_in_ci_95": zero_in_ci,
        "n_bootstrap": n_bootstrap,
    }


def multiple_comparison_correction(p_values: list[float], method: str = "bonferroni") -> dict:
    """Correct for multiple comparisons across tickers.

    Args:
        p_values: list of p-values from individual tests
        method: "bonferroni" or "fdr_bh" (Benjamini-Hochberg)

    Returns corrected significance decisions and adjusted p-values.
    """
    n = len(p_values)
    if n == 0:
        return {"method": method, "n_tests": 0}

    p = np.array(p_values)

    if method == "bonferroni":
        adjusted = np.minimum(p * n, 1.0)
    elif method == "fdr_bh":
        # Benjamini-Hochberg procedure
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        adjusted_unsorted = np.minimum(
            sorted_p * n / np.arange(1, n + 1), 1.0
        )
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            adjusted_unsorted[i] = min(adjusted_unsorted[i], adjusted_unsorted[i + 1])
        adjusted = np.zeros(n)
        adjusted[sorted_idx] = adjusted_unsorted
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "method": method,
        "n_tests": n,
        "adjusted_p_values": [round(float(x), 6) for x in adjusted],
        "n_significant_05": int(np.sum(adjusted < 0.05)),
        "n_significant_01": int(np.sum(adjusted < 0.01)),
    }


def run_dqn_vs_bh_tests(
    dqn_sharpes: list[float],
    bh_sharpes: list[float],
    dqn_returns: list[float],
    bh_returns: list[float],
    seed: int = 42,
) -> dict:
    """Complete statistical test battery for DQN vs BH comparison.

    Returns a structured dict suitable for JSON serialization and academic reporting.
    """
    # Binomial test on Sharpe wins
    n_sharpe_wins = sum(1 for d, b in zip(dqn_sharpes, bh_sharpes) if d > b)
    n_return_wins = sum(1 for d, b in zip(dqn_returns, bh_returns) if d > b)
    n_total = len(dqn_sharpes)

    binomial_sharpe = binomial_test(n_sharpe_wins, n_total)
    binomial_return = binomial_test(n_return_wins, n_total)

    # Paired t-test on Sharpe differences
    t_test = paired_t_test(dqn_sharpes, bh_sharpes)

    # Bootstrap CI on Sharpe differences
    boot_ci = bootstrap_ci(dqn_sharpes, bh_sharpes, seed=seed)

    # Individual ticker p-values (approximate via bootstrap)
    individual_p = []
    rng = np.random.default_rng(seed)
    diffs = np.array(dqn_sharpes) - np.array(bh_sharpes)
    for i in range(n_total):
        boot_diffs = diffs[rng.integers(0, n_total, size=(10_000,))]
        p_val = min(
            np.mean(boot_diffs >= abs(diffs[i])),
            np.mean(boot_diffs <= -abs(diffs[i])),
        ) * 2
        individual_p.append(round(float(min(p_val, 1.0)), 6))

    mc_bonf = multiple_comparison_correction(individual_p, "bonferroni")
    mc_fdr = multiple_comparison_correction(individual_p, "fdr_bh")

    # Interpretation
    if t_test["significant_05"] and not boot_ci["zero_in_ci_95"]:
        interpretation = (
            "DQN significantly outperforms BH (p < 0.05, bootstrap CI excludes zero). "
            f"Effect size: {t_test['effect_size']} (Cohen's d = {t_test['cohens_d']:.3f})."
        )
    elif t_test["significant_05"]:
        interpretation = (
            "DQN Sharpe difference is statistically significant by t-test (p < 0.05) "
            "but bootstrap CI includes zero — effect is fragile."
        )
    elif binomial_sharpe["p_value"] < 0.05:
        interpretation = (
            "Win rate is significant by binomial test but mean Sharpe difference "
            "is not significant by t-test — DQN wins more often but by small margins."
        )
    else:
        interpretation = (
            f"No statistically significant difference between DQN and BH. "
            f"Binomial p = {binomial_sharpe['p_value']:.3f}, "
            f"t-test p = {t_test['p_value']:.3f}. "
            f"The observed {binomial_sharpe['win_rate']:.1%} win rate is consistent with random chance."
        )

    return {
        "binomial_test_sharpe": binomial_sharpe,
        "binomial_test_return": binomial_return,
        "paired_t_test": t_test,
        "bootstrap_ci": boot_ci,
        "multiple_comparison_bonferroni": mc_bonf,
        "multiple_comparison_fdr": mc_fdr,
        "interpretation": interpretation,
    }
