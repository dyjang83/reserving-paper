"""
Evaluation metrics for the loss reserving paper.

This module implements the full evaluation framework described in Section 3
of the paper. Key design principles:

1. All metrics computed on dollar scale (not log scale) — actuarially meaningful
2. Per-line breakdown — shows where ML wins/loses
3. Per-lag breakdown — shows maturity effects
4. Reserve adequacy — did the model over or under-reserve?
5. Mack standard error ratio — how do ML CIs compare to actuarial CIs?
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Point estimate metrics                                              #
# ------------------------------------------------------------------ #

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error in dollars."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error in dollars."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute percentage error. Excludes zero actuals."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def weighted_rmse(y_true: pd.Series, y_pred: pd.Series,
                  weights: pd.Series) -> float:
    """
    Premium-weighted RMSE.

    Weights each company's error by its earned premium, so large insurers
    count more. This matches actuarial practice where reserve adequacy for
    large writers is most consequential.
    """
    w = weights / weights.sum()
    return float(np.sqrt(np.sum(w * (y_true - y_pred) ** 2)))


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Mean signed error (predicted - actual).

    Positive = over-reserved (conservative).
    Negative = under-reserved (aggressive).
    """
    return float(np.mean(y_pred - y_true))


def reserve_adequacy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Percentage of accident years where the reserve is adequate (pred >= actual).
    Higher is more conservative. Industry target is typically 50-70%.
    """
    return float((y_pred >= y_true).mean())


# ------------------------------------------------------------------ #
# Uncertainty / CI metrics                                            #
# ------------------------------------------------------------------ #

def ci_coverage(y_true: pd.Series,
                ci_lower: pd.Series,
                ci_upper: pd.Series) -> float:
    """
    Empirical coverage of bootstrap confidence intervals.

    For 95% CIs, well-calibrated models should have ~95% coverage.
    Under-coverage means CIs are too narrow; over-coverage means too wide.
    """
    within = (y_true >= ci_lower) & (y_true <= ci_upper)
    return float(within.mean())


def ci_width(ci_lower: pd.Series, ci_upper: pd.Series) -> float:
    """Mean width of confidence intervals in dollars."""
    return float((ci_upper - ci_lower).mean())


def ci_width_relative(y_true: pd.Series,
                      ci_lower: pd.Series,
                      ci_upper: pd.Series) -> float:
    """Mean CI width as a fraction of the actual ultimate."""
    mask = y_true != 0
    return float(((ci_upper[mask] - ci_lower[mask]) / y_true[mask]).mean())


# ------------------------------------------------------------------ #
# Aggregate evaluation                                                #
# ------------------------------------------------------------------ #

def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    weights: pd.Series = None,
    ci_lower: pd.Series = None,
    ci_upper: pd.Series = None,
) -> dict:
    """
    Compute the full set of evaluation metrics for one model.

    Returns a dict suitable for building the paper's results table.
    """
    results = {
        "model": model_name,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "reserve_adequacy": reserve_adequacy(y_true, y_pred),
    }

    if weights is not None:
        results["weighted_rmse"] = weighted_rmse(y_true, y_pred, weights)

    if ci_lower is not None and ci_upper is not None:
        results["ci_coverage_95"] = ci_coverage(y_true, ci_lower, ci_upper)
        results["ci_width"] = ci_width(ci_lower, ci_upper)
        results["ci_width_relative"] = ci_width_relative(
            y_true, ci_lower, ci_upper
        )

    return results


def evaluate_by_line(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    lines: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate model performance broken out by line of business.

    This produces the key comparison table showing where ML wins/loses
    (Section 4.2 of the paper).
    """
    rows = []
    for line in sorted(lines.unique()):
        mask = lines == line
        rows.append({
            "model": model_name,
            "line": line,
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "mape": mape(y_true[mask], y_pred[mask]),
            "bias": bias(y_true[mask], y_pred[mask]),
            "reserve_adequacy": reserve_adequacy(y_true[mask], y_pred[mask]),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def evaluate_by_lag(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    dev_lags: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate model performance broken out by development lag.

    Shows whether ML advantage is concentrated at early lags (high uncertainty)
    or persists at late lags (near-fully-developed).
    This is the key analysis for Section 4.3 of the paper.
    """
    rows = []
    for lag in sorted(dev_lags.unique()):
        mask = dev_lags == lag
        rows.append({
            "model": model_name,
            "dev_lag": lag,
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "mape": mape(y_true[mask], y_pred[mask]),
            "bias": bias(y_true[mask], y_pred[mask]),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def improvement_table(
    baseline_results: pd.DataFrame,
    ml_results: pd.DataFrame,
    metric: str = "rmse",
    index_col: str = "line",
) -> pd.DataFrame:
    """
    Build a comparison table showing ML improvement over baseline.

    Parameters
    ----------
    baseline_results : DataFrame from evaluate_by_line or evaluate_by_lag
    ml_results : DataFrame from evaluate_by_line or evaluate_by_lag
    metric : column to compare
    index_col : 'line' or 'dev_lag'

    Returns
    -------
    DataFrame with baseline, ml, and improvement_pct columns.
    """
    base = baseline_results.set_index(index_col)[metric].rename("baseline")
    ml = ml_results.set_index(index_col)[metric].rename("ml")
    combined = pd.concat([base, ml], axis=1)
    combined["improvement_pct"] = (
        (combined["baseline"] - combined["ml"]) / combined["baseline"] * 100
    ).round(1)
    combined["ml_wins"] = combined["ml"] < combined["baseline"]
    return combined.round(2)


def format_results_table(results: list[dict]) -> pd.DataFrame:
    """
    Format a list of evaluate_model() outputs into a publication-ready table.

    Rounds all floats appropriately and orders columns for the paper.
    """
    df = pd.DataFrame(results)
    col_order = [
        "model", "rmse", "mae", "mape", "bias",
        "reserve_adequacy", "weighted_rmse",
        "ci_coverage_95", "ci_width_relative"
    ]
    cols_present = [c for c in col_order if c in df.columns]
    df = df[cols_present]

    # Format for paper
    df["rmse"] = df["rmse"].apply(lambda x: f"${x:,.0f}")
    df["mae"] = df["mae"].apply(lambda x: f"${x:,.0f}")
    df["mape"] = df["mape"].apply(lambda x: f"{x:.1%}")
    df["bias"] = df["bias"].apply(lambda x: f"${x:,.0f}")
    df["reserve_adequacy"] = df["reserve_adequacy"].apply(
        lambda x: f"{x:.1%}"
    )
    if "ci_coverage_95" in df.columns:
        df["ci_coverage_95"] = df["ci_coverage_95"].apply(
            lambda x: f"{x:.1%}"
        )
    if "ci_width_relative" in df.columns:
        df["ci_width_relative"] = df["ci_width_relative"].apply(
            lambda x: f"{x:.1%}"
        )

    return df
