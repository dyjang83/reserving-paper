"""
Actuarial baseline models for the loss reserving paper.

Wraps chain-ladder, Bornhuetter-Ferguson, and Cape Cod from the
actuarial-reserving library (pip install actuarial-reserving) into
the paper's evaluation framework so all five models produce
comparable outputs.

Note: The actuarial-reserving library is a companion package developed
alongside this research. See github.com/dyjang83/reserving.
"""

import pandas as pd
import numpy as np
from typing import Optional


def fit_chain_ladder(
    upper_df: pd.DataFrame,
    line: str,
) -> dict:
    """
    Fit chain-ladder to a single line of business using the upper triangle.

    Parameters
    ----------
    upper_df : DataFrame
        Upper triangle data (calendar_year <= evaluation year).
    line : str
        Line of business to fit.

    Returns
    -------
    dict with keys: factors, factor_lookup
    """
    data = upper_df[upper_df["line"] == line].copy()
    data = data.sort_values(["company_code", "accident_year", "dev_lag"])

    data["next_paid"] = data.groupby(
        ["company_code", "accident_year"]
    )["paid_loss"].shift(-1)

    data = data.dropna(subset=["next_paid"])
    data = data[data["paid_loss"] > 0]
    data["ata"] = data["next_paid"] / data["paid_loss"]

    factors = (
        data.groupby("dev_lag")["ata"]
        .median()
        .reset_index()
    )

    factor_lookup = dict(zip(factors["dev_lag"], factors["ata"]))

    return {"factors": factors, "factor_lookup": factor_lookup, "line": line}


def predict_chain_ladder(
    snapshot_df: pd.DataFrame,
    model: dict,
) -> pd.Series:
    """
    Project paid losses to ultimate using fitted chain-ladder factors.

    For each snapshot at dev_lag k, multiplies paid_loss by:
        ATA(k) * ATA(k+1) * ... * ATA(9)
    """
    factor_lookup = model["factor_lookup"]

    def _chain(row):
        loss = row["paid_loss"]
        for lag in range(int(row["dev_lag"]), 10):
            loss *= factor_lookup.get(lag, 1.0)
        return loss

    return snapshot_df.apply(_chain, axis=1).rename("pred_ultimate")


def fit_bornhuetter_ferguson(
    upper_df: pd.DataFrame,
    line: str,
    apriori_elr: float,
) -> dict:
    """
    Fit Bornhuetter-Ferguson for a single line.

    Parameters
    ----------
    upper_df : DataFrame
        Upper triangle data.
    line : str
        Line of business.
    apriori_elr : float
        A priori expected loss ratio. In the paper, this is set to the
        volume-weighted average paid loss ratio from the training data.

    Returns
    -------
    dict with model components.
    """
    cl_model = fit_chain_ladder(upper_df, line)
    factor_lookup = cl_model["factor_lookup"]

    # Compute CDFs by chaining factors from each lag to ultimate
    cdfs = {}
    lags = sorted(factor_lookup.keys())
    for start_lag in lags:
        cdf = 1.0
        for lag in range(start_lag, 10):
            cdf *= factor_lookup.get(lag, 1.0)
        cdfs[start_lag] = cdf

    return {
        "cl_model": cl_model,
        "cdfs": cdfs,
        "apriori_elr": apriori_elr,
        "line": line,
    }


def predict_bornhuetter_ferguson(
    snapshot_df: pd.DataFrame,
    model: dict,
) -> pd.Series:
    """
    Project to ultimate using Bornhuetter-Ferguson method.

    ultimate = emerged + apriori_ultimate * (1 - pct_reported)
    where pct_reported = 1 / CDF
    """
    cdfs = model["cdfs"]
    elr = model["apriori_elr"]

    def _bf(row):
        lag = int(row["dev_lag"])
        cdf = cdfs.get(lag, 1.0)
        pct_reported = 1.0 / cdf if cdf > 0 else 1.0
        apriori_ult = row["earned_prem_net"] * elr
        return row["paid_loss"] + apriori_ult * (1.0 - pct_reported)

    return snapshot_df.apply(_bf, axis=1).rename("pred_ultimate")


def fit_cape_cod(
    upper_df: pd.DataFrame,
    line: str,
) -> dict:
    """
    Fit Cape Cod for a single line.

    Derives the ELR from the data itself:
        ELR = sum(emerged paid) / sum(used-up premium)
    where used-up premium = premium * pct_reported.
    """
    cl_model = fit_chain_ladder(upper_df, line)
    factor_lookup = cl_model["factor_lookup"]

    cdfs = {}
    lags = sorted(factor_lookup.keys())
    for start_lag in lags:
        cdf = 1.0
        for lag in range(start_lag, 10):
            cdf *= factor_lookup.get(lag, 1.0)
        cdfs[start_lag] = cdf

    # Compute ELR from training data
    data = upper_df[upper_df["line"] == line].copy()

    # Get latest diagonal snapshot per company/accident_year
    latest = (
        data.sort_values("dev_lag")
        .groupby(["company_code", "accident_year"])
        .last()
        .reset_index()
    )
    latest["pct_reported"] = latest["dev_lag"].map(
        lambda lag: 1.0 / cdfs.get(int(lag), 1.0)
    )
    latest["used_up_prem"] = latest["earned_prem_net"] * latest["pct_reported"]

    total_used_up = latest["used_up_prem"].sum()
    total_emerged = latest["paid_loss"].sum()
    elr = total_emerged / total_used_up if total_used_up > 0 else 0.65

    return {
        "cl_model": cl_model,
        "cdfs": cdfs,
        "elr": elr,
        "line": line,
    }


def predict_cape_cod(
    snapshot_df: pd.DataFrame,
    model: dict,
) -> pd.Series:
    """
    Project to ultimate using Cape Cod method.

    ultimate = emerged + ELR * premium * (1 - pct_reported)
    """
    cdfs = model["cdfs"]
    elr = model["elr"]

    def _cc(row):
        lag = int(row["dev_lag"])
        cdf = cdfs.get(lag, 1.0)
        pct_reported = 1.0 / cdf if cdf > 0 else 1.0
        return row["paid_loss"] + elr * row["earned_prem_net"] * (1.0 - pct_reported)

    return snapshot_df.apply(_cc, axis=1).rename("pred_ultimate")


def fit_all_actuarial(
    upper_df: pd.DataFrame,
    lines: Optional[list] = None,
) -> dict:
    """
    Fit all three actuarial methods for all lines.

    Returns a nested dict: {method: {line: model}}
    """
    if lines is None:
        lines = upper_df["line"].unique().tolist()

    models = {"chain_ladder": {}, "bornhuetter_ferguson": {}, "cape_cod": {}}

    for line in lines:
        line_data = upper_df[upper_df["line"] == line]

        # Chain-ladder
        models["chain_ladder"][line] = fit_chain_ladder(upper_df, line)

        # BF — a priori ELR = volume-weighted paid loss ratio from training
        prem = line_data["earned_prem_net"].sum()
        paid = line_data[line_data["dev_lag"] == line_data["dev_lag"].max()]["paid_loss"].sum()
        apriori = paid / prem if prem > 0 else 0.65
        models["bornhuetter_ferguson"][line] = fit_bornhuetter_ferguson(
            upper_df, line, apriori_elr=apriori
        )

        # Cape Cod
        models["cape_cod"][line] = fit_cape_cod(upper_df, line)

    return models


def predict_all_actuarial(
    test_df: pd.DataFrame,
    models: dict,
) -> pd.DataFrame:
    """
    Generate predictions from all three actuarial methods.

    Parameters
    ----------
    test_df : DataFrame
        Test set snapshots.
    models : dict
        Output of fit_all_actuarial().

    Returns
    -------
    DataFrame with columns: pred_cl, pred_bf, pred_cc added to test_df.
    """
    result = test_df.copy()
    result["pred_cl"] = np.nan
    result["pred_bf"] = np.nan
    result["pred_cc"] = np.nan

    for line in test_df["line"].unique():
        mask = result["line"] == line
        subset = result[mask].copy()

        if line in models["chain_ladder"]:
            result.loc[mask, "pred_cl"] = predict_chain_ladder(
                subset, models["chain_ladder"][line]
            ).values

        if line in models["bornhuetter_ferguson"]:
            result.loc[mask, "pred_bf"] = predict_bornhuetter_ferguson(
                subset, models["bornhuetter_ferguson"][line]
            ).values

        if line in models["cape_cod"]:
            result.loc[mask, "pred_cc"] = predict_cape_cod(
                subset, models["cape_cod"][line]
            ).values

    return result
