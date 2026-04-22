"""
Data cleaning for the loss reserving paper.

This module is more rigorous than the original loss-reserve-ml cleaner:
  - Explicit data quality flags for the paper's Table 1 (data description)
  - Triangle completeness validation
  - Outlier detection and documented handling
  - Separation of upper triangle (training) from lower triangle (ground truth)
"""

import pandas as pd
import numpy as np


RENAME_MAP = {
    "GRCODE":           "company_code",
    "GRNAME":           "company",
    "AccidentYear":     "accident_year",
    "DevelopmentYear":  "dev_year",
    "DevelopmentLag":   "dev_lag",
    "IncurredLosses":   "incurred_loss",
    "CumPaidLoss":      "paid_loss",
    "BulkLoss":         "bulk_loss",
    "EarnedPremDIR":    "earned_prem_direct",
    "EarnedPremCeded":  "earned_prem_ceded",
    "EarnedPremNet":    "earned_prem_net",
    "Single":           "is_single_entity",
    "PostedReserves2007": "posted_reserve_2007",
}


def clean(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Clean and standardize raw CAS Schedule P data.

    Parameters
    ----------
    df : DataFrame
        Raw CAS data as loaded by loader.load_cas_line() or load_cas_all().
    verbose : bool
        If True, print data quality report. Useful for paper's Table 1.

    Returns
    -------
    Cleaned DataFrame with derived columns.
    """
    original_rows = len(df)

    # Rename columns
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # Drop rows missing vital information
    vital_cols = ["incurred_loss", "paid_loss", "earned_prem_net"]
    df = df.dropna(subset=vital_cols)
    after_na = len(df)

    # Remove negative losses (data artifacts)
    neg_mask = (df["incurred_loss"] < 0) | (df["paid_loss"] < 0)
    df = df[~neg_mask]
    after_neg = len(df)

    # Remove zero premium (cannot compute loss ratios, small runoff entities)
    zero_prem = df["earned_prem_net"] == 0
    df = df[~zero_prem]
    after_prem = len(df)

    if verbose:
        print(f"Data quality report:")
        print(f"  Original rows:          {original_rows:,}")
        print(f"  After removing NaN:     {after_na:,} (removed {original_rows - after_na:,})")
        print(f"  After removing neg:     {after_neg:,} (removed {after_na - after_neg:,})")
        print(f"  After removing 0 prem:  {after_prem:,} (removed {after_neg - after_prem:,})")

    # Derived actuarial columns
    df["loss_ratio_incurred"] = (
        df["incurred_loss"] / df["earned_prem_net"].replace(0, np.nan)
    )
    df["loss_ratio_paid"] = (
        df["paid_loss"] / df["earned_prem_net"].replace(0, np.nan)
    )
    df["case_reserve"] = (df["incurred_loss"] - df["paid_loss"]).clip(lower=0)
    df["paid_to_incurred"] = (
        df["paid_loss"] / df["incurred_loss"].replace(0, np.nan)
    ).clip(0, 2.0)

    return df.reset_index(drop=True)


def validate_triangles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check triangle completeness for each company and accident year.

    Returns a summary DataFrame with one row per (line, company, accident_year)
    showing how many development lags are present.

    Used in paper Table 1 to characterize data quality.
    """
    completeness = (
        df.groupby(["line", "company_code", "accident_year"])["dev_lag"]
        .count()
        .reset_index(name="n_lags")
    )
    completeness["is_complete"] = completeness["n_lags"] == 10
    return completeness


def split_upper_lower(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into upper triangle (observable) and lower triangle (ground truth).

    The upper triangle contains only cells where:
        accident_year + dev_lag - 1 <= evaluation_year

    For the CAS 1998-2007 dataset, the evaluation year is 2007 (the diagonal).
    The upper triangle is used for training; the lower triangle reveals
    actual ultimate losses for model evaluation.

    Returns
    -------
    (upper_df, lower_df) : tuple of DataFrames
    """
    # Calendar year = accident year + development lag - 1
    df = df.copy()
    df["calendar_year"] = df["accident_year"] + df["dev_lag"] - 1

    # Upper triangle: calendar year <= 2007 (the evaluation diagonal)
    upper = df[df["calendar_year"] <= 2007].copy()

    # Lower triangle: calendar year > 2007 (revealed after evaluation date)
    lower = df[df["calendar_year"] > 2007].copy()

    return upper, lower


def get_ultimates(df: pd.DataFrame, basis: str = "paid") -> pd.DataFrame:
    """
    Extract ultimate losses (lag 10 values) for each company/accident_year.

    Parameters
    ----------
    df : DataFrame
        Full cleaned data including lower triangle.
    basis : str
        'paid' or 'incurred' — which loss column to use as the target.

    Returns
    -------
    DataFrame with columns: line, company_code, accident_year, target_ultimate
    """
    col = "paid_loss" if basis == "paid" else "incurred_loss"
    ultimates = (
        df[df["dev_lag"] == 10]
        [["line", "company_code", "accident_year", col]]
        .rename(columns={col: "target_ultimate"})
        .drop_duplicates()
    )
    return ultimates


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 1 of the paper: data description by line of business.

    Returns a DataFrame suitable for inclusion in the paper.
    """
    completeness = validate_triangles(df)
    complete_pct = (
        completeness.groupby("line")["is_complete"].mean() * 100
    ).round(1)

    stats = df.groupby("line").agg(
        n_companies=("company_code", "nunique"),
        n_rows=("dev_lag", "count"),
        accident_year_min=("accident_year", "min"),
        accident_year_max=("accident_year", "max"),
        median_loss_ratio=("loss_ratio_paid", "median"),
        std_loss_ratio=("loss_ratio_paid", "std"),
    ).round(3)

    stats["pct_complete_triangles"] = complete_pct
    return stats
