"""
Feature engineering for the loss reserving paper.

Design principles (important for peer review):
  1. Every feature has an actuarial interpretation
  2. No lookahead bias — only information available at evaluation date
  3. All transformations reproducible from raw triangle data

Feature groups:
  A. Development features
  B. Loss magnitude features
  C. Actuarial ratio features
  D. Triangle shape features
  E. Line-of-business features
"""

import pandas as pd
import numpy as np


def build_targets(df: pd.DataFrame, basis: str = "paid") -> pd.DataFrame:
    loss_col = "paid_loss" if basis == "paid" else "incurred_loss"
    ultimates = (
        df[df["dev_lag"] == 10]
        [["line", "company_code", "accident_year", loss_col]]
        .rename(columns={loss_col: "target_ultimate"})
        .drop_duplicates(subset=["line", "company_code", "accident_year"])
    )
    ml_df = df[df["dev_lag"] < 10].merge(
        ultimates, on=["line", "company_code", "accident_year"], how="inner"
    )
    ml_df["log_target"] = np.log1p(ml_df["target_ultimate"])
    return ml_df.reset_index(drop=True)


def add_development_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["maturity_pct"] = df["dev_lag"] / 9.0
    df["lags_remaining"] = 10 - df["dev_lag"]
    df["calendar_year"] = df["accident_year"] + df["dev_lag"] - 1
    return df


def add_loss_magnitude_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_paid"] = np.log1p(df["paid_loss"])
    df["log_incurred"] = np.log1p(df["incurred_loss"])
    df["log_premium"] = np.log1p(df["earned_prem_net"])
    return df


def add_actuarial_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["paid_to_incurred"] = (
        df["paid_loss"] / df["incurred_loss"].replace(0, np.nan)
    ).clip(0, 2.0).fillna(0)
    df["case_reserve"] = (df["incurred_loss"] - df["paid_loss"]).clip(lower=0)
    df["log_case_reserve"] = np.log1p(df["case_reserve"])
    df["paid_loss_ratio"] = (
        df["paid_loss"] / df["earned_prem_net"].replace(0, np.nan)
    ).clip(0, 5.0).fillna(0)
    df["incurred_loss_ratio"] = (
        df["incurred_loss"] / df["earned_prem_net"].replace(0, np.nan)
    ).clip(0, 5.0).fillna(0)
    df["bulk_loss_ratio"] = (
        df["bulk_loss"] / df["earned_prem_net"].replace(0, np.nan)
    ).clip(0, 5.0).fillna(0)
    return df


def _compute_ata_volatility(upper_df: pd.DataFrame) -> pd.DataFrame:
    temp = upper_df.sort_values(
        ["line", "company_code", "accident_year", "dev_lag"]
    ).copy()
    temp["next_paid"] = temp.groupby(
        ["line", "company_code", "accident_year"]
    )["paid_loss"].shift(-1)
    temp = temp.dropna(subset=["next_paid"])
    temp = temp[temp["paid_loss"] > 0]
    temp["ata"] = temp["next_paid"] / temp["paid_loss"]
    return (
        temp.groupby(["line", "company_code"])["ata"]
        .std()
        .reset_index(name="ata_volatility")
    )


def _compute_paid_speed(upper_df: pd.DataFrame) -> pd.DataFrame:
    lag5 = upper_df[upper_df["dev_lag"] == 5].copy()
    lag5 = lag5[lag5["incurred_loss"] > 0]
    lag5["speed"] = lag5["paid_loss"] / lag5["incurred_loss"]
    return (
        lag5.groupby(["line", "company_code"])["speed"]
        .median()
        .reset_index(name="avg_paid_speed")
    )


def add_triangle_shape_features(df, upper_df):
    df = df.copy()
    ata_vol = _compute_ata_volatility(upper_df)
    avg_speed = _compute_paid_speed(upper_df)
    df = df.merge(ata_vol, on=["line", "company_code"], how="left")
    df = df.merge(avg_speed, on=["line", "company_code"], how="left")
    df["ata_volatility"] = df["ata_volatility"].fillna(
        df.groupby("line")["ata_volatility"].transform("median")
    )
    df["avg_paid_speed"] = df["avg_paid_speed"].fillna(
        df.groupby("line")["avg_paid_speed"].transform("median")
    )
    return df


def add_lob_features(df):
    return pd.get_dummies(df, columns=["line"], drop_first=False)


def build_features(df, upper_df, basis="paid"):
    ml_df = build_targets(df, basis=basis)
    ml_df = add_development_features(ml_df)
    ml_df = add_loss_magnitude_features(ml_df)
    ml_df = add_actuarial_ratio_features(ml_df)
    ml_df = add_triangle_shape_features(ml_df, upper_df)
    ml_df = add_lob_features(ml_df)
    return ml_df


def get_feature_columns(ml_df):
    line_dummies = [c for c in ml_df.columns if c.startswith("line_")]
    base_features = [
        "dev_lag", "maturity_pct", "lags_remaining",
        "log_paid", "log_incurred", "log_premium",
        "paid_to_incurred", "case_reserve", "log_case_reserve",
        "paid_loss_ratio", "incurred_loss_ratio", "bulk_loss_ratio",
        "ata_volatility", "avg_paid_speed",
    ]
    features = [f for f in base_features if f in ml_df.columns]
    features += [d for d in line_dummies if d in ml_df.columns]
    return features
