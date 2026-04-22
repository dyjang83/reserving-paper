"""
Data loader for the paper:
  'When Does Machine Learning Beat Traditional Actuarial Methods
   for Loss Reserve Estimation? A Systematic Comparison'

Two data sources:
  1. CAS Schedule P (1998-2007) — primary dataset, six lines of business
  2. chainladder package built-ins — validation on independent benchmark data
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"

# CAS Schedule P files
CAS_FILES = {
    "ppauto":  "ppauto_pos.csv",
    "comauto": "comauto_pos.csv",
    "wkcomp":  "wkcomp_pos.csv",
    "medmal":  "medmal_pos.csv",
    "othliab": "othliab_pos.csv",
    "prodliab":"prodliab_pos.csv",
}

# Line-of-business metadata for paper tables
LINE_METADATA = {
    "ppauto":  {"label": "Private passenger auto",     "tail": "short"},
    "comauto": {"label": "Commercial auto",            "tail": "short"},
    "wkcomp":  {"label": "Workers compensation",       "tail": "medium"},
    "medmal":  {"label": "Medical malpractice",        "tail": "long"},
    "othliab": {"label": "Other liability",            "tail": "long"},
    "prodliab":{"label": "Product liability",          "tail": "long"},
}


def load_cas_line(line: str) -> pd.DataFrame:
    """Load one line of business from CAS Schedule P CSVs."""
    path = RAW_DIR / CAS_FILES[line]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"Download from casact.org/publications-research/research/"
            f"research-resources/loss-reserving-data-pulled-naic-schedule-p"
        )
    df = pd.read_csv(path)
    df["line"] = line
    df["line_label"] = LINE_METADATA[line]["label"]
    df["tail_type"] = LINE_METADATA[line]["tail"]
    return df


def load_cas_all() -> pd.DataFrame:
    """Load all six CAS Schedule P lines into a single long-format DataFrame."""
    return pd.concat(
        [load_cas_line(line) for line in CAS_FILES],
        ignore_index=True
    )


def load_chainladder_benchmarks() -> dict:
    """
    Load classic benchmark triangles from the chainladder package.
    Used for validation on independent data.

    Returns dict of {name: Triangle} for:
      - RAA: Mack (1993) benchmark — 10x10, 1981-1990
      - GenIns: Generic insurance triangle
      - MW2014: Meyers-Wacek 2014 benchmark
    """
    try:
        import chainladder as cl
    except ImportError:
        raise ImportError(
            "chainladder package not installed.\n"
            "Run: pip install chainladder"
        )

    benchmarks = {}
    for name in ["RAA", "GenIns", "MW2014", "UKMotor", "ABC"]:
        try:
            benchmarks[name] = cl.load_sample(name)
        except Exception:
            pass

    return benchmarks


def load_meyers_shi() -> dict:
    """
    Load the Meyers-Shi (2011) NAIC Schedule P dataset from chainladder.
    This is the exact data used by DeepTriangle (Kuo 2019) — using it
    gives us a direct comparison point against published benchmarks.

    Returns dict of {line: Triangle}.
    """
    try:
        import chainladder as cl
        clrd = cl.load_sample("clrd")
        return clrd
    except ImportError:
        raise ImportError("pip install chainladder")
    except Exception as e:
        raise RuntimeError(f"Could not load Meyers-Shi data: {e}")
