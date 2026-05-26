"""Load raw CSVs into typed DataFrames.

The schema of martj42's results.csv is:
    date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral
"""

from __future__ import annotations

import pandas as pd

from src.config import GOALSCORERS_FILE, RESULTS_FILE, SHOOTOUTS_FILE
from src.utils import get_logger

log = get_logger(__name__)


def load_results() -> pd.DataFrame:
    """Load all international match results."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file missing at {RESULTS_FILE}. "
            "Run `python -m src.data.download` first."
        )
    df = pd.read_csv(RESULTS_FILE, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df[["home_score", "away_score"]] = df[["home_score", "away_score"]].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    log.info("Loaded %d international matches (%s → %s)",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df


def load_shootouts() -> pd.DataFrame:
    if not SHOOTOUTS_FILE.exists():
        return pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
    return pd.read_csv(SHOOTOUTS_FILE, parse_dates=["date"])


def load_goalscorers() -> pd.DataFrame:
    if not GOALSCORERS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(GOALSCORERS_FILE, parse_dates=["date"])
