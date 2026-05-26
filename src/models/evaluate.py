"""Backtesting evaluation against held-out tournaments.

Reports accuracy, Brier score, log-loss, and "calibration" of W/D/L
probabilities on real World Cup matches in the test window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.features.builder import FEATURE_COLUMNS, load_features
from src.models.predict import _load_artifacts
from src.utils import get_logger

log = get_logger(__name__)


def evaluate_on_world_cup(year: int) -> pd.DataFrame:
    """Evaluate predictions on a specific World Cup. Returns a DataFrame
    of per-match predictions vs actuals."""
    result_model, _, feature_columns, _, _ = _load_artifacts()
    df = load_features()

    mask = (
        (df["tournament"] == "FIFA World Cup")
        & (df["date"].dt.year == year)
    )
    wc = df.loc[mask].copy()
    if wc.empty:
        log.warning("No matches for World Cup %d in dataset.", year)
        return wc

    X = wc[feature_columns].values
    proba = result_model.predict_proba(X)
    preds = proba.argmax(axis=1)
    wc["pred_result"] = preds
    wc["p_home_win"] = proba[:, 0]
    wc["p_draw"] = proba[:, 1]
    wc["p_away_win"] = proba[:, 2]
    wc["correct"] = wc["pred_result"] == wc["result"]

    acc = accuracy_score(wc["result"], wc["pred_result"])
    ll = log_loss(wc["result"], proba, labels=[0, 1, 2])
    brier = ((proba - np.eye(3)[wc["result"].values]) ** 2).sum(axis=1).mean()

    log.info("World Cup %d  | matches=%d | acc=%.3f | logloss=%.3f | brier=%.3f",
             year, len(wc), acc, ll, brier)
    return wc[["date", "home_team", "away_team", "home_score", "away_score",
               "result", "pred_result", "correct",
               "p_home_win", "p_draw", "p_away_win"]]


def summary_table(years: list[int]) -> pd.DataFrame:
    """Per-tournament summary across multiple World Cups."""
    rows = []
    for y in years:
        df = evaluate_on_world_cup(y)
        if df.empty:
            continue
        acc = df["correct"].mean()
        rows.append({"year": y, "matches": len(df), "accuracy": round(acc, 3)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(summary_table([2018, 2022]))
