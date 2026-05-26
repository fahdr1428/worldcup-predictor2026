"""Training pipeline.

Two models are trained:
  1. ``result_model``  — XGBoost classifier predicting W/D/L (3 classes).
  2. ``goals_model``   — multi-output gradient-boosted regressor predicting
                         (home_goals, away_goals) as Poisson means.

Both consume the same feature matrix produced by ``features.builder``.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    FEATURE_META_FILE,
    GOALS_MODEL_FILE,
    RANDOM_STATE,
    RESULT_MODEL_FILE,
    TEST_START_DATE,
    TRAIN_START_DATE,
)
from src.features.builder import FEATURE_COLUMNS, build_features, load_features
from src.utils import get_logger, timer

log = get_logger(__name__)


@dataclass
class TrainingReport:
    train_size: int
    test_size: int
    accuracy: float
    log_loss: float
    home_goals_mae: float
    away_goals_mae: float

    def __str__(self) -> str:
        return (
            f"Training report\n"
            f"  train rows : {self.train_size:,}\n"
            f"  test rows  : {self.test_size:,}\n"
            f"  accuracy   : {self.accuracy:.3f}\n"
            f"  log-loss   : {self.log_loss:.3f}\n"
            f"  home goals MAE: {self.home_goals_mae:.3f}\n"
            f"  away goals MAE: {self.away_goals_mae:.3f}\n"
        )


def _build_or_load_features() -> pd.DataFrame:
    try:
        return load_features()
    except FileNotFoundError:
        log.info("Cached features not found — building from scratch.")
        df, _ = build_features(save=True)
        return df


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df["date"] >= TRAIN_START_DATE].copy()
    train = df[df["date"] < TEST_START_DATE]
    test = df[df["date"] >= TEST_START_DATE]
    return train, test


def train(save: bool = True) -> TrainingReport:
    """Train both models and report performance on the held-out window."""
    df = _build_or_load_features()
    train_df, test_df = _split(df)
    log.info("Split: %d train / %d test", len(train_df), len(test_df))

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["result"].values
    g_train = train_df[["home_score", "away_score"]].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["result"].values
    g_test = test_df[["home_score", "away_score"]].values

    # ------------------------------------------------------------------
    # Result classifier
    # ------------------------------------------------------------------
    with timer("train result classifier"):
        result_model = XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        result_model.fit(X_train, y_train)

    y_pred = result_model.predict(X_test)
    y_proba = result_model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba, labels=[0, 1, 2])

    # ------------------------------------------------------------------
    # Goal regressors (Poisson)
    # ------------------------------------------------------------------
    with timer("train goals regressors"):
        goals_model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="count:poisson",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            n_jobs=2,
        )
        goals_model.fit(X_train, g_train)

    g_pred = goals_model.predict(X_test)
    hmae = mean_absolute_error(g_test[:, 0], g_pred[:, 0])
    amae = mean_absolute_error(g_test[:, 1], g_pred[:, 1])

    report = TrainingReport(
        train_size=len(train_df),
        test_size=len(test_df),
        accuracy=accuracy,
        log_loss=ll,
        home_goals_mae=hmae,
        away_goals_mae=amae,
    )
    log.info("\n%s", report)

    if save:
        joblib.dump(result_model, RESULT_MODEL_FILE)
        joblib.dump(goals_model, GOALS_MODEL_FILE)
        joblib.dump({"feature_columns": FEATURE_COLUMNS}, FEATURE_META_FILE)
        log.info("Saved models → %s", RESULT_MODEL_FILE.parent)

    return report


if __name__ == "__main__":
    train()
