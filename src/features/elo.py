"""ELO rating system for international football.

This is a custom variant of the World Football Elo Ratings approach:
  - Home advantage added to expected score
  - K-factor scaled by tournament importance and goal margin
  - Pre-match ELO returned for every match so we can use it as a feature
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    ELO_HOME_ADVANTAGE,
    ELO_K_BASE,
    ELO_START,
    TOURNAMENT_WEIGHTS,
    TOURNAMENT_WEIGHT_DEFAULT,
)
from src.utils import get_logger

log = get_logger(__name__)


def expected_score(rating_a: float, rating_b: float, home_adv: float = 0.0) -> float:
    """Standard ELO expectation with optional home advantage."""
    return 1.0 / (1.0 + 10 ** ((rating_b - (rating_a + home_adv)) / 400))


def goal_margin_multiplier(margin: int) -> float:
    """Inflates K for decisive results. Standard FIFA/Elo formula."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8.0  # 3→1.75, 4→1.875, 5→2.0, ...


@dataclass
class EloTable:
    """In-memory ELO ratings for every team encountered so far."""

    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, ELO_START)

    def set(self, team: str, value: float) -> None:
        self.ratings[team] = value

    def as_series(self) -> pd.Series:
        return pd.Series(self.ratings, name="elo").sort_values(ascending=False)


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, EloTable]:
    """Walk every match in chronological order, returning:
       * a copy of `matches` with ``home_elo`` and ``away_elo`` columns
         containing pre-match ratings
       * the final ELO table
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    table = EloTable()

    home_elos = np.empty(len(matches), dtype=np.float32)
    away_elos = np.empty(len(matches), dtype=np.float32)

    for i, row in enumerate(matches.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        ra, rb = table.get(home), table.get(away)
        home_elos[i] = ra
        away_elos[i] = rb

        # Home advantage applies only when not on neutral ground.
        home_adv = 0.0 if row.neutral else ELO_HOME_ADVANTAGE

        # Actual score: 1 win, 0.5 draw, 0 loss (from home POV)
        if row.home_score > row.away_score:
            actual = 1.0
        elif row.home_score < row.away_score:
            actual = 0.0
        else:
            actual = 0.5

        expected = expected_score(ra, rb, home_adv=home_adv)
        margin = abs(row.home_score - row.away_score)

        weight = TOURNAMENT_WEIGHTS.get(row.tournament, TOURNAMENT_WEIGHT_DEFAULT)
        k = ELO_K_BASE * weight * goal_margin_multiplier(margin)

        delta = k * (actual - expected)
        table.set(home, ra + delta)
        table.set(away, rb - delta)

    out = matches.copy()
    out["home_elo"] = home_elos
    out["away_elo"] = away_elos
    out["elo_diff"] = home_elos - away_elos
    log.info("ELO computed over %d matches | %d teams tracked",
             len(out), len(table.ratings))
    return out, table
