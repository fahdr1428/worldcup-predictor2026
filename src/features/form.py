"""Rolling-form features for each team.

For every match we compute, *from each team's perspective using only
prior matches*, things like recent win-rate, goals scored, conceded, and
head-to-head record vs the opponent.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.config import FORM_WINDOW
from src.utils import get_logger

log = get_logger(__name__)


def _result_points(scored: int, conceded: int) -> int:
    if scored > conceded:
        return 3
    if scored == conceded:
        return 1
    return 0


def add_form_features(matches: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Add rolling form features. Input must be chronologically sorted."""
    matches = matches.sort_values("date").reset_index(drop=True)

    # Per-team deques of recent (scored, conceded, points)
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=5))
    appearances: dict[str, int] = defaultdict(int)

    cols = {
        "home_form_points": np.zeros(len(matches), dtype=np.float32),
        "away_form_points": np.zeros(len(matches), dtype=np.float32),
        "home_form_gs": np.zeros(len(matches), dtype=np.float32),
        "away_form_gs": np.zeros(len(matches), dtype=np.float32),
        "home_form_gc": np.zeros(len(matches), dtype=np.float32),
        "away_form_gc": np.zeros(len(matches), dtype=np.float32),
        "h2h_home_winrate": np.full(len(matches), 0.5, dtype=np.float32),
        "home_matches_played": np.zeros(len(matches), dtype=np.int32),
        "away_matches_played": np.zeros(len(matches), dtype=np.int32),
    }

    def _summary(team: str) -> tuple[float, float, float]:
        h = history[team]
        if not h:
            return 1.0, 1.0, 1.0  # neutral priors
        pts = sum(x[2] for x in h) / len(h)
        gs = sum(x[0] for x in h) / len(h)
        gc = sum(x[1] for x in h) / len(h)
        return pts, gs, gc

    for i, row in enumerate(matches.itertuples(index=False)):
        home, away = row.home_team, row.away_team

        # Pre-match form (uses only previously seen matches)
        h_pts, h_gs, h_gc = _summary(home)
        a_pts, a_gs, a_gc = _summary(away)
        cols["home_form_points"][i] = h_pts
        cols["away_form_points"][i] = a_pts
        cols["home_form_gs"][i] = h_gs
        cols["away_form_gs"][i] = a_gs
        cols["home_form_gc"][i] = h_gc
        cols["away_form_gc"][i] = a_gc
        cols["home_matches_played"][i] = appearances[home]
        cols["away_matches_played"][i] = appearances[away]

        # Head-to-head (use canonical sorted key so direction doesn't matter)
        key = tuple(sorted([home, away]))
        h2h_list = h2h[key]
        if h2h_list:
            home_wins = sum(1 for winner in h2h_list if winner == home)
            cols["h2h_home_winrate"][i] = home_wins / len(h2h_list)

        # Update state
        history[home].append((row.home_score, row.away_score,
                              _result_points(row.home_score, row.away_score)))
        history[away].append((row.away_score, row.home_score,
                              _result_points(row.away_score, row.home_score)))
        appearances[home] += 1
        appearances[away] += 1
        winner = (home if row.home_score > row.away_score
                  else away if row.home_score < row.away_score
                  else "draw")
        h2h[key].append(winner)

    for k, v in cols.items():
        matches[k] = v

    matches["form_points_diff"] = matches["home_form_points"] - matches["away_form_points"]
    matches["form_gs_diff"] = matches["home_form_gs"] - matches["away_form_gs"]
    matches["form_gc_diff"] = matches["away_form_gc"] - matches["home_form_gc"]  # smaller gc is better

    log.info("Form features added (window=%d)", window)
    return matches
