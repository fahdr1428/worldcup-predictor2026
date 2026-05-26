"""Prediction interface.

Wraps the trained models so callers can ask "what happens if Argentina
plays France in the World Cup?" and get back:
  * Probability of home win / draw / away win
  * Expected goals for each side
  * Full score-line probability matrix (via Poisson)
  * Knockout-stage winner probabilities (incorporating penalty shootouts)
  * Key influencing players from each squad
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import (
    FEATURE_META_FILE,
    GOALS_MODEL_FILE,
    HOST_COUNTRIES,
    RESULT_MODEL_FILE,
    SQUAD_STRENGTH_CENTER,
    TOURNAMENT_WEIGHT_DEFAULT,
    TOURNAMENT_WEIGHTS,
)
from src.features.builder import load_features, load_team_ratings
from src.players.loader import load_effective_players
from src.players.strength import load_squad_strength
from src.utils import get_logger

log = get_logger(__name__)

MAX_GOALS = 8


@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    expected_home_goals: float
    expected_away_goals: float
    score_grid: np.ndarray = field(repr=False)
    most_likely_score: tuple[int, int] = (0, 0)
    home_key_players: list[dict] = field(default_factory=list)
    away_key_players: list[dict] = field(default_factory=list)

    @property
    def p_home_advance(self) -> float:
        return self.p_home_win + 0.5 * self.p_draw

    def as_dict(self) -> dict:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "p_home_win": self.p_home_win,
            "p_draw": self.p_draw,
            "p_away_win": self.p_away_win,
            "expected_home_goals": self.expected_home_goals,
            "expected_away_goals": self.expected_away_goals,
            "most_likely_score": self.most_likely_score,
            "home_key_players": self.home_key_players,
            "away_key_players": self.away_key_players,
        }


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[object, object, list[str], pd.Series, pd.DataFrame]:
    if not RESULT_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained models not found — run `python -m scripts.03_train_model`."
        )
    result_model = joblib.load(RESULT_MODEL_FILE)
    goals_model = joblib.load(GOALS_MODEL_FILE)
    meta = joblib.load(FEATURE_META_FILE)
    feature_columns = meta["feature_columns"]
    ratings = load_team_ratings()
    features_df = load_features()
    return result_model, goals_model, feature_columns, ratings, features_df


@lru_cache(maxsize=1)
def _squad_strength_dict() -> dict[str, dict[str, float]]:
    """Cache squad strength as a dict-of-dicts keyed by country.

    Bumping the cache: call ``refresh_squad_strength_cache()``.
    """
    df = load_squad_strength()
    return {row["country"]: row.to_dict() for _, row in df.iterrows()}


# Cache invalidation: when the user edits player data via the UI, the
# squad-strength file changes. We expose a function to refresh that.
def refresh_squad_strength_cache() -> None:
    """Force the predictor to re-read squad-strength data on the next call.

    Called after the UI saves overrides or availability changes.
    """
    from src.players.strength import build_squad_strength
    build_squad_strength(save=True)
    _squad_strength_dict.cache_clear()
    _key_players.cache_clear()


@lru_cache(maxsize=512)
def _latest_form_cached(team: str) -> tuple[float, float, float, int]:
    _, _, _, _, features_df = _load_artifacts()
    home_rows = features_df[features_df["home_team"] == team]
    away_rows = features_df[features_df["away_team"] == team]
    if home_rows.empty and away_rows.empty:
        return (1.0, 1.0, 1.0, 0)

    pieces = []
    if not home_rows.empty:
        pieces.append(home_rows.assign(
            form_points=home_rows["home_form_points"],
            form_gs=home_rows["home_form_gs"],
            form_gc=home_rows["home_form_gc"],
            matches_played=home_rows["home_matches_played"],
        )[["date", "form_points", "form_gs", "form_gc", "matches_played"]])
    if not away_rows.empty:
        pieces.append(away_rows.assign(
            form_points=away_rows["away_form_points"],
            form_gs=away_rows["away_form_gs"],
            form_gc=away_rows["away_form_gc"],
            matches_played=away_rows["away_matches_played"],
        )[["date", "form_points", "form_gs", "form_gc", "matches_played"]])
    combined = pd.concat(pieces).sort_values("date")
    last = combined.iloc[-1]
    return (
        float(last["form_points"]),
        float(last["form_gs"]),
        float(last["form_gc"]),
        int(last["matches_played"]),
    )


def _latest_form(features_df: pd.DataFrame, team: str) -> dict[str, float]:
    pts, gs, gc, mp = _latest_form_cached(team)
    return {
        "form_points": pts,
        "form_gs": gs,
        "form_gc": gc,
        "matches_played": mp,
    }


@lru_cache(maxsize=2048)
def _h2h_winrate_cached(team_a: str, team_b: str) -> float:
    _, _, _, _, features_df = _load_artifacts()
    mask = (
        ((features_df["home_team"] == team_a) & (features_df["away_team"] == team_b))
        | ((features_df["home_team"] == team_b) & (features_df["away_team"] == team_a))
    )
    rel = features_df.loc[mask].sort_values("date").tail(5)
    if rel.empty:
        return 0.5
    wins_a = 0
    for row in rel.itertuples(index=False):
        if row.home_team == team_a and row.home_score > row.away_score:
            wins_a += 1
        elif row.away_team == team_a and row.away_score > row.home_score:
            wins_a += 1
    return wins_a / len(rel)


def _h2h_winrate(features_df: pd.DataFrame, team_a: str, team_b: str) -> float:
    return _h2h_winrate_cached(team_a, team_b)


def _squad_features_for(country: str) -> dict[str, float]:
    """Pull live squad-strength stats for a country (cached)."""
    strength = _squad_strength_dict()
    row = strength.get(country)
    if row is None:
        return {
            "squad_strength": SQUAD_STRENGTH_CENTER,
            "attack": SQUAD_STRENGTH_CENTER,
            "midfield": SQUAD_STRENGTH_CENTER,
            "defence": SQUAD_STRENGTH_CENTER,
            "gk": SQUAD_STRENGTH_CENTER,
            "n_unavailable": 0.0,
        }
    return {
        "squad_strength": float(row["squad_strength"]),
        "attack": float(row["attack"]),
        "midfield": float(row["midfield"]),
        "defence": float(row["defence"]),
        "gk": float(row["gk"]),
        "n_unavailable": float(row["n_unavailable"]),
    }


@lru_cache(maxsize=64)
def _key_players(country: str, k: int = 5) -> tuple:
    """Return the top-k effective-rated players for a team (cached as tuple)."""
    players = load_effective_players()
    sub = players[players["country"] == country].sort_values(
        "effective_rating", ascending=False
    ).head(k)
    return tuple(
        sub[["name", "position", "rating", "effective_rating",
             "availability", "club"]].to_dict("records")
    )


def build_match_features(
    home_team: str,
    away_team: str,
    *,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
) -> pd.DataFrame:
    _, _, feature_columns, ratings, features_df = _load_artifacts()

    home_elo = float(ratings.get(home_team, 1500.0))
    away_elo = float(ratings.get(away_team, 1500.0))
    home_form = _latest_form(features_df, home_team)
    away_form = _latest_form(features_df, away_team)
    h2h = _h2h_winrate(features_df, home_team, away_team)

    h_sq = _squad_features_for(home_team)
    a_sq = _squad_features_for(away_team)

    is_wc = int("World Cup" in tournament)
    is_friendly = int(tournament == "Friendly")
    is_continental = int(tournament in {
        "UEFA Euro", "Copa América", "Africa Cup of Nations",
        "AFC Asian Cup", "CONCACAF Championship",
    })

    row = {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_form_points": home_form["form_points"],
        "away_form_points": away_form["form_points"],
        "home_form_gs": home_form["form_gs"],
        "away_form_gs": away_form["form_gs"],
        "home_form_gc": home_form["form_gc"],
        "away_form_gc": away_form["form_gc"],
        "form_points_diff": home_form["form_points"] - away_form["form_points"],
        "form_gs_diff": home_form["form_gs"] - away_form["form_gs"],
        "form_gc_diff": away_form["form_gc"] - home_form["form_gc"],
        "h2h_home_winrate": h2h,
        "home_matches_played": home_form["matches_played"],
        "away_matches_played": away_form["matches_played"],
        "is_neutral": int(neutral),
        "is_friendly": is_friendly,
        "is_world_cup": is_wc,
        "is_continental": is_continental,
        "tournament_weight": TOURNAMENT_WEIGHTS.get(
            tournament, TOURNAMENT_WEIGHT_DEFAULT
        ),
        "home_is_host": int(home_team in HOST_COUNTRIES),
        "away_is_host": int(away_team in HOST_COUNTRIES),
        # Squad
        "home_squad_strength": h_sq["squad_strength"],
        "away_squad_strength": a_sq["squad_strength"],
        "squad_strength_diff": h_sq["squad_strength"] - a_sq["squad_strength"],
        "home_attack": h_sq["attack"],
        "away_attack": a_sq["attack"],
        "home_defence": h_sq["defence"],
        "away_defence": a_sq["defence"],
        "home_midfield": h_sq["midfield"],
        "away_midfield": a_sq["midfield"],
        "home_gk": h_sq["gk"],
        "away_gk": a_sq["gk"],
        "attack_vs_defence_diff": h_sq["attack"] - a_sq["defence"],
        "away_attack_vs_home_defence_diff": a_sq["attack"] - h_sq["defence"],
        "home_n_unavailable": h_sq["n_unavailable"],
        "away_n_unavailable": a_sq["n_unavailable"],
    }
    return pd.DataFrame([row])[feature_columns]


def _poisson_score_grid(lam_home: float, lam_away: float) -> np.ndarray:
    home_probs = poisson.pmf(np.arange(MAX_GOALS + 1), lam_home)
    away_probs = poisson.pmf(np.arange(MAX_GOALS + 1), lam_away)
    return np.outer(home_probs, away_probs)


def predict_match(
    home_team: str,
    away_team: str,
    *,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
    include_key_players: bool = True,
) -> MatchPrediction:
    result_model, goals_model, _, _, _ = _load_artifacts()
    X = build_match_features(home_team, away_team,
                             tournament=tournament, neutral=neutral)

    proba = result_model.predict_proba(X.values)[0]
    lam_home, lam_away = goals_model.predict(X.values)[0]
    lam_home = max(lam_home, 0.05)
    lam_away = max(lam_away, 0.05)

    grid = _poisson_score_grid(lam_home, lam_away)
    most_likely = np.unravel_index(np.argmax(grid), grid.shape)

    home_key, away_key = [], []
    if include_key_players:
        try:
            home_key = _key_players(home_team)
            away_key = _key_players(away_team)
        except Exception as e:  # noqa: BLE001
            log.warning("Key-player lookup failed: %s", e)

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        p_home_win=float(proba[0]),
        p_draw=float(proba[1]),
        p_away_win=float(proba[2]),
        expected_home_goals=float(lam_home),
        expected_away_goals=float(lam_away),
        score_grid=grid,
        most_likely_score=(int(most_likely[0]), int(most_likely[1])),
        home_key_players=home_key,
        away_key_players=away_key,
    )


def list_known_teams() -> list[str]:
    _, _, _, ratings, _ = _load_artifacts()
    return sorted(ratings.index.tolist())


def team_rating(team: str) -> float | None:
    _, _, _, ratings, _ = _load_artifacts()
    return float(ratings[team]) if team in ratings.index else None
