"""End-to-end feature builder.

Takes raw match data and produces the modelling-ready DataFrame the
training pipeline consumes. Now includes squad-strength features computed
from player ratings, so the model actually weights "this team has Messi"
into its predictions.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    HOST_COUNTRIES,
    PROCESSED_FEATURES_FILE,
    SQUAD_STRENGTH_CENTER,
    SQUAD_STRENGTH_SCALE,
    TEAM_RATINGS_FILE,
)
from src.data.loaders import load_results
from src.features.elo import compute_elo
from src.features.form import add_form_features
from src.utils import get_logger, timer

log = get_logger(__name__)


FEATURE_COLUMNS: list[str] = [
    # Team strength (ELO)
    "home_elo",
    "away_elo",
    "elo_diff",
    # Recent form
    "home_form_points",
    "away_form_points",
    "home_form_gs",
    "away_form_gs",
    "home_form_gc",
    "away_form_gc",
    "form_points_diff",
    "form_gs_diff",
    "form_gc_diff",
    # H2H
    "h2h_home_winrate",
    "home_matches_played",
    "away_matches_played",
    # Context
    "is_neutral",
    "is_friendly",
    "is_world_cup",
    "is_continental",
    "tournament_weight",
    "home_is_host",
    "away_is_host",
    # Squad strength (NEW — derived from player ratings)
    "home_squad_strength",
    "away_squad_strength",
    "squad_strength_diff",
    "home_attack",
    "away_attack",
    "home_defence",
    "away_defence",
    "home_midfield",
    "away_midfield",
    "home_gk",
    "away_gk",
    "attack_vs_defence_diff",  # home attack - away defence
    "away_attack_vs_home_defence_diff",  # away attack - home defence
    "home_n_unavailable",
    "away_n_unavailable",
]

_CONTINENTAL = {
    "UEFA Euro", "Copa América", "Africa Cup of Nations",
    "AFC Asian Cup", "CONCACAF Championship",
}


def _add_match_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_neutral"] = df["neutral"].astype(int)
    df["is_friendly"] = (df["tournament"] == "Friendly").astype(int)
    df["is_world_cup"] = df["tournament"].str.contains(
        "World Cup", case=False, na=False
    ).astype(int)
    df["is_continental"] = df["tournament"].isin(_CONTINENTAL).astype(int)
    from src.config import TOURNAMENT_WEIGHTS, TOURNAMENT_WEIGHT_DEFAULT
    df["tournament_weight"] = df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(
        TOURNAMENT_WEIGHT_DEFAULT
    )
    df["home_is_host"] = df["home_team"].isin(HOST_COUNTRIES).astype(int)
    df["away_is_host"] = df["away_team"].isin(HOST_COUNTRIES).astype(int)

    diff = df["home_score"] - df["away_score"]
    df["result"] = 1
    df.loc[diff > 0, "result"] = 0
    df.loc[diff < 0, "result"] = 2
    return df


def _add_squad_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join in squad-strength features from the players module.

    For historical matches we don't have era-specific squad data, so we
    use *current* squad strength as a proxy. This is fine because:
      1. The supervised model only trains on matches from 1990+, where
         national team strength has been reasonably consistent within an
         era.
      2. The dominant signal for old matches is still ELO + form. Squad
         strength is most useful for *new* matches (the World Cup we're
         actually predicting).

    Anything we don't have squad data for (small nations, defunct teams)
    falls back to the central value (treated as average).
    """
    from src.players.strength import load_squad_strength

    strength = load_squad_strength().set_index("country")

    def _col(team_col: str, src_col: str, default: float) -> pd.Series:
        return df[team_col].map(strength[src_col].to_dict()).fillna(default)

    df = df.copy()
    df["home_squad_strength"] = _col("home_team", "squad_strength", SQUAD_STRENGTH_CENTER)
    df["away_squad_strength"] = _col("away_team", "squad_strength", SQUAD_STRENGTH_CENTER)
    df["squad_strength_diff"] = df["home_squad_strength"] - df["away_squad_strength"]

    df["home_attack"] = _col("home_team", "attack", SQUAD_STRENGTH_CENTER)
    df["away_attack"] = _col("away_team", "attack", SQUAD_STRENGTH_CENTER)
    df["home_defence"] = _col("home_team", "defence", SQUAD_STRENGTH_CENTER)
    df["away_defence"] = _col("away_team", "defence", SQUAD_STRENGTH_CENTER)
    df["home_midfield"] = _col("home_team", "midfield", SQUAD_STRENGTH_CENTER)
    df["away_midfield"] = _col("away_team", "midfield", SQUAD_STRENGTH_CENTER)
    df["home_gk"] = _col("home_team", "gk", SQUAD_STRENGTH_CENTER)
    df["away_gk"] = _col("away_team", "gk", SQUAD_STRENGTH_CENTER)

    # Tactical match-ups: home attack vs away defence and vice versa
    df["attack_vs_defence_diff"] = df["home_attack"] - df["away_defence"]
    df["away_attack_vs_home_defence_diff"] = df["away_attack"] - df["home_defence"]

    df["home_n_unavailable"] = _col("home_team", "n_unavailable", 0).astype(int)
    df["away_n_unavailable"] = _col("away_team", "n_unavailable", 0).astype(int)

    return df


def build_features(save: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full features DataFrame and persist it."""
    with timer("load match data"):
        df = load_results()

    with timer("compute ELO"):
        df, elo_table = compute_elo(df)

    with timer("rolling form features"):
        df = add_form_features(df)

    with timer("match context"):
        df = _add_match_context(df)

    with timer("squad strength features"):
        df = _add_squad_features(df)

    if save:
        df.to_parquet(PROCESSED_FEATURES_FILE, index=False)
        elo_table.as_series().to_frame().to_parquet(TEAM_RATINGS_FILE)
        log.info("Saved features → %s", PROCESSED_FEATURES_FILE)
        log.info("Saved team ratings → %s", TEAM_RATINGS_FILE)

    return df, elo_table.as_series()


def load_features() -> pd.DataFrame:
    if not PROCESSED_FEATURES_FILE.exists():
        raise FileNotFoundError(
            "Processed features missing. Run "
            "`python -m scripts.02_build_features` first."
        )
    return pd.read_parquet(PROCESSED_FEATURES_FILE)


def load_team_ratings() -> pd.Series:
    if not TEAM_RATINGS_FILE.exists():
        raise FileNotFoundError("Team ratings missing — build features first.")
    return pd.read_parquet(TEAM_RATINGS_FILE).iloc[:, 0]


if __name__ == "__main__":
    build_features()
