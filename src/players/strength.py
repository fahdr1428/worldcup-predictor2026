"""Squad-strength feature computation.

Turns a DataFrame of players into a single number per country: the
aggregated "squad rating" used as a model feature. Also surfaces
per-position strength (attack / midfield / defence / GK) so the UI can
show which area is weak.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    BENCH_WEIGHT,
    REPLACEMENT_DISCOUNT,
    SQUAD_SIZE_TOP,
    SQUAD_SIZE_TOTAL,
    SQUAD_STRENGTH_CENTER,
    SQUAD_STRENGTH_FILE,
    TOP_XI_DECAY,
    TOP_XI_WEIGHT,
    WORLD_CUP_TEAMS,
)
from src.players.loader import load_effective_players
from src.utils import get_logger

log = get_logger(__name__)


def _weighted_top_xi(sorted_ratings: pd.Series) -> float:
    """Top 11 with geometric decay so the star players matter more."""
    if sorted_ratings.empty:
        return SQUAD_STRENGTH_CENTER
    top = sorted_ratings.head(SQUAD_SIZE_TOP).reset_index(drop=True)
    weights = (TOP_XI_DECAY ** top.index.to_series().astype(float))
    weights = weights / weights.sum()
    return float((top * weights).sum())


def _bench_mean(sorted_ratings: pd.Series) -> float:
    """Average of squad positions 12-23, modelling depth."""
    bench = sorted_ratings.iloc[SQUAD_SIZE_TOP:SQUAD_SIZE_TOTAL]
    if bench.empty:
        # Pad with replacements at REPLACEMENT_DISCOUNT × last starter
        if len(sorted_ratings) > 0:
            return float(sorted_ratings.iloc[-1] * REPLACEMENT_DISCOUNT)
        return SQUAD_STRENGTH_CENTER * REPLACEMENT_DISCOUNT
    return float(bench.mean())


def compute_team_strength(players: pd.DataFrame) -> pd.DataFrame:
    """Compute one strength row per country.

    Players who are 'out' or 'suspended' (effective_rating == 0) are
    excluded — their slot is treated as filled by an implicit backup we
    can't enumerate. A small flat penalty is applied per missing player
    so unavailability still costs something, but losing a star doesn't
    destroy a 12-strong squad.

    Doubtful players stay in the active pool at their reduced rating.

    Columns: country, squad_strength, attack, midfield, defence, gk,
             n_players, n_unavailable
    """
    rows = []
    for country, group in players.groupby("country"):
        # Drop players whose effective rating is zero — they're not
        # influencing the match in any form.
        active = group[group["effective_rating"] > 0].copy()
        n_out = int((group["effective_rating"] <= 0).sum())

        sorted_ratings = active.sort_values(
            "effective_rating", ascending=False
        )["effective_rating"]
        overall = (
            TOP_XI_WEIGHT * _weighted_top_xi(sorted_ratings)
            + BENCH_WEIGHT * _bench_mean(sorted_ratings)
        )

        # Each fully-out player costs ~0.4 strength points, capped at 3.0
        # so a team with 5 reserves out doesn't get wiped.
        overall -= min(n_out * 0.4, 3.0)

        def _pos_mean(pos: str) -> float:
            r = active.loc[active["position"] == pos, "effective_rating"]
            return float(r.mean()) if not r.empty else SQUAD_STRENGTH_CENTER

        n_unavail = int((group["availability"] != "available").sum())

        rows.append({
            "country": country,
            "squad_strength": round(overall, 2),
            "attack": round(_pos_mean("FWD"), 2),
            "midfield": round(_pos_mean("MID"), 2),
            "defence": round(_pos_mean("DEF"), 2),
            "gk": round(_pos_mean("GK"), 2),
            "n_players": int(len(group)),
            "n_unavailable": n_unavail,
        })

    df = pd.DataFrame(rows).sort_values("squad_strength", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


def build_squad_strength(save: bool = True) -> pd.DataFrame:
    players = load_effective_players()
    df = compute_team_strength(players)

    # Make sure every WC team gets a row, even if seed missed them
    missing = set(WORLD_CUP_TEAMS) - set(df["country"])
    if missing:
        log.warning("No squad data for %d teams (%s) — using centre value.",
                    len(missing), sorted(missing))
        pad_rows = pd.DataFrame([
            {"country": t, "squad_strength": SQUAD_STRENGTH_CENTER,
             "attack": SQUAD_STRENGTH_CENTER, "midfield": SQUAD_STRENGTH_CENTER,
             "defence": SQUAD_STRENGTH_CENTER, "gk": SQUAD_STRENGTH_CENTER,
             "n_players": 0, "n_unavailable": 0}
            for t in missing
        ])
        df = pd.concat([df, pad_rows], ignore_index=True)

    if save:
        df.to_csv(SQUAD_STRENGTH_FILE, index=False)
        log.info("Saved squad strength → %s", SQUAD_STRENGTH_FILE)
    return df


def load_squad_strength() -> pd.DataFrame:
    """Load cached squad strength, building it if missing."""
    if not SQUAD_STRENGTH_FILE.exists():
        return build_squad_strength(save=True)
    return pd.read_csv(SQUAD_STRENGTH_FILE)


def get_team_strength(country: str) -> float:
    """Convenience: look up one team's overall squad strength."""
    df = load_squad_strength()
    row = df.loc[df["country"] == country]
    if row.empty:
        return SQUAD_STRENGTH_CENTER
    return float(row.iloc[0]["squad_strength"])


if __name__ == "__main__":
    df = build_squad_strength(save=True)
    print(df.head(20).to_string(index=False))
