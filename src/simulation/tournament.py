"""Monte Carlo simulation of the 2026 FIFA World Cup.

48-team format → 12 groups of 4 → top 2 + 8 best third-placed → 32-team
knockout bracket → champion.

We simulate the tournament thousands of times, sampling each match from
the trained model's probabilities. The result is a probability for every
team to reach each round + win the trophy.

Performance notes:
  - We pre-compute model predictions for every plausible matchup ONCE,
    in a single batched call. This is the dominant cost (~10-30 sec).
  - The actual simulations are then pure numpy/python and run at
    ~10K trials per 10-30 seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import DEFAULT_SIMULATIONS, HOST_COUNTRIES, WORLDCUP_2026_GROUPS
from src.utils import get_logger

log = get_logger(__name__)

ROUNDS = ["group", "r32", "r16", "qf", "sf", "final", "champion"]


@dataclass
class CachedMatch:
    """Pre-computed match distribution used during simulation."""
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    lam_home: float
    lam_away: float


class MatchCache:
    """Avoid re-running the model for the same matchup repeatedly."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, bool], CachedMatch] = {}

    def get(self, home: str, away: str, *, neutral: bool = True) -> CachedMatch:
        key = (home, away, neutral)
        if key not in self._cache:
            from src.models.predict import predict_match
            p = predict_match(home, away, neutral=neutral,
                              include_key_players=False)
            self._cache[key] = CachedMatch(
                home=home, away=away,
                p_home=p.p_home_win, p_draw=p.p_draw, p_away=p.p_away_win,
                lam_home=p.expected_home_goals, lam_away=p.expected_away_goals,
            )
        return self._cache[key]

    def has(self, home: str, away: str, *, neutral: bool = True) -> bool:
        return (home, away, neutral) in self._cache

    def put(self, m: CachedMatch, *, neutral: bool = True) -> None:
        self._cache[(m.home, m.away, neutral)] = m


def _is_neutral(home: str, away: str) -> bool:
    """A match is neutral unless one of the teams is a host playing at home."""
    return not (home in HOST_COUNTRIES or away in HOST_COUNTRIES)


def _batch_precompute_group_matches(cache: MatchCache) -> None:
    """Pre-compute every plausible matchup the simulator could need.

    We don't know which teams will face which in knockouts, so we
    pre-compute all 1128 unique team pairings (48*47/2) in one batched
    model call. This is the single most important performance fix —
    eliminates slow per-call predictions during knockout rounds.
    """
    from src.models.predict import build_match_features, _load_artifacts

    all_teams = sorted({t for teams in WORLDCUP_2026_GROUPS.values() for t in teams})

    matchups = []
    feature_rows = []
    # For each unordered pair, build features in BOTH directions (since
    # home/away ordering affects the prediction). 48*47 = 2256 matchups.
    for i, a in enumerate(all_teams):
        for b in all_teams[i + 1:]:
            for home, away in [(a, b), (b, a)]:
                neutral = _is_neutral(home, away)
                matchups.append((home, away, neutral))
                X = build_match_features(home, away, neutral=neutral)
                feature_rows.append(X.iloc[0])

    if not feature_rows:
        return

    big_X = pd.DataFrame(feature_rows)
    result_model, goals_model, _, _, _ = _load_artifacts()

    probas = result_model.predict_proba(big_X.values)
    goals = goals_model.predict(big_X.values)

    for i, (home, away, neutral) in enumerate(matchups):
        cache.put(
            CachedMatch(
                home=home, away=away,
                p_home=float(probas[i][0]),
                p_draw=float(probas[i][1]),
                p_away=float(probas[i][2]),
                lam_home=max(float(goals[i][0]), 0.05),
                lam_away=max(float(goals[i][1]), 0.05),
            ),
            neutral=neutral,
        )


def _sample_group_match(cache: MatchCache, home: str, away: str,
                        rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Return (home_pts, away_pts, home_goals, away_goals)."""
    neutral = _is_neutral(home, away)
    m = cache.get(home, away, neutral=neutral)
    hg = int(rng.poisson(m.lam_home))
    ag = int(rng.poisson(m.lam_away))
    if hg > ag:
        return 3, 0, hg, ag
    if hg < ag:
        return 0, 3, hg, ag
    return 1, 1, hg, ag


def _sample_knockout_winner(cache: MatchCache, home: str, away: str,
                            rng: np.random.Generator) -> str:
    """Knockout — must produce a winner. Draws split 50/50 (PK shootout)."""
    neutral = _is_neutral(home, away)
    m = cache.get(home, away, neutral=neutral)
    r = rng.random()
    if r < m.p_home:
        return home
    if r < m.p_home + m.p_away:
        return away
    return home if rng.random() < 0.5 else away


def _simulate_group(cache: MatchCache, teams: list[str],
                    rng: np.random.Generator) -> list[tuple[str, int, int]]:
    """Round-robin. Returns list of (team, points, goal_diff) sorted."""
    pts = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            pa, pb, ga, gb = _sample_group_match(cache, a, b, rng)
            pts[a] += pa
            pts[b] += pb
            gd[a] += ga - gb
            gd[b] += gb - ga
            gf[a] += ga
            gf[b] += gb
    standings = sorted(
        teams,
        key=lambda t: (pts[t], gd[t], gf[t], rng.random()),
        reverse=True,
    )
    return [(t, pts[t], gd[t]) for t in standings]


def _build_knockout_bracket(group_results) -> list[str]:
    """Reduce 12 groups → 32 advancing teams (1st + 2nd + top 8 of 3rd)."""
    firsts, seconds, thirds = [], [], []
    for group, standings in group_results.items():
        firsts.append(standings[0][0])
        seconds.append(standings[1][0])
        thirds.append((standings[2][0], standings[2][1], standings[2][2]))
    thirds.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_thirds = [t[0] for t in thirds[:8]]
    return firsts + seconds + best_thirds


def _run_knockout(cache: MatchCache, bracket: list[str],
                  rng: np.random.Generator,
                  results: dict[str, dict[str, int]]) -> str:
    round_names = ["r16", "qf", "sf", "final"]
    round_idx = 0
    current = list(bracket)
    for t in current:
        results[t]["r32"] += 1
    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), 2):
            winner = _sample_knockout_winner(cache, current[i], current[i + 1], rng)
            next_round.append(winner)
        current = next_round
        if round_idx < len(round_names):
            for t in current:
                results[t][round_names[round_idx]] += 1
        round_idx += 1
    champion = current[0]
    results[champion]["champion"] += 1
    return champion


def simulate_tournament(
    n_simulations: int = DEFAULT_SIMULATIONS,
    seed: int = 42,
    progress: bool = True,
) -> pd.DataFrame:
    """Run the full Monte Carlo simulation."""
    rng = np.random.default_rng(seed)
    cache = MatchCache()

    log.info("Pre-computing model predictions for group-stage matches …")
    _batch_precompute_group_matches(cache)
    log.info("  done — %d unique matchups cached", len(cache._cache))

    all_teams = [t for teams in WORLDCUP_2026_GROUPS.values() for t in teams]
    results = {t: {r: 0 for r in ROUNDS} for t in all_teams}

    iterator = range(n_simulations)
    if progress:
        iterator = tqdm(iterator, desc="Simulating", unit="trial")

    for _ in iterator:
        group_results = {}
        for group, teams in WORLDCUP_2026_GROUPS.items():
            standings = _simulate_group(cache, teams, rng)
            group_results[group] = standings
            for team, _, _ in standings[:2]:
                results[team]["group"] += 1

        bracket = _build_knockout_bracket(group_results)
        for t in bracket:
            if results[t]["group"] == 0:
                results[t]["group"] += 1

        _run_knockout(cache, bracket, rng, results)

    rows = []
    for team, counts in results.items():
        row = {"team": team}
        for r in ROUNDS:
            row[f"p_{r}"] = counts[r] / n_simulations
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("p_champion", ascending=False)
    out.reset_index(drop=True, inplace=True)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-simulations", type=int,
                        default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = simulate_tournament(args.n_simulations, args.seed)
    print(df.head(20).to_string(index=False))
