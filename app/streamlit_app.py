"""Streamlit front-end for the World Cup predictor with player ratings.

Five tabs:
    1. Match predictor — pick teams, see W/D/L + score grid + KEY PLAYERS
    2. Team explorer  — ELO leaderboard, recent matches
    3. Squad ratings  — browse all 48 squads, see strength per position
    4. Players & overrides — edit ratings, mark injuries/suspensions
    5. Tournament simulator — Monte Carlo with squad-aware predictions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    AVAILABILITY_FILE,
    DEFAULT_SIMULATIONS,
    PLAYER_OVERRIDES_FILE,
    RESULT_MODEL_FILE,
    WORLD_CUP_TEAMS,
    WORLDCUP_2026_GROUPS,
)
from src.features.builder import load_features
from src.models.predict import (
    list_known_teams,
    predict_match,
    refresh_squad_strength_cache,
    team_rating,
)
from src.players.loader import (
    load_availability,
    load_effective_players,
    load_overrides,
    save_availability,
    save_overrides,
)
from src.players.schema import AvailabilityRecord
from src.players.strength import build_squad_strength, load_squad_strength

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

if not RESULT_MODEL_FILE.exists():
    st.title("⚽ World Cup 2026 Predictor")
    st.warning(
        "**Models not trained yet.** Run the pipeline first:\n\n"
        "```bash\npython -m scripts.run_pipeline\n```\n\n"
        "Then refresh this page."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def cached_teams() -> list[str]:
    return list_known_teams()


@st.cache_data
def cached_features() -> pd.DataFrame:
    return load_features()


@st.cache_data(show_spinner=False)
def cached_prediction(home: str, away: str, neutral: bool, cache_buster: int):
    """cache_buster lets us bust this cache when overrides change."""
    pred = predict_match(home, away, neutral=neutral)
    return pred.as_dict(), pred.score_grid


@st.cache_data(show_spinner="Simulating tournament — this can take a minute on the free tier …")
def cached_simulation(n: int, seed: int, cache_buster: int) -> pd.DataFrame:
    from src.simulation.tournament import simulate_tournament
    # Fast path: if the user is using the default (5K sims, seed 42) AND
    # has not edited any player data, return the pre-shipped result.
    if n == DEFAULT_SIMULATIONS and seed == 42:
        from src.config import DEFAULT_SIM_CACHE_FILE
        if DEFAULT_SIM_CACHE_FILE.exists():
            n_overrides = len(load_overrides())
            n_avail_changes = sum(
                1 for r in load_availability().values() if r.status != "available"
            )
            if n_overrides == 0 and n_avail_changes == 0:
                try:
                    return pd.read_parquet(DEFAULT_SIM_CACHE_FILE)
                except Exception:
                    pass  # fall through to live simulation
    return simulate_tournament(n_simulations=n, seed=seed, progress=False)


def _cache_buster() -> int:
    """Returns a value that changes when player data changes, so we can
    invalidate prediction caches by passing it in."""
    parts = []
    for f in (PLAYER_OVERRIDES_FILE, AVAILABILITY_FILE):
        parts.append(str(f.stat().st_mtime) if f.exists() else "0")
    return hash(tuple(parts)) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚽ Predictor")
st.sidebar.markdown(
    "AI match predictor for the 2026 FIFA World Cup, trained on every "
    "international match since 1872 (~45K games) plus live squad ratings "
    "for all 48 nations (~576 players)."
)
st.sidebar.markdown("---")
n_active_overrides = len(load_overrides())
n_active_avail = sum(1 for r in load_availability().values()
                     if r.status != "available")
st.sidebar.metric("Rating overrides active", n_active_overrides)
st.sidebar.metric("Players unavailable", n_active_avail)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with XGBoost · Squad strength weighted by position · "
    "Poisson goal models · Monte Carlo simulation."
)

teams = cached_teams()
features_df = cached_features()
buster = _cache_buster()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_match, tab_team, tab_squads, tab_players, tab_sim = st.tabs([
    "🎯 Match Predictor",
    "📊 Team Explorer",
    "🛡️ Squad Ratings",
    "👤 Players & Overrides",
    "🏆 Tournament Simulator",
])

# ===========================================================================
# Tab 1 — Match Predictor
# ===========================================================================
with tab_match:
    st.header("Match predictor")
    st.caption("Now squad-aware: predictions weight current player ratings, "
               "injuries, and suspensions.")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        default_home = "Argentina" if "Argentina" in teams else teams[0]
        home = st.selectbox("Home team", teams, index=teams.index(default_home))
    with col_b:
        default_away = "France" if "France" in teams else teams[1]
        away = st.selectbox("Away team", teams, index=teams.index(default_away))
    with col_c:
        neutral = st.checkbox("Neutral venue", value=True)

    if home == away:
        st.error("Pick two different teams.")
    else:
        pred_dict, score_grid = cached_prediction(home, away, neutral, buster)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{home} win", f"{pred_dict['p_home_win']*100:.1f}%")
        m2.metric("Draw", f"{pred_dict['p_draw']*100:.1f}%")
        m3.metric(f"{away} win", f"{pred_dict['p_away_win']*100:.1f}%")
        mls = pred_dict["most_likely_score"]
        m4.metric("Most likely score", f"{mls[0]}–{mls[1]}")

        prob_df = pd.DataFrame({
            "outcome": [f"{home} win", "Draw", f"{away} win"],
            "probability": [pred_dict["p_home_win"], pred_dict["p_draw"],
                            pred_dict["p_away_win"]],
        })
        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("probability:Q", axis=alt.Axis(format=".0%"),
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("outcome:N", sort=None),
                color=alt.Color("outcome:N", legend=None),
                tooltip=["outcome", alt.Tooltip("probability:Q", format=".1%")],
            )
            .properties(height=180)
        )
        st.altair_chart(chart, use_container_width=True)

        # Side-by-side: key players + heatmap
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Key players")
            st.markdown(f"**{home}**")
            for p in pred_dict.get("home_key_players", []):
                badge = "" if p["availability"] == "available" else f" ⚠️ *{p['availability']}*"
                st.markdown(
                    f"- {p['name']} ({p['position']}) — "
                    f"**{p['effective_rating']:.0f}**{badge}"
                )
            st.markdown(f"**{away}**")
            for p in pred_dict.get("away_key_players", []):
                badge = "" if p["availability"] == "available" else f" ⚠️ *{p['availability']}*"
                st.markdown(
                    f"- {p['name']} ({p['position']}) — "
                    f"**{p['effective_rating']:.0f}**{badge}"
                )

        with col_right:
            st.subheader("Strength comparison")
            strength = load_squad_strength().set_index("country")
            if home in strength.index and away in strength.index:
                comp_df = pd.DataFrame({
                    "category": ["Squad", "Attack", "Midfield", "Defence", "GK"],
                    home: [
                        strength.loc[home, "squad_strength"],
                        strength.loc[home, "attack"],
                        strength.loc[home, "midfield"],
                        strength.loc[home, "defence"],
                        strength.loc[home, "gk"],
                    ],
                    away: [
                        strength.loc[away, "squad_strength"],
                        strength.loc[away, "attack"],
                        strength.loc[away, "midfield"],
                        strength.loc[away, "defence"],
                        strength.loc[away, "gk"],
                    ],
                })
                melted = comp_df.melt(id_vars="category", var_name="team",
                                       value_name="rating")
                bars = (
                    alt.Chart(melted)
                    .mark_bar()
                    .encode(
                        x=alt.X("rating:Q", scale=alt.Scale(domain=[60, 95])),
                        y=alt.Y("category:N", sort=None),
                        color="team:N",
                        yOffset="team:N",
                        tooltip=["team", "category",
                                 alt.Tooltip("rating:Q", format=".1f")],
                    )
                    .properties(height=240)
                )
                st.altair_chart(bars, use_container_width=True)
            else:
                st.info("Squad data unavailable for one of the teams.")

        st.subheader("Score-line probability heatmap")
        grid_df = (
            pd.DataFrame(score_grid)
            .reset_index()
            .melt(id_vars="index", var_name="away_goals", value_name="prob")
            .rename(columns={"index": "home_goals"})
        )
        heatmap = (
            alt.Chart(grid_df)
            .mark_rect()
            .encode(
                x=alt.X("away_goals:O", title=f"{away} goals"),
                y=alt.Y("home_goals:O", title=f"{home} goals", sort="descending"),
                color=alt.Color("prob:Q", scale=alt.Scale(scheme="greens"),
                                legend=alt.Legend(format=".1%", title="Probability")),
                tooltip=[
                    alt.Tooltip("home_goals:O", title=f"{home}"),
                    alt.Tooltip("away_goals:O", title=f"{away}"),
                    alt.Tooltip("prob:Q", format=".2%"),
                ],
            )
            .properties(height=380)
        )
        st.altair_chart(heatmap, use_container_width=True)

# ===========================================================================
# Tab 2 — Team Explorer
# ===========================================================================
with tab_team:
    st.header("Team explorer")
    default_t = "Brazil" if "Brazil" in teams else teams[0]
    team = st.selectbox("Team", teams, index=teams.index(default_t),
                        key="team_explorer_team")

    rating = team_rating(team)
    c1, c2 = st.columns(2)
    c1.metric(f"{team} ELO rating", f"{rating:.0f}" if rating else "n/a")
    try:
        squad_row = load_squad_strength().set_index("country").loc[team]
        c2.metric(f"{team} squad strength", f"{squad_row['squad_strength']:.1f}")
    except KeyError:
        c2.metric(f"{team} squad strength", "n/a")

    team_matches = features_df[
        (features_df["home_team"] == team) | (features_df["away_team"] == team)
    ].sort_values("date").tail(20).copy()

    if team_matches.empty:
        st.info("No historical matches for that team.")
    else:
        def _summarise(row):
            if row["home_team"] == team:
                return pd.Series({
                    "date": row["date"].date(),
                    "opponent": row["away_team"],
                    "venue": "home" if not row["neutral"] else "neutral",
                    "goals_for": row["home_score"],
                    "goals_against": row["away_score"],
                    "tournament": row["tournament"],
                })
            return pd.Series({
                "date": row["date"].date(),
                "opponent": row["home_team"],
                "venue": "away" if not row["neutral"] else "neutral",
                "goals_for": row["away_score"],
                "goals_against": row["home_score"],
                "tournament": row["tournament"],
            })

        display = team_matches.apply(_summarise, axis=1)
        display["result"] = np.where(
            display["goals_for"] > display["goals_against"], "W",
            np.where(display["goals_for"] < display["goals_against"], "L", "D"),
        )
        st.subheader("Last 20 matches")
        st.dataframe(display, hide_index=True, use_container_width=True)

        st.subheader("Goals timeline")
        goals_df = display.melt(
            id_vars=["date"],
            value_vars=["goals_for", "goals_against"],
            var_name="type", value_name="goals",
        )
        line = (
            alt.Chart(goals_df)
            .mark_line(point=True)
            .encode(
                x="date:T", y="goals:Q", color="type:N",
                tooltip=["date:T", "type:N", "goals:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(line, use_container_width=True)

    st.subheader("Top 25 nations (current ELO)")
    from src.features.builder import load_team_ratings
    ratings = load_team_ratings().sort_values(ascending=False).head(25)
    st.dataframe(
        ratings.rename("ELO").round(0).astype(int).reset_index().rename(
            columns={"index": "team"}
        ),
        hide_index=True, use_container_width=True,
    )

# ===========================================================================
# Tab 3 — Squad ratings (browse all 48 teams)
# ===========================================================================
with tab_squads:
    st.header("Squad ratings — all 48 World Cup nations")
    strength_df = load_squad_strength().copy()
    strength_df = strength_df.sort_values("squad_strength", ascending=False)

    st.caption("Squad strength = 70% top-XI (with rating decay) + 30% bench "
               "depth. Adjusted for injuries and suspensions.")

    st.dataframe(
        strength_df.round(1),
        hide_index=True, use_container_width=True,
    )

    st.subheader("Strength leaderboard")
    top = strength_df.head(20)
    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("squad_strength:Q", scale=alt.Scale(domain=[60, 90])),
            y=alt.Y("country:N", sort="-x"),
            color=alt.Color("squad_strength:Q",
                            scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["country", "squad_strength", "attack", "defence",
                     "n_unavailable"],
        )
        .properties(height=500)
    )
    st.altair_chart(chart, use_container_width=True)

# ===========================================================================
# Tab 4 — Players & overrides
# ===========================================================================
with tab_players:
    st.header("Players & overrides")
    st.caption(
        "Browse every player, adjust ratings up/down, and mark "
        "injuries/suspensions. Predictions update instantly."
    )

    players_df = load_effective_players()

    # Pick a country
    countries = sorted(players_df["country"].unique())
    default_c = "Argentina" if "Argentina" in countries else countries[0]
    country = st.selectbox("Team", countries,
                           index=countries.index(default_c),
                           key="players_country")

    team_players = (
        players_df[players_df["country"] == country]
        .sort_values("effective_rating", ascending=False)
        .reset_index(drop=True)
    )

    if team_players.empty:
        st.info("No player data for this team.")
    else:
        st.subheader(f"{country} squad ({len(team_players)} players)")

        # Top metrics for this team
        strength = load_squad_strength().set_index("country")
        if country in strength.index:
            row = strength.loc[country]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Squad strength", f"{row['squad_strength']:.1f}")
            c2.metric("Attack", f"{row['attack']:.1f}")
            c3.metric("Defence", f"{row['defence']:.1f}")
            c4.metric("Unavailable", int(row['n_unavailable']))

        # Editable table
        st.markdown("**Edit:** change a rating delta to bump a player up/down, "
                    "or change their availability. Click *Save changes* to apply.")

        overrides = load_overrides()
        avail = load_availability()

        editable = team_players[
            ["player_id", "name", "position", "rating", "age", "club"]
        ].copy()
        editable["override_delta"] = editable["player_id"].map(overrides).fillna(0.0)
        editable["availability"] = editable["player_id"].map(
            lambda pid: avail.get(pid, AvailabilityRecord()).status
        )

        edited = st.data_editor(
            editable.drop(columns=["player_id"]),
            column_config={
                "rating": st.column_config.NumberColumn(
                    "Base rating", min_value=40, max_value=99, step=1,
                    disabled=True,
                ),
                "override_delta": st.column_config.NumberColumn(
                    "Δ override", min_value=-30, max_value=30, step=1,
                    help="Adjustment to base rating, e.g. +3 for hot form, -5 for poor form",
                ),
                "availability": st.column_config.SelectboxColumn(
                    "Availability",
                    options=["available", "doubtful", "out", "suspended"],
                    required=True,
                ),
                "age": st.column_config.NumberColumn("Age", disabled=True),
                "name": st.column_config.TextColumn("Player", disabled=True),
                "position": st.column_config.TextColumn("Pos", disabled=True),
                "club": st.column_config.TextColumn("Club", disabled=True),
            },
            hide_index=True, use_container_width=True,
            key=f"editor_{country}",
        )

        col_save, col_reset, _ = st.columns([1, 1, 3])
        if col_save.button("💾 Save changes", type="primary"):
            new_overrides = overrides.copy()
            new_overrides_full = {pid: {"delta": new_overrides[pid]}
                                  for pid in new_overrides}
            new_avail = {pid: rec for pid, rec in avail.items()}

            for idx, row in edited.iterrows():
                pid = editable.iloc[idx]["player_id"]
                delta = float(row["override_delta"])
                if delta != 0.0:
                    new_overrides_full[pid] = {"delta": delta, "note": ""}
                else:
                    new_overrides_full.pop(pid, None)

                status = row["availability"]
                if status != "available":
                    new_avail[pid] = AvailabilityRecord(status=status)
                else:
                    new_avail.pop(pid, None)

            save_overrides(new_overrides_full)
            save_availability(new_avail)

            # Rebuild squad strength file so predictions pick up changes
            with st.spinner("Recomputing squad strengths …"):
                build_squad_strength(save=True)
                refresh_squad_strength_cache()

            # Bust Streamlit caches
            cached_prediction.clear()
            cached_simulation.clear()

            st.success("Saved. Match predictions and simulations updated.")
            st.rerun()

        if col_reset.button("↺ Reset this team"):
            new_overrides_full = {pid: {"delta": v}
                                  for pid, v in overrides.items()}
            new_avail = dict(avail)
            for pid in editable["player_id"]:
                new_overrides_full.pop(pid, None)
                new_avail.pop(pid, None)
            save_overrides(new_overrides_full)
            save_availability(new_avail)
            build_squad_strength(save=True)
            cached_prediction.clear()
            cached_simulation.clear()
            st.success(f"Reset all overrides for {country}.")
            st.rerun()

# ===========================================================================
# Tab 5 — Tournament simulator
# ===========================================================================
with tab_sim:
    st.header("World Cup 2026 — Monte Carlo simulator")
    st.caption(
        "Now squad-aware: simulations factor in current player ratings, "
        "injuries, and suspensions across all 48 nations."
    )

    col_s, col_b = st.columns([1, 3])
    with col_s:
        n_sims = st.select_slider(
            "Simulations",
            options=[500, 1_000, 2_500, 5_000],
            value=DEFAULT_SIMULATIONS,
            help="Higher = more reliable probabilities. Capped at 5,000 to fit "
                 "Streamlit Cloud's free tier — running locally you can do more.",
        )
        seed = st.number_input("Random seed", value=42, step=1,
                               help="Default 42 loads instantly (pre-computed). "
                                    "Other values trigger a fresh ~30-60s simulation.")
        run = st.button("Run simulation", type="primary")
        st.caption("💡 Default settings (5,000 sims, seed 42) load instantly. "
                   "Other combinations take 30-60 seconds.")
    with col_b:
        with st.expander("View groups — official 2026 draw"):
            grp_df = pd.DataFrame(WORLDCUP_2026_GROUPS)
            st.dataframe(grp_df, hide_index=True, use_container_width=True)

    if run:
        sim = cached_simulation(int(n_sims), int(seed), buster)

        st.subheader("🏆 Champion probabilities (top 16)")
        top = sim.head(16).copy()
        top["champion_pct"] = (top["p_champion"] * 100).round(2)
        bars = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("champion_pct:Q", title="P(champion) %"),
                y=alt.Y("team:N", sort="-x"),
                tooltip=["team", "champion_pct"],
                color=alt.Color("champion_pct:Q",
                                scale=alt.Scale(scheme="oranges"), legend=None),
            )
            .properties(height=440)
        )
        st.altair_chart(bars, use_container_width=True)

        st.subheader("Path-to-glory table")
        path_cols = ["team", "p_group", "p_r32", "p_r16", "p_qf", "p_sf",
                     "p_final", "p_champion"]
        display = sim[path_cols].copy()
        for c in path_cols[1:]:
            display[c] = (display[c] * 100).round(1)
        display = display.rename(columns={
            "p_group": "Adv. (%)", "p_r32": "R32 (%)", "p_r16": "R16 (%)",
            "p_qf": "QF (%)", "p_sf": "SF (%)", "p_final": "Final (%)",
            "p_champion": "Champion (%)",
        })
        st.dataframe(display, hide_index=True, use_container_width=True)
    else:
        st.info("Press **Run simulation** to compute tournament probabilities.")
