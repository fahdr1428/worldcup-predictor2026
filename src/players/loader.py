"""Player data loader.

Single source of truth for "what players do we have, and what are their
current effective ratings?". Layered:

  1. Start from the seed dataset (always available)
  2. If a scraped CSV exists at PLAYERS_FILE, prefer it
  3. Apply per-player overrides from the JSON file
  4. Apply availability statuses (injury/suspension) to derive *effective*
     ratings

The result is a single tidy DataFrame the rest of the project consumes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import (
    AVAILABILITY_FILE,
    PLAYERS_FILE,
    PLAYER_OVERRIDES_FILE,
)
from src.players.schema import PLAYER_COLUMNS, AvailabilityRecord
from src.players.seed_data import SEED_PLAYERS
from src.utils import get_logger

log = get_logger(__name__)


def _make_player_id(name: str, country: str) -> str:
    """Deterministic id from name+country so overrides survive re-scrapes."""
    key = f"{name.lower()}|{country.lower()}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _seed_to_dataframe() -> pd.DataFrame:
    rows = []
    for country, players in SEED_PLAYERS.items():
        for name, position, rating, age, club in players:
            rows.append({
                "player_id": _make_player_id(name, country),
                "name": name,
                "country": country,
                "position": position,
                "rating": float(rating),
                "age": int(age),
                "club": club,
            })
    return pd.DataFrame(rows, columns=PLAYER_COLUMNS)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        log.warning("Couldn't parse %s (%s) — treating as empty.", path.name, e)
        return {}


def load_base_players() -> pd.DataFrame:
    """Load player ratings from CSV if present, else from seed."""
    if PLAYERS_FILE.exists():
        try:
            df = pd.read_csv(PLAYERS_FILE)
            if {"name", "country", "rating"}.issubset(df.columns):
                # Ensure player_id column exists
                if "player_id" not in df.columns:
                    df["player_id"] = df.apply(
                        lambda r: _make_player_id(r["name"], r["country"]),
                        axis=1,
                    )
                df["rating"] = df["rating"].astype(float)
                df["age"] = df.get("age", 28).astype(int)
                df["club"] = df.get("club", "").fillna("")
                df["position"] = df.get("position", "MID")
                log.info("Loaded %d players from scraped CSV", len(df))
                return df[PLAYER_COLUMNS]
        except Exception as e:  # noqa: BLE001
            log.warning("Couldn't read %s (%s) — falling back to seed.",
                        PLAYERS_FILE, e)

    log.debug("Using seed player dataset (576 players across 48 nations).")
    return _seed_to_dataframe()


def load_overrides() -> dict[str, float]:
    """Return a mapping player_id → rating delta."""
    raw = _load_json(PLAYER_OVERRIDES_FILE)
    return {pid: float(v.get("delta", 0.0)) for pid, v in raw.items()}


def save_overrides(overrides: dict[str, dict]) -> None:
    """Persist overrides as JSON. ``overrides`` is player_id → {delta, note}."""
    PLAYER_OVERRIDES_FILE.write_text(
        json.dumps(overrides, indent=2, ensure_ascii=False)
    )


def load_availability() -> dict[str, AvailabilityRecord]:
    """Return a mapping player_id → AvailabilityRecord."""
    raw = _load_json(AVAILABILITY_FILE)
    out: dict[str, AvailabilityRecord] = {}
    for pid, info in raw.items():
        out[pid] = AvailabilityRecord(
            status=info.get("status", "available"),
            note=info.get("note", ""),
        )
    return out


def save_availability(records: dict[str, AvailabilityRecord]) -> None:
    serialised = {
        pid: {"status": rec.status, "note": rec.note}
        for pid, rec in records.items()
    }
    AVAILABILITY_FILE.write_text(
        json.dumps(serialised, indent=2, ensure_ascii=False)
    )


def load_effective_players() -> pd.DataFrame:
    """The main entry point. Returns the players DataFrame with extra columns:

    - ``override_delta``    — rating adjustment from the user
    - ``availability``      — status string
    - ``avail_factor``      — multiplier from the status
    - ``effective_rating``  — (base + override) × availability factor
    """
    df = load_base_players().copy()
    overrides = load_overrides()
    avail = load_availability()

    df["override_delta"] = df["player_id"].map(overrides).fillna(0.0)
    df["availability"] = df["player_id"].map(
        lambda pid: avail.get(pid, AvailabilityRecord()).status
    )
    df["avail_factor"] = df["player_id"].map(
        lambda pid: avail.get(pid, AvailabilityRecord()).factor()
    )
    df["effective_rating"] = (
        (df["rating"] + df["override_delta"]) * df["avail_factor"]
    ).clip(lower=0, upper=99)

    return df
