"""Player data schemas.

A "player" record is a dict-like row with at minimum:
    player_id   : stable string id
    name        : display name
    country     : national team
    position    : one of {GK, DEF, MID, FWD}
    rating      : overall rating (0-99, FIFA-style)
    age         : int
    club        : club team (best-effort)

Availability is stored separately in a JSON keyed by player_id so the
ratings file stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Position = Literal["GK", "DEF", "MID", "FWD"]
Status = Literal["available", "doubtful", "out", "suspended"]

PLAYER_COLUMNS: list[str] = [
    "player_id",
    "name",
    "country",
    "position",
    "rating",
    "age",
    "club",
]


@dataclass(frozen=True)
class Player:
    player_id: str
    name: str
    country: str
    position: Position
    rating: float
    age: int
    club: str = ""


@dataclass
class AvailabilityRecord:
    status: Status = "available"
    note: str = ""

    def factor(self) -> float:
        """Return the rating multiplier for this availability status."""
        from src.config import (
            INJURY_OUT_FACTOR,
            INJURY_PLAYING_FACTOR,
            SUSPENSION_FACTOR,
        )
        if self.status == "available":
            return 1.0
        if self.status == "doubtful":
            return INJURY_PLAYING_FACTOR
        if self.status == "out":
            return INJURY_OUT_FACTOR
        if self.status == "suspended":
            return SUSPENSION_FACTOR
        return 1.0


@dataclass
class Override:
    """User-supplied adjustment to a player's rating."""
    delta: float = 0.0  # added to base rating, can be negative
    note: str = ""
