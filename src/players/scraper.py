"""SoFIFA scraper (Path A).

⚠️ HONEST CAVEAT: SoFIFA actively blocks scrapers. This module is a
best-effort attempt — if it works on your network and IP, great; if not,
the rest of the system falls back to the seed dataset and you can edit
ratings via the in-app UI.

What it does:
  * Iterates over each World Cup nation
  * Fetches the national team search page on sofifa.com
  * Parses the top ~25 highest-rated players
  * Writes the result to PLAYERS_FILE

How to use:
    python -m src.players.scraper

If SoFIFA blocks the requests (403s, captchas, etc.), the script logs the
failure for each country and continues — partial results are still useful.
At the end, anything successfully scraped overwrites the seed for that
country; everything else stays on the seed dataset.
"""

from __future__ import annotations

import re
import time
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    PLAYERS_FILE,
    PLAYERS_RAW_FILE,
    SCRAPER_MAX_RETRIES,
    SCRAPER_REQUEST_DELAY,
    SCRAPER_TIMEOUT,
    SCRAPER_USER_AGENT,
    SOFIFA_BASE,
    WORLD_CUP_TEAMS,
)
from src.players.loader import _make_player_id, load_base_players
from src.players.schema import PLAYER_COLUMNS
from src.utils import get_logger

log = get_logger(__name__)

HEADERS = {
    "User-Agent": SCRAPER_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}


def _get(url: str, *, params: dict | None = None) -> requests.Response | None:
    """GET with retries and polite delay."""
    last_err: Exception | None = None
    for attempt in range(SCRAPER_MAX_RETRIES):
        try:
            response = requests.get(
                url, params=params, headers=HEADERS, timeout=SCRAPER_TIMEOUT
            )
            time.sleep(SCRAPER_REQUEST_DELAY)
            if response.status_code == 200:
                return response
            log.warning("  %s returned %s (attempt %d)", url,
                        response.status_code, attempt + 1)
            if response.status_code in (403, 429):
                time.sleep(5)  # back off on rate-limit / block
        except requests.RequestException as e:
            last_err = e
            log.warning("  request failed: %s (attempt %d)", e, attempt + 1)
            time.sleep(3)
    if last_err:
        log.error("  giving up: %s", last_err)
    return None


# Map SoFIFA's positional codes to our four buckets
_POSITION_BUCKETS = {
    "GK": "GK",
    "CB": "DEF", "RB": "DEF", "LB": "DEF", "RWB": "DEF", "LWB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "RM": "MID", "LM": "MID",
    "CF": "FWD", "ST": "FWD", "LW": "FWD", "RW": "FWD",
}


def _bucket_position(raw: str) -> str:
    raw = (raw or "").strip().upper()
    return _POSITION_BUCKETS.get(raw, "MID")


def _parse_player_rows(soup: BeautifulSoup, country: str) -> list[dict]:
    """SoFIFA returns a table of players; parse the relevant cells."""
    rows = []
    table = soup.select_one("table")
    if table is None:
        return rows

    for tr in table.select("tbody tr"):
        try:
            name_cell = tr.select_one("td.col-name a[data-tippy-content], "
                                       "td.col-name a[href*='/player/']")
            if name_cell is None:
                continue
            name = name_cell.get_text(strip=True)
            if not name:
                continue

            # Overall rating
            ovr_cell = tr.select_one("td.col-oa, td.col-ovr")
            ovr = int(re.search(r"\d+", ovr_cell.text).group()) if ovr_cell else 0
            if ovr == 0:
                continue

            # Position: span.pos or first pos in the cell
            pos_span = tr.select_one("span.pos")
            position = _bucket_position(pos_span.text if pos_span else "")

            # Age
            age_cell = tr.select_one("td.col-ae, td.col-age")
            age = int(re.search(r"\d+", age_cell.text).group()) if age_cell else 28

            # Club
            club_cell = tr.select_one("td.col-name a[href*='/team/']")
            club = club_cell.get_text(strip=True) if club_cell else ""

            rows.append({
                "player_id": _make_player_id(name, country),
                "name": name,
                "country": country,
                "position": position,
                "rating": float(ovr),
                "age": age,
                "club": club,
            })
        except Exception as e:  # noqa: BLE001
            log.debug("  row parse failed: %s", e)
            continue
    return rows


def scrape_team(country: str, *, max_players: int = 25) -> list[dict]:
    """Try to scrape top players for one national team."""
    # SoFIFA's players search supports nationality filter via the `na`
    # parameter, but the country name → id mapping is internal. We use
    # the search URL with the country name as text.
    url = f"{SOFIFA_BASE}/players"
    params = {
        "type": "all",
        "ae[0]": "16",  # min age, ensures real players only
        "ae[1]": "45",
        "showCol[0]": "ae", "showCol[1]": "oa", "showCol[2]": "pt",
        "r": "250026",  # internal FIFA edition id; may change yearly
        "s": "oa", "d": "desc",  # sort overall descending
        "country": country,
    }
    response = _get(url, params=params)
    if response is None:
        return []

    soup = BeautifulSoup(response.text, "lxml")
    rows = _parse_player_rows(soup, country)
    if not rows:
        log.warning("  no rows parsed for %s — site structure may have changed",
                    country)
    return rows[:max_players]


def scrape_all(teams: Iterable[str] | None = None,
               *, max_players: int = 25) -> pd.DataFrame:
    """Scrape every team and merge with the seed dataset.

    Teams the scraper succeeds on overwrite the seed. Teams it fails on
    keep their seed data.
    """
    teams = list(teams) if teams is not None else WORLD_CUP_TEAMS

    log.info("Scraping SoFIFA for %d teams (max %d players each) …",
             len(teams), max_players)
    log.info("⚠️ If SoFIFA blocks us, fallback is the seed dataset — "
             "that's by design, not an error.")

    scraped: list[dict] = []
    n_success = 0
    for i, country in enumerate(teams, 1):
        log.info("[%d/%d] %s", i, len(teams), country)
        rows = scrape_team(country, max_players=max_players)
        if rows:
            scraped.extend(rows)
            n_success += 1

    log.info("Scrape complete: %d/%d teams had usable results.",
             n_success, len(teams))

    if not scraped:
        log.error("Nothing scraped — keeping the seed dataset as-is. "
                  "(This is fine; SoFIFA is likely blocking us.)")
        # Still write seed to PLAYERS_FILE so downstream code can read it
        base = load_base_players()
        base.to_csv(PLAYERS_FILE, index=False)
        return base

    scraped_df = pd.DataFrame(scraped)
    if not PLAYERS_RAW_FILE.exists() or scraped_df.shape[0]:
        scraped_df.to_csv(PLAYERS_RAW_FILE, index=False)

    # Merge: scraped wins where present, seed fills the rest
    seed = load_base_players()
    scraped_countries = set(scraped_df["country"].unique())
    seed_remaining = seed[~seed["country"].isin(scraped_countries)]
    merged = pd.concat([scraped_df[PLAYER_COLUMNS], seed_remaining],
                       ignore_index=True)
    merged.to_csv(PLAYERS_FILE, index=False)
    log.info("Wrote merged dataset (%d players) → %s", len(merged), PLAYERS_FILE)
    return merged


if __name__ == "__main__":
    scrape_all()
