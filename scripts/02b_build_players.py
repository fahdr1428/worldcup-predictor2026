"""Step 02b — Build the players + squad-strength data.

Two modes:
    python -m scripts.02b_build_players              # seed only (fast, safe)
    python -m scripts.02b_build_players --scrape     # try SoFIFA first

If the scrape fails or returns nothing useful, the seed dataset is used
instead. Either way, the squad-strength file is written so the rest of
the pipeline has something to consume.
"""

from __future__ import annotations

import argparse

from src.players.scraper import scrape_all
from src.players.strength import build_squad_strength
from src.utils import get_logger

log = get_logger("players")


def main(scrape: bool = False) -> None:
    if scrape:
        log.info("Attempting SoFIFA scrape — may take 1–3 minutes.")
        scrape_all()
    else:
        log.info("Using seed player dataset (no scraping). "
                 "Re-run with --scrape to try SoFIFA.")

    df = build_squad_strength(save=True)
    log.info("\nTop 15 squads by strength:\n%s",
             df.head(15).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrape", action="store_true",
                        help="Attempt to refresh player data from SoFIFA")
    args = parser.parse_args()
    main(scrape=args.scrape)
