"""Run the full pipeline end-to-end:
    1. Download match data
    2. Build players + squad strength (uses seed dataset)
    3. Build features (now includes squad strength)
    4. Train models
    5. Simulate the World Cup AND ship the default result for instant loading

Usage:
    python -m scripts.run_pipeline                # safe defaults
    python -m scripts.run_pipeline --scrape       # also try SoFIFA scrape
"""

from __future__ import annotations

import argparse

from src.config import DEFAULT_SIM_CACHE_FILE, DEFAULT_SIMULATIONS
from src.data.download import download_all
from src.features.builder import build_features
from src.models.train import train
from src.players.scraper import scrape_all
from src.players.strength import build_squad_strength
from src.simulation.tournament import simulate_tournament
from src.utils import get_logger, timer

log = get_logger("pipeline")


def main(n_simulations: int = DEFAULT_SIMULATIONS, scrape: bool = False) -> None:
    with timer("01 · download match data"):
        download_all()

    with timer("02 · build player + squad data"):
        if scrape:
            scrape_all()
        build_squad_strength(save=True)

    with timer("03 · build features"):
        build_features()

    with timer("04 · train models"):
        train()

    with timer("05 · simulate world cup (default cache)"):
        df = simulate_tournament(n_simulations=n_simulations, seed=42, progress=False)
        # Save the default simulation so Streamlit Cloud loads it instantly
        df.to_parquet(DEFAULT_SIM_CACHE_FILE, index=False)
        log.info("Default sim cached at %s", DEFAULT_SIM_CACHE_FILE)
        log.info("\nTop 15 contenders:\n%s",
                 df.head(15).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-simulations", type=int,
                        default=DEFAULT_SIMULATIONS)
    parser.add_argument("--scrape", action="store_true",
                        help="Try to scrape live player ratings from SoFIFA")
    args = parser.parse_args()
    main(n_simulations=args.n_simulations, scrape=args.scrape)
