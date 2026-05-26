"""Download raw match data from public sources.

We use martj42's `international_results` dataset on GitHub — the de facto
open dataset for international football matches (45K+ matches, MIT-licensed).
"""

from __future__ import annotations

from pathlib import Path

import requests

from src.config import (
    GOALSCORERS_FILE,
    GOALSCORERS_URL,
    RESULTS_FILE,
    RESULTS_URL,
    SHOOTOUTS_FILE,
    SHOOTOUTS_URL,
)
from src.utils import get_logger

log = get_logger(__name__)


_SOURCES: list[tuple[str, Path]] = [
    (RESULTS_URL, RESULTS_FILE),
    (SHOOTOUTS_URL, SHOOTOUTS_FILE),
    (GOALSCORERS_URL, GOALSCORERS_FILE),
]


def _download_one(url: str, dest: Path, *, timeout: int = 60) -> None:
    log.info("Downloading %s → %s", url, dest.name)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    dest.write_bytes(response.content)
    log.info("  %s saved (%.1f KB)", dest.name, dest.stat().st_size / 1024)


def download_all(force: bool = False) -> None:
    """Download every source file. Skips files that already exist unless
    ``force`` is True."""
    for url, dest in _SOURCES:
        if dest.exists() and not force:
            log.info("Skipping %s (already cached). Use force=True to refresh.",
                     dest.name)
            continue
        _download_one(url, dest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download raw match data")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached files exist")
    args = parser.parse_args()
    download_all(force=args.force)
