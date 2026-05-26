"""Central configuration: paths, data sources, model + tournament setup."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PLAYERS_DIR = DATA_DIR / "players"
MODELS_DIR = PROJECT_ROOT / "models"

for _p in (RAW_DIR, PROCESSED_DIR, PLAYERS_DIR, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/shootouts.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/goalscorers.csv"
)

RESULTS_FILE = RAW_DIR / "results.csv"
SHOOTOUTS_FILE = RAW_DIR / "shootouts.csv"
GOALSCORERS_FILE = RAW_DIR / "goalscorers.csv"

PROCESSED_FEATURES_FILE = PROCESSED_DIR / "features.parquet"
TEAM_RATINGS_FILE = PROCESSED_DIR / "team_ratings.parquet"

# Player data files
PLAYERS_RAW_FILE = PLAYERS_DIR / "players_raw.csv"
PLAYERS_FILE = PLAYERS_DIR / "players.csv"
SQUADS_FILE = PLAYERS_DIR / "squads.json"
AVAILABILITY_FILE = PLAYERS_DIR / "availability.json"
PLAYER_OVERRIDES_FILE = PLAYERS_DIR / "overrides.json"
SQUAD_STRENGTH_FILE = PLAYERS_DIR / "squad_strength.csv"

# Model artifacts
RESULT_MODEL_FILE = MODELS_DIR / "result_model.joblib"
GOALS_MODEL_FILE = MODELS_DIR / "goals_model.joblib"
FEATURE_META_FILE = MODELS_DIR / "feature_meta.joblib"
DEFAULT_SIM_CACHE_FILE = MODELS_DIR / "default_simulation.parquet"

# ---------------------------------------------------------------------------
# ELO settings
# ---------------------------------------------------------------------------
ELO_START = 1500.0
ELO_K_BASE = 30.0
ELO_HOME_ADVANTAGE = 65.0
TOURNAMENT_WEIGHTS: dict[str, float] = {
    "Friendly": 1.0,
    "FIFA World Cup": 1.6,
    "FIFA World Cup qualification": 1.3,
    "UEFA Euro": 1.5,
    "UEFA Euro qualification": 1.2,
    "Copa América": 1.4,
    "Africa Cup of Nations": 1.4,
    "AFC Asian Cup": 1.3,
    "CONCACAF Championship": 1.2,
    "Confederations Cup": 1.4,
    "UEFA Nations League": 1.25,
}
TOURNAMENT_WEIGHT_DEFAULT = 1.1

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TRAIN_START_DATE = "1990-01-01"
TEST_START_DATE = "2022-01-01"
FORM_WINDOW = 10
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Player-rating settings
# ---------------------------------------------------------------------------
TOP_XI_WEIGHT = 0.70
BENCH_WEIGHT = 0.30
SQUAD_SIZE_TOP = 11
SQUAD_SIZE_TOTAL = 23
TOP_XI_DECAY = 0.95

INJURY_PLAYING_FACTOR = 0.60
INJURY_OUT_FACTOR = 0.0
SUSPENSION_FACTOR = 0.0
REPLACEMENT_DISCOUNT = 0.85

SQUAD_STRENGTH_CENTER = 75.0
SQUAD_STRENGTH_SCALE = 10.0

# ---------------------------------------------------------------------------
# 2026 World Cup — official 48-team draw (confirmed 5 Dec 2025,
# with playoff winners added 31 Mar 2026).
# Source: FIFA official draw / Wikipedia / ESPN.
# ---------------------------------------------------------------------------
WORLDCUP_2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

HOST_COUNTRIES = {"United States", "Canada", "Mexico"}

WORLD_CUP_TEAMS: list[str] = sorted(
    {t for teams in WORLDCUP_2026_GROUPS.values() for t in teams}
)

DEFAULT_SIMULATIONS = 5_000

# ---------------------------------------------------------------------------
# Scraper settings
# ---------------------------------------------------------------------------
SOFIFA_BASE = "https://sofifa.com"
SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
SCRAPER_REQUEST_DELAY = 1.5
SCRAPER_TIMEOUT = 30
SCRAPER_MAX_RETRIES = 3
