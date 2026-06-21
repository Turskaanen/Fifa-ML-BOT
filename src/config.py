from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "footybot.db"

TEAM_STATS_CSV = DATA_DIR / "team_stats.csv"

RANDOM_STATE = 42
N_MONTE_CARLO_SIMS = 10_000
