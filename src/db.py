import sqlite3
from pathlib import Path
from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    xg_home REAL,
    xg_away REAL,
    home_goals INTEGER,
    away_goals INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = get_connection()
    with conn:
        conn.executescript(SCHEMA)
    conn.close()

def insert_match(home_team, away_team, xg_home, xg_away, home_goals=None, away_goals=None):
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO matches (home_team, away_team, xg_home, xg_away, home_goals, away_goals)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (home_team, away_team, xg_home, xg_away, home_goals, away_goals),
        )
    conn.close()
