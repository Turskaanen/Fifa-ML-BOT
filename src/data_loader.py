import pandas as pd
from .config import TEAM_STATS_CSV

def load_team_stats() -> pd.DataFrame:
    df = pd.read_csv(TEAM_STATS_CSV)
    # esim. varmistetaan pari perusjuttua
    if "team_name" not in df.columns:
        raise ValueError("team_name -kolumni puuttuu CSV:stä")
    return df

def get_team_row(df: pd.DataFrame, team_name: str) -> pd.Series:
    row = df[df["team_name"] == team_name]
    if row.empty:
        raise ValueError(f"Joukkuetta ei löydy: {team_name}")
    return row.iloc[0]
