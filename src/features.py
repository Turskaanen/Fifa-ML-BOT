import pandas as pd

# Tässä valitaan featuret xG / OU / scoreline-malleille
# Voit myöhemmin säätää listaa datan perusteella
XG_FEATURES = [
    "attempt_at_goal",
    "attempt_at_goal_on_target",
    "xg",  # jos lisäät myöhemmin tai lasket
]

OU_FEATURES = [
    "attempt_at_goal",
    "attempt_at_goal_against",
    "goals",
    "goals_conceded",
]

SCORELINE_FEATURES = [
    "attempt_at_goal",
    "attempt_at_goal_against",
    "goals",
    "goals_conceded",
]

def build_features(df: pd.DataFrame, feature_cols: list, target_col: str | None = None):
    X = df[feature_cols].copy()
    y = df[target_col] if target_col is not None else None
    return X, y
