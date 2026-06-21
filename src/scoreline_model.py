import numpy as np
import pandas as pd
from .xg_model import predict_team_xg

def expected_scoreline(
    xg_home: float,
    xg_away: float,
    max_goals: int = 6,
) -> dict:
    """
    Palauttaa todennäköisyysjakauman tuloksille (0–max_goals).
    Käytetään Poisson-jakaumaa molemmille joukkueille.
    """
    home_goals = np.arange(0, max_goals + 1)
    away_goals = np.arange(0, max_goals + 1)

    from scipy.stats import poisson

    p_home = poisson.pmf(home_goals, xg_home)
    p_away = poisson.pmf(away_goals, xg_away)

    matrix = np.outer(p_home, p_away)

    result = {
        "matrix": matrix,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "p_home_win": float(np.tril(matrix, -1).sum()),
        "p_draw": float(np.trace(matrix)),
        "p_away_win": float(np.triu(matrix, 1).sum()),
    }
    return result
