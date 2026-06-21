import numpy as np
from scipy.stats import poisson
from .config import N_MONTE_CARLO_SIMS

def simulate_match_monte_carlo(xg_home: float, xg_away: float, n_sims: int = N_MONTE_CARLO_SIMS):
    home_goals = poisson.rvs(mu=xg_home, size=n_sims)
    away_goals = poisson.rvs(mu=xg_away, size=n_sims)

    home_wins = (home_goals > away_goals).mean()
    draws = (home_goals == away_goals).mean()
    away_wins = (home_goals < away_goals).mean()

    avg_home = home_goals.mean()
    avg_away = away_goals.mean()

    return {
        "home_win_prob": float(home_wins),
        "draw_prob": float(draws),
        "away_win_prob": float(away_wins),
        "avg_home_goals": float(avg_home),
        "avg_away_goals": float(avg_away),
    }
