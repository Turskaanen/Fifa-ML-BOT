import argparse
from .data_loader import load_team_stats, get_team_row
from .xg_model import train_xg_model, load_xg_model, predict_team_xg
from .ou_model import train_ou_model
from .scoreline_model import expected_scoreline
from .monte_carlo import simulate_match_monte_carlo
from .db import init_db, insert_match

def main():
    parser = argparse.ArgumentParser(prog="footybot", description="FootyBot v1.3")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init-db")

    train_xg_p = sub.add_parser("train-xg")
    train_xg_p.add_argument("--target", default="goals")

    train_ou_p = sub.add_parser("train-ou")
    train_ou_p.add_argument("--target", default="goals_total")

    sim_p = sub.add_parser("simulate")
    sim_p.add_argument("home_team")
    sim_p.add_argument("away_team")

    args = parser.parse_args()

    if args.command == "init-db":
        init_db()
        print("DB init done.")
    elif args.command == "train-xg":
        df = load_team_stats()
        _, score = train_xg_model(df, target_col=args.target)
        print(f"xG-malli treenattu, score: {score:.3f}")
    elif args.command == "train-ou":
        df = load_team_stats()
        _, score = train_ou_model(df, target_col=args.target)
        print(f"OU-malli treenattu, score: {score:.3f}")
    elif args.command == "simulate":
        df = load_team_stats()
        model = load_xg_model()

        home_row = get_team_row(df, args.home_team)
        away_row = get_team_row(df, args.away_team)

        xg_home = predict_team_xg(model, home_row)
        xg_away = predict_team_xg(model, away_row)

        print(f"xG {args.home_team}: {xg_home:.2f}, {args.away_team}: {xg_away:.2f}")

        scoreline = expected_scoreline(xg_home, xg_away)
        mc = simulate_match_monte_carlo(xg_home, xg_away)

        print("Poisson scoreline probs:", scoreline["p_home_win"], scoreline["p_draw"], scoreline["p_away_win"])
        print("Monte Carlo:", mc)

        insert_match(args.home_team, args.away_team, xg_home, xg_away)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
