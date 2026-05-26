"""Step 4 — Monte-Carlo simulate the 2026 World Cup and print probabilities."""

import argparse

from src.simulation.tournament import simulate_tournament

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = simulate_tournament(args.n_simulations, args.seed)
    print("\nTop 20 contenders:\n")
    print(df.head(20).to_string(index=False))
