"""Production CLI entrypoint for Steam recommendation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.baselines import (
    evaluate_recommendations,
    popularity_recommendations,
    random_recommendations,
)
from src.data_loader import (
    build_interaction_matrices,
    filter_recommendations,
    leave_one_out_split,
)
from src.matrix_factorization import (
    MatrixFactorizationALS,
    MatrixFactorizationSGD,
    build_test_items_by_user,
    evaluate_mf_leave_one_out,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Steam recommendation models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recommendations.csv"),
        help="Path to raw recommendations.csv input data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/model_results.csv"),
        help="Path to save the evaluation results table.",
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=Path("data/interactions_filtered.csv"),
        help="Path to save the filtered interactions file.",
    )
    parser.add_argument(
        "--min-user-reviews",
        type=int,
        default=50,
        help="Minimum interactions required per user.",
    )
    parser.add_argument(
        "--min-game-reviews",
        type=int,
        default=500,
        help="Minimum interactions required per game.",
    )
    return parser.parse_args()


def _format_metrics_row(name: str, metrics_10: dict, metrics_20: dict) -> dict:
    return {
        "model": name,
        "hit_rate@10": metrics_10["hit_rate_at_k"],
        "hit_rate@20": metrics_20["hit_rate_at_k"],
        "ndcg@10": metrics_10["ndcg_at_k"],
        "ndcg@20": metrics_20["ndcg_at_k"],
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    if not args.input.exists():
        raise FileNotFoundError(
            f"Input not found: {args.input}. "
            "Place recommendations.csv there or pass --input."
        )

    print("Loading data...")
    interactions = pd.read_csv(args.input)
    interactions = filter_recommendations(
        interactions,
        min_user_reviews=args.min_user_reviews,
        min_game_reviews=args.min_game_reviews,
    )

    args.filtered_output.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(args.filtered_output, index=False)
    print(f"Saved filtered interactions: {args.filtered_output}")

    print("Splitting (leave-one-out on positive interactions)...")
    train_df, test_df = leave_one_out_split(interactions)
    print(f"  Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    print("Building matrices...")
    train_data = build_interaction_matrices(train_df)
    user_ids = sorted(test_df["user_id"].unique().tolist())
    all_game_ids = sorted(train_df["app_id"].unique().tolist())
    test_items_by_user_id = test_df.groupby("user_id")["app_id"].apply(list).to_dict()
    test_items_by_user_idx = build_test_items_by_user(
        test_df, train_data.user_to_idx, train_data.game_to_idx
    )

    results = []

    print("\nRunning baselines...")
    pop_recs = popularity_recommendations(train_df, user_ids=user_ids, k=20)
    pop_10 = evaluate_recommendations(pop_recs, test_items_by_user_id, k=10)
    pop_20 = evaluate_recommendations(pop_recs, test_items_by_user_id, k=20)
    results.append(_format_metrics_row("Popularity", pop_10, pop_20))
    print(
        f"  Popularity  -> hit_rate@10: {pop_10['hit_rate_at_k']:.6f}  "
        f"hit_rate@20: {pop_20['hit_rate_at_k']:.6f}  "
        f"ndcg@10: {pop_10['ndcg_at_k']:.6f}  ndcg@20: {pop_20['ndcg_at_k']:.6f}"
    )

    rand_recs = random_recommendations(
        train_df, user_ids=user_ids, all_game_ids=all_game_ids, k=20, seed=42
    )
    rand_10 = evaluate_recommendations(rand_recs, test_items_by_user_id, k=10)
    rand_20 = evaluate_recommendations(rand_recs, test_items_by_user_id, k=20)
    results.append(_format_metrics_row("Random", rand_10, rand_20))
    print(
        f"  Random      -> hit_rate@10: {rand_10['hit_rate_at_k']:.6f}  "
        f"hit_rate@20: {rand_20['hit_rate_at_k']:.6f}  "
        f"ndcg@10: {rand_10['ndcg_at_k']:.6f}  ndcg@20: {rand_20['ndcg_at_k']:.6f}"
    )

    print("\nTraining SGD (k=50) for comparison...")
    sgd = MatrixFactorizationSGD(
        k=50, reg=0.01, learning_rate=0.005, epochs=50, random_state=42
    )
    sgd.fit(train_data.hours_matrix, verbose=True)
    sgd_10 = evaluate_mf_leave_one_out(sgd, train_data.hours_matrix, test_items_by_user_idx, k=10)
    sgd_20 = evaluate_mf_leave_one_out(sgd, train_data.hours_matrix, test_items_by_user_idx, k=20)
    results.append(_format_metrics_row("SGD (k=50)", sgd_10, sgd_20))
    print(
        f"  SGD (k=50)  -> hit_rate@10: {sgd_10['hit_rate_at_k']:.6f}  "
        f"hit_rate@20: {sgd_20['hit_rate_at_k']:.6f}  "
        f"ndcg@10: {sgd_10['ndcg_at_k']:.6f}  ndcg@20: {sgd_20['ndcg_at_k']:.6f}"
    )

    print("\nTraining best model: ALS (k=200)...")
    als = MatrixFactorizationALS(k=200, reg=0.01, iterations=50, random_state=42)
    als.fit(train_data.hours_matrix, verbose=True)
    als_10 = evaluate_mf_leave_one_out(als, train_data.hours_matrix, test_items_by_user_idx, k=10)
    als_20 = evaluate_mf_leave_one_out(als, train_data.hours_matrix, test_items_by_user_idx, k=20)
    results.append(_format_metrics_row("ALS (k=200)", als_10, als_20))
    print(
        f"  ALS (k=200) -> hit_rate@10: {als_10['hit_rate_at_k']:.6f}  "
        f"hit_rate@20: {als_20['hit_rate_at_k']:.6f}  "
        f"ndcg@10: {als_10['ndcg_at_k']:.6f}  ndcg@20: {als_20['ndcg_at_k']:.6f}"
    )

    results_df = (
        pd.DataFrame(results)
        .sort_values("hit_rate@10", ascending=False)
        .reset_index(drop=True)
    )
    return results_df[["model", "hit_rate@10", "hit_rate@20", "ndcg@10", "ndcg@20"]]


def print_summary(results_df: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    pop = results_df[results_df["model"] == "Popularity"].iloc[0]
    best = results_df.iloc[0]

    def pct_improvement(best_val: float, base_val: float) -> float:
        if base_val == 0:
            return 0.0
        return (best_val - base_val) / base_val * 100.0

    print(f"\nBest model:                              {best['model']}")
    print(
        "hit_rate@10 improvement over popularity: "
        f"+{pct_improvement(best['hit_rate@10'], pop['hit_rate@10']):.1f}%"
    )
    print(
        "hit_rate@20 improvement over popularity: "
        f"+{pct_improvement(best['hit_rate@20'], pop['hit_rate@20']):.1f}%"
    )
    print(
        "ndcg@10     improvement over popularity: "
        f"+{pct_improvement(best['ndcg@10'], pop['ndcg@10']):.1f}%"
    )
    print(
        "ndcg@20     improvement over popularity: "
        f"+{pct_improvement(best['ndcg@20'], pop['ndcg@20']):.1f}%"
    )


def main() -> None:
    args = parse_args()
    results_df = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print_summary(results_df)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()