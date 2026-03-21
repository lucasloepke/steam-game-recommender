"""Command-line experiment runner for Steam recommendation models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.baselines import evaluate_recommendations, popularity_recommendations, random_recommendations
from src.data_loader import (
    build_interaction_matrices,
    filter_recommendations,
    leave_one_out_split,
)
from src.matrix_factorization import MatrixFactorizationSGD, build_test_items_by_user, evaluate_mf_leave_one_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline and matrix factorization experiments on Steam interactions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/interactions_filtered.csv"),
        help="Path to filtered interactions CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/model_results.csv"),
        help="Path to save output results table CSV.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K threshold for Precision@K and Recall@K.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs for matrix factorization.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for SGD matrix factorization.",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=0.01,
        help="L2 regularization coefficient for matrix factorization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="List of latent dimensions for matrix factorization trials.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing raw Kaggle CSV files.",
    )
    parser.add_argument(
        "--min-user-reviews",
        type=int,
        default=20,
        help="Minimum interactions per user during filtering.",
    )
    parser.add_argument(
        "--min-game-reviews",
        type=int,
        default=200,
        help="Minimum interactions per game during filtering.",
    )
    return parser.parse_args()


def ensure_filtered_interactions(args: argparse.Namespace) -> Path:
    if args.input.exists():
        return args.input

    data_dir = args.data_dir
    recs_path = data_dir / "recommendations.csv"

    if not recs_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. To auto-create it, place "
            f"`recommendations.csv` in `{data_dir}`."
        )

    recommendations = pd.read_csv(recs_path)
    filtered = filter_recommendations(
        recommendations,
        min_user_reviews=args.min_user_reviews,
        min_game_reviews=args.min_game_reviews,
    )
    args.input.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.input, index=False)
    print(
        f"Created filtered interactions at {args.input} "
        f"(rows={len(filtered):,}, min_user_reviews={args.min_user_reviews}, "
        f"min_game_reviews={args.min_game_reviews})."
    )
    return args.input


def run_experiments(args: argparse.Namespace) -> pd.DataFrame:
    input_path = ensure_filtered_interactions(args)
    interactions = pd.read_csv(input_path)
    train_df, test_df = leave_one_out_split(interactions)

    train_data = build_interaction_matrices(train_df)
    user_ids = sorted(test_df["user_id"].unique().tolist())
    all_game_ids = sorted(train_df["app_id"].unique().tolist())
    test_items_by_user_id = test_df.groupby("user_id")["app_id"].apply(list).to_dict()

    rows: List[dict] = []

    pop_recs = popularity_recommendations(train_df, user_ids=user_ids, k=args.top_k)
    pop_scores = evaluate_recommendations(pop_recs, test_items_by_user_id, k=args.top_k)
    rows.append(
        {
            "model": "Popularity",
            f"precision@{args.top_k}": pop_scores["precision_at_k"],
            f"recall@{args.top_k}": pop_scores["recall_at_k"],
        }
    )

    rand_recs = random_recommendations(
        train_df,
        user_ids=user_ids,
        all_game_ids=all_game_ids,
        k=args.top_k,
        seed=args.seed,
    )
    rand_scores = evaluate_recommendations(rand_recs, test_items_by_user_id, k=args.top_k)
    rows.append(
        {
            "model": "Random",
            f"precision@{args.top_k}": rand_scores["precision_at_k"],
            f"recall@{args.top_k}": rand_scores["recall_at_k"],
        }
    )

    test_items_by_user_idx = build_test_items_by_user(
        test_df, train_data.user_to_idx, train_data.game_to_idx
    )
    for latent_k in args.latent_dims:
        model = MatrixFactorizationSGD(
            k=latent_k,
            reg=args.reg,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            random_state=args.seed,
        )
        model.fit(train_data.positive_matrix, verbose=True)
        mf_scores = evaluate_mf_leave_one_out(
            model=model,
            train_matrix=train_data.binary_matrix,
            test_items_by_user=test_items_by_user_idx,
            k=args.top_k,
        )
        rows.append(
            {
                "model": f"MF (k={latent_k})",
                f"precision@{args.top_k}": mf_scores["precision_at_k"],
                f"recall@{args.top_k}": mf_scores["recall_at_k"],
            }
        )

    score_col = f"precision@{args.top_k}"
    results_df = pd.DataFrame(rows).sort_values(score_col, ascending=False).reset_index(drop=True)
    return results_df


def main() -> None:
    args = parse_args()
    results_df = run_experiments(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    print("Model evaluation results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()

