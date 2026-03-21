"""Baseline recommenders and top-K evaluation metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def precision_at_k(recommended: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Compute Precision@K for one user."""
    if k <= 0:
        return 0.0
    rec_k = list(recommended)[:k]
    if not rec_k:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in rec_k if item in relevant_set)
    return hits / float(k)


def recall_at_k(recommended: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Compute Recall@K for one user."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    rec_k = list(recommended)[:k]
    hits = sum(1 for item in rec_k if item in relevant_set)
    return hits / float(len(relevant_set))


def evaluate_recommendations(
    recommendations_by_user: Dict[int, Sequence[int]],
    test_items_by_user: Dict[int, Sequence[int]],
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate user recommendations with mean Precision@K and Recall@K."""
    precisions: List[float] = []
    recalls: List[float] = []

    for user_id, relevant_items in test_items_by_user.items():
        recommended_items = recommendations_by_user.get(user_id, [])
        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items, k))

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
    }


def build_user_history(train_df: pd.DataFrame) -> Dict[int, set[int]]:
    """Build a mapping from user id to interacted game ids in the train set."""
    history = train_df.groupby("user_id")["app_id"].apply(set).to_dict()
    return {int(user_id): set(map(int, games)) for user_id, games in history.items()}


def popularity_ranking(train_df: pd.DataFrame) -> List[int]:
    """Rank games by total number of positive recommendations."""
    popular = (
        train_df[train_df["is_recommended"] == 1]
        .groupby("app_id")
        .size()
        .sort_values(ascending=False)
    )
    return [int(game_id) for game_id in popular.index.to_list()]


def popularity_recommendations(
    train_df: pd.DataFrame,
    user_ids: Iterable[int],
    k: int = 10,
) -> Dict[int, List[int]]:
    """Recommend top-K popular unseen games to each user."""
    ranking = popularity_ranking(train_df)
    history = build_user_history(train_df)
    recs: Dict[int, List[int]] = {}

    for uid in user_ids:
        user_history = history.get(int(uid), set())
        unseen_ranked = [gid for gid in ranking if gid not in user_history]
        recs[int(uid)] = unseen_ranked[:k]
    return recs


def random_recommendations(
    train_df: pd.DataFrame,
    user_ids: Iterable[int],
    all_game_ids: Sequence[int],
    k: int = 10,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Recommend K random unseen games to each user."""
    rng = np.random.default_rng(seed)
    history = build_user_history(train_df)
    all_games = np.array(list(map(int, all_game_ids)), dtype=np.int64)
    recs: Dict[int, List[int]] = {}

    for uid in user_ids:
        user_history = history.get(int(uid), set())
        candidates = [gid for gid in all_games if int(gid) not in user_history]
        if not candidates:
            recs[int(uid)] = []
            continue
        sample_size = min(k, len(candidates))
        recs[int(uid)] = [int(x) for x in rng.choice(candidates, size=sample_size, replace=False)]
    return recs

