"""Data loading and preprocessing utilities for Steam recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass
class InteractionData:
    """Container for sparse interaction matrices and id mappings."""

    binary_matrix: csr_matrix
    hours_matrix: csr_matrix
    positive_matrix: csr_matrix
    user_to_idx: Dict[int, int]
    idx_to_user: Dict[int, int]
    game_to_idx: Dict[int, int]
    idx_to_game: Dict[int, int]


def load_steam_data(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load recommendations, games, and users CSV files from a data directory."""
    data_path = Path(data_dir)
    recs = pd.read_csv(data_path / "recommendations.csv")
    games = pd.read_csv(data_path / "games.csv")
    users = pd.read_csv(data_path / "users.csv")
    return recs, games, users


def filter_recommendations(
    recommendations: pd.DataFrame,
    min_user_reviews: int = 5,
    min_game_reviews: int = 50,
) -> pd.DataFrame:
    """Filter interactions to users and games with enough review activity."""
    df = recommendations.copy()

    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_reviews].index
    df = df[df["user_id"].isin(valid_users)]

    game_counts = df["app_id"].value_counts()
    valid_games = game_counts[game_counts >= min_game_reviews].index
    df = df[df["app_id"].isin(valid_games)]

    df = df.reset_index(drop=True)
    print(
        "Filtered interactions:"
        f" rows={len(df):,},"
        f" users={df['user_id'].nunique():,},"
        f" games={df['app_id'].nunique():,}"
    )
    return df


def build_interaction_matrices(interactions: pd.DataFrame) -> InteractionData:
    """Build sparse user--game matrices from interactions.

    - ``binary_matrix``: ``is_recommended`` as stored values (0/1) for every row.
    - ``hours_matrix``: ``log1p(hours)`` for every row.
    - ``positive_matrix``: only rows with ``is_recommended == 1``, values ``1.0``.
    """
    users = np.unique(interactions["user_id"].to_numpy())
    games = np.unique(interactions["app_id"].to_numpy())

    user_to_idx = {int(uid): idx for idx, uid in enumerate(users)}
    game_to_idx = {int(gid): idx for idx, gid in enumerate(games)}
    idx_to_user = {idx: int(uid) for uid, idx in user_to_idx.items()}
    idx_to_game = {idx: int(gid) for gid, idx in game_to_idx.items()}

    rows = interactions["user_id"].map(user_to_idx).to_numpy(dtype=np.int64)
    cols = interactions["app_id"].map(game_to_idx).to_numpy(dtype=np.int64)
    binary_values = interactions["is_recommended"].astype(np.float64).to_numpy()
    hours_values = np.log1p(interactions["hours"].astype(np.float64).to_numpy())

    shape = (len(users), len(games))
    binary_matrix = csr_matrix((binary_values, (rows, cols)), shape=shape)
    hours_matrix = csr_matrix((hours_values, (rows, cols)), shape=shape)

    positive_mask = interactions["is_recommended"].to_numpy() == 1
    pos_rows = rows[positive_mask]
    pos_cols = cols[positive_mask]
    pos_values = np.ones(pos_rows.shape[0], dtype=np.float64)
    positive_matrix = csr_matrix((pos_values, (pos_rows, pos_cols)), shape=shape)

    return InteractionData(
        binary_matrix=binary_matrix,
        hours_matrix=hours_matrix,
        positive_matrix=positive_matrix,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        game_to_idx=game_to_idx,
        idx_to_game=idx_to_game,
    )


def compute_dataset_statistics(matrix: csr_matrix) -> Dict[str, float]:
    """Compute number of users, games, interactions, and sparsity."""
    m, n = matrix.shape
    nnz = int(matrix.nnz)
    sparsity = 1.0 - (nnz / float(m * n))
    return {"users": m, "games": n, "interactions": nnz, "sparsity": sparsity}


def print_dataset_statistics(matrix: csr_matrix, title: str = "Dataset statistics") -> None:
    """Print formatted summary statistics for a sparse interaction matrix."""
    stats = compute_dataset_statistics(matrix)
    print(f"{title}:")
    print(f"  Users (m): {stats['users']:,}")
    print(f"  Games (n): {stats['games']:,}")
    print(f"  Interactions (nnz): {stats['interactions']:,}")
    print(f"  Sparsity: {stats['sparsity']:.6f}")


def leave_one_out_split(
    interactions: pd.DataFrame,
    time_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent positive interaction as the test item.

    Everything that occurred AFTER that held-out interaction is also removed
    from training to prevent any future leakage. This means:
      - Test set:  one row per user — their last positive interaction
      - Train set: all rows that occurred ON OR BEFORE the date of the
                   held-out item, excluding the held-out item itself.

    Users with no positive interactions are excluded from the test set
    entirely and remain fully in train.
    """
    df = interactions.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(["user_id", time_col]).reset_index(drop=True)

    # Find each user's most recent positive interaction
    positive_df = df[df["is_recommended"] == 1]
    test_idx = positive_df.groupby("user_id").tail(1).index
    test_df = df.loc[test_idx].reset_index(drop=True)

    # Build a lookup: user_id -> date of their held-out item
    cutoff_dates = test_df.set_index("user_id")[time_col]

    # For each row in the full df, keep it in train only if:
    #   1. It is not the held-out row itself, AND
    #   2. Its date is <= the cutoff date for that user
    #      (or the user has no test item, in which case keep everything)
    df["_cutoff"] = df["user_id"].map(cutoff_dates)

    train_mask = (
        (~df.index.isin(test_idx)) &
        (df[time_col].isna() | df["_cutoff"].isna() | (df[time_col] <= df["_cutoff"]))
    )

    train_df = df[train_mask].drop(columns=["_cutoff"]).reset_index(drop=True)

    print(
        f"Split: train={len(train_df):,} rows, test={len(test_df):,} rows"
        f" ({test_df['user_id'].nunique():,} users)"
    )
    return train_df, test_df
