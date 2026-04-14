"""SGD-based and ALS-based matrix factorization for collaborative filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import trange
import implicit

from .baselines import evaluate_recommendations


@dataclass
class MatrixFactorizationSGD:
    """Matrix factorization model trained with SGD."""

    k: int = 50
    reg: float = 0.01
    learning_rate: float = 0.0005
    epochs: int = 20
    random_state: int = 42
    fit_batch_size: int | None = 500_000

    def __post_init__(self) -> None:
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None

    def fit(self, matrix: csr_matrix, verbose: bool = True) -> "MatrixFactorizationSGD":
        """SGD on observed entries; shuffled epochs, chunked by fit_batch_size."""
        m, n = matrix.shape
        rng = np.random.default_rng(self.random_state)
        self.P = 0.01 * rng.standard_normal((m, self.k))
        self.Q = 0.01 * rng.standard_normal((n, self.k))

        coo = matrix.tocoo()
        rows = coo.row.astype(np.int64)
        cols = coo.col.astype(np.int64)
        vals = coo.data.astype(np.float64)
        num_obs = len(vals)
        batch_size = num_obs if self.fit_batch_size is None else int(self.fit_batch_size)
        if batch_size <= 0:
            batch_size = num_obs

        iterator = trange(self.epochs, desc="MF epochs", disable=not verbose)
        for _ in iterator:
            order = rng.permutation(num_obs)
            for start in range(0, num_obs, batch_size):
                idx_block = order[start : start + batch_size]
                r = rows[idx_block]
                c = cols[idx_block]
                v = vals[idx_block]

                preds = np.sum(self.P[r] * self.Q[c], axis=1)
                errors = v - preds

                delta_p = self.learning_rate * (
                    errors[:, None] * self.Q[c] - self.reg * self.P[r]
                )
                delta_q = self.learning_rate * (
                    errors[:, None] * self.P[r] - self.reg * self.Q[c]
                )

                max_norm = 5.0
                dp_norm = np.linalg.norm(delta_p)
                if dp_norm > max_norm:
                    delta_p = delta_p * (max_norm / dp_norm)
                dq_norm = np.linalg.norm(delta_q)
                if dq_norm > max_norm:
                    delta_q = delta_q * (max_norm / dq_norm)

                unique_r, inv_r = np.unique(r, return_inverse=True)
                p_updates = np.zeros((len(unique_r), self.k), dtype=np.float64)
                np.add.at(p_updates, inv_r, delta_p)
                self.P[unique_r] += p_updates

                unique_c, inv_c = np.unique(c, return_inverse=True)
                q_updates = np.zeros((len(unique_c), self.k), dtype=np.float64)
                np.add.at(q_updates, inv_c, delta_q)
                self.Q[unique_c] += q_updates

            if np.isnan(self.P).any():
                print("Warning: NaN detected in P; stopping early.")
                break

        return self

    def recommend_user(self, user_idx: int, train_matrix: csr_matrix, k: int = 10) -> List[int]:
        """Recommend top-K unseen item indices for a user index."""
        if self.P is None or self.Q is None:
            raise ValueError("Model must be fitted before calling recommend_user.")

        scores = self.Q @ self.P[user_idx]
        seen_items = train_matrix[user_idx].indices
        scores[seen_items] = -np.inf

        top_k = min(k, int((scores != -np.inf).sum()))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]
        return sorted_indices.tolist()


class MatrixFactorizationALS:
    """ALS matrix factorization using the implicit library.

    We bypass implicit's recommend() method entirely because it has confusing
    orientation requirements. Instead we extract the learned user and item
    factor matrices directly and do scoring ourselves — same math, no confusion.

    implicit.fit() expects item x user (games x users), so we pass matrix.T.
    After fitting:
      - model.user_factors has shape (n_games, k)  [implicit calls items "users"]
      - model.item_factors has shape (n_users, k)  [implicit calls users "items"]
    So we swap them: our user factors = model.item_factors,
                     our item factors = model.user_factors.
    """

    def __init__(self, k: int = 50, reg: float = 0.01, iterations: int = 50, random_state: int = 42):
        self.k = k
        self.reg = reg
        self.iterations = iterations
        self.random_state = random_state
        self.model = implicit.als.AlternatingLeastSquares(
            factors=k,
            regularization=reg,
            iterations=iterations,
            random_state=random_state,
        )
        self.user_factors: np.ndarray | None = None  # shape: (n_users, k)
        self.item_factors: np.ndarray | None = None  # shape: (n_items, k)

    def fit(self, matrix: csr_matrix, verbose: bool = True) -> "MatrixFactorizationALS":
        """Fit ALS model. matrix must be users x items."""
        # implicit expects item x user, so we pass the transpose
        item_user = csr_matrix(matrix.T)
        self.model.fit(item_user, show_progress=verbose)
        # implicit swaps naming: what it calls "user_factors" are actually item factors
        # and what it calls "item_factors" are actually user factors.
        self.user_factors = np.array(self.model.item_factors)  # (n_users, k)
        self.item_factors = np.array(self.model.user_factors)  # (n_items, k)
        return self

    def recommend_user(self, user_idx: int, train_matrix: csr_matrix, k: int = 10) -> List[int]:
        """Recommend top-K unseen items by scoring directly with factor matrices."""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model must be fitted before calling recommend_user.")

        # Score all items for this user
        scores = self.item_factors @ self.user_factors[user_idx]

        # Mask seen items
        seen_items = train_matrix[user_idx].indices
        scores[seen_items] = -np.inf

        top_k = min(k, int((scores != -np.inf).sum()))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]
        return sorted_indices.tolist()


def build_test_items_by_user(
    test_df: pd.DataFrame,
    user_to_idx: Dict[int, int],
    game_to_idx: Dict[int, int],
) -> Dict[int, Sequence[int]]:
    """Convert leave-one-out test rows into user-indexed item-index targets."""
    df = test_df[
        test_df["user_id"].isin(user_to_idx) & test_df["app_id"].isin(game_to_idx)
    ].copy()
    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["game_idx"] = df["app_id"].map(game_to_idx)
    return df.groupby("user_idx")["game_idx"].apply(list).to_dict()


def evaluate_mf_leave_one_out(
    model,
    train_matrix: csr_matrix,
    test_items_by_user: Dict[int, Sequence[int]],
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate a model with Precision@K and Recall@K."""
    predictions: Dict[int, Sequence[int]] = {}
    for user_idx in test_items_by_user:
        predictions[user_idx] = model.recommend_user(user_idx, train_matrix, k=k)
    return evaluate_recommendations(predictions, test_items_by_user, k=k)
