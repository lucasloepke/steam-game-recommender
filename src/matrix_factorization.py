"""SGD-based matrix factorization for collaborative filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import trange

from .baselines import evaluate_recommendations


@dataclass
class MatrixFactorizationSGD:
    """Matrix factorization model trained with SGD."""

    k: int = 50
    reg: float = 0.01
    learning_rate: float = 0.0005
    epochs: int = 20
    random_state: int = 42
    # Max observed pairs per vectorized chunk (None = one pass over all pairs; may use a lot of RAM).
    fit_batch_size: int | None = 500_000

    def __post_init__(self) -> None:
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None

    def fit(self, matrix: csr_matrix, verbose: bool = True) -> "MatrixFactorizationSGD":
        """SGD on observed entries; shuffled epochs, chunked by ``fit_batch_size``."""
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

                delta_p = np.clip(delta_p, -1.0, 1.0)
                delta_q = np.clip(delta_q, -1.0, 1.0)

                np.add.at(self.P, r, delta_p)
                np.add.at(self.Q, c, delta_q)

            if np.isnan(self.P).any():
                print(
                    "Warning: NaN detected in user factors (P) after an epoch; "
                    "stopping matrix factorization training early."
                )
                break

        return self

    def recommend_user(self, user_idx: int, train_matrix: csr_matrix, k: int = 10) -> List[int]:
        """Recommend top-K unseen item indices for a user index."""
        if self.P is None or self.Q is None:
            raise ValueError("Model must be fitted before calling recommend_user.")

        user_vector = self.P[user_idx]
        scores = self.Q @ user_vector

        seen_items = set(train_matrix[user_idx].indices.tolist())
        candidate_indices = [idx for idx in range(train_matrix.shape[1]) if idx not in seen_items]
        if not candidate_indices:
            return []

        candidate_scores = scores[candidate_indices]
        top_k = min(k, len(candidate_indices))
        top_local = np.argpartition(-candidate_scores, top_k - 1)[:top_k]
        sorted_local = top_local[np.argsort(-candidate_scores[top_local])]
        return [int(candidate_indices[i]) for i in sorted_local]


def build_test_items_by_user(
    test_df: pd.DataFrame,
    user_to_idx: Dict[int, int],
    game_to_idx: Dict[int, int],
) -> Dict[int, Sequence[int]]:
    """Convert leave-one-out test rows into user-indexed item-index targets."""
    out: Dict[int, Sequence[int]] = {}
    for _, row in test_df.iterrows():
        user_id = int(row["user_id"])
        game_id = int(row["app_id"])
        if user_id in user_to_idx and game_id in game_to_idx:
            out[user_to_idx[user_id]] = [game_to_idx[game_id]]
    return out


def evaluate_mf_leave_one_out(
    model: MatrixFactorizationSGD,
    train_matrix: csr_matrix,
    test_items_by_user: Dict[int, Sequence[int]],
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate matrix factorization model with Precision@K and Recall@K."""
    predictions: Dict[int, Sequence[int]] = {}
    for user_idx in test_items_by_user:
        predictions[user_idx] = model.recommend_user(user_idx, train_matrix, k=k)
    return evaluate_recommendations(predictions, test_items_by_user, k=k)

