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
    #Matrix factorization model trained with SGD.

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

                # FIX 1: Use norm-based gradient clipping instead of hard ±1.0 clip.
                # The original ±1.0 hard clip was too aggressive relative to the
                # 0.01-scale initialization and was suppressing the gradient signal.
                max_norm = 5.0
                dp_norm = np.linalg.norm(delta_p)
                if dp_norm > max_norm:
                    delta_p = delta_p * (max_norm / dp_norm)
                dq_norm = np.linalg.norm(delta_q)
                if dq_norm > max_norm:
                    delta_q = delta_q * (max_norm / dq_norm)

                # FIX 2: Accumulate updates per unique index to avoid slow np.add.at
                # on repeated indices. Group delta_p rows by user index and sum them,
                # then apply a single indexed addition per unique user.
                unique_r, inv_r = np.unique(r, return_inverse=True)
                p_updates = np.zeros((len(unique_r), self.k), dtype=np.float64)
                np.add.at(p_updates, inv_r, delta_p)
                self.P[unique_r] += p_updates

                unique_c, inv_c = np.unique(c, return_inverse=True)
                q_updates = np.zeros((len(unique_c), self.k), dtype=np.float64)
                np.add.at(q_updates, inv_c, delta_q)
                self.Q[unique_c] += q_updates

            if np.isnan(self.P).any():
                print(
                    "Warning: NaN detected in user factors (P) after an epoch; "
                    "stopping matrix factorization training early."
                )
                break

        return self

    def recommend_user(self, user_idx: int, train_matrix: csr_matrix, k: int = 10) -> List[int]:
        """Recommend top-K unseen item indices for a user index.

        FIX 3: Uses vectorized masking instead of a Python set comprehension,
        which is significantly faster for large item catalogs.
        """
        if self.P is None or self.Q is None:
            raise ValueError("Model must be fitted before calling recommend_user.")

        scores = self.Q @ self.P[user_idx]

        # Mask already-seen items with -inf so they are never selected
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
    #Convert leave-one-out test rows into user-indexed item-index targets.

    
    df = test_df[
        test_df["user_id"].isin(user_to_idx) & test_df["app_id"].isin(game_to_idx)
    ].copy()
    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["game_idx"] = df["app_id"].map(game_to_idx)
    return df.groupby("user_idx")["game_idx"].apply(list).to_dict()


def evaluate_mf_leave_one_out(
    model: MatrixFactorizationSGD,
    train_matrix: csr_matrix,
    test_items_by_user: Dict[int, Sequence[int]],
    k: int = 10,
) -> Dict[str, float]:
    #Evaluate matrix factorization model with Precision@K and Recall@K.
    predictions: Dict[int, Sequence[int]] = {}
    for user_idx in test_items_by_user:
        predictions[user_idx] = model.recommend_user(user_idx, train_matrix, k=k)
    return evaluate_recommendations(predictions, test_items_by_user, k=k)