"""
Steam Game Recommender — Final Evaluation
==========================================
Best model: ALS (k=200, reg=0.01, iterations=50)
Trained on log1p(hours_played) implicit feedback signal.

Results vs baselines:
  - ALS (k=200): precision@10 ~0.0086  (+281% over popularity)
  - SGD (k=50):  precision@10 ~0.0028  (+23%  over popularity)
  - Popularity:  precision@10 ~0.0023
  - Random:      precision@10 ~0.0008
"""

import pandas as pd
import sys
from pathlib import Path

ROOT = Path('.')
sys.path.append(str(ROOT))

from src.baselines import evaluate_recommendations, popularity_recommendations, random_recommendations
from src.data_loader import build_interaction_matrices, leave_one_out_split, filter_recommendations
from src.matrix_factorization import MatrixFactorizationALS, MatrixFactorizationSGD, build_test_items_by_user, evaluate_mf_leave_one_out

# ── Data loading ───────────────────────────────────────────────────────────────
print('Loading data...')
interactions = pd.read_csv('data/recommendations.csv')
interactions = filter_recommendations(interactions, min_user_reviews=50, min_game_reviews=500)
interactions.to_csv('data/interactions_filtered.csv', index=False)

# ── Train/test split ───────────────────────────────────────────────────────────
print('Splitting (leave-one-out on positive interactions)...')
train_df, test_df = leave_one_out_split(interactions)
print(f"  Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

# ── Build interaction matrices ─────────────────────────────────────────────────
print('Building matrices...')
train_data = build_interaction_matrices(train_df)
user_ids = sorted(test_df['user_id'].unique().tolist())
all_game_ids = sorted(train_df['app_id'].unique().tolist())
test_items_by_user_id = test_df.groupby('user_id')['app_id'].apply(list).to_dict()
test_items_by_user_idx = build_test_items_by_user(test_df, train_data.user_to_idx, train_data.game_to_idx)

k_eval = 10
results = []

# ── Baselines ──────────────────────────────────────────────────────────────────
print('\nRunning baselines...')

pop_recs = popularity_recommendations(train_df, user_ids=user_ids, k=k_eval)
pop_scores = evaluate_recommendations(pop_recs, test_items_by_user_id, k=k_eval)
results.append({'model': 'Popularity', 'precision@10': pop_scores['precision_at_k'], 'recall@10': pop_scores['recall_at_k']})
print(f"  Popularity  -> precision@10: {pop_scores['precision_at_k']:.6f}")

rand_recs = random_recommendations(train_df, user_ids=user_ids, all_game_ids=all_game_ids, k=k_eval, seed=42)
rand_scores = evaluate_recommendations(rand_recs, test_items_by_user_id, k=k_eval)
results.append({'model': 'Random', 'precision@10': rand_scores['precision_at_k'], 'recall@10': rand_scores['recall_at_k']})
print(f"  Random      -> precision@10: {rand_scores['precision_at_k']:.6f}")

# ── SGD baseline (for comparison) ─────────────────────────────────────────────
print('\nTraining SGD (k=50) for comparison...')
sgd = MatrixFactorizationSGD(k=50, reg=0.01, learning_rate=0.005, epochs=50, random_state=42)
sgd.fit(train_data.hours_matrix, verbose=True)
sgd_scores = evaluate_mf_leave_one_out(sgd, train_data.hours_matrix, test_items_by_user_idx, k=k_eval)
results.append({'model': 'SGD (k=50)', 'precision@10': sgd_scores['precision_at_k'], 'recall@10': sgd_scores['recall_at_k']})
print(f"  SGD (k=50)  -> precision@10: {sgd_scores['precision_at_k']:.6f}")

# ── Best model: ALS k=200 ──────────────────────────────────────────────────────
print('\nTraining best model: ALS (k=200)...')
als = MatrixFactorizationALS(k=200, reg=0.01, iterations=50, random_state=42)
als.fit(train_data.hours_matrix, verbose=True)
als_scores = evaluate_mf_leave_one_out(als, train_data.hours_matrix, test_items_by_user_idx, k=k_eval)
results.append({'model': 'ALS (k=200)', 'precision@10': als_scores['precision_at_k'], 'recall@10': als_scores['recall_at_k']})
print(f"  ALS (k=200) -> precision@10: {als_scores['precision_at_k']:.6f}")

# ── Final results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values('precision@10', ascending=False).reset_index(drop=True)
print()
print('=' * 50)
print('FINAL RESULTS')
print('=' * 50)
print(results_df.to_string())

pop_precision = results_df[results_df['model'] == 'Popularity']['precision@10'].values[0]
best = results_df.iloc[0]
improvement = (best['precision@10'] - pop_precision) / pop_precision * 100
print(f"\nBest model:                  {best['model']}")
print(f"Best precision@10:           {best['precision@10']:.6f}")
print(f"Improvement over popularity: +{improvement:.1f}%")
