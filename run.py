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

results = []

# ── Baselines ──────────────────────────────────────────────────────────────────
print('\nRunning baselines...')

pop_recs = popularity_recommendations(train_df, user_ids=user_ids, k=20)
pop_10 = evaluate_recommendations(pop_recs, test_items_by_user_id, k=10)
pop_20 = evaluate_recommendations(pop_recs, test_items_by_user_id, k=20)
results.append({'model': 'Popularity', 'hit_rate@10': pop_10['hit_rate_at_k'], 'hit_rate@20': pop_20['hit_rate_at_k'], 'ndcg@10': pop_10['ndcg_at_k'], 'ndcg@20': pop_20['ndcg_at_k']})
print(f"  Popularity  -> hit_rate@10: {pop_10['hit_rate_at_k']:.6f}  hit_rate@20: {pop_20['hit_rate_at_k']:.6f}  ndcg@10: {pop_10['ndcg_at_k']:.6f}  ndcg@20: {pop_20['ndcg_at_k']:.6f}")

rand_recs = random_recommendations(train_df, user_ids=user_ids, all_game_ids=all_game_ids, k=20, seed=42)
rand_10 = evaluate_recommendations(rand_recs, test_items_by_user_id, k=10)
rand_20 = evaluate_recommendations(rand_recs, test_items_by_user_id, k=20)
results.append({'model': 'Random', 'hit_rate@10': rand_10['hit_rate_at_k'], 'hit_rate@20': rand_20['hit_rate_at_k'], 'ndcg@10': rand_10['ndcg_at_k'], 'ndcg@20': rand_20['ndcg_at_k']})
print(f"  Random      -> hit_rate@10: {rand_10['hit_rate_at_k']:.6f}  hit_rate@20: {rand_20['hit_rate_at_k']:.6f}  ndcg@10: {rand_10['ndcg_at_k']:.6f}  ndcg@20: {rand_20['ndcg_at_k']:.6f}")

# ── SGD (for comparison) ───────────────────────────────────────────────────────
print('\nTraining SGD (k=50) for comparison...')
sgd = MatrixFactorizationSGD(k=50, reg=0.01, learning_rate=0.005, epochs=50, random_state=42)
sgd.fit(train_data.hours_matrix, verbose=True)
sgd_10 = evaluate_mf_leave_one_out(sgd, train_data.hours_matrix, test_items_by_user_idx, k=10)
sgd_20 = evaluate_mf_leave_one_out(sgd, train_data.hours_matrix, test_items_by_user_idx, k=20)
results.append({'model': 'SGD (k=50)', 'hit_rate@10': sgd_10['hit_rate_at_k'], 'hit_rate@20': sgd_20['hit_rate_at_k'], 'ndcg@10': sgd_10['ndcg_at_k'], 'ndcg@20': sgd_20['ndcg_at_k']})
print(f"  SGD (k=50)  -> hit_rate@10: {sgd_10['hit_rate_at_k']:.6f}  hit_rate@20: {sgd_20['hit_rate_at_k']:.6f}  ndcg@10: {sgd_10['ndcg_at_k']:.6f}  ndcg@20: {sgd_20['ndcg_at_k']:.6f}")

# ── Best model: ALS k=200 ──────────────────────────────────────────────────────
print('\nTraining best model: ALS (k=200)...')
als = MatrixFactorizationALS(k=200, reg=0.01, iterations=50, random_state=42)
als.fit(train_data.hours_matrix, verbose=True)
als_10 = evaluate_mf_leave_one_out(als, train_data.hours_matrix, test_items_by_user_idx, k=10)
als_20 = evaluate_mf_leave_one_out(als, train_data.hours_matrix, test_items_by_user_idx, k=20)
results.append({'model': 'ALS (k=200)', 'hit_rate@10': als_10['hit_rate_at_k'], 'hit_rate@20': als_20['hit_rate_at_k'], 'ndcg@10': als_10['ndcg_at_k'], 'ndcg@20': als_20['ndcg_at_k']})
print(f"  ALS (k=200) -> hit_rate@10: {als_10['hit_rate_at_k']:.6f}  hit_rate@20: {als_20['hit_rate_at_k']:.6f}  ndcg@10: {als_10['ndcg_at_k']:.6f}  ndcg@20: {als_20['ndcg_at_k']:.6f}")

# ── Final results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values('hit_rate@10', ascending=False).reset_index(drop=True)
results_df = results_df[['model', 'hit_rate@10', 'hit_rate@20', 'ndcg@10', 'ndcg@20']]
print()
print('=' * 70)
print('FINAL RESULTS')
print('=' * 70)
print(results_df.to_string())

pop_hr_10  = results_df[results_df['model'] == 'Popularity']['hit_rate@10'].values[0]
pop_hr_20  = results_df[results_df['model'] == 'Popularity']['hit_rate@20'].values[0]
pop_ndcg_10 = results_df[results_df['model'] == 'Popularity']['ndcg@10'].values[0]
pop_ndcg_20 = results_df[results_df['model'] == 'Popularity']['ndcg@20'].values[0]
best = results_df.iloc[0]
print(f"\nBest model:                              {best['model']}")
print(f"hit_rate@10 improvement over popularity: +{(best['hit_rate@10'] - pop_hr_10)   / pop_hr_10   * 100:.1f}%")
print(f"hit_rate@20 improvement over popularity: +{(best['hit_rate@20'] - pop_hr_20)   / pop_hr_20   * 100:.1f}%")
print(f"ndcg@10     improvement over popularity: +{(best['ndcg@10']     - pop_ndcg_10) / pop_ndcg_10 * 100:.1f}%")
print(f"ndcg@20     improvement over popularity: +{(best['ndcg@20']     - pop_ndcg_20) / pop_ndcg_20 * 100:.1f}%")