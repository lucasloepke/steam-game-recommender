import pandas as pd
import sys
from pathlib import Path

ROOT = Path('.')
sys.path.append(str(ROOT))

from src.baselines import evaluate_recommendations, popularity_recommendations, random_recommendations
from src.data_loader import build_interaction_matrices, leave_one_out_split, filter_recommendations
from src.matrix_factorization import MatrixFactorizationSGD, build_test_items_by_user, evaluate_mf_leave_one_out

print('Loading data...')
interactions = pd.read_csv('data/recommendations.csv')
interactions = filter_recommendations(interactions, min_user_reviews=50, min_game_reviews=500)
interactions.to_csv('data/interactions_filtered.csv', index=False)

print('Splitting...')
train_df, test_df = leave_one_out_split(interactions)

print('Building matrices...')
train_data = build_interaction_matrices(train_df)
user_ids = sorted(test_df['user_id'].unique().tolist())
all_game_ids = sorted(train_df['app_id'].unique().tolist())
test_items_by_user_id = test_df.groupby('user_id')['app_id'].apply(list).to_dict()
test_items_by_user_idx = build_test_items_by_user(test_df, train_data.user_to_idx, train_data.game_to_idx)

k_eval = 10
results = []

print('Running popularity baseline...')
pop_recs = popularity_recommendations(train_df, user_ids=user_ids, k=k_eval)
pop_scores = evaluate_recommendations(pop_recs, test_items_by_user_id, k=k_eval)
results.append({'model': 'Popularity', 'precision@10': pop_scores['precision_at_k'], 'recall@10': pop_scores['recall_at_k']})

print('Running random baseline...')
rand_recs = random_recommendations(train_df, user_ids=user_ids, all_game_ids=all_game_ids, k=k_eval, seed=42)
rand_scores = evaluate_recommendations(rand_recs, test_items_by_user_id, k=k_eval)
results.append({'model': 'Random', 'precision@10': rand_scores['precision_at_k'], 'recall@10': rand_scores['recall_at_k']})

print('Training MF models...')
for latent_k in [20, 50, 100]:
    model = MatrixFactorizationSGD(k=latent_k, reg=0.01, learning_rate=0.01, epochs=20, random_state=42)
    model.fit(train_data.hours_matrix, verbose=True)
    mf_scores = evaluate_mf_leave_one_out(model, train_data.hours_matrix, test_items_by_user_idx, k=k_eval)
    results.append({
        'model': f'MF (k={latent_k})',
        'precision@10': mf_scores['precision_at_k'],
        'recall@10': mf_scores['recall_at_k'],
    })

results_df = pd.DataFrame(results).sort_values('precision@10', ascending=False).reset_index(drop=True)
print()
print(results_df.to_string())
