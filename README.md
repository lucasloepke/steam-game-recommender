# Steam Game Recommender

Collaborative filtering project for Steam game recommendation using the Kaggle dataset **Game Recommendations on Steam** by Anton Kozyriev.

## Project Structure

```text
steam-game-recommender/
├── data/               # local datasets (gitignored)
├── notebooks/
│   ├── eda.ipynb       # exploratory data analysis
│   └── modeling.ipynb  # model training and evaluation
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── baselines.py
│   ├── matrix_factorization.py
│   ├── download_data.py
│   └── run_modeling.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset Files

Place the following files in `data/`:

- `recommendations.csv` with columns: `app_id`, `user_id`, `is_recommended`, `hours`, `date`
- `games.csv` with columns: `app_id`, `title`, `rating`, `positive_ratio`, `user_reviews`, `price_final`
- `users.csv` with columns: `user_id`, `products`, `reviews`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset (KaggleHub)

Install dependencies first, then run:

```bash
python -m src.download_data
```

To also copy the downloaded CSV files into your local `data/` folder:

```bash
python -m src.download_data --copy-to-data
```

This script downloads the Kaggle dataset:
- `antonkozyriev/game-recommendations-on-steam`

and validates these files exist:
- `recommendations.csv`
- `games.csv`
- `users.csv`

## Implementation

- Data loading and filtering (defaults in `filter_recommendations`: `user >= 5`, `game >= 50`; CLI runner uses stricter defaults)
- Sparse interaction matrices:
  - Binary: `M_ij = 1` if recommended, else `0`
  - Implicit: `M_ij = hours`
- Dataset statistics and sparsity computation
- Leave-one-out split (most recent item per user in test)
- Baselines:
  - Popularity baseline
  - Random baseline
- Matrix factorization with SGD:
  - Objective:  
    `L = Σ_(i,j)∈Ω (M_ij - p_i^T q_j)^2 + λ(||p_i||^2 + ||q_j||^2)`
- Evaluation: `Precision@K` and `Recall@K`

## Notebooks

- `notebooks/eda.ipynb`
  - Loads data and prints statistics
  - Plots review-count distributions (log scale)
  - Plots sparsity heatmap for random `500x500` submatrix
  - Saves filtered interactions to `data/interactions_filtered.csv`
- `notebooks/modeling.ipynb`
  - Loads filtered interactions
  - Runs popularity and random baselines
  - Trains matrix factorization with `k ∈ {20, 50, 100}`
  - Produces results table with `Precision@10` and `Recall@10`

## CLI Experiment Runner

Run the full modeling pipeline from the terminal:

```bash
python -m src.run_modeling --input data/interactions_filtered.csv --output data/model_results.csv
```

If `data/interactions_filtered.csv` is missing, the runner builds it from `data/recommendations.csv` using `--min-user-reviews` and `--min-game-reviews` (defaults: 20 and 200).

If you did **not** copy files into local `data/`, point to the KaggleHub cache path:

```bash
python -m src.run_modeling \
  --data-dir "/absolute/path/printed/by/src.download_data" \
  --input data/interactions_filtered.csv \
  --output data/model_results.csv
```

Optional overrides:

```bash
python -m src.run_modeling \
  --top-k 10 \
  --latent-dims 20 50 100 \
  --epochs 20 \
  --learning-rate 0.01 \
  --reg 0.01 \
  --seed 42
```
