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
│   └── download_data.py
├── run.py              # production experiment entrypoint
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

## Data

`run.py` expects Steam interactions in:

- `data/recommendations.csv`

with at least these columns:

- `app_id`
- `user_id`
- `is_recommended`
- `hours`
- `date`

You can create `data/recommendations.csv` in either of two ways:

1) Manually place Kaggle CSV files in `data/`
2) Download via KaggleHub

### Download Dataset (KaggleHub)

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

## Run Procedure

Standard run:

```bash
python run.py
```

This pipeline performs:

- Filtering (`min_user_reviews=50`, `min_game_reviews=500`)
- Leave-one-out split on positive interactions
- Baselines (Popularity and Random)
- Matrix factorization models (SGD and ALS)
- Ranking evaluation with `HitRate@10/20` and `NDCG@10/20`

Outputs:

- `data/interactions_filtered.csv`
- `data/model_results.csv`

Optional CLI overrides:

```bash
python run.py \
  --input data/recommendations.csv \
  --filtered-output data/interactions_filtered.csv \
  --output data/model_results.csv \
  --min-user-reviews 50 \
  --min-game-reviews 500
```

## Notebooks

- `notebooks/eda.ipynb`: exploratory data analysis
- `notebooks/modeling.ipynb`: iterative modeling experiments
