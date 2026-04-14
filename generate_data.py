import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from src.data_loader import filter_recommendations

rng = np.random.default_rng(42)

n_users = 5000
n_games = 500
n_interactions = 80000

user_ids = rng.integers(10000, 99999, size=n_users)
app_ids = rng.integers(100, 9999, size=n_games)

rows = {
    "user_id": rng.choice(user_ids, size=n_interactions),
    "app_id": rng.choice(app_ids, size=n_interactions),
    "is_recommended": rng.integers(0, 2, size=n_interactions),
    "hours": np.round(rng.exponential(scale=20, size=n_interactions), 1),
    "date": pd.date_range("2020-01-01", periods=n_interactions, freq="1h").strftime("%Y-%m-%d"),
}

recs = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "app_id"]).reset_index(drop=True)

games = pd.DataFrame({
    "app_id": app_ids,
    "title": [f"Game {i}" for i in app_ids],
})

users = pd.DataFrame({
    "user_id": user_ids,
    "products": rng.integers(1, 100, size=n_users),
    "reviews": rng.integers(0, 20, size=n_users),
})

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

recs.to_csv(data_dir / "recommendations.csv", index=False)
games.to_csv(data_dir / "games.csv", index=False)
users.to_csv(data_dir / "users.csv", index=False)

filtered = filter_recommendations(recs, min_user_reviews=3, min_game_reviews=10)
filtered.to_csv(data_dir / "interactions_filtered.csv", index=False)

print("Done! Files created:")
for f in sorted(data_dir.iterdir()):
    print(f"  {f.name}: {f.stat().st_size:,} bytes")
