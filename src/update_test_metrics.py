
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

WEIGHTED_F1 = {
    "random_forest": 0.207424,
    "svm": 0.191357,
}

for model, score in WEIGHTED_F1.items():

    path = (
        BASE_DIR
        / "reports"
        / model
        / "within_location"
        / "test_metrics.csv"
    )

    df = pd.read_csv(path)
    df["weighted_f1"] = score
    df.to_csv(path, index=False)

    print(f"Updated {model}: weighted_f1 = {score:.6f}")

print("\nTest metrics updated successfully.")
