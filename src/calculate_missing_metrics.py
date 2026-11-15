
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score


BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTION_FILES = {
    "Random Forest": (
        "reports/random_forest/within_location/test_predictions.csv",
        "actual_activity",
        "predicted_activity",
    ),
    "SVM": (
        "reports/svm/within_location/test_predictions.csv",
        "true_label",
        "predicted_label",
    ),
    "1D-CNN": (
        "reports/cnn/within_location/test_predictions.csv",
        "actual_label",
        "predicted_label",
    ),
}


print("=" * 65)
print("WITHIN-LOCATION WEIGHTED-F1 CALCULATION")
print("=" * 65)

for model, (relative_path, true_col, pred_col) in PREDICTION_FILES.items():

    path = BASE_DIR / relative_path
    df = pd.read_csv(path)

    y_true = df[true_col].astype(str).str.strip()
    y_pred = df[pred_col].astype(str).str.strip()

    score = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"\nModel: {model}")
    print(f"Test samples: {len(df)}")
    print(f"Weighted-F1: {score:.6f} ({score * 100:.2f}%)")
