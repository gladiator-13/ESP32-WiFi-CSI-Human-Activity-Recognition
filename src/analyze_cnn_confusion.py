
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIRS = {
    "Random Forest": ROOT / "reports/random_forest/within_location",
    "SVM": ROOT / "reports/svm/within_location",
    "1D-CNN": ROOT / "reports/cnn/within_location",
}

OUTPUT_DIR = ROOT / "reports/model_comparison/within_location"

ACTIVITIES = {
    1: "Standing",
    2: "Sitting",
    3: "Walk",
    4: "Jog in place",
    5: "Turn around",
    6: "Sit down, stand up",
    7: "Jump in place",
    8: "Bend and pick up",
    9: "Lunges",
    10: "Wave",
    11: "Clap",
    12: "Boxing",
    13: "Arms up and down",
    14: "Push forward",
    15: "Arm circles",
    16: "Typing",
    17: "Phone call",
    18: "Writing notes",
    19: "Reading a book",
    20: "Combing hair",
}



def load_predictions(model_name, directory):
    path = directory / "test_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = pd.read_csv(path)

    # Handle different column conventions across models.
    if model_name == "Random Forest":
        actual_col = "actual_activity"
        predicted_col = "predicted_activity"

    elif model_name == "SVM":
        actual_col = "true_label"
        predicted_col = "predicted_label"

    elif model_name == "1D-CNN":
        actual_col = "actual_label"
        predicted_col = "predicted_label"

    else:
        raise ValueError(f"Unknown model: {model_name}")

    required = {actual_col, predicted_col}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{model_name}: missing columns {required}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert activity names such as "A01" or "Activity_01"
    # into integer labels 1–20.
    def convert_label(value):
        if pd.isna(value):
            raise ValueError(f"{model_name}: missing activity label")

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, (float, np.floating)) and value.is_integer():
            return int(value)

        value = str(value).strip()

        import re
        match = re.search(r"(\d+)$", value)

        if match:
            return int(match.group(1))

        raise ValueError(
            f"{model_name}: cannot interpret activity label {value}"
        )

    y_true = df[actual_col].map(convert_label).to_numpy(dtype=int)
    y_pred = df[predicted_col].map(convert_label).to_numpy(dtype=int)

    if len(y_true) != len(y_pred):
        raise ValueError(f"{model_name}: prediction length mismatch")

    if not set(np.unique(y_true)).issubset(ACTIVITIES):
        raise ValueError(f"{model_name}: unexpected actual labels")

    if not set(np.unique(y_pred)).issubset(ACTIVITIES):
        raise ValueError(f"{model_name}: unexpected predicted labels")

    print(f"{model_name}: loaded {len(df)} predictions")

    return y_true, y_pred


def analyze_model(model_name, y_true, y_pred):

    labels = list(ACTIVITIES.keys())

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=[f"A{i:02d}" for i in labels],
        columns=[f"A{i:02d}" for i in labels]
    )

    cm_df.to_csv(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_').replace('-', '_')}_test_confusion_matrix.csv"
    )

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    class_rows = []

    for label in labels:
        key = str(label)
        class_rows.append({
            "model": model_name,
            "activity_id": f"A{label:02d}",
            "activity": ACTIVITIES[label],
            "precision": report[key]["precision"],
            "recall": report[key]["recall"],
            "f1_score": report[key]["f1-score"],
            "support": report[key]["support"],
        })

    # Misclassification pairs
    mistakes = []

    for actual_idx, predicted_idx in zip(*np.where(cm > 0)):
        if actual_idx == predicted_idx:
            continue

        actual = labels[actual_idx]
        predicted = labels[predicted_idx]

        mistakes.append({
            "model": model_name,
            "actual_id": f"A{actual:02d}",
            "actual_activity": ACTIVITIES[actual],
            "predicted_id": f"A{predicted:02d}",
            "predicted_activity": ACTIVITIES[predicted],
            "misclassified_windows": int(cm[actual_idx, predicted_idx]),
            "percentage_of_actual_class": (
                cm[actual_idx, predicted_idx]
                / cm[actual_idx].sum()
                * 100
            ),
        })

    mistakes_df = pd.DataFrame(mistakes)

    if not mistakes_df.empty:
        mistakes_df = mistakes_df.sort_values(
            "misclassified_windows",
            ascending=False
        )

    return class_rows, mistakes_df, cm


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_class_rows = []
    all_mistakes = []
    confusion_matrices = {}

    # Analyze each model
    for model_name, directory in REPORT_DIRS.items():

        print("\n" + "=" * 65)
        print(f"ANALYZING: {model_name}")
        print("=" * 65)

        y_true, y_pred = load_predictions(model_name, directory)

        class_rows, mistakes_df, cm = analyze_model(
            model_name,
            y_true,
            y_pred
        )

        all_class_rows.extend(class_rows)
        all_mistakes.append(mistakes_df)
        confusion_matrices[model_name] = cm

        print("\nTop 10 misclassifications:")
        print(mistakes_df.head(10).to_string(index=False))

    # Combined per-class metrics
    class_df = pd.DataFrame(all_class_rows)

    class_df.to_csv(
        OUTPUT_DIR / "all_models_test_class_metrics.csv",
        index=False
    )

    # Combined misclassification table
    mistakes_df = pd.concat(
        all_mistakes,
        ignore_index=True
    )

    mistakes_df.to_csv(
        OUTPUT_DIR / "all_models_test_misclassifications.csv",
        index=False
    )

    # Compare average F1 across models
    f1_pivot = class_df.pivot(
        index=["activity_id", "activity"],
        columns="model",
        values="f1_score"
    ).reset_index()

    f1_pivot["average_f1"] = f1_pivot[
        ["Random Forest", "SVM", "1D-CNN"]
    ].mean(axis=1)

    f1_pivot["consistently_difficult"] = (
        f1_pivot[
            ["Random Forest", "SVM", "1D-CNN"]
        ].lt(0.10).all(axis=1)
    )

    f1_pivot = f1_pivot.sort_values("average_f1")

    f1_pivot.to_csv(
        OUTPUT_DIR / "all_models_activity_f1_comparison.csv",
        index=False
    )

    # Find common directed misclassification pairs
    pair_frames = []

    for model_name, cm in confusion_matrices.items():

        for i, actual in enumerate(ACTIVITIES):
            for j, predicted in enumerate(ACTIVITIES):

                if i == j or cm[i, j] == 0:
                    continue

                pair_frames.append({
                    "model": model_name,
                    "actual_id": f"A{actual:02d}",
                    "actual_activity": ACTIVITIES[actual],
                    "predicted_id": f"A{predicted:02d}",
                    "predicted_activity": ACTIVITIES[predicted],
                    "count": int(cm[i, j]),
                })

    pairs_df = pd.DataFrame(pair_frames)

    common_pairs = (
        pairs_df.groupby(
            [
                "actual_id",
                "actual_activity",
                "predicted_id",
                "predicted_activity",
            ]
        )
        .agg(
            total_misclassified_windows=("count", "sum"),
            models_confused=("model", "nunique"),
            models=("model", lambda x: ", ".join(sorted(x))),
        )
        .reset_index()
        .sort_values(
            ["models_confused", "total_misclassified_windows"],
            ascending=False
        )
    )

    common_pairs.to_csv(
        OUTPUT_DIR / "common_misclassification_pairs.csv",
        index=False
    )

    # Final summary
    print("\n" + "=" * 65)
    print("CROSS-MODEL CONFUSION ANALYSIS")
    print("=" * 65)

    print("\nLowest average F1 activities:")
    print(f1_pivot.head(10).round(4).to_string(index=False))

    print("\nActivities with F1 below 0.10 in ALL models:")
    difficult = f1_pivot[f1_pivot["consistently_difficult"]]

    if difficult.empty:
        print("None")
    else:
        print(difficult.to_string(index=False))

    print("\nMost common cross-model misclassification pairs:")
    print(common_pairs.head(15).to_string(index=False))

    print("\nSaved analysis files:")
    for path in sorted(OUTPUT_DIR.glob("*")):
        if path.is_file():
            print(path)

    print("\nCross-model confusion analysis completed!")


if __name__ == "__main__":
    main()
