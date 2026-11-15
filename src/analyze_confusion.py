
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RF_DIR = ROOT / "reports" / "random_forest" / "within_location"
SVM_DIR = ROOT / "reports" / "svm" / "within_location"

OUTPUT_DIR = ROOT / "reports" / "model_comparison" / "within_location"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_confusion_matrix(path):

    df = pd.read_csv(path, index_col=0)

    # Convert row and column labels to integers.
    # Supports both "1" and "Activity_01" formats.

    def extract_label(value):
        value = str(value).strip()

        # Handle Actual_1, Predicted_1, Activity_01, etc.
        if "_" in value:
            value = value.rsplit("_", 1)[-1]

        return int(value)


    df.index = [extract_label(x) for x in df.index]
    df.columns = [extract_label(x) for x in df.columns]

    df = df.sort_index(axis=0).sort_index(axis=1)

    return df.to_numpy(dtype=int)


def analyze_model(model_name, cm):

    records = []

    total_errors = int(cm.sum() - np.trace(cm))

    print("\n" + "=" * 75)
    print(f"{model_name.upper()} - TOP ACTIVITY CONFUSIONS")
    print("=" * 75)

    for actual in range(1, 21):

        row = cm[actual - 1]
        total_actual = int(row.sum())

        incorrect = []

        for predicted in range(1, 21):

            if predicted == actual:
                continue

            count = int(cm[actual - 1, predicted - 1])

            if count > 0:
                percentage = (
                    count / total_actual * 100
                    if total_actual else 0
                )

                incorrect.append({
                    "model": model_name,
                    "actual_label": f"A{actual:02d}",
                    "actual_activity": ACTIVITIES[actual],
                    "predicted_label": f"A{predicted:02d}",
                    "predicted_activity": ACTIVITIES[predicted],
                    "misclassified_windows": count,
                    "percentage_of_actual_class": percentage,
                })

        incorrect.sort(
            key=lambda x: x["misclassified_windows"],
            reverse=True,
        )

        print(
            f"\nA{actual:02d} - {ACTIVITIES[actual]} "
            f"(Total windows: {total_actual})"
        )

        if not incorrect:
            print("No misclassifications.")
            continue

        for rank, item in enumerate(incorrect[:3], start=1):

            print(
                f"  {rank}. Predicted as "
                f"{item['predicted_label']} "
                f"({item['predicted_activity']}): "
                f"{item['misclassified_windows']} windows "
                f"({item['percentage_of_actual_class']:.2f}%)"
            )

        records.extend(incorrect[:3])

    print("\nOverall misclassified windows:", total_errors)

    return records


def get_global_confusions(model_name, cm):

    records = []

    for actual in range(1, 21):

        for predicted in range(1, 21):

            if actual == predicted:
                continue

            count = int(cm[actual - 1, predicted - 1])

            if count == 0:
                continue

            total_actual = int(cm[actual - 1].sum())

            records.append({
                "model": model_name,
                "actual_label": f"A{actual:02d}",
                "actual_activity": ACTIVITIES[actual],
                "predicted_label": f"A{predicted:02d}",
                "predicted_activity": ACTIVITIES[predicted],
                "misclassified_windows": count,
                "percentage_of_actual_class": (
                    count / total_actual * 100
                    if total_actual else 0
                ),
            })

    records.sort(
        key=lambda x: x["misclassified_windows"],
        reverse=True,
    )

    return records


def main():

    print("=" * 75)
    print("NUMERICAL CONFUSION ANALYSIS")
    print("=" * 75)

    # Load confusion matrices
    rf_cm = load_confusion_matrix(
        RF_DIR / "test_confusion_matrix.csv"
    )

    svm_cm = load_confusion_matrix(
        SVM_DIR / "test_confusion_matrix.csv"
    )

    assert rf_cm.shape == (20, 20)
    assert svm_cm.shape == (20, 20)

    print("\nConfusion matrices loaded successfully.")

    # Analyze each model
    rf_records = analyze_model("Random Forest", rf_cm)
    svm_records = analyze_model("SVM", svm_cm)

    # Save top 3 confusions per activity
    top_df = pd.DataFrame(rf_records + svm_records)

    top_df.to_csv(
        OUTPUT_DIR / "top_confusions_per_activity.csv",
        index=False,
    )

    # Global strongest confusions
    global_records = (
        get_global_confusions("Random Forest", rf_cm)
        + get_global_confusions("SVM", svm_cm)
    )

    global_df = pd.DataFrame(global_records)

    global_df.to_csv(
        OUTPUT_DIR / "all_confusions.csv",
        index=False,
    )

    # Save top 20 directed confusions per model
    for model_name in ["Random Forest", "SVM"]:

        model_df = global_df[
            global_df["model"] == model_name
        ].head(20)

        filename = (
            "random_forest_top_confusions.csv"
            if model_name == "Random Forest"
            else "svm_top_confusions.csv"
        )

        model_df.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

        print(f"\nTop 10 global confusions - {model_name}")

        print(
            model_df.head(10)[
                [
                    "actual_label",
                    "actual_activity",
                    "predicted_label",
                    "predicted_activity",
                    "misclassified_windows",
                    "percentage_of_actual_class",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print("\nReports saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
