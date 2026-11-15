
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RF_DIR = ROOT / "reports" / "random_forest" / "within_location"
SVM_DIR = ROOT / "reports" / "svm" / "within_location"

OUTPUT_DIR = ROOT / "reports" / "model_comparison" / "within_location"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ACTIVITIES = [
    "Standing",
    "Sitting",
    "Walk",
    "Jog in place",
    "Turn around",
    "Sit down, stand up",
    "Jump in place",
    "Bend and pick up",
    "Lunges",
    "Wave",
    "Clap",
    "Boxing",
    "Arms up and down",
    "Push forward",
    "Arm circles",
    "Typing",
    "Phone call",
    "Writing notes",
    "Reading a book",
    "Combing hair",
]

LABELS = [f"A{i:02d}" for i in range(1, 21)]


def load_confusion_matrix(path):
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(dtype=int)


def load_classification_report(path):
    return pd.read_csv(path, index_col=0)


def normalize_confusion_matrix(cm):
    row_sums = cm.sum(axis=1, keepdims=True)

    return np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )


def plot_confusion_matrix(cm, title, output_path):

    normalized_cm = normalize_confusion_matrix(cm)

    fig, ax = plt.subplots(figsize=(12, 10))

    image = ax.imshow(
        normalized_cm,
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(image, ax=ax, label="Proportion of actual activity")

    ax.set_title(title, fontsize=15, pad=15)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("Actual activity")

    ax.set_xticks(np.arange(20))
    ax.set_yticks(np.arange(20))

    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)

    ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(rf_cm, svm_cm):

    rf_normalized = normalize_confusion_matrix(rf_cm)
    svm_normalized = normalize_confusion_matrix(svm_cm)

    fig, axes = plt.subplots(1, 2, figsize=(21, 9))

    matrices = [
        (rf_normalized, "Random Forest"),
        (svm_normalized, "SVM"),
    ]

    for ax, (matrix, title) in zip(axes, matrices):

        image = ax.imshow(
            matrix,
            interpolation="nearest",
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Predicted activity")
        ax.set_ylabel("Actual activity")

        ax.set_xticks(np.arange(20))
        ax.set_yticks(np.arange(20))

        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_yticklabels(LABELS)

    cbar_ax = fig.add_axes([0.93, 0.20, 0.015, 0.60])

    fig.colorbar(
        image,
        cax=cbar_ax,
        label="Proportion of actual activity",
    )

    fig.suptitle(
        "Confusion Matrix Comparison: Random Forest vs SVM",
        fontsize=17,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.91,
        bottom=0.15,
        top=0.85,
        wspace=0.30,
    )
    
    fig.savefig(
        OUTPUT_DIR / "confusion_matrix_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def get_activity_f1(report, activity_number):
    possible_labels = [
        str(activity_number),
        f"Activity_{activity_number:02d}",
    ]

    for label in possible_labels:
        if label in report.index:
            return report.loc[label, "f1-score"]

    raise ValueError(
        f"Activity {activity_number} not found in classification report."
    )


def plot_f1_comparison(rf_report, svm_report):

    rf_f1 = [
        get_activity_f1(rf_report, i)
        for i in range(1, 21)
    ]

    svm_f1 = [
        get_activity_f1(svm_report, i)
        for i in range(1, 21)
    ]

    x = np.arange(20)
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.bar(
        x - width / 2,
        rf_f1,
        width,
        label="Random Forest",
    )

    ax.bar(
        x + width / 2,
        svm_f1,
        width,
        label="SVM",
    )

    ax.set_title(
        "Per-Activity F1-score: Random Forest vs SVM",
        fontsize=15,
    )

    ax.set_xlabel("Activity")
    ax.set_ylabel("F1-score")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"A{i:02d}" for i in range(1, 21)],
        rotation=45,
    )

    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "per_activity_f1_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    print("=" * 65)
    print("CLASSICAL MODEL COMPARISON")
    print("=" * 65)

    # Load confusion matrices
    rf_cm = load_confusion_matrix(
        RF_DIR / "test_confusion_matrix.csv"
    )

    svm_cm = load_confusion_matrix(
        SVM_DIR / "test_confusion_matrix.csv"
    )

    # Load classification reports
    rf_report = load_classification_report(
        RF_DIR / "test_classification_report.csv"
    )

    svm_report = load_classification_report(
        SVM_DIR / "test_classification_report.csv"
    )

    # Validate dimensions
    assert rf_cm.shape == (20, 20)
    assert svm_cm.shape == (20, 20)

    print("\nData loaded successfully.")
    print("Random Forest confusion matrix:", rf_cm.shape)
    print("SVM confusion matrix:", svm_cm.shape)

    # Individual confusion matrices
    plot_confusion_matrix(
        rf_cm,
        "Random Forest - Normalized Confusion Matrix",
        OUTPUT_DIR / "random_forest_confusion_matrix.png",
    )

    plot_confusion_matrix(
        svm_cm,
        "SVM - Normalized Confusion Matrix",
        OUTPUT_DIR / "svm_confusion_matrix.png",
    )

    # Side-by-side comparison
    plot_side_by_side(rf_cm, svm_cm)

    # Per-activity F1 comparison
    plot_f1_comparison(rf_report, svm_report)

    # Print per-class comparison
    comparison = pd.DataFrame({
        "Activity": ACTIVITIES,
        "Random_Forest_F1": [
            get_activity_f1(rf_report, i)
            for i in range(1, 21)
        ],
        "SVM_F1": [
            get_activity_f1(svm_report, i)
            for i in range(1, 21)
        ],
    })

    comparison["Difference_RF_minus_SVM"] = (
        comparison["Random_Forest_F1"] - comparison["SVM_F1"]
    )

    comparison.to_csv(
        OUTPUT_DIR / "per_activity_f1_comparison.csv",
        index=False,
    )

    print("\nPer-activity F1 comparison:")
    print(comparison.round(3).to_string(index=False))

    print("\nVisualizations saved to:")
    print(OUTPUT_DIR)

    print("\nComparison completed successfully!")


if __name__ == "__main__":
    main()
