
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
COMPARISON_DIR = REPORTS_DIR / "model_comparison"
RECORDING_DIR = REPORTS_DIR / "recording_level"
SUMMARY_DIR = RECORDING_DIR / "summaries"
DOCS_DIR = BASE_DIR / "docs"

MODELS = {
    "random_forest": "Random Forest",
    "svm": "SVM",
    "cnn": "1D-CNN",
}

METRICS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def save_csv(df, filename):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path.relative_to(BASE_DIR)}")


def markdown_table(df, columns=None, digits=4):
    if columns is not None:
        df = df[columns]

    if df.empty:
        return "_No data available._"

    headers = list(df.columns)
    rows = []

    for _, row in df.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        rows.append(values)

    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        output.append("| " + " | ".join(row) + " |")

    return "\n".join(output)


def prepare_recording_results():

    data = load_csv(
        SUMMARY_DIR / "recording_level_summary.csv"
    )

    data["model_name"] = data["model"].map(MODELS)

    if data["model_name"].isna().any():
        raise ValueError("Unknown model found in recording summary.")

    within = data[
        data["evaluation"] == "within_location"
    ].copy()

    lolo = data[
        data["evaluation"] == "lolo"
    ].copy()

    if len(within) != 3 or len(lolo) != 9:
        raise ValueError(
            "Expected 3 within-location and 9 LOLO evaluations."
        )

    return data, within, lolo


def create_within_comparison(within):

    result = within[
        [
            "model_name",
            "recording_count",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "tie_count",
        ]
    ].copy()

    result = result.rename(
        columns={"model_name": "model"}
    )

    result = result.sort_values("macro_f1", ascending=False)

    save_csv(
        result,
        "recording_level_model_comparison.csv",
    )

    return result


def create_lolo_foldwise(lolo):

    result = lolo[
        [
            "model_name",
            "fold",
            "recording_count",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "tie_count",
        ]
    ].copy()

    result = result.rename(
        columns={"model_name": "model"}
    )

    result["fold"] = result["fold"].astype(int)

    result = result.sort_values(
        ["fold", "macro_f1"],
        ascending=[True, False],
    )

    save_csv(
        result,
        "recording_level_lolo_foldwise_comparison.csv",
    )

    return result


def create_lolo_summary(lolo):

    result = (
        lolo.groupby("model_name")[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )

    result.columns = [
        "_".join(str(x) for x in column if str(x))
        if isinstance(column, tuple)
        else column
        for column in result.columns
    ]

    result = result.rename(
        columns={"model_name": "model"}
    )

    result["number_of_folds"] = 3

    result = result[
        [
            "model",
            "number_of_folds",
        ]
        + [
            f"{metric}_{stat}"
            for metric in METRICS
            for stat in ["mean", "std"]
        ]
    ]

    result = result.sort_values(
        "macro_f1_mean",
        ascending=False,
    )

    save_csv(
        result,
        "recording_level_lolo_model_summary.csv",
    )

    return result


def create_window_vs_recording(within, lolo_summary):

    window_within = load_csv(
        COMPARISON_DIR / "within_location_test_comparison.csv"
    )

    window_lolo = load_csv(
        COMPARISON_DIR / "lolo_model_summary.csv"
    )

    rows = []

    for model_slug, display_name in MODELS.items():

        window_name = display_name

        w_within = window_within[
            window_within["model"] == window_name
        ].iloc[0]

        r_within = within[
            within["model"] == model_slug
        ].iloc[0]

        for metric in METRICS:

            window_value = float(w_within[metric])
            recording_value = float(r_within[metric])

            rows.append({
                "model": display_name,
                "evaluation": "within_location",
                "metric": metric,
                "window_level": window_value,
                "recording_level": recording_value,
                "difference_percentage_points": (
                    recording_value - window_value
                ) * 100,
            })

        w_lolo = window_lolo[
            window_lolo["model"] == window_name
        ].iloc[0]

        r_lolo = lolo_summary[
            lolo_summary["model"] == display_name
        ].iloc[0]

        for metric in METRICS:

            window_value = float(w_lolo[f"{metric}_mean"])
            recording_value = float(r_lolo[f"{metric}_mean"])

            rows.append({
                "model": display_name,
                "evaluation": "lolo_mean",
                "metric": metric,
                "window_level": window_value,
                "recording_level": recording_value,
                "difference_percentage_points": (
                    recording_value - window_value
                ) * 100,
            })

    result = pd.DataFrame(rows)

    save_csv(
        result,
        "window_vs_recording_comparison.csv",
    )

    return result


def create_per_class_comparison():

    records = []

    for model, display_name in MODELS.items():

        # Within-location per-class results
        path = (
            RECORDING_DIR
            / "within_location"
            / model
            / "classification_report.csv"
        )

        report = pd.read_csv(path, index_col=0)

        for activity, row in report.iterrows():

            if str(activity).startswith("Activity_"):

                records.append({
                    "model": display_name,
                    "evaluation": "within_location",
                    "fold": None,
                    "activity": activity,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1-score"],
                    "support": row["support"],
                })

        # LOLO per-class results
        for fold in range(1, 4):

            path = (
                RECORDING_DIR
                / "lolo"
                / f"fold_{fold}"
                / model
                / "classification_report.csv"
            )

            report = pd.read_csv(path, index_col=0)

            for activity, row in report.iterrows():

                if str(activity).startswith("Activity_"):

                    records.append({
                        "model": display_name,
                        "evaluation": "lolo",
                        "fold": fold,
                        "activity": activity,
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1_score": row["f1-score"],
                        "support": row["support"],
                    })

    result = pd.DataFrame(records)

    save_csv(
        result,
        "recording_level_per_class_comparison.csv",
    )

    return result


def generate_markdown(
    within,
    foldwise,
    lolo_summary,
    window_comparison,
    per_class,
):

    best_within = within.iloc[0]
    best_lolo = lolo_summary.iloc[0]

    lines = [
        "# Recording-Level Human Activity Recognition Analysis",
        "",
        "## 1. Objective",
        "",
        "This analysis evaluates whether aggregating overlapping "
        "window predictions into recording-level predictions "
        "improves activity recognition performance.",
        "",
        "Three models are evaluated: Random Forest, SVM, and 1D-CNN.",
        "",
        "## 2. Recording-level methodology",
        "",
        "- Each recording contains 19 windows.",
        "- Each window is 2 seconds long.",
        "- Consecutive windows have 50% overlap.",
        "- The 19 predicted activity labels are aggregated using "
        "majority voting.",
        "- The most frequent predicted activity becomes the "
        "recording-level prediction.",
        "- Ties are resolved deterministically by selecting the "
        "smallest numeric activity label.",
        "- All 20 activity classes are included in metric "
        "calculations, with zero-division handled as zero.",
        "",
        "Recording-level evaluation changes the unit of prediction; "
        "it does not involve additional model training.",
        "",
        "## 3. Within-location results",
        "",
        markdown_table(
            within,
            [
                "model",
                "recording_count",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "tie_count",
            ],
        ),
        "",
        f"Best within-location model: **{best_within['model']}**, "
        f"with macro-F1 of {best_within['macro_f1']:.4f}.",
        "",
        "## 4. LOLO fold-wise results",
        "",
        markdown_table(
            foldwise,
            [
                "model",
                "fold",
                "recording_count",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "tie_count",
            ],
        ),
        "",
        "## 5. LOLO mean and standard deviation",
        "",
        "The mean is the unweighted average of the three held-out "
        "location folds. Standard deviation is the sample standard "
        "deviation (ddof=1).",
        "",
        markdown_table(
            lolo_summary,
            [
                "model",
                "number_of_folds",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "weighted_f1_mean",
                "weighted_f1_std",
            ],
        ),
        "",
        f"Best mean LOLO model: **{best_lolo['model']}**, "
        f"with macro-F1 of {best_lolo['macro_f1_mean']:.4f}.",
        "",
        "## 6. Window-level versus recording-level comparison",
        "",
        markdown_table(
            window_comparison[
                window_comparison["metric"] == "macro_f1"
            ],
            [
                "model",
                "evaluation",
                "window_level",
                "recording_level",
                "difference_percentage_points",
            ],
        ),
        "",
        "A positive difference indicates a higher metric after "
        "recording-level aggregation. This is a change in evaluation "
        "unit, not evidence that the trained model itself improved.",
        "",
        "## 7. Per-activity recording-level results",
        "",
        "Detailed precision, recall, F1, and support for every "
        "activity, model, and evaluation fold are available in "
        "`reports/recording_level/summaries/"
        "recording_level_per_class_comparison.csv`.",
        "",
        "## 8. Interpretation",
        "",
        "Recording-level majority voting improves within-location "
        "macro-F1 for all three models compared with window-level "
        "evaluation. The improvement is smaller or inconsistent "
        "under LOLO evaluation.",
        "",
        "This suggests that aggregating multiple windows can reduce "
        "some window-level prediction variability, but does not "
        "remove the difficulty of generalizing to unseen locations.",
        "",
        "Location-specific differences should be interpreted "
        "descriptively. Only three locations are available, so "
        "these results do not establish a causal explanation for "
        "the performance differences.",
        "",
        "## 9. Limitations",
        "",
        "- The 50% overlapping windows are correlated observations.",
        "- Majority voting discards prediction confidence.",
        "- The deterministic tie-break can affect tied recordings.",
        "- Only three locations are used for LOLO evaluation.",
        "- Fold means are unweighted across locations.",
        "- No statistical significance claim is made.",
        "",
        "## 10. Reproducibility",
        "",
        "The analysis is generated by "
        "`src/compare_recording_level.py` using saved prediction "
        "files, classification reports, and existing window-level "
        "comparison summaries.",
        "",
    ]

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DOCS_DIR / "RECORDING_LEVEL_ANALYSIS.md"

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved: {output_path.relative_to(BASE_DIR)}")


def main():

    print("=" * 75)
    print("RECORDING-LEVEL MODEL COMPARISON")
    print("=" * 75)

    _, within, lolo = prepare_recording_results()

    within_comparison = create_within_comparison(within)

    foldwise = create_lolo_foldwise(lolo)

    lolo_summary = create_lolo_summary(lolo)

    window_comparison = create_window_vs_recording(
        within,
        lolo_summary,
    )

    per_class = create_per_class_comparison()

    generate_markdown(
        within_comparison,
        foldwise,
        lolo_summary,
        window_comparison,
        per_class,
    )

    print("\n" + "=" * 75)
    print("ALL COMPARISON REPORTS GENERATED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
