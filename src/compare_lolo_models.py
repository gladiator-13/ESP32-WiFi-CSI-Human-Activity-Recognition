
"""
Unified comparison of Random Forest, SVM and 1D-CNN
for within-location and Leave-One-Location-Out (LOLO)
ESP32 Wi-Fi CSI Human Activity Recognition.

This script does not train models. It reads existing evaluation
reports and generates comparison CSV files and a Markdown report.
"""

import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

MODELS = [
    "random_forest",
    "svm",
    "cnn"
]

MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "svm": "SVM",
    "cnn": "1D-CNN"
}

METRICS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1"
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1"
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def read_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_metric(data, metric, prefix=None):
    """
    Find a metric in a dictionary or pandas Series.

    Supports keys such as:
        accuracy
        test_accuracy
        validation_accuracy
        macro_f1
        test_macro_f1
    """

    if isinstance(data, pd.Series):
        data = data.to_dict()

    aliases = {
        "accuracy": [
            "accuracy",
            "test_accuracy",
            "validation_accuracy"
        ],
        "macro_precision": [
            "macro_precision",
            "test_macro_precision",
            "validation_macro_precision"
        ],
        "macro_recall": [
            "macro_recall",
            "test_macro_recall",
            "validation_macro_recall"
        ],
        "macro_f1": [
            "macro_f1",
            "test_macro_f1",
            "validation_macro_f1"
        ],
        "weighted_f1": [
            "weighted_f1",
            "test_weighted_f1",
            "validation_weighted_f1"
        ]
    }

    keys = aliases[metric]

    if prefix:
        keys = [
            f"{prefix}_{metric}",
            *keys
        ]

    normalized = {
        str(key).strip().lower(): value
        for key, value in data.items()
    }

    for key in keys:
        key = key.lower()

        if key in normalized:
            value = normalized[key]

            if pd.notna(value):
                return float(value)

    raise KeyError(
        f"Could not find metric '{metric}'. "
        f"Available keys: {list(data.keys())}"
    )


def find_column(df, candidates):
    """Find a dataframe column using case-insensitive matching."""

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


# ============================================================
# LOAD LOLO RESULTS
# ============================================================

def load_lolo_results(model):

    path = (
        REPORTS_DIR
        / model
        / "lolo"
        / "lolo_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"LOLO summary not found: {path}"
        )

    df = pd.read_csv(path)

    fold_column = find_column(
        df,
        ["fold"]
    )

    location_column = find_column(
        df,
        ["held_out_location", "location"]
    )

    if fold_column is None:
        raise ValueError(
            f"No fold column found in {path}"
        )

    if location_column is None:
        raise ValueError(
            f"No held-out location column found in {path}"
        )

    results = []

    for _, row in df.iterrows():

        fold_value = str(row[fold_column]).strip()

        # Handles both "fold_1" and "1".
        fold_number = int(fold_value.split("_")[-1])

        result = {
            "model": MODEL_DISPLAY_NAMES[model],
            "fold": fold_number,
            "held_out_location": str(row[location_column])
        }

        for metric in METRICS:
            result[metric] = find_metric(row, metric)

        results.append(result)

    return results


# ============================================================
# LOAD WITHIN-LOCATION TEST RESULTS
# ============================================================



def load_within_location_results(model):
    """Load within-location test metrics from CSV or JSON."""

    report_dir = (
        REPORTS_DIR
        / model
        / "within_location"
    )

    # Random Forest and SVM store metrics as CSV.
    if model in ["random_forest", "svm"]:

        path = report_dir / "test_metrics.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Required CSV file not found: {path}"
            )

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"Metrics CSV is empty: {path}")

        data = df.iloc[0].to_dict()

    # CNN stores metrics as JSON.
    elif model == "cnn":

        path = report_dir / "test_metrics.json"
        data = read_json(path)

    else:
        raise ValueError(f"Unknown model: {model}")

    result = {
        "model": MODEL_DISPLAY_NAMES[model],
        "accuracy": float(
            data.get("accuracy", data.get("test_accuracy"))
        ),
        "macro_precision": float(
            data.get("macro_precision", data.get("test_macro_precision"))
        ),
        "macro_recall": float(
            data.get("macro_recall", data.get("test_macro_recall"))
        ),
        "macro_f1": float(
            data.get("macro_f1", data.get("test_macro_f1"))
        ),
        "weighted_f1": float(
            data.get("weighted_f1", data.get("test_weighted_f1", float("nan")))
        ),
    }

    print(
        f"  Loaded {result['model']}: "
        f"test macro-F1 = {result['macro_f1']:.4f}"
    )

    return result



# ============================================================
# FORMAT TABLES FOR MARKDOWN
# ============================================================

def markdown_table(df, percentage_columns=None):

    percentage_columns = percentage_columns or []

    display_df = df.copy()

    for column in percentage_columns:

        if column in display_df.columns:

            display_df[column] = display_df[column].apply(
                lambda value: f"{value * 100:.2f}%"
            )

    return display_df.to_markdown(
        index=False
    )


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():

    print("=" * 70)
    print("UNIFIED MODEL COMPARISON")
    print("=" * 70)

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD ALL LOLO RESULTS
    # --------------------------------------------------------

    lolo_results = []

    for model in MODELS:

        print(
            f"\nLoading LOLO results: "
            f"{MODEL_DISPLAY_NAMES[model]}"
        )

        lolo_results.extend(
            load_lolo_results(model)
        )

    lolo_df = pd.DataFrame(lolo_results)

    # --------------------------------------------------------
    # SAVE FOLD-WISE COMPARISON
    # --------------------------------------------------------

    foldwise_df = lolo_df[
        [
            "fold",
            "held_out_location",
            "model",
            *METRICS
        ]
    ].sort_values(
        ["fold", "model"]
    )

    foldwise_path = (
        REPORTS_DIR
        / "model_comparison"
        / "lolo_foldwise_comparison.csv"
    )

    foldwise_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    foldwise_df.to_csv(
        foldwise_path,
        index=False
    )

    print("\nFOLD-WISE LOLO COMPARISON")

    print(
        foldwise_df[
            [
                "fold",
                "held_out_location",
                "model",
                "macro_f1"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # LOLO MEAN AND STANDARD DEVIATION
    # --------------------------------------------------------

    summary_rows = []

    for model in MODELS:

        model_name = MODEL_DISPLAY_NAMES[model]

        model_df = lolo_df[
            lolo_df["model"] == model_name
        ]

        row = {
            "model": model_name,
            "number_of_folds": len(model_df)
        }

        for metric in METRICS:

            row[f"{metric}_mean"] = (
                model_df[metric].mean()
            )

            row[f"{metric}_std"] = (
                model_df[metric].std(ddof=1)
            )

        summary_rows.append(row)

    lolo_summary_df = pd.DataFrame(summary_rows)

    lolo_summary_path = (
        REPORTS_DIR
        / "model_comparison"
        / "lolo_model_summary.csv"
    )

    lolo_summary_df.to_csv(
        lolo_summary_path,
        index=False
    )

    print("\nLOLO MEAN AND STANDARD DEVIATION")

    print(
        lolo_summary_df[
            [
                "model",
                "macro_f1_mean",
                "macro_f1_std"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # LOAD WITHIN-LOCATION RESULTS
    # --------------------------------------------------------

    within_results = []

    for model in MODELS:

        print(
            f"\nLoading within-location results: "
            f"{MODEL_DISPLAY_NAMES[model]}"
        )

        within_results.append(
            load_within_location_results(model)
        )

    within_df = pd.DataFrame(within_results)

    within_path = (
        REPORTS_DIR
        / "model_comparison"
        / "within_location_test_comparison.csv"
    )

    within_df.to_csv(
        within_path,
        index=False
    )

    # --------------------------------------------------------
    # WITHIN-LOCATION VS LOLO COMPARISON
    # --------------------------------------------------------

    comparison_rows = []

    for model in MODELS:

        model_name = MODEL_DISPLAY_NAMES[model]

        within_row = within_df[
            within_df["model"] == model_name
        ].iloc[0]

        lolo_row = lolo_summary_df[
            lolo_summary_df["model"] == model_name
        ].iloc[0]

        row = {
            "model": model_name
        }

        for metric in METRICS:

            within_value = float(
                within_row[metric]
            )

            lolo_value = float(
                lolo_row[f"{metric}_mean"]
            )

            row[f"within_{metric}"] = within_value
            row[f"lolo_{metric}"] = lolo_value

            # Positive value means performance decreased
            # when moving from within-location to LOLO.
            row[f"{metric}_drop_percentage_points"] = (
                within_value - lolo_value
            ) * 100

        comparison_rows.append(row)

    generalization_df = pd.DataFrame(
        comparison_rows
    )

    generalization_path = (
        REPORTS_DIR
        / "model_comparison"
        / "within_vs_lolo_comparison.csv"
    )

    generalization_df.to_csv(
        generalization_path,
        index=False
    )

    # --------------------------------------------------------
    # GENERATE MARKDOWN REPORT
    # --------------------------------------------------------

    foldwise_display = foldwise_df[
        [
            "fold",
            "held_out_location",
            "model",
            "accuracy",
            "macro_f1",
            "weighted_f1"
        ]
    ].copy()

    summary_display = lolo_summary_df[
        [
            "model",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "weighted_f1_mean",
            "weighted_f1_std"
        ]
    ].copy()

    generalization_display = generalization_df[
        [
            "model",
            "within_macro_f1",
            "lolo_macro_f1",
            "macro_f1_drop_percentage_points"
        ]
    ].copy()

    within_display = within_df[
        [
            "model",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1"
        ]
    ].copy()

    report = f"""# Unified Model Comparison and LOLO Analysis

## 1. Objective

This analysis compares three machine learning models for ESP32 Wi-Fi
CSI Human Activity Recognition:

- Random Forest
- Support Vector Machine (SVM)
- 1D Convolutional Neural Network (CNN)

Two evaluation settings are considered:

1. **Within-location:** Participant-wise train, validation and test
   splits are created within each recording location.
2. **Leave-One-Location-Out (LOLO):** Each fold holds out one entire
   recording location for testing. Training and validation data come
   from the other two locations.

The primary comparison metric is macro-F1 because it gives equal
importance to all 20 activity classes.

## 2. LOLO Experimental Results

### 2.1 Fold-wise results

{markdown_table(
    foldwise_display,
    ["accuracy", "macro_f1", "weighted_f1"]
)}

### 2.2 Mean and standard deviation across folds

{markdown_table(
    summary_display,
    [
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "weighted_f1_mean",
        "weighted_f1_std"
    ]
)}

The mean is calculated as the unweighted arithmetic average of the
three location-specific fold results. Standard deviation uses the
sample standard deviation (ddof = 1).

### 2.3 Model comparison

The mean LOLO macro-F1 values are:

{chr(10).join(
    f"- **{row['model']}:** "
    f"{row['macro_f1_mean'] * 100:.2f}% "
    f"(standard deviation: "
    f"{row['macro_f1_std'] * 100:.2f} percentage points)"
    for _, row in lolo_summary_df.iterrows()
)}

"""

    # --------------------------------------------------------
    # LOCATION-WISE INTERPRETATION
    # --------------------------------------------------------

    location_lines = []

    for fold_number in sorted(lolo_df["fold"].unique()):

        fold_df = lolo_df[
            lolo_df["fold"] == fold_number
        ]

        location = fold_df[
            "held_out_location"
        ].iloc[0]

        best_row = fold_df.loc[
            fold_df["macro_f1"].idxmax()
        ]

        location_lines.append(
            f"- **{location}:** Best macro-F1 was achieved by "
            f"{best_row['model']} "
            f"({best_row['macro_f1'] * 100:.2f}%)."
        )

    report += f"""## 3. Location-wise Analysis

{chr(10).join(location_lines)}

The fold-wise results show how model performance changes when the
held-out recording location changes. A location with consistently
lower performance across models may represent a more challenging
generalization setting.

These results do not, by themselves, establish why a location is
more difficult.

## 4. Within-Location vs Cross-Location Generalization

### 4.1 Within-location test performance

{markdown_table(
    within_display,
    [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1"
    ]
)}

### 4.2 Macro-F1 comparison

{markdown_table(
    generalization_display,
    [
        "within_macro_f1",
        "lolo_macro_f1"
    ]
)}

The macro-F1 drop is calculated as:

**Within-location test macro-F1 minus mean LOLO test macro-F1.**

A positive drop indicates lower performance when evaluating on an
entirely held-out location.

"""

    # --------------------------------------------------------
    # AUTOMATIC FINDINGS
    # --------------------------------------------------------

    best_lolo = lolo_summary_df.loc[
        lolo_summary_df["macro_f1_mean"].idxmax()
    ]

    worst_lolo = lolo_summary_df.loc[
        lolo_summary_df["macro_f1_mean"].idxmin()
    ]

    hardest_location_rows = (
        lolo_df.groupby("held_out_location")["macro_f1"]
        .mean()
        .sort_values()
    )

    hardest_location = hardest_location_rows.index[0]

    rf_mean = float(
        lolo_summary_df.loc[
            lolo_summary_df["model"] == "Random Forest",
            "macro_f1_mean"
        ].iloc[0]
    )

    svm_mean = float(
        lolo_summary_df.loc[
            lolo_summary_df["model"] == "SVM",
            "macro_f1_mean"
        ].iloc[0]
    )

    cnn_mean = float(
        lolo_summary_df.loc[
            lolo_summary_df["model"] == "1D-CNN",
            "macro_f1_mean"
        ].iloc[0]
    )

    report += f"""## 5. Key Findings

1. **Best average LOLO model:** {best_lolo['model']} with a mean
   macro-F1 of {best_lolo['macro_f1_mean'] * 100:.2f}%.
2. **Lowest average LOLO model:** {worst_lolo['model']} with a mean
   macro-F1 of {worst_lolo['macro_f1_mean'] * 100:.2f}%.
3. **Most challenging held-out location by average macro-F1:**
   {hardest_location}.
4. Random Forest mean LOLO macro-F1:
   {rf_mean * 100:.2f}%.
5. SVM mean LOLO macro-F1:
   {svm_mean * 100:.2f}%.
6. CNN mean LOLO macro-F1:
   {cnn_mean * 100:.2f}%.

"""

    if abs(rf_mean - svm_mean) < 0.005:

        report += (
            "Random Forest and SVM have very similar average LOLO "
            "macro-F1. Their small difference should not be treated "
            "as evidence of a meaningful performance advantage.\n\n"
        )

    if cnn_mean < rf_mean and cnn_mean < svm_mean:

        report += (
            "The CNN has lower mean LOLO macro-F1 than both "
            "classical models in these experiments.\n\n"
        )

    report += """## 6. Discussion

The difference between within-location and LOLO performance
indicates that performance measured using participant-wise splits
within a location does not necessarily transfer to a completely
unseen recording location.

Potential contributors include differences in the recording
environment, CSI propagation conditions, device placement, and
participant characteristics. However, the present experiments
cannot isolate or prove any of these causes.

The results should therefore be interpreted as evidence of a
cross-location generalization challenge, rather than proof of a
specific source of domain shift.

Macro-F1 is emphasized because overall accuracy and weighted-F1
can obscure poor recognition of individual activities.

## 7. Limitations

- Only three recording locations are available.
- Each LOLO fold tests on one held-out location.
- The fold-wise test windows overlap by 50%, so window-level
  predictions are not statistically independent.
- The mean LOLO score is an unweighted average across locations.
- These experiments do not establish performance on arbitrary
  unseen environments or hardware configurations.
- No causal conclusions about environmental or signal differences
  can be drawn from these metrics alone.

## 8. Conclusion

The within-location and LOLO experiments reveal a substantial
difference between participant-wise evaluation within known
locations and evaluation on an entirely unseen location.

Random Forest and SVM show similar average LOLO macro-F1, while
the CNN achieves a lower average score in the current experiments.

These findings demonstrate that cross-location generalization is
a major challenge for the current ESP32 Wi-Fi CSI human activity
recognition pipeline.

Further work should investigate recording-level evaluation,
location-related signal variation, and methods to improve
cross-location generalization without using held-out test data
for model selection.

## 9. Generated Files

- `reports/model_comparison/lolo_foldwise_comparison.csv`
- `reports/model_comparison/lolo_model_summary.csv`
- `reports/model_comparison/within_location_test_comparison.csv`
- `reports/model_comparison/within_vs_lolo_comparison.csv`
- `docs/LOLO_ANALYSIS.md`
"""

    report_path = DOCS_DIR / "LOLO_ANALYSIS.md"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 70)

    print("\nGenerated CSV files:")
    print(foldwise_path)
    print(lolo_summary_path)
    print(within_path)
    print(generalization_path)

    print("\nGenerated Markdown report:")
    print(report_path)

    print("\nBest mean LOLO model:", best_lolo["model"])
    print(
        "Best mean LOLO macro-F1:",
        f"{best_lolo['macro_f1_mean'] * 100:.2f}%"
    )

    print("\nMost challenging held-out location:")
    print(hardest_location)

    print("\nAll comparisons completed successfully.")


if __name__ == "__main__":
    main()
