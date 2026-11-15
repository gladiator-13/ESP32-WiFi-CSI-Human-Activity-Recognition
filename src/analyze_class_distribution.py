
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WINDOWS_DIR = PROJECT_ROOT / "data" / "processed" / "windows"

ACTIVITIES = list(range(1, 21))


def analyze_distribution(metadata, title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    counts = pd.crosstab(
        metadata["split"],
        metadata["activity"]
    )

    counts = counts.reindex(
        index=["train", "validation", "test"],
        columns=ACTIVITIES,
        fill_value=0
    )

    print("\nWindow counts per activity:")
    print(counts.to_string())

    print("\nTotal windows per split:")
    print(counts.sum(axis=1))

    for split in counts.index:

        class_counts = counts.loc[split]

        minimum = class_counts.min()
        maximum = class_counts.max()

        imbalance_ratio = (
            maximum / minimum if minimum > 0 else float("inf")
        )

        print(f"\n{split.upper()}")
        print(f"Minimum class windows: {minimum}")
        print(f"Maximum class windows: {maximum}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    return counts


def main():

    # Within-location dataset
    within_path = (
        WINDOWS_DIR
        / "within_location"
        / "train"
        / "metadata.csv"
    )

    # Load all three splits
    within_metadata = pd.concat(
        [
            pd.read_csv(
                WINDOWS_DIR
                / "within_location"
                / split
                / "metadata.csv"
            )
            for split in ["train", "validation", "test"]
        ],
        ignore_index=True
    )

    analyze_distribution(
        within_metadata,
        "WITHIN-LOCATION CLASS DISTRIBUTION"
    )

    # LOLO datasets
    for fold in [1, 2, 3]:

        fold_metadata = pd.concat(
            [
                pd.read_csv(
                    WINDOWS_DIR
                    / "lolo"
                    / f"fold_{fold}"
                    / split
                    / "metadata.csv"
                )
                for split in ["train", "validation", "test"]
            ],
            ignore_index=True
        )

        analyze_distribution(
            fold_metadata,
            f"LOLO FOLD {fold} CLASS DISTRIBUTION"
        )

    print("\n" + "=" * 75)
    print("CLASS DISTRIBUTION ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
