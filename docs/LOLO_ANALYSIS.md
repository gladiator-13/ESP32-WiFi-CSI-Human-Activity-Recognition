# Unified Model Comparison and LOLO Analysis

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

|   fold | held_out_location   | model         | accuracy   | macro_f1   | weighted_f1   |
|-------:|:--------------------|:--------------|:-----------|:-----------|:--------------|
|      1 | Location_1          | 1D-CNN        | 7.92%      | 4.59%      | 4.63%         |
|      1 | Location_1          | Random Forest | 11.07%     | 8.72%      | 8.79%         |
|      1 | Location_1          | SVM           | 11.92%     | 10.01%     | 10.10%        |
|      2 | Location_2          | 1D-CNN        | 7.97%      | 3.05%      | 3.11%         |
|      2 | Location_2          | Random Forest | 6.46%      | 3.06%      | 3.08%         |
|      2 | Location_2          | SVM           | 6.20%      | 2.96%      | 2.98%         |
|      3 | Location_3          | 1D-CNN        | 7.33%      | 4.61%      | 4.63%         |
|      3 | Location_3          | Random Forest | 9.48%      | 7.31%      | 7.24%         |
|      3 | Location_3          | SVM           | 8.29%      | 6.04%      | 5.93%         |

### 2.2 Mean and standard deviation across folds

| model         | accuracy_mean   | accuracy_std   | macro_f1_mean   | macro_f1_std   | weighted_f1_mean   | weighted_f1_std   |
|:--------------|:----------------|:---------------|:----------------|:---------------|:-------------------|:------------------|
| Random Forest | 9.00%           | 2.34%          | 6.36%           | 2.95%          | 6.37%              | 2.96%             |
| SVM           | 8.80%           | 2.89%          | 6.34%           | 3.53%          | 6.33%              | 3.58%             |
| 1D-CNN        | 7.74%           | 0.36%          | 4.08%           | 0.90%          | 4.12%              | 0.88%             |

The mean is calculated as the unweighted arithmetic average of the
three location-specific fold results. Standard deviation uses the
sample standard deviation (ddof = 1).

### 2.3 Model comparison

The mean LOLO macro-F1 values are:

- **Random Forest:** 6.36% (standard deviation: 2.95 percentage points)
- **SVM:** 6.34% (standard deviation: 3.53 percentage points)
- **1D-CNN:** 4.08% (standard deviation: 0.90 percentage points)

## 3. Location-wise Analysis

- **Location_1:** Best macro-F1 was achieved by SVM (10.01%).
- **Location_2:** Best macro-F1 was achieved by Random Forest (3.06%).
- **Location_3:** Best macro-F1 was achieved by Random Forest (7.31%).

The fold-wise results show how model performance changes when the
held-out recording location changes. A location with consistently
lower performance across models may represent a more challenging
generalization setting.

These results do not, by themselves, establish why a location is
more difficult.

## 4. Within-Location vs Cross-Location Generalization

### 4.1 Within-location test performance

| model         | accuracy   | macro_precision   | macro_recall   | macro_f1   | weighted_f1   |
|:--------------|:-----------|:------------------|:---------------|:-----------|:--------------|
| Random Forest | 21.31%     | 20.83%            | 21.33%         | 20.75%     | 20.74%        |
| SVM           | 19.58%     | 19.04%            | 19.67%         | 19.19%     | 19.14%        |
| 1D-CNN        | 20.29%     | 20.16%            | 20.38%         | 19.65%     | 19.65%        |

### 4.2 Macro-F1 comparison

| model         | within_macro_f1   | lolo_macro_f1   |   macro_f1_drop_percentage_points |
|:--------------|:------------------|:----------------|----------------------------------:|
| Random Forest | 20.75%            | 6.36%           |                           14.3902 |
| SVM           | 19.19%            | 6.34%           |                           12.8556 |
| 1D-CNN        | 19.65%            | 4.08%           |                           15.568  |

The macro-F1 drop is calculated as:

**Within-location test macro-F1 minus mean LOLO test macro-F1.**

A positive drop indicates lower performance when evaluating on an
entirely held-out location.

## 5. Key Findings

1. **Best average LOLO model:** Random Forest with a mean
   macro-F1 of 6.36%.
2. **Lowest average LOLO model:** 1D-CNN with a mean
   macro-F1 of 4.08%.
3. **Most challenging held-out location by average macro-F1:**
   Location_2.
4. Random Forest mean LOLO macro-F1:
   6.36%.
5. SVM mean LOLO macro-F1:
   6.34%.
6. CNN mean LOLO macro-F1:
   4.08%.

Random Forest and SVM have very similar average LOLO macro-F1. Their small difference should not be treated as evidence of a meaningful performance advantage.

The CNN has lower mean LOLO macro-F1 than both classical models in these experiments.

## 6. Discussion

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
