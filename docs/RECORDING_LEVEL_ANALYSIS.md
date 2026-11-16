# Recording-Level Human Activity Recognition Analysis

## 1. Objective

This analysis evaluates whether aggregating overlapping window predictions into recording-level predictions improves activity recognition performance.

Four models are evaluated: Random Forest, SVM, 1D-CNN, and CNN-BiLSTM-Attention.

The dataset contains four recording locations.

## 2. Recording-level methodology

- Each recording contains 19 windows.
- Each window is 2 seconds long.
- Consecutive windows have 50% overlap.
- The 19 predicted activity labels are aggregated using majority voting.
- The most frequent predicted activity becomes the recording-level prediction.
- Ties are resolved deterministically by selecting the smallest numeric activity label.
- All 20 activity classes are included in metric calculations, with zero-division handled as zero.

Recording-level evaluation changes the unit of prediction; it does not involve additional model training.

## 3. Within-location results

| model | recording_count | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | tie_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 168 | 0.2738 | 0.3013 | 0.2778 | 0.2609 | 0.2585 | 18 |
| SVM | 168 | 0.2619 | 0.2590 | 0.2653 | 0.2403 | 0.2383 | 14 |
| CNN-BiLSTM-Attention | 168 | 0.2321 | 0.2157 | 0.2361 | 0.2149 | 0.2123 | 18 |
| 1D-CNN | 168 | 0.2083 | 0.2282 | 0.2111 | 0.1970 | 0.1956 | 7 |

Best within-location model: **Random Forest**, with macro-F1 of 0.2609.

## 4. LOLO fold-wise results

| model | fold | recording_count | accuracy | macro_f1 | weighted_f1 | tie_count |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 1 | 204 | 0.1569 | 0.1227 | 0.1226 | 21 |
| SVM | 1 | 204 | 0.1520 | 0.1141 | 0.1150 | 13 |
| 1D-CNN | 1 | 204 | 0.1275 | 0.0996 | 0.0993 | 13 |
| CNN-BiLSTM-Attention | 1 | 204 | 0.1029 | 0.0784 | 0.0773 | 10 |
| Random Forest | 2 | 185 | 0.0432 | 0.0298 | 0.0306 | 11 |
| SVM | 2 | 185 | 0.0378 | 0.0220 | 0.0220 | 7 |
| 1D-CNN | 2 | 185 | 0.0432 | 0.0197 | 0.0197 | 7 |
| CNN-BiLSTM-Attention | 2 | 185 | 0.0324 | 0.0096 | 0.0093 | 6 |
| Random Forest | 3 | 186 | 0.1022 | 0.0866 | 0.0880 | 16 |
| SVM | 3 | 186 | 0.1022 | 0.0776 | 0.0755 | 17 |
| 1D-CNN | 3 | 186 | 0.0806 | 0.0758 | 0.0742 | 12 |
| CNN-BiLSTM-Attention | 3 | 186 | 0.0645 | 0.0549 | 0.0539 | 11 |
| SVM | 4 | 192 | 0.0990 | 0.0458 | 0.0437 | 13 |
| 1D-CNN | 4 | 192 | 0.1042 | 0.0444 | 0.0416 | 9 |
| CNN-BiLSTM-Attention | 4 | 192 | 0.1094 | 0.0417 | 0.0391 | 5 |
| Random Forest | 4 | 192 | 0.0781 | 0.0357 | 0.0335 | 6 |

## 5. LOLO mean and standard deviation

The mean is the unweighted average of the four held-out location folds. Standard deviation is the sample standard deviation (ddof=1).

| model | number_of_folds | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | weighted_f1_mean | weighted_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 4 | 0.0951 | 0.0478 | 0.0687 | 0.0441 | 0.0687 | 0.0446 |
| SVM | 4 | 0.0977 | 0.0467 | 0.0649 | 0.0400 | 0.0641 | 0.0404 |
| 1D-CNN | 4 | 0.0889 | 0.0359 | 0.0599 | 0.0351 | 0.0587 | 0.0351 |
| CNN-BiLSTM-Attention | 4 | 0.0773 | 0.0359 | 0.0461 | 0.0287 | 0.0449 | 0.0285 |

Best mean LOLO model: **Random Forest**, with macro-F1 of 0.0687.

## 6. Window-level versus recording-level comparison

| model | evaluation | window_level | recording_level | difference_percentage_points |
| --- | --- | --- | --- | --- |
| Random Forest | within_location | 0.2075 | 0.2609 | 5.3329 |
| Random Forest | lolo_mean | 0.0678 | 0.0687 | 0.0891 |
| SVM | within_location | 0.1919 | 0.2403 | 4.8388 |
| SVM | lolo_mean | 0.0641 | 0.0649 | 0.0784 |
| 1D-CNN | within_location | 0.1965 | 0.1970 | 0.0501 |
| 1D-CNN | lolo_mean | 0.0604 | 0.0599 | -0.0551 |
| CNN-BiLSTM-Attention | within_location | 0.1813 | 0.2149 | 3.3537 |
| CNN-BiLSTM-Attention | lolo_mean | 0.0502 | 0.0461 | -0.4098 |

A positive difference indicates a higher metric after recording-level aggregation. This is a change in evaluation unit, not evidence that the trained model itself improved.

## 7. Per-activity recording-level results

Detailed precision, recall, F1, and support for every activity, model, and evaluation fold are available in `reports/recording_level/summaries/recording_level_per_class_comparison.csv`.

## 8. Interpretation

Within-location and cross-location performance are reported separately. Recording-level majority voting can reduce some window-level prediction variability, but it does not necessarily remove the difficulty of generalizing to unseen locations.

Location-specific differences should be interpreted descriptively. Four locations are available, and the observed differences do not establish a causal explanation for performance variation.

## 9. Limitations

- The 50% overlapping windows are correlated observations.
- Majority voting discards prediction confidence.
- The deterministic tie-break can affect tied recordings.
- Only four locations are used for LOLO evaluation.
- Fold means are unweighted across locations.
- No statistical significance claim is made.

## 10. Reproducibility

The analysis is generated by `src/compare_recording_level.py` using saved prediction files, classification reports, and existing window-level comparison summaries.
