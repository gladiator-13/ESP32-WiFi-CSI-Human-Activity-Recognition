# Unified Model Comparison and LOLO Analysis

## 1. Objective

This analysis compares four machine learning models for ESP32 Wi-Fi
CSI Human Activity Recognition:

- Random Forest
- Support Vector Machine (SVM)
- 1D Convolutional Neural Network (CNN)
- CNN-BiLSTM-Attention

Two evaluation settings are considered:

1. **Within-location:** Participant-wise train, validation and test
   splits are created within each recording location.
2. **Leave-One-Location-Out (LOLO):** Each fold holds out one entire
   recording location for testing. Training and validation data come
   from the other three locations.

The primary comparison metric is macro-F1 because it gives equal
importance to all 20 activity classes.

## 2. LOLO Experimental Results

### 2.1 Fold-wise results

|   fold | held_out_location   | model                | accuracy   | macro_f1   | weighted_f1   |
|-------:|:--------------------|:---------------------|:-----------|:-----------|:--------------|
|      1 | Location_1          | 1D-CNN               | 11.30%     | 9.08%      | 9.05%         |
|      1 | Location_1          | CNN-BiLSTM-Attention | 9.55%      | 7.69%      | 7.65%         |
|      1 | Location_1          | Random Forest        | 12.56%     | 10.79%     | 10.81%        |
|      1 | Location_1          | SVM                  | 11.97%     | 9.66%      | 9.68%         |
|      2 | Location_2          | 1D-CNN               | 4.55%      | 3.17%      | 3.17%         |
|      2 | Location_2          | CNN-BiLSTM-Attention | 3.41%      | 1.57%      | 1.56%         |
|      2 | Location_2          | Random Forest        | 4.04%      | 2.83%      | 2.86%         |
|      2 | Location_2          | SVM                  | 4.58%      | 3.29%      | 3.33%         |
|      3 | Location_3          | 1D-CNN               | 7.41%      | 6.89%      | 6.80%         |
|      3 | Location_3          | CNN-BiLSTM-Attention | 6.79%      | 5.87%      | 5.82%         |
|      3 | Location_3          | Random Forest        | 9.39%      | 8.55%      | 8.56%         |
|      3 | Location_3          | SVM                  | 8.04%      | 6.87%      | 6.80%         |
|      4 | Location_4          | 1D-CNN               | 9.90%      | 5.03%      | 4.79%         |
|      4 | Location_4          | CNN-BiLSTM-Attention | 10.86%     | 4.97%      | 4.71%         |
|      4 | Location_4          | Random Forest        | 8.88%      | 4.97%      | 4.78%         |
|      4 | Location_4          | SVM                  | 10.47%     | 5.80%      | 5.60%         |

### 2.2 Mean and standard deviation across folds

| model                | accuracy_mean   | accuracy_std   | macro_f1_mean   | macro_f1_std   | weighted_f1_mean   | weighted_f1_std   |
|:---------------------|:----------------|:---------------|:----------------|:---------------|:-------------------|:------------------|
| Random Forest        | 8.72%           | 3.52%          | 6.78%           | 3.56%          | 6.75%              | 3.60%             |
| SVM                  | 8.76%           | 3.23%          | 6.41%           | 2.64%          | 6.35%              | 2.64%             |
| 1D-CNN               | 8.29%           | 2.97%          | 6.04%           | 2.53%          | 5.95%              | 2.54%             |
| CNN-BiLSTM-Attention | 7.65%           | 3.29%          | 5.02%           | 2.57%          | 4.93%              | 2.56%             |

The mean is calculated as the unweighted arithmetic average of the
four location-specific fold results. Standard deviation uses the
sample standard deviation (ddof = 1).

### 2.3 Model comparison

The mean LOLO macro-F1 values are:

- **Random Forest:** 6.78% (standard deviation: 3.56 percentage points)
- **SVM:** 6.41% (standard deviation: 2.64 percentage points)
- **1D-CNN:** 6.04% (standard deviation: 2.53 percentage points)
- **CNN-BiLSTM-Attention:** 5.02% (standard deviation: 2.57 percentage points)

## 3. Location-wise Analysis

- **Location_1:** Best macro-F1 was achieved by Random Forest (10.79%).
- **Location_2:** Best macro-F1 was achieved by SVM (3.29%).
- **Location_3:** Best macro-F1 was achieved by Random Forest (8.55%).
- **Location_4:** Best macro-F1 was achieved by SVM (5.80%).

The fold-wise results show how model performance changes when the
held-out recording location changes. A location with consistently
lower performance across models may represent a more challenging
generalization setting.

These results do not, by themselves, establish why a location is
more difficult.

## 4. Within-Location vs Cross-Location Generalization

### 4.1 Within-location test performance

| model                | accuracy   | macro_precision   | macro_recall   | macro_f1   | weighted_f1   |
|:---------------------|:-----------|:------------------|:---------------|:-----------|:--------------|
| Random Forest        | 21.31%     | 20.83%            | 21.33%         | 20.75%     | 20.74%        |
| SVM                  | 19.58%     | 19.04%            | 19.67%         | 19.19%     | 19.14%        |
| 1D-CNN               | 20.29%     | 20.16%            | 20.38%         | 19.65%     | 19.65%        |
| CNN-BiLSTM-Attention | 18.52%     | 18.79%            | 18.71%         | 18.13%     | 18.00%        |

### 4.2 Macro-F1 comparison

| model                | within_macro_f1   | lolo_macro_f1   |   macro_f1_drop_percentage_points |
|:---------------------|:------------------|:----------------|----------------------------------:|
| Random Forest        | 20.75%            | 6.78%           |                           13.9711 |
| SVM                  | 19.19%            | 6.41%           |                           12.7862 |
| 1D-CNN               | 19.65%            | 6.04%           |                           13.6069 |
| CNN-BiLSTM-Attention | 18.13%            | 5.02%           |                           13.1083 |

The macro-F1 drop is calculated as:

**Within-location test macro-F1 minus mean LOLO test macro-F1.**

A positive drop indicates lower performance when evaluating on an
entirely held-out location.

## 5. Key Findings

1. **Best average LOLO model:** Random Forest with a mean
   macro-F1 of 6.78%.
2. **Lowest average LOLO model:** CNN-BiLSTM-Attention with a mean
   macro-F1 of 5.02%.
3. **Most challenging held-out location by average macro-F1:**
   Location_2.
4. Model-wise mean LOLO macro-F1:
       - Random Forest: 6.78%.
   - SVM: 6.41%.
   - 1D-CNN: 6.04%.
   - CNN-BiLSTM-Attention: 5.02%.

Random Forest and SVM have very similar average LOLO macro-F1. Their small difference should not be treated as evidence of a meaningful performance advantage.

The best-performing model by mean LOLO macro-F1 is Random Forest. This ranking describes the current experimental results and does not establish statistical significance.

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

- Only four recording locations are available.
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
