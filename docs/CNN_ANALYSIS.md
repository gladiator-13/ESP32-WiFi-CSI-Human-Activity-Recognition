
# CNN and Cross-Model Confusion Analysis

## 1. Objective

This analysis investigates the test-set classification errors of three
models used for Wi-Fi CSI-based human activity recognition:

- Random Forest
- Support Vector Machine (RBF SVM)
- 1D Convolutional Neural Network

The objective is to identify activities that are difficult to recognize,
examine recurring misclassification patterns, and determine whether
classification difficulties are specific to a model or shared across
different model architectures.

All results in this document refer to the within-location test split.

## 2. Evaluation Setup

Each model was evaluated on the same within-location test set containing
2,375 windows from 20 activity classes.

The test predictions were generated using the saved models and were
analyzed using:

- Per-class precision, recall, and F1-score
- Confusion matrices
- Directed actual-to-predicted misclassification counts
- Cross-model comparison of per-class F1-scores

The three prediction CSV files used different column conventions.
These were standardized during analysis without modifying the original
prediction files.

### Important methodological consideration

The windowing procedure uses 2-second windows with 50% overlap.
Consequently, adjacent windows are correlated and should not be treated
as statistically independent observations.

Misclassification counts in this analysis represent window-level
errors, not independent recording-level or participant-level errors.

## 3. Overall Test Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest | 21.31% | 20.83% | 21.33% | 20.75% |
| SVM | 19.58% | 19.04% | 19.67% | 19.19% |
| 1D-CNN | 20.29% | 20.16% | 20.38% | 19.65% |

Random Forest achieved the highest test macro-F1 among the three models.
The CNN performed better than SVM but did not surpass Random Forest.

The CNN achieved a validation macro-F1 of approximately 20.94%, compared
with a test macro-F1 of approximately 19.65%.

This difference demonstrates that validation performance does not
necessarily translate into equivalent test performance.

## 4. Activities Consistently Difficult Across Models

Two activities achieved an F1-score below 0.10 for all three models.

| Activity | Random Forest | SVM | 1D-CNN | Average F1 |
|---|---:|---:|---:|---:|
| A20 - Combing hair | 0.0905 | 0.0000 | 0.0093 | 0.0333 |
| A13 - Arms up and down | 0.0730 | 0.0359 | 0.0408 | 0.0499 |

These activities represent the most consistent classification
difficulties across the evaluated model families.

Other activities also show relatively weak performance:

- A12 - Boxing: particularly weak performance from Random Forest and SVM.
- A17 - Phone call: particularly weak performance from CNN.
- A11 - Clap: particularly weak performance from CNN.
- A15 - Arm circles: relatively weak performance across all three models.

The distinction is important: not every difficult activity is
consistently difficult for every model.

## 5. Major Cross-Model Misclassification Patterns

The following directed activity pairs were among the most frequent
misclassifications and appeared in all three models.

| Actual Activity | Predicted Activity | Combined Errors |
|---|---|---:|
| A13 - Arms up and down | A14 - Push forward | 113 |
| A19 - Reading a book | A18 - Writing notes | 85 |
| A07 - Jump in place | A04 - Jog in place | 84 |
| A18 - Writing notes | A02 - Sitting | 73 |
| A16 - Typing | A18 - Writing notes | 66 |
| A16 - Typing | A19 - Reading a book | 64 |
| A07 - Jump in place | A09 - Lunges | 63 |
| A17 - Phone call | A11 - Clap | 63 |
| A04 - Jog in place | A07 - Jump in place | 61 |
| A09 - Lunges | A06 - Sit down, stand up | 59 |

Combined errors are sums of window-level misclassifications across
the three models. They are not unique-window counts.

### 5.1 Upper-body movement confusion

The strongest shared confusion was:

A13 - Arms up and down -> A14 - Push forward

This pair accumulated 113 misclassified windows across the models.

Both activities involve arm movements. Similarity in their temporal
CSI patterns is a possible explanation for the confusion.

However, the confusion matrix alone cannot establish the physical
or signal-level cause.

### 5.2 Desk-related activity confusion

Several recurring patterns involve typing, reading, and writing:

- A19 - Reading a book -> A18 - Writing notes
- A16 - Typing -> A18 - Writing notes
- A16 - Typing -> A19 - Reading a book
- A18 - Writing notes -> A02 - Sitting

These activities may produce relatively subtle differences in body
movement and posture. The current representation may not capture
sufficiently distinctive characteristics for reliable classification.

This is a hypothesis requiring further investigation.

### 5.3 Dynamic lower-body activity confusion

Two recurring patterns involve jumping, jogging, and lunges:

- A07 - Jump in place -> A04 - Jog in place
- A04 - Jog in place -> A07 - Jump in place
- A07 - Jump in place -> A09 - Lunges
- A09 - Lunges -> A06 - Sit down, stand up

The reciprocal confusion between jumping and jogging suggests
difficulty distinguishing these dynamic movement patterns.

Temporal feature extraction and sequence modeling may be worth
investigating in subsequent experiments.

### 5.4 Combing hair

A20 - Combing hair consistently achieved very low F1-scores.

The activity also exhibited confusion with activities such as:

- A10 - Wave
- A15 - Arm circles

These errors may indicate difficulty distinguishing certain
upper-body movement patterns using the current CSI representation.

Further investigation of the corresponding recordings is required.

## 6. Interpretation Across Model Families

The presence of recurring confusion patterns across Random Forest,
SVM, and CNN is an important observation.

The models use different learning approaches:

- Random Forest operates on handcrafted statistical features.
- SVM uses an RBF kernel on handcrafted statistical features.
- CNN learns temporal and subcarrier representations directly from
  normalized CSI windows.

Despite these differences, several activity pairs remain difficult.

This suggests that the classification problem may involve limitations
that are not resolved simply by changing the model family.

Potential factors for further investigation include:

1. Similarity of CSI responses across different activities.
2. Insufficiently distinctive movement patterns in individual windows.
3. Variation between participants.
4. Effects of recording location and environmental conditions.
5. Limitations of the current temporal window representation.
6. Effects of preprocessing and subcarrier selection.

These are possible contributing factors, not established causal findings.

## 7. Limitations

The following limitations should be considered when interpreting
the analysis:

1. The analysis uses window-level predictions with 50% overlap.
2. Misclassification counts are not independent observations.
3. Confusion patterns do not establish the physical cause of an error.
4. Within-location evaluation does not establish generalization to
   unseen locations.
5. The current results do not establish statistical significance
   between model performance differences.
6. Recording-level and participant-level performance have not yet
   been evaluated in this analysis.

## 8. Conclusions

The cross-model analysis identifies two consistently difficult
activities: A20 - Combing hair and A13 - Arms up and down.

The most prominent shared directed confusion is A13 - Arms up and
down being predicted as A14 - Push forward.

Other recurring difficulties involve reading versus writing,
typing versus writing, jumping versus jogging, and phone calls versus
clapping.

Random Forest achieved the highest test macro-F1 of 20.75%, followed
by the 1D-CNN at 19.65% and SVM at 19.19%.

The results indicate that improving the representation of
activity-specific temporal and spatial CSI characteristics may be
more important than simply changing the classifier.

Further experiments should investigate recording-level performance,
cross-location generalization, and temporal modeling approaches.
