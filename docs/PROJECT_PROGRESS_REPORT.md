# ESP32 Wi-Fi CSI Human Activity Recognition

## Project Progress Report

**Project:** Human Activity Recognition using ESP32 Wi-Fi Channel State
Information (CSI)\
**Status:** Data preparation and feature engineering completed; baseline
model development next.

------------------------------------------------------------------------

## 1. Project Overview

The objective of this project is to develop a Human Activity Recognition
(HAR) system using Channel State Information (CSI) collected from Wi-Fi
signals using ESP32 devices.

Wi-Fi CSI captures how wireless signals change as they interact with
people and objects in an environment. These signal variations can be
used to identify human activities without requiring wearable sensors or
cameras.

The system aims to recognize 20 human activities from CSI amplitude
measurements using machine-learning and deep-learning approaches.

The project focuses on evaluating recognition performance under two
different scenarios:

1.  **Within-location evaluation:** Training, validation and testing
    participants belong to the same physical environment, while
    remaining participant-independent.
2.  **Leave-One-Location-Out (LOLO) evaluation:** The model is trained
    on two locations and evaluated on a completely unseen third
    location.

The second scenario is particularly important because it evaluates the
ability of the model to generalize across different physical
environments.

------------------------------------------------------------------------

## 2. Dataset Understanding

### 2.1 Dataset Structure

The dataset contains recordings from 28 participants distributed across
three locations.

  Location     Participants      Count
  ------------ -------------- --------
  Location 1   P00--P09             10
  Location 2   P10--P18              9
  Location 3   P19--P27              9
  **Total**    **P00--P27**     **28**

The dataset contains 20 human activity classes, labelled 1--20.

Baseline recordings are also available and are assigned activity label
`0`.

For the primary experiment, we are considering only the 20 activity
classes. Baseline recordings are retained in the processed metadata but
excluded from the current classification pipeline.

### 2.2 Initial Dataset Statistics

  Recording category             Count
  -------------------------- ---------
  Activity recordings              577
  Baseline recordings               75
  **Total raw recordings**     **652**

Every participant has recordings for all 20 activity classes in the
original dataset.

------------------------------------------------------------------------

## 3. Initial Dataset Inspection

We inspected the raw CSV files to understand their structure and CSI
representation.

### CSV structure

The recordings contain:

-   `timestamp`: packet timestamp.
-   `csi_line`: CSI measurements represented as an array.

Each CSI packet contains 64 complex subcarrier values, represented by
128 integers.

The dataset uses one antenna (`ant=0`), so the initial implementation
focuses on CSI amplitude rather than dual-antenna phase differences.

Phase information and dual-antenna phase differences are not included in
the current pipeline.

### Sample inspection

A representative recording, `P00_A01_R1_1790671306.csv`, contained:

-   2,228 packets.
-   Approximately 22 seconds of recording.
-   An average sampling rate close to 101 Hz.
-   Irregular packet arrival intervals.
-   One malformed CSI packet.

This inspection established that packet timestamps could not simply be
assumed to have perfectly uniform intervals.

------------------------------------------------------------------------

## 4. CSI Preprocessing

The preprocessing pipeline was implemented in:

-   `src/preprocessing.py`
-   `src/batch_preprocessing.py`
-   `notebooks/02_csi_preprocessing.ipynb`

The preprocessing pipeline was implemented in Python using NumPy,
Pandas, SciPy, and related scientific computing libraries.

### 4.1 CSI Parsing

Each CSI string was parsed into 128 numerical values and reshaped into a
64 × 2 array.

The two columns represent the imaginary and real components of each
complex CSI measurement.

Each CSI packet contains 64 complex subcarrier values represented by 128
integer values.

The real and imaginary components are parsed and converted into complex
values.

### 4.2 CSI Amplitude Extraction

CSI amplitude was calculated using:

$$
A = \sqrt{Re^2 + Im^2}
$$

This produces amplitude measurements for each subcarrier.

Only one antenna (`ant=0`) is used in the current experiment.

### 4.3 Subcarrier Selection

The following subcarrier indices are removed:

`[0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]`

This removes 12 subcarriers, leaving:

$$
64-12=52
$$

usable CSI amplitude features per packet.

### 4.4 Signal Cleaning and Processing

The implemented preprocessing operations include:

1.  CSI parsing and malformed-row removal.
2.  Packet timestamp inspection.
3.  Packet-gap handling and interpolation.
4.  Hampel filtering for outlier removal.
5.  Savitzky-Golay filtering for signal smoothing.
6.  Uniform resampling to 100 Hz.
7.  Trimming each recording to 20 seconds.
8.  Saving the resulting processed signals.

The pipeline also:

-   Removes invalid timestamps.
-   Sorts packets by timestamp.
-   Removes duplicate timestamps.
-   Handles packet gaps through interpolation where appropriate.

### 4.5 Hampel Filtering

A Hampel filter was applied to reduce outliers.

Parameters:

-   Window size: 10.
-   Threshold: 3 × MAD.
-   MAD is scaled using the robust Gaussian consistency factor.

### 4.6 Savitzky-Golay Smoothing

Savitzky-Golay filtering was applied to reduce high-frequency noise
while retaining useful signal variations.

Parameters:

-   Window length: 11.
-   Polynomial order: 3.

### 4.7 Uniform Resampling

The original packet timestamps were irregular.

Signals were interpolated onto a uniform 100 Hz time grid.

The target duration was fixed at 20 seconds.

### 4.8 Fixed-Length Output

Each processed recording was converted into a matrix with dimensions:

`(2000, 52)`

This represents:

-   2,000 time samples.
-   52 usable subcarriers.
-   20 seconds at 100 Hz.

  Operation               Configuration
  ----------------------- -------------------------------
  Hampel filter           Window 10, threshold 3 × MAD
  Savitzky-Golay filter   Window 11, polynomial order 3
  Resampling frequency    100 Hz
  Recording duration      20 seconds
  Samples per recording   2,000
  Selected CSI features   52

Every successfully processed recording has the shape:

$$
(2000,52)
$$

No global normalization or PCA was fitted during this stage. These
operations must be fitted using training data only after the dataset
split.

------------------------------------------------------------------------

## 5. Problems Encountered and Solutions

### Problem 1: CSI parsing failures

**Issue:** The initial parser using `np.fromstring()` generated warnings
and failed to correctly parse some CSI strings because of inconsistent
string formatting.

**Solution:** Replaced the parser with regular-expression-based
extraction. The updated parser:

-   Extracts numerical values from the CSI array.
-   Validates that exactly 128 values are present.
-   Converts them into a NumPy array.
-   Reshapes the result into 64 × 2.

This substantially improved parsing reliability.

### Problem 2: Baseline filename format

**Issue:** Baseline recordings used filenames containing `BASE` rather
than the activity naming convention `A00`.

**Solution:** Updated the metadata extraction function to recognize both
formats:

-   `P00_A01_R1_...csv`
-   `P00_BASE_R1_...csv`

Baseline recordings are assigned activity label 0 and recording type
`baseline`.

### Problem 3: Irregular packet gaps

**Issue:** Some recordings contained unusually large gaps between
consecutive packet timestamps.

The initial maximum-gap threshold of 100 ms rejected 16 recordings.

**Solution:** Investigated the packet-gap distribution and provisionally
increased the maximum allowed gap to 200 ms.

This recovered four additional recordings with gaps between
approximately 157 and 180 ms.

Two recordings remained excluded because their maximum gaps were
approximately:

-   P26 Activity 18: 376.8 ms.
-   P27 Activity 19: 297.3 ms.

These recordings were excluded rather than forcing interpolation across
excessively large gaps.

### Problem 4: Missing participant/activity combinations in metadata

**Issue:** An initial coverage analysis reported 24 missing
participant/activity combinations.

**Investigation:** We compared the original dataset against the
processed metadata and discovered that the analysis was using an
outdated in-memory `activity_metadata` variable.

**Solution:** Reloaded the metadata directly from disk and repeated the
analysis using the current dataset.

The corrected analysis confirmed:

-   28 participants.
-   20 activity classes.
-   Zero missing participant/activity combinations.

### Problem 5: Incorrect file paths during validation

**Issue:** An initial raw-file comparison reported zero original
recordings and 650 missing signal files.

**Investigation:** The notebook was running from the `notebooks/`
directory, while the validation script used paths relative to the
project root.

**Solution:** Identified the project root automatically using the
existence of the raw data directory and metadata file, then constructed
absolute paths.

The corrected validation confirmed that the paths and processed files
were correct.

------------------------------------------------------------------------

## 6. Final Preprocessing Results

The batch preprocessing pipeline processed 650 of the 652 original
recordings.

  Recording category     Original   Processed
  -------------------- ---------- -----------
  Activity                    577         575
  Baseline                     75          75
  Total                       652         650

Two activity recordings were excluded due to excessive packet gaps.

The packet-gap threshold was provisionally set to 200 ms after examining
the recordings.

Two recordings were excluded because of unusually large packet gaps:

-   `P26_A18_R1_1794407419.csv` --- maximum gap: 376.8 ms
-   `P27_A19_R1_1794408319.csv` --- maximum gap: 297.3 ms

### Quality Assurance Results

  Validation                                  Result
  ------------------------------------------- --------
  Raw recordings                              652
  Processed metadata entries                  650
  Processed NumPy arrays                      650
  Missing metadata entries                    2
  Unexpected metadata entries                 0
  Missing signal files                        0
  Participants                                28
  Activity classes                            20
  Missing participant/activity combinations   0
  Invalid array shapes                        0
  Arrays containing NaN or infinity           0

All successfully processed arrays have shape `(2000, 52)`.

The original dataset was also independently checked to confirm that all
28 participants have recordings for all 20 activities.

A separate dataset inspection verified the processed recordings against
the original dataset.

  Validation                                                 Result
  ---------------------------------------------------------- --------
  Participant folders identified                             28
  Activity classes identified                                20
  Missing participant/activity combinations in raw dataset   0
  Unexpected processed recordings                            0
  Missing processed signal files                             0
  Invalid signal shapes                                      0
  Non-finite signal values                                   0
  Processed recordings                                       650

**Status: PASSED**

------------------------------------------------------------------------

## 7. Files and Code Implemented

The following project components have been implemented:

-   `src/preprocessing.py`: Handles preprocessing of an individual CSI
    recording.
-   `src/batch_preprocessing.py`: Applies preprocessing to all raw
    recordings and generates metadata.
-   `notebooks/02_csi_preprocessing.ipynb`: Used for experimentation,
    inspection, and preprocessing validation.
-   `data/processed/metadata.csv`: Contains recording-level metadata.
-   `data/processed/preprocessing_errors.csv`: Records preprocessing
    failures.
-   `src/create_splits.py`
-   `src/create_lolo_splits.py`
-   `src/create_windows.py`
-   `src/validate_windows.py`
-   `src/analyze_class_distribution.py`
-   `src/feature_normalization.py`
-   `src/validate_normalization.py`
-   `src/extract_handcrafted_features.py`
-   `src/validate_handcrafted_features.py`

Processed signal arrays are stored under:

`data/processed/signals/`

Raw recordings are stored under:

`data/data/`

Raw recordings and generated signal arrays are excluded from Git
tracking to avoid unnecessarily large repository uploads.

------------------------------------------------------------------------

## 8. Dataset Splitting

The dataset is divided into training, validation and test sets at the
participant level before window generation.

This is essential because randomly splitting overlapping windows could
introduce data leakage.

Two evaluation strategies have been implemented.

### 8.1 Within-Location Participant-Wise Split

Implemented in:

`src/create_splits.py`

Configuration:

-   Random seed: 42
-   Approximately 60% training participants
-   Approximately 20% validation participants
-   Approximately 20% testing participants

Participant allocation:

  Location        Train   Validation    Test
  ------------ -------- ------------ -------
  Location 1          6            2       2
  Location 2          5            2       2
  Location 3          5            2       2
  **Total**      **16**        **6**   **6**

Recording distribution:

  Split          Activity recordings   Baseline recordings
  ------------ --------------------- ---------------------
  Train                          328                    34
  Validation                     122                    17
  Test                           125                    24
  **Total**                  **575**                **75**

All 20 activity classes are represented in every split.

No participant appears in more than one split.

### 8.2 Leave-One-Location-Out (LOLO)

Implemented in:

`src/create_lolo_splits.py`

Three independent evaluation folds were created.

  Fold     Training locations        Testing location
  -------- ------------------------- ------------------
  Fold 1   Location 2 + Location 3   Location 1
  Fold 2   Location 1 + Location 3   Location 2
  Fold 3   Location 1 + Location 2   Location 3

Validation participants are selected only from the two training
locations.

The held-out location is used exclusively for testing.

All three folds passed participant separation and activity class
coverage checks.

**Important:** Each LOLO fold has its own training, validation and test
allocation. The held-out test location never contributes to training or
validation.

### 8.3 Split Files

Within-location:

-   `data/processed/splits/within_location_split.csv`
-   `data/processed/splits/recording_splits.csv`
-   `data/processed/splits/activity_class_coverage.csv`

LOLO:

-   `data/processed/splits/lolo_participant_splits.csv`
-   `data/processed/splits/lolo_recording_splits.csv`
-   `data/processed/splits/lolo_activity_class_coverage.csv`

**Status: ALL SPLIT VALIDATIONS PASSED**

------------------------------------------------------------------------

## 9. Window Generation

Implemented in:

`src/create_windows.py`

The processed recordings are converted into fixed-length windows for
classification.

### 9.1 Window Configuration

  Parameter                         Value
  ----------------------- ---------------
  Recording length          2,000 samples
  Window length               200 samples
  Window duration               2 seconds
  Sampling frequency               100 Hz
  Overlap                             50%
  Stride                      100 samples
  Features per timestep                52

Each window has the shape:

$$
(200,52)
$$

Number of windows per recording:

$$
N =
\left\lfloor
\frac{2000-200}{100}
\right\rfloor+1
=19
$$

Since there are 575 activity recordings:

$$
575 \times 19 = 10,925
$$

Therefore, the within-location dataset contains 10,925 activity windows.

Baseline recordings are excluded from the current 20-class
classification task.

### 9.2 Within-Location Window Distribution

  Split             Windows Shape
  ------------ ------------ -------------------
  Train               6,232 `(6232, 200, 52)`
  Validation          2,318 `(2318, 200, 52)`
  Test                2,375 `(2375, 200, 52)`
  **Total**      **10,925** 

### 9.3 LOLO Window Distribution

  Fold       Train   Validation    Test    Total
  -------- ------- ------------ ------- --------
  Fold 1     5,472        1,577   3,876   10,925
  Fold 2     5,852        1,558   3,515   10,925
  Fold 3     5,852        1,539   3,534   10,925

The same underlying recordings are reassigned across folds, so the total
of 32,775 LOLO windows represents three fold-specific datasets, not
32,775 unique windows.

### 9.4 Window Storage

The generated data is saved under:

`data/processed/windows/`

Structure:

``` text
windows/
├── within_location/
│   ├── train/
│   ├── validation/
│   └── test/
└── lolo/
    ├── fold_1/
    │   ├── train/
    │   ├── validation/
    │   └── test/
    ├── fold_2/
    └── fold_3/
```

Each split contains:

-   `X.npy`
-   `y.npy`
-   `metadata.csv`

### 9.5 Window Quality Assurance

Implemented in:

`src/validate_windows.py`

The validation checks included:

-   Correct array dimensions.
-   Correct window length and feature count.
-   Finite numerical values.
-   Activity labels matching metadata.
-   Exactly 19 windows per recording.
-   Correct window numbering.
-   Correct starting and ending sample positions.
-   Participant separation.
-   All 20 activity classes represented.
-   Correct held-out location for each LOLO fold.
-   Correct total recording and window counts.

**Status: ALL WITHIN-LOCATION AND LOLO WINDOW VALIDATIONS PASSED**

------------------------------------------------------------------------

## 10. Class Distribution Analysis

Implemented in:

`src/analyze_class_distribution.py`

Class distribution was examined across all splits and all three LOLO
folds.

The imbalance ratio is calculated as:

$$
IR =
\frac{\text{Maximum class count}}
{\text{Minimum class count}}
$$

### 10.1 Within-Location

  -----------------------------------------------------------------------
  Split                     Minimum            Maximum    Imbalance ratio
                      windows/class      windows/class 
  -------------- ------------------ ------------------ ------------------
  Train                         304                342               1.12

  Validation                    114                133               1.17

  Test                          114                133               1.17
  -----------------------------------------------------------------------

### 10.2 LOLO

  Fold   Split          Minimum   Maximum   Imbalance ratio
  ------ ------------ --------- --------- -----------------
  1      Train              266       304              1.14
  1      Validation          76        95              1.25
  1      Test               190       209              1.10
  2      Train              285       323              1.13
  2      Validation          76        95              1.25
  2      Test               171       190              1.11
  3      Train              285       323              1.13
  3      Validation          76        95              1.25
  3      Test               171       209              1.22

### 10.3 Class Imbalance Decision

The maximum observed imbalance ratio is 1.25.

This indicates relatively mild class imbalance.

For the initial experiments:

-   No SMOTE will be applied.
-   No artificial duplication of test samples will be performed.
-   Initial models will be trained without class weighting.
-   Macro-F1 and per-class recall will be reported.
-   Class weighting can be investigated later if model performance
    indicates a need.

Note that overlapping windows from the same recording are correlated.
Window counts should not be interpreted as independent recording counts.

**Status: CLASS DISTRIBUTION ANALYSIS COMPLETED**

------------------------------------------------------------------------

## 11. Feature Normalization

Implemented in:

`src/feature_normalization.py`

CSI amplitude values can have different scales across subcarriers.

StandardScaler (Z-score normalization) was used:

$$
z = \frac{x-\mu}{\sigma}
$$

Each of the 52 CSI features is normalized independently.

### 11.1 Leakage Prevention

Normalization is fitted exclusively on training data.

The fitted training scaler is then applied to:

-   Training data
-   Validation data
-   Test data

For LOLO, a separate scaler is fitted for each fold.

The held-out test location is never used to calculate normalization
statistics.

### 11.2 Normalized Dataset Structure

Output directory:

`data/processed/normalized/`

``` text
normalized/
├── within_location/
│   ├── scaler.joblib
│   ├── train/
│   ├── validation/
│   └── test/
└── lolo/
    ├── fold_1/
    ├── fold_2/
    └── fold_3/
```

Each split contains:

-   `X.npy`
-   `y.npy`
-   `metadata.csv`

Each evaluation setup also stores its fitted `scaler.joblib`.

### 11.3 Normalization Quality Assurance

Implemented in:

`src/validate_normalization.py`

Training-set validation results:

  Dataset             Maximum absolute mean   Maximum std. error
  ----------------- ----------------------- --------------------
  Within-location                0.00000014           0.00000006
  LOLO Fold 1                    0.00000014           0.00000004
  LOLO Fold 2                    0.00000022           0.00000005
  LOLO Fold 3                    0.00000013           0.00000005

All errors are extremely close to zero.

All validation and test arrays were also checked for finite values and
correct dimensions.

**Status: ALL NORMALIZATION VALIDATIONS PASSED**

------------------------------------------------------------------------

## 12. Handcrafted Feature Extraction

Implemented in:

`src/extract_handcrafted_features.py`

Random Forest and other traditional machine-learning algorithms require
a two-dimensional feature matrix.

The normalized CSI windows have shape:

$$
(N,200,52)
$$

Therefore, statistical features are extracted from each window.

### 12.1 Extracted Features

Nine statistical features are calculated for each of the 52 subcarriers.

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  Mean                                Average amplitude

  Standard deviation                  Signal variability

  Minimum                             Minimum amplitude

  Maximum                             Maximum amplitude

  Median                              Median amplitude

  25th percentile                     Lower distribution boundary

  75th percentile                     Upper distribution boundary

  RMS                                 Root mean square

  Mean absolute difference            Average absolute change between
                                      consecutive samples
  -----------------------------------------------------------------------

Total features:

$$
52 \times 9 = 468
$$

Every window is therefore converted into a 468-dimensional feature
vector.

### 12.2 Feature Dataset Structure

Output directory:

`data/processed/handcrafted_features/`

``` text
handcrafted_features/
├── within_location/
│   ├── train/
│   ├── validation/
│   └── test/
└── lolo/
    ├── fold_1/
    ├── fold_2/
    └── fold_3/
```

Each split contains:

-   `X_features.npy`
-   `y.npy`
-   `metadata.csv`
-   `feature_names.csv`

Expected shapes:

  Dataset           Split        Shape
  ----------------- ------------ ---------------
  Within-location   Train        `(6232, 468)`
  Within-location   Validation   `(2318, 468)`
  Within-location   Test         `(2375, 468)`
  LOLO Fold 1       Train        `(5472, 468)`
  LOLO Fold 2       Train        `(5852, 468)`
  LOLO Fold 3       Train        `(5852, 468)`

### 12.3 Feature Quality Assurance

Implemented in:

`src/validate_handcrafted_features.py`

Validation checks:

-   Correct two-dimensional feature matrices.
-   Exactly 468 features per window.
-   No NaN or infinite values.
-   Labels match metadata.
-   All 20 activity classes are represented.
-   Feature names are unique.
-   Metadata is present.
-   Window counts remain unchanged.

**Status: ALL HANDCRAFTED FEATURE VALIDATIONS PASSED**

------------------------------------------------------------------------

## 13. Current Project Structure

The relevant project structure is now:

``` text
Project/
│
├── data/
│   ├── data/
│   │   ├── P00/
│   │   ├── P01/
│   │   └── ...
│   │
│   └── processed/
│       ├── metadata.csv
│       ├── preprocessing_errors.csv
│       │
│       ├── signals/
│       │
│       ├── splits/
│       │   ├── within_location_split.csv
│       │   ├── recording_splits.csv
│       │   ├── activity_class_coverage.csv
│       │   ├── lolo_participant_splits.csv
│       │   ├── lolo_recording_splits.csv
│       │   └── lolo_activity_class_coverage.csv
│       │
│       ├── windows/
│       │   ├── within_location/
│       │   └── lolo/
│       │
│       ├── normalized/
│       │   ├── within_location/
│       │   └── lolo/
│       │
│       └── handcrafted_features/
│           ├── within_location/
│           └── lolo/
│
├── notebooks/
│   └── 02_csi_preprocessing.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── batch_preprocessing.py
│   ├── create_splits.py
│   ├── create_lolo_splits.py
│   ├── create_windows.py
│   ├── validate_windows.py
│   ├── analyze_class_distribution.py
│   ├── feature_normalization.py
│   ├── validate_normalization.py
│   ├── extract_handcrafted_features.py
│   └── validate_handcrafted_features.py
│
├── .gitignore
└── PROJECT_PROGRESS_REPORT.md
```

Generated datasets and raw recordings are excluded from GitHub using
`.gitignore`.

------------------------------------------------------------------------

## 14. Completed Milestones

    No. Milestone                               Status
  ----- --------------------------------------- -----------
      1 Dataset exploration                     Completed
      2 CSI parsing and preprocessing           Completed
      3 Preprocessing quality assurance         Passed
      4 Within-location participant splitting   Passed
      5 LOLO participant splitting              Passed
      6 Window generation                       Completed
      7 Window quality assurance                Passed
      8 Class distribution analysis             Completed
      9 Training-only normalization             Completed
     10 Normalization quality assurance         Passed
     11 Handcrafted feature extraction          Completed
     12 Handcrafted feature quality assurance   Passed

**Overall status: DATA PREPARATION AND FEATURE ENGINEERING COMPLETED.**

------------------------------------------------------------------------

## 15. Remaining Work

The next stage is model development and evaluation.

### 15.1 Traditional Machine Learning

Planned models:

1.  Random Forest
2.  Support Vector Machine (SVM)

Both will use the 468 handcrafted features.

Experiments will include:

-   Hyperparameter tuning using training and validation data.
-   Final evaluation using the test set.
-   Accuracy.
-   Macro precision.
-   Macro recall.
-   Macro-F1.
-   Confusion matrices.
-   Per-class performance analysis.

### 15.2 Deep Learning

Planned neural-network architectures:

1.  1D-CNN
2.  CNN-BiLSTM with Attention

These models will use the normalized original CSI windows:

$$
(N,200,52)
$$

The handcrafted 468-dimensional features will not replace the original
time-series input for these neural-network models.

### 15.3 Additional Experiments

Planned investigations include:

-   PCA versus no PCA.
-   Effect of removing the first principal component.
-   Comparison between handcrafted and learned features.
-   Model generalization across locations.
-   Comparison of within-location and LOLO performance.
-   Recording-level and participant-level aggregated evaluation.

PCA, if used, must be fitted using training data only, independently for
each evaluation setup and LOLO fold.

### 15.4 Final Comparison

A final comparison table will summarize the performance of all models
under both evaluation strategies.

The LOLO results will be reported separately for all three held-out
locations, along with an aggregate result.

------------------------------------------------------------------------

## 16. Important Experimental Principles

The following rules will be maintained throughout model development:

1.  Split participants before window generation.
2.  Never allow a participant to appear in multiple splits within one
    evaluation fold.
3.  Never fit normalization or PCA using validation or test data.
4.  Fit a separate preprocessing transformation for every LOLO fold.
5.  Do not use the held-out location during LOLO training or validation.
6.  Do not artificially balance the test set.
7.  Preserve the original processed signals and window datasets.
8.  Use validation data for model selection and hyperparameter tuning.
9.  Use the test set only for final evaluation.
10. Report macro-F1 and per-class metrics in addition to accuracy.
11. Consider the correlation between overlapping windows when
    interpreting evaluation results.
12. Where appropriate, aggregate window predictions at recording or
    participant level to provide additional evaluation.

------------------------------------------------------------------------

## 17. Immediate Next Step

**Start Random Forest baseline development.**

The handcrafted feature matrices are ready:

-   Input: 468 features per window.
-   Output classes: 20 activities.
-   Evaluation: Within-location and three LOLO folds.

The first experiment will establish a traditional machine-learning
baseline before moving to SVM and deep-learning architectures.
