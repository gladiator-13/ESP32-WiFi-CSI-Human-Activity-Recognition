# ESP32 Wi-Fi CSI-Based Human Activity Recognition

A Human Activity Recognition (HAR) project using Wi-Fi Channel State Information (CSI) collected through ESP32 devices.

This project investigates whether variations in Wi-Fi CSI amplitude can be used to recognize human activities without relying on wearable sensors or cameras. It includes signal preprocessing, feature extraction, machine learning and deep learning models, participant-wise evaluation, and cross-location generalization experiments.

**GitHub Repository:** [ESP32-WiFi-CSI-Human-Activity-Recognition](https://github.com/gladiator-13/ESP32-WiFi-CSI-Human-Activity-Recognition)

---

## 1. Project Objectives

The objectives of this project are to:

- Process raw ESP32 Wi-Fi CSI recordings.
- Extract meaningful amplitude-based CSI representations.
- Develop machine learning and deep learning models for human activity recognition.
- Evaluate recognition performance across different participants.
- Investigate generalization across different physical locations.
- Compare window-level and recording-level predictions.
- Study the challenges of environment-independent activity recognition using Wi-Fi sensing.

## 2. Dataset Description

The dataset contains recordings from **37 participants across four physical locations**.

| Location | Participants | Count |
|---|---|---:|
| Location 1 | P00–P09 | 10 |
| Location 2 | P10–P18 | 9 |
| Location 3 | P19–P27 | 9 |
| Location 4 | P28–P36 | 9 |
| **Total** | **P00–P36** | **37** |

### Activity classes

The dataset contains 20 human activity classes, labelled A01–A20, and baseline recordings labelled BASE.

The primary classification task considers the 20 activity classes. Baseline recordings are retained in the processed dataset but excluded from the primary 20-class classification experiments.

### Dataset statistics

| Description | Count |
|---|---:|
| Original CSV recordings | 862 |
| Successfully processed recordings | 860 |
| Activity recordings processed | 767 |
| Baseline recordings processed | 93 |
| Excluded recordings | 2 |
| Participants | 37 |
| Activity classes | 20 |
| Physical locations | 4 |

Every participant has recordings for all 20 activity classes.

Two recordings were excluded because their maximum packet gaps exceeded the preprocessing threshold.

## 3. CSI Data Representation

Each CSI packet contains 64 complex subcarrier measurements represented using 128 integers.

Each complex measurement consists of:
- Real component.
- Imaginary component.

The CSI amplitude is calculated as:

$$
A_k = \sqrt{\operatorname{Re}_k^2 + \operatorname{Im}_k^2}
$$

where \(k\) represents the subcarrier index.

The preprocessing pipeline removes the following subcarrier indices:

`[0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]`

This leaves **52 usable subcarriers** per CSI packet.

The initial implementation uses amplitude measurements from one antenna.

## 4. Signal Preprocessing Pipeline

The preprocessing pipeline consists of the following operations:

1. Read raw CSV recordings.
2. Parse CSI strings into numerical complex arrays.
3. Validate CSI array lengths.
4. Remove malformed packets.
5. Sort timestamps and handle duplicate timestamps.
6. Extract CSI amplitude.
7. Remove invalid subcarriers.
8. Apply Hampel filtering to reduce outliers.
9. Apply Savitzky–Golay smoothing.
10. Resample signals uniformly at 100 Hz.
11. Interpolate small packet gaps.
12. Trim recordings to a fixed duration of 20 seconds.
13. Save processed signals as NumPy arrays.

### Preprocessing configuration

| Parameter | Value |
|---|---|
| Original subcarriers | 64 |
| Removed subcarriers | 12 |
| Retained subcarriers | 52 |
| Sampling frequency | 100 Hz |
| Recording duration | 20 seconds |
| Samples per recording | 2,000 |
| Hampel window | 10 |
| Hampel threshold | 3 × MAD |
| Savitzky–Golay window | 11 |
| Savitzky–Golay polynomial order | 3 |
| Maximum packet gap | 200 ms |

### Processed signal representation

Each successfully processed recording is stored with shape:

`(2000, 52)`

Rows represent time samples, and columns represent CSI subcarriers.

### Preprocessing exclusions

Two recordings were excluded because their packet gaps exceeded the maximum allowed threshold:

- `P26_A18_R1_1794407419.csv` — approximately 376.8 ms.
- `P27_A19_R1_1794408319.csv` — approximately 297.3 ms.

These exclusions are documented in the preprocessing error report.

### Quality assurance

The processed dataset passed the following checks:

- All 37 participants are represented.
- All 20 activity classes are represented.
- Every participant has all 20 activity classes.
- Processed metadata entries have corresponding signal files.
- All processed signals have the expected shape of `(2000, 52)`.
- No processed arrays contain NaN or infinite values.
- The two excluded recordings are documented.

## 5. Dataset Splitting and Evaluation Protocols

To evaluate recognition performance while avoiding participant-level data leakage, the project uses participant-wise dataset splitting.

Two evaluation protocols are implemented.

### 5.1 Within-Location Evaluation

Participants from all four locations are divided into training, validation, and test groups.

Participants do not overlap between these groups.

The split is performed independently within each location:

| Location | Training Participants | Validation Participants | Test Participants |
|---|---:|---:|---:|
| Location 1 | 6 | 2 | 2 |
| Location 2 | 5 | 2 | 2 |
| Location 3 | 5 | 2 | 2 |
| Location 4 | 5 | 2 | 2 |
| **Total** | **21** | **8** | **8** |

### 5.2 Leave-One-Location-Out (LOLO) Evaluation

LOLO evaluation investigates whether models trained on three locations can recognize activities in an unseen fourth location.

Four folds are used:

| Fold | Training Locations | Held-Out Location |
|---|---|---|
| 1 | Locations 2, 3, 4 | Location 1 |
| 2 | Locations 1, 3, 4 | Location 2 |
| 3 | Locations 1, 2, 4 | Location 3 |
| 4 | Locations 1, 2, 3 | Location 4 |

For each fold, the held-out location is excluded from model training and validation.

This evaluation provides a more challenging test of cross-environment generalization.

## 6. Window Generation and Feature Extraction

Each 20-second processed recording is divided into overlapping windows.

| Parameter | Value |
|---|---|
| Sampling frequency | 100 Hz |
| Window duration | 2 seconds |
| Window length | 200 samples |
| Overlap | 50% |
| Stride | 100 samples |
| Subcarriers | 52 |
| Windows per recording | 19 |

Each generated CSI window has shape:

`(200, 52)`

Window generation is performed separately for the training, validation, and test splits.

### Handcrafted features

For the traditional machine learning models, statistical features are extracted from each window.

Nine statistical features are calculated for each of the 52 subcarriers, resulting in:

`9 × 52 = 468 features per window`

The resulting feature representation has 468 dimensions.

### Normalization

StandardScaler is fitted using training windows only.

The fitted transformation is then applied to validation and test data.

For LOLO evaluation, each fold uses its own independently fitted scaler to prevent information leakage from the held-out location.

## 7. Models Implemented

Four classification approaches are implemented and evaluated.

| Model | Input Representation | Approach |
|---|---|---|
| Random Forest | 468 handcrafted features | Ensemble learning |
| Support Vector Machine (SVM) | 468 handcrafted features | Kernel-based classification |
| 1D Convolutional Neural Network | `(200, 52)` | Temporal and subcarrier feature learning |
| CNN-BiLSTM with Attention | `(200, 52)` | Convolutional, recurrent, and attention-based learning |

Hyperparameter selection and model training are performed using the training and validation datasets.

The final models are evaluated on the corresponding held-out test datasets.

## 8. Evaluation Metrics

The project evaluates model performance using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix

Two levels of evaluation are considered:

**Window-level evaluation:** Each 2-second CSI window is classified independently.

**Recording-level evaluation:** Predictions from the 19 windows belonging to a recording are combined using majority voting. Ties are resolved using the smallest numeric activity label.

This allows comparison between individual window predictions and final recording-level activity recognition.

## 9. Experimental Results

### 9.1 Within-Location Window-Level Results

The following table summarizes the test-set macro F1-scores.

| Model | Macro F1-score |
|---|---:|
| Random Forest | 20.75% |
| 1D-CNN | 19.65% |
| SVM | 19.19% |
| CNN-BiLSTM-Attention | 18.13% |

### 9.2 Within-Location Recording-Level Results

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Random Forest | 27.38% | 26.09% |
| SVM | 26.19% | 24.03% |
| CNN-BiLSTM-Attention | 23.21% | 21.49% |
| 1D-CNN | 20.83% | 19.70% |

Recording-level majority voting improves the results for several models compared with independent window-level predictions.

### 9.3 LOLO Window-Level Results

Mean macro F1-score across the four held-out locations:

| Model | Mean Macro F1 | Standard Deviation |
|---|---:|---:|
| Random Forest | 6.78% | 3.56 percentage points |
| SVM | 6.41% | 2.64 percentage points |
| 1D-CNN | 6.04% | 2.53 percentage points |
| CNN-BiLSTM-Attention | 5.02% | 2.57 percentage points |

### 9.4 LOLO Recording-Level Results

Mean recording-level macro F1-score across the four held-out locations:

| Model | Mean Macro F1 |
|---|---:|
| Random Forest | Approximately 6.87% |
| SVM | Approximately 6.49% |
| 1D-CNN | Approximately 5.99% |
| CNN-BiLSTM-Attention | Approximately 4.61% |

### 9.5 Observations

- Random Forest achieves the strongest overall results in the reported within-location and LOLO comparisons.
- Recording-level majority voting improves recognition performance for several models.
- LOLO performance is substantially lower than within-location performance.
- Location 2 is particularly challenging across the evaluated models.
- The results highlight the difficulty of generalizing Wi-Fi CSI-based activity recognition across different physical environments.

These results are experimental findings for the current dataset and pipeline. They should not be interpreted as evidence of reliable real-time or environment-independent recognition.

## 10. Project Structure

The repository contains source code, notebooks, documentation, and experiment reports. Generated datasets and model binaries are excluded from Git tracking.

A simplified representation of the repository is:

```text
ESP32-WiFi-CSI-Human-Activity-Recognition/
│
├── archive/
│   └── three_location/
│
├── data/
│   └── processed/
│       ├── metadata.csv
│       ├── splits/
│       └── ...
│
├── docs/
│   ├── LOLO_ANALYSIS.md
│   └── RECORDING_LEVEL_ANALYSIS.md
│
├── notebooks/
│   └── 02_csi_preprocessing.ipynb
│
├── reports/
│   ├── random_forest/
│   ├── svm/
│   ├── cnn/
│   ├── cnn_bilstm_attention/
│   ├── model_comparison/
│   └── recording_level/
│
├── src/
│   ├── preprocessing.py
│   ├── batch_preprocessing.py
│   ├── model_layers.py
│   ├── train_cnn_bilstm_attention.py
│   ├── train_cnn_bilstm_attention_lolo.py
│   ├── evaluate_recording_level.py
│   ├── compare_lolo_models.py
│   └── compare_recording_level.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

This is a simplified overview; the repository contains additional scripts and experiment-specific reports.

## 11. Installation and Setup

### Clone the repository

```bash
git clone https://github.com/gladiator-13/ESP32-WiFi-CSI-Human-Activity-Recognition.git

cd ESP32-WiFi-CSI-Human-Activity-Recognition
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

The project uses scientific Python and machine learning libraries, including:

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Joblib

The exact dependency versions are maintained in `requirements.txt`.

### Dataset setup

Place the original dataset in the expected raw-data directory:

```text
data/data/
```

The directory should contain participant folders named `P00` through `P36`, with their original CSV recordings.

**The raw dataset is not included in the Git repository.** Access to the original recordings is required to reproduce the complete preprocessing and experiment pipeline.

## 12. Reproducibility and Data Integrity

The project follows these practices:

- Fixed random seeds are used where applicable.
- Participants are kept separate across training, validation, and test splits.
- LOLO folds hold out an entire physical location.
- Normalization is fitted on training data only.
- Each LOLO fold uses an independent normalization transformation.
- Window generation is performed after dataset splitting.
- Processed metadata and split assignments are preserved.
- Generated arrays and trained model binaries are excluded from Git tracking.
- Experiment reports and evaluation summaries are maintained for analysis.

## 13. Current Development Status

| Component | Status |
|---|---|
| Dataset inspection | Completed |
| CSI parsing and amplitude extraction | Completed |
| Subcarrier selection | Completed |
| Signal filtering and smoothing | Completed |
| Uniform resampling | Completed |
| Batch preprocessing | Completed |
| Preprocessing quality assurance | Completed |
| Four-location dataset preparation | Completed |
| Participant-wise splitting | Completed |
| LOLO evaluation splits | Completed |
| Window generation | Completed |
| Training-only normalization | Completed |
| Handcrafted feature extraction | Completed |
| Random Forest implementation and evaluation | Completed |
| SVM implementation and evaluation | Completed |
| 1D-CNN implementation and evaluation | Completed |
| CNN-BiLSTM-Attention implementation and evaluation | Completed |
| Window-level model comparison | Completed |
| Recording-level evaluation | Completed |
| LOLO model comparison | Completed |
| Recording-level comparison and reporting | Completed |
| Final research analysis and documentation | In progress |


## 14. Limitations and Future Work

The current experiments reveal several challenges:

- Recognition performance decreases substantially when testing on an unseen physical location.
- CSI amplitude patterns can vary with environmental conditions and recording locations.
- The current implementation focuses on offline processing and classification.
- Real-time inference on ESP32 hardware has not yet been implemented.

Potential future directions include:

- Investigating more robust cross-location representations.
- Exploring domain adaptation and domain generalization methods.
- Studying the effects of environmental and hardware variations.
- Evaluating additional signal representations and model architectures.
- Exploring real-time CSI acquisition and activity recognition.

## 15. License and Dataset Attribution

Dataset ownership, usage permissions, and redistribution conditions should follow the original dataset provider's terms.

---

**Project Status:** The four-location CSI preprocessing, dataset preparation, four-model evaluation, LOLO experiments, recording-level evaluation, and comparative reporting have been completed. Final research analysis and documentation are in progress.