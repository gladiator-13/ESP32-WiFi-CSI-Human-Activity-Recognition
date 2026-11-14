# ESP32 Wi-Fi CSI Human Activity Recognition

A Human Activity Recognition (HAR) project using Wi-Fi Channel State Information (CSI) collected through ESP32 devices.

The project investigates how variations in Wi-Fi CSI amplitude can be used to recognize human activities without relying on wearable sensors or cameras.

**GitHub Repository:** [ESP32-WiFi-CSI-Human-Activity-Recognition](https://github.com/gladiator-13/ESP32-WiFi-CSI-Human-Activity-Recognition)

---

## 1. Project Objectives

The main objectives are:

- Process raw ESP32 Wi-Fi CSI recordings.
- Extract meaningful CSI amplitude features.
- Develop machine learning and deep learning models for human activity recognition.
- Evaluate recognition performance across different participants.
- Investigate generalization across different physical locations.
- Explore the potential for real-time activity recognition using ESP32 hardware.

## 2. Dataset Description

The dataset consists of CSI recordings collected from 28 participants across three locations.

| Location | Participants | Count |
|---|---|---:|
| Location 1 | P00–P09 | 10 |
| Location 2 | P10–P18 | 9 |
| Location 3 | P19–P27 | 9 |
| **Total** | **P00–P27** | **28** |

### Activity classes

The dataset contains 20 human activity classes, labelled 1–20, and baseline recordings labelled 0.

The primary HAR task uses 20 activity classes. Baseline recordings are retained for possible future experiments.

### Dataset statistics

| Description | Count |
|---|---:|
| Original recordings | 652 |
| Activity recordings | 577 |
| Baseline recordings | 75 |
| Successfully processed | 650 |
| Excluded recordings | 2 |

All participants have original recordings for all 20 activities.

## 3. CSI Data Representation

Each CSI packet contains 64 complex subcarrier measurements represented using 128 integers.

Each complex measurement contains:
- Real component.
- Imaginary component.

The CSI amplitude is calculated as:

\[
A_k = \sqrt{Re_k^2 + Im_k^2}
\]

where \(k\) represents the subcarrier index.

The initial implementation uses amplitude measurements from one antenna.

After subcarrier selection, each packet contains 52 usable amplitude measurements.

## 4. Preprocessing Pipeline

The preprocessing pipeline consists of the following operations:

1. Read raw CSV recordings.
2. Parse CSI strings into numerical arrays.
3. Validate CSI array length.
4. Remove malformed packets.
5. Sort timestamps and remove duplicates.
6. Extract CSI amplitude.
7. Remove invalid subcarriers.
8. Apply Hampel filtering.
9. Apply Savitzky–Golay smoothing.
10. Resample signals uniformly at 100 Hz.
11. Interpolate small packet gaps.
12. Trim signals to a fixed duration of 20 seconds.
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

### Processed signal shape

Each recording is stored as:

`(2000, 52)`

Rows represent time samples and columns represent CSI subcarriers.

## 5. Preprocessing Results

The batch preprocessing pipeline successfully processed 650 recordings.

| Category | Count |
|---|---:|
| Processed activity recordings | 575 |
| Processed baseline recordings | 75 |
| Total processed recordings | 650 |
| Excluded recordings | 2 |

### Excluded recordings

Two recordings were excluded because their packet gaps exceeded the maximum allowed threshold:

- `P26_A18_R1_1794407419.csv` — approximately 376.8 ms.
- `P27_A19_R1_1794408319.csv` — approximately 297.3 ms.

These exclusions are documented in the preprocessing error report.

### Quality assurance

The processed dataset passed the following checks:

- All 28 participants are represented.
- All 20 activity classes are represented.
- Every participant has all 20 activity classes.
- No processed metadata entries are missing corresponding signal files.
- Every processed signal has shape `(2000, 52)`.
- No processed arrays contain NaN or infinite values.

## 6. Project Structure

```text
ESP32-WiFi-CSI-Human-Activity-Recognition/
│
├── data/
│   ├── data/
│   │   ├── P00/
│   │   ├── P01/
│   │   ├── ...
│   │   └── P27/
│   │
│   └── processed/
│       ├── signals/
│       ├── metadata.csv
│       ├── preprocessing_errors.csv
│       └── splits/
│
├── notebooks/
│   └── 02_csi_preprocessing.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── batch_preprocessing.py
│   └── create_splits.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Some components, such as `create_splits.py`, split files, and `requirements.txt`, are planned or under development and may not yet exist in the repository.

## 7. Installation and Setup

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

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

The preprocessing implementation uses scientific Python libraries, including:

- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter

The exact project dependencies should be maintained in `requirements.txt`.

### Dataset setup

Place the original dataset in the following directory:

```text
data/data/
```

The dataset should contain participant folders named `P00` through `P27`.

Each participant folder should contain their original CSV recordings.

The raw dataset is not included in the Git repository.

## 8. Running Preprocessing

From the project root, execute:

```bash
python src/batch_preprocessing.py
```

The batch preprocessing script processes the raw CSV recordings and generates:

```text
data/processed/
├── signals/
├── metadata.csv
└── preprocessing_errors.csv
```

Each successfully processed recording is saved as a NumPy `.npy` file.

## 9. Planned Dataset Splitting Strategy

To prevent participant-specific information from leaking between training and evaluation, the project uses participant-wise splitting.

The initial benchmark will use train, validation, and test participants drawn from each location.

A separate Leave-One-Location-Out (LOLO) evaluation will investigate cross-location generalization.

The split assignments will be created before window generation.

## 10. Planned Window Generation

After participant-wise splitting, each 20-second recording will be divided into overlapping windows.

| Parameter | Value |
|---|---|
| Sampling frequency | 100 Hz |
| Window duration | 2 seconds |
| Window length | 200 samples |
| Overlap | 50% |
| Stride | 100 samples |
| Features | 52 subcarriers |

Each generated window will have shape:

`(200, 52)`

Window generation will be performed separately for the assigned dataset splits.

Normalization and dimensionality reduction will be fitted using training data only.

## 11. Planned Models

The following models are planned for investigation:

- Random Forest.
- Support Vector Machine.
- 1D Convolutional Neural Network.
- CNN-BiLSTM with Attention.

The deep learning models will learn temporal and spatial patterns from the CSI amplitude sequences.

## 12. Evaluation Metrics

Model performance will be evaluated using:

- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Confusion matrix.

Both within-location and cross-location evaluation will be considered.

## 13. Current Development Status

| Component | Status |
|---|---|
| Dataset inspection | Completed |
| CSI parsing | Completed |
| Amplitude extraction | Completed |
| Subcarrier selection | Completed |
| Signal filtering | Completed |
| Uniform resampling | Completed |
| Batch preprocessing | Completed |
| Metadata generation | Completed |
| Preprocessing validation | Completed |
| Participant-wise splitting | Next |
| LOLO evaluation splits | Planned |
| Window generation | Planned |
| Model development | Planned |
| Model evaluation | Planned |
| ESP32 deployment | Planned |

## 14. Reproducibility and Data Integrity

- Random seeds will be fixed for reproducible dataset splitting.
- Participants will not overlap across train, validation, and test sets.
- Preprocessing metadata and split assignments will be preserved.
- Training-only transformations will prevent information leakage.
- Raw data and generated NumPy arrays will remain outside Git tracking.

## 15. License and Dataset Attribution

Dataset ownership, usage permissions, and redistribution conditions should be documented according to the original dataset provider's terms.

---

**Project status:** CSI preprocessing and quality assurance completed. Participant-wise dataset splitting is the next development milestone.
