# ESP32 Wi-Fi CSI Human Activity Recognition
## Project Progress Report — Phase 1: Dataset Analysis and Preprocessing

**Project Repository:** https://github.com/gladiator-13/ESP32-WiFi-CSI-Human-Activity-Recognition

**Current Phase:** Dataset inspection, CSI preprocessing, and preprocessing validation.

---

## 1. Project Overview

The objective of this project is to develop a Human Activity Recognition (HAR) system using Channel State Information (CSI) collected from Wi-Fi signals using ESP32 devices.

Wi-Fi CSI captures how wireless signals change as they interact with people and objects in an environment. These signal variations can be used to identify human activities without requiring wearable sensors or cameras.

The project aims to investigate activity recognition using CSI amplitude measurements and deep learning models, with particular attention to generalization across different participants and physical locations.

## 2. Dataset Overview

The dataset contains recordings from 28 participants, distributed across three locations.

| Location | Participants | Participant IDs |
|---|---:|---|
| Location 1 | 10 | P00–P09 |
| Location 2 | 9 | P10–P18 |
| Location 3 | 9 | P19–P27 |
| Total | 28 | P00–P27 |

The dataset contains:
- 20 human activity classes, labelled 1–20.
- Baseline recordings, labelled 0.
- 652 original CSV recordings.
- 577 activity recordings.
- 75 baseline recordings.

Every participant has at least one original recording for each of the 20 activities.

## 3. Initial Dataset Inspection

We inspected the raw CSV files to understand their structure and CSI representation.

### CSV structure

The recordings contain:
- `timestamp`: packet timestamp.
- `csi_line`: CSI measurements represented as an array.

Each CSI packet contains 64 complex subcarrier values, represented by 128 integers.

The dataset uses one antenna (`ant=0`), so the initial implementation focuses on CSI amplitude rather than dual-antenna phase differences.

### Sample inspection

A representative recording, `P00_A01_R1_1790671306.csv`, contained:
- 2,228 packets.
- Approximately 22 seconds of recording.
- An average sampling rate close to 101 Hz.
- Irregular packet arrival intervals.
- One malformed CSI packet.

This inspection established that packet timestamps could not simply be assumed to have perfectly uniform intervals.

## 4. Preprocessing Pipeline Implemented

The preprocessing pipeline was implemented in Python using NumPy, Pandas, SciPy, and related scientific computing libraries.

### Step 1: CSI parsing

Each CSI string was parsed into 128 numerical values and reshaped into a 64 × 2 array.

The two columns represent the imaginary and real components of each complex CSI measurement.

### Step 2: CSI amplitude extraction

CSI amplitude was calculated using:

Amplitude = sqrt(Real² + Imaginary²)

This produces amplitude measurements for each subcarrier.

### Step 3: Subcarrier selection

The following subcarrier indices were removed:

`[0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]`

This removes 12 subcarriers, leaving 52 usable subcarriers.

### Step 4: Data cleaning

The pipeline:
- Removes malformed CSI packets.
- Removes invalid timestamps.
- Sorts packets by timestamp.
- Removes duplicate timestamps.
- Handles packet gaps through interpolation where appropriate.

### Step 5: Hampel filtering

A Hampel filter was applied to reduce outliers.

Parameters:
- Window size: 10.
- Threshold: 3 × MAD.
- MAD is scaled using the robust Gaussian consistency factor.

### Step 6: Savitzky–Golay smoothing

Savitzky–Golay filtering was applied to reduce high-frequency noise while retaining useful signal variations.

Parameters:
- Window length: 11.
- Polynomial order: 3.

### Step 7: Uniform resampling

The original packet timestamps were irregular.

Signals were interpolated onto a uniform 100 Hz time grid.

The target duration was fixed at 20 seconds.

### Step 8: Fixed-length output

Each processed recording was converted into a matrix with dimensions:

`(2000, 52)`

This represents:
- 2,000 time samples.
- 52 usable subcarriers.
- 20 seconds at 100 Hz.

No global normalization or PCA was fitted during this stage. These operations must be fitted using training data only after the dataset split.

## 5. Problems Encountered and Solutions

### Problem 1: CSI parsing failures

**Issue:** The initial parser using `np.fromstring()` generated warnings and failed to correctly parse some CSI strings because of inconsistent string formatting.

**Solution:** Replaced the parser with regular-expression-based extraction. The updated parser:
- Extracts numerical values from the CSI array.
- Validates that exactly 128 values are present.
- Converts them into a NumPy array.
- Reshapes the result into 64 × 2.

This substantially improved parsing reliability.

### Problem 2: Baseline filename format

**Issue:** Baseline recordings used filenames containing `BASE` rather than the activity naming convention `A00`.

**Solution:** Updated the metadata extraction function to recognize both formats:
- `P00_A01_R1_...csv`
- `P00_BASE_R1_...csv`

Baseline recordings are assigned activity label 0 and recording type `baseline`.

### Problem 3: Irregular packet gaps

**Issue:** Some recordings contained unusually large gaps between consecutive packet timestamps.

The initial maximum-gap threshold of 100 ms rejected 16 recordings.

**Solution:** Investigated the packet-gap distribution and provisionally increased the maximum allowed gap to 200 ms.

This recovered four additional recordings with gaps between approximately 157 and 180 ms.

Two recordings remained excluded because their maximum gaps were approximately:
- P26 Activity 18: 376.8 ms.
- P27 Activity 19: 297.3 ms.

These recordings were excluded rather than forcing interpolation across excessively large gaps.

### Problem 4: Missing participant/activity combinations in metadata

**Issue:** An initial coverage analysis reported 24 missing participant/activity combinations.

**Investigation:** We compared the original dataset against the processed metadata and discovered that the analysis was using an outdated in-memory `activity_metadata` variable.

**Solution:** Reloaded the metadata directly from disk and repeated the analysis using the current dataset.

The corrected analysis confirmed:
- 28 participants.
- 20 activity classes.
- Zero missing participant/activity combinations.

### Problem 5: Incorrect file paths during validation

**Issue:** An initial raw-file comparison reported zero original recordings and 650 missing signal files.

**Investigation:** The notebook was running from the `notebooks/` directory, while the validation script used paths relative to the project root.

**Solution:** Identified the project root automatically using the existence of the raw data directory and metadata file, then constructed absolute paths.

The corrected validation confirmed that the paths and processed files were correct.

## 6. Final Preprocessing Results

The batch preprocessing pipeline processed 650 of the 652 original recordings.

| Recording category | Original | Processed |
|---|---:|---:|
| Activity | 577 | 575 |
| Baseline | 75 | 75 |
| Total | 652 | 650 |

Two activity recordings were excluded due to excessive packet gaps.

### Quality assurance results

| Validation | Result |
|---|---|
| Raw recordings | 652 |
| Processed metadata entries | 650 |
| Processed NumPy arrays | 650 |
| Missing metadata entries | 2 |
| Unexpected metadata entries | 0 |
| Missing signal files | 0 |
| Participants | 28 |
| Activity classes | 20 |
| Missing participant/activity combinations | 0 |
| Invalid array shapes | 0 |
| Arrays containing NaN or infinity | 0 |

Every processed recording has dimensions `(2000, 52)`.

The original dataset was also independently checked to confirm that all 28 participants have recordings for all 20 activities.

## 7. Files and Code Implemented

The following project components have been implemented:

- `src/preprocessing.py`: Handles preprocessing of an individual CSI recording.
- `src/batch_preprocessing.py`: Applies preprocessing to all raw recordings and generates metadata.
- `notebooks/02_csi_preprocessing.ipynb`: Used for experimentation, inspection, and preprocessing validation.
- `data/processed/metadata.csv`: Contains recording-level metadata.
- `data/processed/preprocessing_errors.csv`: Records preprocessing failures.

Processed signal arrays are stored under:

`data/processed/signals/`

Raw recordings are stored under:

`data/data/`

Raw recordings and generated signal arrays are excluded from Git tracking to avoid unnecessarily large repository uploads.

## 8. Current Project Status

**Completed:**
- Dataset structure inspection.
- CSI packet parsing.
- CSI amplitude extraction.
- Subcarrier selection.
- Malformed packet handling.
- Hampel filtering.
- Savitzky–Golay smoothing.
- Uniform resampling to 100 Hz.
- Fixed-duration signal generation.
- Batch preprocessing.
- Metadata generation.
- Raw-to-processed reconciliation.
- Participant/activity coverage validation.
- Processed array shape and numerical validation.

**Not yet completed:**
- Participant-wise train/validation/test splitting.
- Leave-One-Location-Out evaluation splits.
- Window generation.
- Training-only normalization and PCA experiments.
- Feature extraction.
- Classical machine learning models.
- Deep learning models.
- Model evaluation and comparison.
- ESP32 deployment or real-time inference.

## 9. Planned Next Steps

1. Create participant-wise train, validation, and test assignments.
2. Validate class coverage in each split.
3. Create Leave-One-Location-Out evaluation folds.
4. Generate 2-second windows with 50% overlap.
5. Fit normalization and dimensionality reduction using training data only.
6. Establish classical machine learning baselines.
7. Implement and train deep learning models.
8. Evaluate performance using accuracy, macro precision, macro recall, macro F1-score, and confusion matrices.
9. Compare within-location and cross-location generalization.

## 10. Conclusion

The first phase of the project successfully established a reproducible CSI preprocessing pipeline.

A total of 650 recordings were successfully converted into fixed-size numerical representations, while two recordings with excessively large packet gaps were excluded.

All 28 participants and all 20 activity classes are represented in the processed dataset, with no missing participant/activity combinations. All processed arrays passed shape and numerical integrity checks.

The project is now ready to proceed to participant-wise dataset splitting and window generation.
