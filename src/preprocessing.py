
from pathlib import Path
import re

import numpy as np
import pandas as pd

from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


# ==========================================
# CONFIGURATION
# ==========================================

RAW_FS = 100
RECORDING_DURATION = 20

N_SUBCARRIERS = 64

#Guard and DC bands are removed, leaving 52 subcarriers
REMOVE_INDICES = [
    0, 1, 2, 3, 4, 5,
    32, #DC
    59, 60, 61, 62, 63
]

KEEP_INDICES = [
    i for i in range(N_SUBCARRIERS)
    if i not in REMOVE_INDICES
]

HAMPel_WINDOW = 10
HAMPel_THRESHOLD = 3.0

SG_WINDOW = 11
SG_POLYORDER = 3

MAX_GAP_SECONDS = 0.1


# ==========================================
# 1. PARSE CSI PACKETS
# ==========================================

def parse_csi_line(csi_line):

    match = re.search(r"\[(.*?)\]", str(csi_line))

    if match is None:
        return None

    values = np.fromstring(match.group(1), sep=" ")

    if len(values) != 128:
        return None

    return values.reshape(64, 2)


# ==========================================
# 2. EXTRACT AMPLITUDE
# ==========================================

def extract_amplitude(csi_array):

    imaginary = csi_array[:, 0]
    real = csi_array[:, 1]

    amplitude = np.sqrt(
        real ** 2 + imaginary ** 2
    )

    return amplitude[KEEP_INDICES]


# ==========================================
# 3. PARSE COMPLETE RECORDING
# ==========================================

def load_raw_recording(csv_path):

    df = pd.read_csv(csv_path)

    if not {"timestamp", "csi_line"}.issubset(df.columns):
        raise ValueError("Required CSV columns are missing.")

    timestamps = []
    amplitudes = []

    for _, row in df.iterrows():

        timestamp = row["timestamp"]

        if pd.isna(timestamp):
            continue

        csi = parse_csi_line(row["csi_line"])

        if csi is None:
            continue

        amplitude = extract_amplitude(csi)

        if not np.all(np.isfinite(amplitude)):
            continue

        timestamps.append(float(timestamp))
        amplitudes.append(amplitude)

    if len(timestamps) < 2:
        raise ValueError("Insufficient valid CSI packets.")

    timestamps = np.asarray(timestamps, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)

    # Sort timestamps
    order = np.argsort(timestamps)

    timestamps = timestamps[order]
    amplitudes = amplitudes[order]

    # Remove duplicate timestamps
    timestamps, unique_indices = np.unique(
        timestamps,
        return_index=True
    )

    amplitudes = amplitudes[unique_indices]

    return timestamps, amplitudes


# ==========================================
# 4. RESAMPLE TO UNIFORM 100 HZ
# ==========================================

def resample_csi(timestamps, amplitudes):

    relative_time = timestamps - timestamps[0]

    # Reject recordings with large packet gaps
    if np.max(np.diff(relative_time)) > MAX_GAP_SECONDS:
        raise ValueError("Large packet gap detected.")

    if relative_time[-1] < RECORDING_DURATION:
        raise ValueError("Recording is shorter than 20 seconds.")

    # 20 seconds at 100 Hz = 2000 samples
    target_time = np.arange(
        0,
        RECORDING_DURATION,
        1 / RAW_FS
    )

    resampled = np.zeros(
        (len(target_time), len(KEEP_INDICES)),
        dtype=np.float32
    )

    for channel in range(len(KEEP_INDICES)):

        resampled[:, channel] = np.interp(
            target_time,
            relative_time,
            amplitudes[:, channel]
        )

    return resampled


# ==========================================
# 5. HAMPEL FILTER
# ==========================================

def hampel_filter(signal):

    signal = signal.copy()

    # Apply median filtering along the time axis
    local_median = median_filter(
        signal,
        size=(HAMPel_WINDOW, 1),
        mode="nearest"
    )

    deviations = np.abs(signal - local_median)

    local_mad = median_filter(
        deviations,
        size=(HAMPel_WINDOW, 1),
        mode="nearest"
    )

    # Robust estimate of standard deviation
    local_sigma = 1.4826 * local_mad

    threshold = HAMPel_THRESHOLD * local_sigma

    outliers = deviations > np.maximum(
        threshold,
        1e-8
    )

    signal[outliers] = local_median[outliers]

    return signal


# ==========================================
# 6. SAVITZKY-GOLAY SMOOTHING
# ==========================================

def smooth_csi(signal):

    return savgol_filter(
        signal,
        window_length=SG_WINDOW,
        polyorder=SG_POLYORDER,
        axis=0
    ).astype(np.float32)


# ==========================================
# 7. COMPLETE PREPROCESSING PIPELINE
# ==========================================

def preprocess_recording(csv_path):

    timestamps, amplitudes = load_raw_recording(
        csv_path
    )

    resampled = resample_csi(
        timestamps,
        amplitudes
    )

    filtered = hampel_filter(resampled)

    smoothed = smooth_csi(filtered)

    if smoothed.shape != (2000, 52):
        raise ValueError(
            f"Unexpected output shape: {smoothed.shape}"
        )

    return smoothed


# ==========================================
# 8. TEST ONE RECORDING
# ==========================================

if __name__ == "__main__":

    csv_path = Path(
        "data/data/P00/P00_A01_R1_1790671306.csv"
    )

    processed = preprocess_recording(csv_path)

    output_dir = Path("data/processed/signals")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        csv_path.stem + ".npy"
    )

    np.save(output_path, processed)

    print("Preprocessing successful!")
    print("Output shape:", processed.shape)
    print("Saved:", output_path)
