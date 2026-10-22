# -*- coding: utf-8 -*-
"""Train separate 5-layer CNN models for back-azimuth (BAZ) and incidence (INC).

This is the original iterative training workflow with expanded documentation.
Each iteration fits two independent models on three-component P-wave windows:
    BAZ model: outputs [sin(BAZ), cos(BAZ)] to avoid the 0/360 degree jump.
    INC model: outputs the incidence angle (degrees) by regression.
Repeated iterations use different stochastic initializations, creating an ensemble.

Input catalog: TRAIN_CSV (CSV containing at least):
    time          Event time, used to locate the event MiniSEED file (UTC).
    P_time        P-arrival time, used to extract the input waveform (UTC).
    azimuth_deg   Reference back-azimuth, in degrees.
    inc_angle_deg Reference incidence angle, in degrees.

Input waveform filename convention:
    <STATION>.YYYYMMDDTHHMMSS.mseed
For NN.WAK, this is e.g., NN.WAK.20210708T123456.mseed. Each file should
contain at least three consistently ordered components at 100 samples/s.
The first three traces are used as stored (they are NOT sorted by channel name).

Outputs, by default in the current working directory:
    model_baz_5layer_iter01.h5, model_inc_5layer_iter01.h5, etc.

Configure paths and N_ITER in the USER SETTINGS section before running.
This file does not construct the training/testing split; that is supplied by
TRAIN_CSV. The separate testing script evaluates held-out testing_events.csv.
"""

import pandas as pd
import numpy as np
import os
from obspy import read
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------
# USER SETTINGS: edit these for your own machine/dataset.
# Relative paths are resolved from the directory where you RUN Python.
# -----------------------
TRAIN_CSV = "training_events.csv"  # Catalog produced by an external train/test split.
MSEED_DIR = r"YOUR_EVENT_MSEED_PATH"  # Folder of per-event MiniSEED files.
STATION = "NN.WAK"  # Filename prefix: NN.WAK.YYYYMMDDTHHMMSS.mseed.

# Input is 1.0 s of P-wave motion, starting exactly at the P pick.
# The original waveform files should already be sampled at 100 Hz: this script
# does not resample, even though it reads each trace's actual sample rate.
SAMPLE_RATE = 100
BANDPASS = (10, 30)  # Hz; 4-corner, zero-phase Butterworth filter below.
PRE_P = 0.0         # Seconds before P pick.
POST_P = 1.0        # Seconds after P pick.
BATCH_SIZE = 16
EPOCHS = 30
BAZ_BIN_SIZE = 15    # Degrees per bin for balancing azimuthal coverage.

# Use 1 for a quick test; use 50 for the full 50-run ensemble used in the study.
N_ITER = 1

FIXED_LEN = int((PRE_P + POST_P) * SAMPLE_RATE)  # 100 samples at 100 Hz.

# -----------------------
# Load the PREDEFINED training set; do not include testing events here.
# -----------------------
df = pd.read_csv(TRAIN_CSV)

# -----------------------
# Resolve the event waveform. The event timestamp in the CSV is rounded down
# to a second; adjacent seconds are also checked to tolerate filename rounding.
# -----------------------
def get_mseed_file(row):
    """Return the event MiniSEED path, or None if none of three names exists."""
    dt = pd.to_datetime(row["time"]).floor("S")
    for offset in [0, -1, 1]:
        dt_try = dt + pd.Timedelta(seconds=offset)
        fname = f"{STATION}.{dt_try.strftime('%Y%m%dT%H%M%S')}.mseed"
        fpath = os.path.join(MSEED_DIR, fname)
        if os.path.exists(fpath):
            return fpath
    return None

# -----------------------
# Convert a 3-component event waveform to shape (100 samples, 3 components).
# Processing is applied to the waveform BEFORE window extraction.
# -----------------------
def load_fixed_p_wave(row):
    """Extract, zero-pad when needed, and normalize each P-wave component."""
    mseed = get_mseed_file(row)
    if mseed is None:
        return None

    st = read(mseed)
    if len(st) < 3:
        return None

    # IMPORTANT: trace order is preserved from MiniSEED. Testing/prediction must
    # use the same component order, or the trained model inputs will differ.
    st.detrend("demean")
    st.filter("bandpass",
              freqmin=BANDPASS[0],
              freqmax=BANDPASS[1],
              corners=4,
              zerophase=True)

    # UTC timestamps are assumed for event and phase picks.
    p_time = pd.to_datetime(row["P_time"]).timestamp()
    fs = st[0].stats.sampling_rate

    data = []
    for tr in st[:3]:
        t0 = tr.stats.starttime.timestamp
        i0 = int((p_time - PRE_P - t0) * fs)
        i1 = i0 + FIXED_LEN

        # Missing samples outside the trace bounds remain zero-padded.
        seg = np.zeros(FIXED_LEN, dtype=np.float32)

        valid_start = max(0, i0)
        valid_end = min(i1, len(tr.data))
        seg_start = valid_start - i0
        seg_end = seg_start + (valid_end - valid_start)

        seg[seg_start:seg_end] = tr.data[valid_start:valid_end]

        # Normalize EACH component by its own absolute maximum, not by a
        # common 3-component amplitude. Testing uses the same normalization.
        maxamp = np.max(np.abs(seg))
        if maxamp > 0:
            seg /= maxamp

        data.append(seg)

    if any(d.shape[0] != FIXED_LEN for d in data):
        return None

    return np.stack(data, axis=-1)

# -----------------------
# Balance azimuthal training coverage by repeating waveforms in sparsely
# populated 15-degree BAZ bins. These are exact repeats, not rotated signals
# or independently generated waveform augmentations.
# -----------------------
X_list = []
baz_sin_list = []
baz_cos_list = []
inc_list = []

# Bin counts are based on the input catalog BEFORE excluding missing waveforms.
df["baz_bin"] = (df["azimuth_deg"] // BAZ_BIN_SIZE).astype(int)
bin_counts = df["baz_bin"].value_counts()
max_count = bin_counts.max()

print("Preparing training data...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    seg = load_fixed_p_wave(row)
    if seg is None:
        continue

    # Events in a less populated bin receive a larger integer repeat factor.
    factor = int(np.ceil(max_count / bin_counts[row["baz_bin"]]))
    # Learn [sin, cos] instead of angle directly so 359 and 1 deg stay close.
    baz_rad = np.deg2rad(row["azimuth_deg"])
    baz_sin = np.sin(baz_rad)
    baz_cos = np.cos(baz_rad)

    for _ in range(factor):
        X_list.append(seg)
        baz_sin_list.append(baz_sin)
        baz_cos_list.append(baz_cos)
        inc_list.append(row["inc_angle_deg"])

# CNN input X: [N, 100, 3]; BAZ labels: [N, 2]; INC labels: [N].
X = np.array(X_list, dtype=np.float32)
y_baz = np.column_stack([baz_sin_list, baz_cos_list]).astype(np.float32)
y_inc = np.array(inc_list, dtype=np.float32)

print(f"Final X shape: {X.shape}")

# -----------------------
# Five 1D convolutional layers, with max pooling after the first two, followed
# by global average pooling, two dense hidden layers, and a linear output.
# output_dim=2 for BAZ [sin,cos], output_dim=1 for INC in degrees.
# Both tasks use Adam + mean squared error (MSE).
# -----------------------
def build_cnn(output_dim):
    """Build the fixed 5-layer 1D-CNN used for either BAZ or INC."""
    model = models.Sequential([
        layers.Input(shape=(FIXED_LEN, 3)),

        # Layer 1
        layers.Conv1D(32, 16, padding="same", activation="relu"),
        layers.MaxPooling1D(2),

        # Layer 2
        layers.Conv1D(64, 9, padding="same", activation="relu"),
        layers.MaxPooling1D(2),

        # Layer 3
        layers.Conv1D(128, 5, padding="same", activation="relu"),

        # Layer 4
        layers.Conv1D(128, 5, padding="same", activation="relu"),

        # Layer 5
        layers.Conv1D(128, 3, padding="same", activation="relu"),

        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(output_dim)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# -----------------------
# Train independent BAZ and INC networks for each ensemble iteration.
# Different stochastic initialization/optimization yields model variability;
# the seed below is randomized, so runs are NOT guaranteed bit-reproducible.
# NOTE: Keras validation_split is an internal training diagnostic. As training
# data have been oversampled by repeating waveforms, that validation portion
# should not be treated as an independent performance test. Use the separate
# testing_events.csv and v3_test_iteratively.py for held-out evaluation.
# -----------------------
for i in range(1, N_ITER + 1):

    print("\n====================================")
    print(f" Iteration {i}/{N_ITER}")
    print("====================================")

    # Optional: change seed for stochastic variation
    tf.keras.backend.clear_session()
    tf.random.set_seed(np.random.randint(0, 100000))

    # ---- BAZ: two continuous outputs representing sin and cos of the angle. ----
    baz_model = build_cnn(output_dim=2)
    baz_model.fit(
        X, y_baz,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    # Save to the CURRENT WORKING DIRECTORY, not MSEED_DIR.
    # For testing, point MODEL_DIR at the directory holding these *.h5 files.
    baz_model.save(f"model_baz_5layer_iter{i:02d}.h5")

    # ---- INC: one continuous output, an angle in degrees. ----
    inc_model = build_cnn(output_dim=1)
    inc_model.fit(
        X, y_inc,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    # The filename suffix must match the corresponding BAZ iteration number.
    inc_model.save(f"model_inc_5layer_iter{i:02d}.h5")

    print(f"Saved iteration {i}")
