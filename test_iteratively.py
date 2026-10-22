# -*- coding: utf-8 -*-
"""Evaluate independently trained 5-layer BAZ and INC CNNs on held-out events.

This is the original testing/evaluation workflow with expanded documentation.
It loads each available BAZ/INC model pair, predicts reference back-azimuth
(BAZ) and incidence (INC), summarizes absolute errors for each model, then
computes a prediction averaged across the available ensemble members.

Inputs:
    CSV_TEST: testing catalog with columns
        time, P_time, azimuth_deg, inc_angle_deg, S_minus_P
    MSEED_DIR: per-event, three-component MiniSEED waveforms named
        <STATION>.YYYYMMDDTHHMMSS.mseed
    MODEL_DIR: matching model_baz_5layer_iterNN.h5 and
        model_inc_5layer_iterNN.h5 files produced by the training script.

Outputs under OUTPUT_DIR:
    ensemble_statistics_per_model.csv
    ensemble_statistics_mean.csv

Waveform preprocessing must match training: first three consistently ordered
traces, 10-30 Hz bandpass, 1 s P window at 100 Hz, independently normalized
three-component traces. This script measures prediction errors; it does not
perform earthquake detection, phase association or hypocenter inversion.
"""

import pandas as pd
import numpy as np
from obspy import read
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# -----------------------
# USER SETTINGS: point to YOUR catalog, per-event waveforms, and saved models.
# Relative paths are resolved from the directory where you RUN Python.
# For the Antelope Valley NN.WAK release, use NN.WAK waveforms/model weights.
# -----------------------
CSV_TEST = "testing_events.csv"          # Held-out events only; not training CSV.

MODEL_DIR = r"YOUR_TRAINED_MODELS_PATH"   # Folder containing both BAZ/INC .h5 models.
N_ENSEMBLE = 50                          # Maximum index to search (01...50).

MSEED_DIR = r"YOUR_WAK_EVENT_MSEED_PATH"  # Folder of NN.WAK per-event MiniSEED files.
STATION = "NN.WAK"                       # Must match waveform filename prefix.

# Keep the parameters IDENTICAL to those used to train these CNN models.
# Input MiniSEED files should already be sampled at 100 Hz (not resampled here).
FS = 100
FREQMIN = 10
FREQMAX = 30
PRE_P = 0.0        # P-window begins at the P pick.
POST_P = 1.0       # P-window extends 1.0 s after pick.
FIXED_LEN = int((PRE_P + POST_P) * FS)  # 100 samples.

OUTPUT_DIR = "ensemble_test_results_revised"  # Written under working directory.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Plotting-font settings retained from the original script. Note that this
# particular version writes CSV metrics but does not generate any plots.
# -----------------------
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20
TICK_FONTSIZE = 16
CBAR_LABEL_FONTSIZE = 16
CBAR_TICK_FONTSIZE = 16

# -----------------------
# Load events set aside during train/test splitting. The CSV labels provide
# ground truth for the error metrics, not inputs to either CNN.
# -----------------------
test_df = pd.read_csv(CSV_TEST)
print(f"Number of test events: {len(test_df)}")

# -----------------------
# Make one [100 samples, 3 components] test example from a P pick.
# The waveform processing matches training, with ONE filename difference:
# this tester looks only for the exact floored-second filename, whereas the
# trainer also searches neighboring seconds (-1, +1).
# -----------------------
def load_waveform(row):
    """Return a fixed-size normalized P-wave segment or None if unavailable."""

    dt = pd.to_datetime(row["time"]).floor("s")
    t_str = dt.strftime("%Y%m%dT%H%M%S")
    mseed = os.path.join(MSEED_DIR, f"{STATION}.{t_str}.mseed")

    if not os.path.exists(mseed):
        return None

    st = read(mseed)
    if len(st) < 3:
        return None

    # Preserve the trace order in the MiniSEED file; it MUST match training.
    # Filter all three component traces before extracting the P-wave window.
    st.detrend("demean")
    st.filter("bandpass",
              freqmin=FREQMIN,
              freqmax=FREQMAX,
              corners=4,
              zerophase=True)

    # Input timestamps are assumed to describe UTC arrivals, as in training.
    p_time = pd.to_datetime(row["P_time"]).timestamp()
    fs = st[0].stats.sampling_rate

    data = []

    for tr in st[:3]:

        t0 = tr.stats.starttime.timestamp
        i0 = int((p_time - PRE_P - t0) * fs)
        i1 = i0 + FIXED_LEN

        # Copy only available trace samples; missing edges remain zero-padded.
        seg = np.zeros(FIXED_LEN, dtype=np.float32)

        valid_start = max(0, i0)
        valid_end = min(i1, len(tr.data))

        if valid_end > valid_start:
            seg_start = valid_start - i0
            seg_end = seg_start + (valid_end - valid_start)
            seg[seg_start:seg_end] = tr.data[valid_start:valid_end]

        # Normalize each channel independently by its maximum absolute value.
        maxamp = np.max(np.abs(seg))
        if maxamp > 0:
            seg /= maxamp

        data.append(seg)

    if any(d.shape[0] != FIXED_LEN for d in data):
        return None

    return np.stack(data, axis=-1)

# -----------------------
# Assemble X_test: [N_events, 100_samples, 3_components].
# Events with missing/incomplete MiniSEED data are skipped, with the reference
# BAZ/INC labels kept in the SAME order as the usable waveform rows.
# -----------------------
X_test = []
baz_true = []
inc_true = []
sp_values = []

print("Loading test waveforms...")

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    seg = load_waveform(row)
    if seg is None:
        continue

    X_test.append(seg)
    baz_true.append(row["azimuth_deg"])
    inc_true.append(row["inc_angle_deg"])
    # S-minus-P is retained here for compatibility with other analyses; it is
    # not used by the BAZ/INC CNNs or the error statistics in this script.
    sp_values.append(row["S_minus_P"])

if len(X_test) == 0:
    raise RuntimeError("No valid test samples loaded.")

X_test = np.array(X_test, dtype=np.float32)
baz_true = np.array(baz_true, dtype=np.float32)
inc_true = np.array(inc_true, dtype=np.float32)
sp_values = np.array(sp_values, dtype=np.float32)

print(f"Total usable test samples: {len(X_test)}")

# -----------------------
# All error arrays passed here are ABSOLUTE errors (degrees):
# BAZ uses shortest circular separation, INC uses ordinary absolute difference.
# The "2sigma" entry is 2 x sample standard deviation, not a confidence interval.
# -----------------------
def compute_stats(err):
    """Return summary statistics of an array of absolute angular errors."""
    return {
        "mean": np.mean(err),
        "median": np.median(err),
        "std": np.std(err),
        "2sigma": 2 * np.std(err),
        "p68": np.percentile(err, 68),
        "p84": np.percentile(err, 84),
        "p95": np.percentile(err, 95)
    }

# -----------------------
# Collect predictions from each available model pair. Model files are named
# consistently with the training script's iteration number (01, 02, ...).
# -----------------------
all_sin = []
all_cos = []
all_inc = []
stats_rows = []

# =========================
# Loop through requested ensemble indices. Missing BAZ/INC pairs are skipped;
# this run requires at least one complete pair to compute ensemble statistics.
# =========================
for n in range(1, N_ENSEMBLE + 1):

    print(f"\n=== Testing model {n}/{N_ENSEMBLE} ===")

    # --- FIXED filename matching training ---
    model_baz_path = os.path.join(MODEL_DIR, f"model_baz_5layer_iter{n:02d}.h5")
    model_inc_path = os.path.join(MODEL_DIR, f"model_inc_5layer_iter{n:02d}.h5")

    if not os.path.exists(model_baz_path) or not os.path.exists(model_inc_path):
        print(f"Model {n} missing, skipping.")
        continue

    model_baz = load_model(model_baz_path, compile=False)
    model_inc = load_model(model_inc_path, compile=False)

    # Network outputs: BAZ is [sin, cos]; INC is a scalar angle in degrees.
    baz_vec = model_baz.predict(X_test, batch_size=32, verbose=0)
    inc_pred = model_inc.predict(X_test, batch_size=32, verbose=0).squeeze()

    sin_pred = baz_vec[:, 0]
    cos_pred = baz_vec[:, 1]

    # atan2 reconstructs the circular BAZ; wrap negative values to [0, 360).
    baz_pred = np.degrees(np.arctan2(sin_pred, cos_pred))
    baz_pred = (baz_pred + 360) % 360

    # Absolute circular error is the smaller of delta and 360 - delta.
    baz_err = np.abs(baz_pred - baz_true)
    baz_err = np.minimum(baz_err, 360 - baz_err)
    inc_err = np.abs(inc_pred - inc_true)

    all_sin.append(sin_pred)
    all_cos.append(cos_pred)
    all_inc.append(inc_pred)

    baz_stats = compute_stats(baz_err)
    inc_stats = compute_stats(inc_err)

    stats_rows.append({
        "model": n,
        **{f"baz_{k}": v for k, v in baz_stats.items()},
        **{f"inc_{k}": v for k, v in inc_stats.items()}
    })

# =========================
# Ensemble result across successfully loaded models:
#   - BAZ: average predicted sine and cosine, normalize the resulting vector,
#          then convert the averaged direction back to degrees.
#   - INC: arithmetic mean of predicted angle values.
# There is no per-event outlier trimming here; this is a simple ensemble mean.
# =========================
print("\n=== Computing ensemble mean ===")

all_sin = np.array(all_sin)
all_cos = np.array(all_cos)
all_inc = np.array(all_inc)

mean_sin = np.mean(all_sin, axis=0)
mean_cos = np.mean(all_cos, axis=0)

# Normalize the MEAN 2-D direction vector before recovering BAZ. A zero-length
# vector is protected from division by zero. (Magnitude itself is not used.)
norm = np.sqrt(mean_sin**2 + mean_cos**2)
norm[norm == 0] = 1
mean_sin /= norm
mean_cos /= norm

baz_ens = np.degrees(np.arctan2(mean_sin, mean_cos))
baz_ens = (baz_ens + 360) % 360
inc_ens = np.mean(all_inc, axis=0)

# ---- Compute errors against the held-out reference labels, in degrees. ----
baz_err_ens = np.abs(baz_ens - baz_true)
baz_err_ens = np.minimum(baz_err_ens, 360 - baz_err_ens)
inc_err_ens = np.abs(inc_ens - inc_true)

print("Ensemble BAZ median:", np.median(baz_err_ens))
print("Ensemble INC median:", np.median(inc_err_ens))

# -----------------------
# Save only summary statistics (not individual-event predictions) as CSV files.
# To reproduce catalog predictions, adapt this workflow to write event-level
# values or use the separate prediction/location pipeline.
# -----------------------
stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(OUTPUT_DIR, "ensemble_statistics_per_model.csv"), index=False)

ensemble_stats = pd.DataFrame({
    "baz_mean": [np.mean(baz_err_ens)],
    "baz_median": [np.median(baz_err_ens)],
    "baz_std": [np.std(baz_err_ens)],
    "inc_mean": [np.mean(inc_err_ens)],
    "inc_median": [np.median(inc_err_ens)],
    "inc_std": [np.std(inc_err_ens)]
})

ensemble_stats.to_csv(os.path.join(OUTPUT_DIR, "ensemble_statistics_mean.csv"), index=False)

print("\nAll results saved to:", OUTPUT_DIR)
