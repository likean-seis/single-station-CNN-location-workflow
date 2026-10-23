# CNN training and testing — v1 (BAZ and INC)

This folder contains the five-layer, one-dimensional convolutional neural network (1D-CNN) training and testing workflow used for the **near-source NN.WAK station** in the 2021 *Mw* 6.0 Antelope Valley earthquake-sequence study. Two separate CNNs estimate earthquake **back-azimuth (BAZ)** and **incidence angle (INC)** from the first second of three-component P-wave recordings.

The folder includes the predefined training/testing catalogs, **one** saved pair of BAZ and INC models, and a compressed archive of per-event NN.WAK MiniSEED waveforms (`mseed.zip`). It does **not** include the complete 50-run ensemble used in the study.

## Contents

| File | Description |
| --- | --- |
| `train_iteratively.py` | Trains independent BAZ and INC CNNs, with optional repeated random initializations. |
| `test_iteratively.py` | Evaluates one or more saved model pairs against held-out events; saves error summaries. |
| `training_events.csv` | **4,385** reference events designated for training. |
| `testing_events.csv` | **610** held-out reference events for testing. |
| `model_baz_5layer_iter01.h5` | Trained BAZ network, iteration 01. |
| `model_inc_5layer_iter01.h5` | Trained INC network, iteration 01. |
| `README.md` | This guide. |

### Waveform Data
**Waveform Data Attribution**

The MiniSEED waveforms distributed in mseed.zip are event-based waveform segments from station NN.WAK of the Nevada Seismic Network, operated by the University of Nevada, Reno (UNR). The original seismic data were accessed through the EarthScope Consortium (formerly IRIS) data services.

**Original data source:**

Nevada Seismic Network (NN): https://doi.org/10.7914/SN/NN

EarthScope Consortium: https://www.earthscope.org/

The waveform files were extracted and organized into event-based segments for this research. EarthScope's default data license is Creative Commons Attribution 4.0 (CC BY 4.0), unless more restrictive terms have been specified by the original data provider.

Original data ownership and attribution remain with the contributing seismic network.

The waveform archive (`mseed.zip`) is available from the
[GitHub Releases page](https://github.com/likean-seis/single-station-CNN-location-workflow/releases).

The archive contains:
- 7,179 per-event NN.WAK MiniSEED waveform files.
- A waveform download log (`mseed/download_log.txt`).

**To use the waveform data:**
1. Open the Releases page linked above.
2. Download `mseed.zip` from the release assets.
3. Extract the archive into this folder.
4. The waveforms should then be located in `mseed/WAK/`.


The two CSV files contain **4,995 distinct catalog entries** in total. Relevant columns for these scripts are:

| Column | Usage |
| --- | --- |
| `time` | Catalog event origin time (UTC); used to locate the event MiniSEED filename. |
| `P_time` | P-phase arrival time (UTC); start of the CNN input window. |
| `azimuth_deg` | Reference BAZ angle in degrees. |
| `inc_angle_deg` | Reference INC angle in degrees. |
| `S_minus_P` | S–P arrival-time difference in seconds; loaded by the test script but **not** used in the CNN input or its reported metrics. |

Other catalog columns include event coordinates, depth, magnitude, P/S picking statistics, source-to-station distances, and grid identifiers. The scripts take their angular training/testing labels from the supplied catalogs; they do not recompute these labels from earthquake locations.

## Bundled waveform data: extract before running

The waveform input is supplied as **`mseed.zip`**, a ZIP archive inside this folder. It contains **7,179** event MiniSEED files under `mseed/WAK/`, plus `mseed/download_log.txt`. The archive has more waveforms than the 4,995 entries in the supplied training and testing catalogs; the scripts select events **only from their respective CSV files**.

I verified the filename coverage of the bundled archive against both supplied catalogs: **4,385/4,385 training events** and **610/610 testing events** have an exact matching NN.WAK MiniSEED filename. This confirms file presence, not that every waveform will necessarily pass all processing checks on every system.

**Extract `mseed.zip` into this directory before running the scripts.** From a terminal whose working directory is this folder:

```bash
python -m zipfile -e mseed.zip .
```

Or extract the archive here using your operating system's ZIP utility. After extraction, the layout should be:

```text
cnn_training_v1_baz_inc/
├── train_iteratively.py
├── test_iteratively.py
├── training_events.csv
├── testing_events.csv
├── model_baz_5layer_iter01.h5
├── model_inc_5layer_iter01.h5
├── mseed.zip
└── mseed/
    ├── download_log.txt
    └── WAK/
        ├── NN.WAK.YYYYMMDDTHHMMSS.mseed
        └── ... (7,179 event files in total)
```

Both scripts should then use this setting, **assuming they are run from this folder**:

```python
MSEED_DIR = "mseed/WAK"
```

The generic `YOUR_..._PATH` strings in the Python scripts are **placeholders**, not actual directories. Replace them with the path above (or an absolute path to your extracted `mseed/WAK` directory).

### Waveform naming and processing assumptions

The waveform filenames follow:

```text
NN.WAK.YYYYMMDDTHHMMSS.mseed
```

The timestamp is derived from the catalog event `time` rounded **down** to the nearest whole second, not from `P_time`. The training script looks for that filename and also checks offsets of −1 and +1 seconds. The testing script checks **only** the exact rounded-second filename. All listed training and testing events have that exact filename in the bundled ZIP, so the difference in file lookup behavior should not affect this supplied dataset.

Each MiniSEED input should contain at least three traces. Both scripts use the **first three traces in file order** rather than sorting channels; their component ordering must therefore remain consistent across training and testing. The code expects **100 Hz** data and does not resample. Events with missing/unusable inputs may be skipped.

## Environment

Required Python packages:

```text
numpy
pandas
obspy
tensorflow
tqdm
matplotlib
```

TensorFlow provides Keras for building, saving, and loading the `.h5` networks. `matplotlib` is imported in the testing script, but that script currently exports only CSV results (no figures). Exact dependency versions are not pinned here; saved HDF5/Keras model compatibility may depend on your TensorFlow/Keras version.

## CNN input and preprocessing

Each input example is a `100 × 3` array (**100 samples × 3 components**) beginning at the P pick:

1. Read the three-component event MiniSEED waveform.
2. Remove the mean from each trace.
3. Apply a **10–30 Hz**, four-corner, zero-phase Butterworth bandpass filter.
4. Extract the **1.0-s** P-wave window (`PRE_P = 0.0`, `POST_P = 1.0`). Samples beyond the available trace are zero-padded.
5. Normalize **each component separately** by its maximum absolute amplitude within the window.

The two CNNs share the same feature-extraction architecture: five `Conv1D` layers (32, 64, 128, 128, 128 filters; kernel sizes 16, 9, 5, 5, 3), max-pooling after the first two convolutional layers, global-average pooling, and two dense layers (128 and 64 units).

- **BAZ model:** two output values, `[sin(BAZ), cos(BAZ)]`, to avoid an artificial discontinuity at 0°/360°.
- **INC model:** one regression output, the angle in degrees.

The models are fitted using Adam optimization and mean-squared-error loss.

## Step 1 — Train models (optional)

Open `train_iteratively.py` and adjust the **USER SETTINGS** section:

```python
TRAIN_CSV = "training_events.csv"
MSEED_DIR = "mseed/WAK"    # After extracting mseed.zip into this folder
STATION = "NN.WAK"
N_ITER = 1
```

Use `N_ITER = 1` to train one pair of models, or `N_ITER = 50` to train 50 separate BAZ/INC pairs. Other default parameters are:

```text
Sampling frequency: 100 Hz
Window:             0.0–1.0 s relative to the P arrival
Frequency band:     10–30 Hz
BAZ oversampling:   15-degree azimuth bins
Epochs:             30
Batch size:         16
Validation split:   10%
```

Training examples from less-populated BAZ bins are **repeated** to balance their representation; the script does not synthetically rotate waveforms. The `validation_split` is an internal training diagnostic applied after this repetition and should not be treated as independent, held-out validation. Use `testing_events.csv` for independent model evaluation.

From within this folder, run:

```bash
python train_iteratively.py
```

Each iteration saves the following files **in the current working directory**:

```text
model_baz_5layer_iter01.h5
model_inc_5layer_iter01.h5
model_baz_5layer_iter02.h5
model_inc_5layer_iter02.h5
...
```

Repeated runs use stochastic initializations/optimization; the script does not guarantee that retrained models will be numerically identical to the supplied models. **Warning:** running training from this folder with `N_ITER = 1` writes the same filenames as the distributed pretrained models and can overwrite them. Back up the included `.h5` files first or train in a separate working directory.

## Step 2 — Evaluate saved models

To evaluate the **single included model pair**, open `test_iteratively.py` and set:

```python
CSV_TEST = "testing_events.csv"
MODEL_DIR = "."                          # Included .h5 files, if run from this folder
N_ENSEMBLE = 1                           # Only iteration 01 is distributed here
MSEED_DIR = "mseed/WAK"                 # After extracting mseed.zip
STATION = "NN.WAK"
```

The distributed testing script defaults to a generic `MODEL_DIR` placeholder and `N_ENSEMBLE = 50`. You **must** configure `MODEL_DIR` before running it. To evaluate more models, place the corresponding numbered BAZ/INC `.h5` pairs in `MODEL_DIR` and increase `N_ENSEMBLE` to the desired highest index. Missing model pairs are skipped, but at least one complete pair is needed.

Run from within this folder:

```bash
python test_iteratively.py
```

The test script reports:

- **BAZ error:** smallest absolute circular angular difference between the predicted and reference directions, in degrees.
- **INC error:** absolute difference between predicted and reference incidence angles, in degrees.
- **Per-model statistics:** mean, median, standard deviation, twice the standard deviation, and 68th, 84th, and 95th percentiles.
- **Ensemble prediction:** mean sine and cosine predictions (renormalized before conversion to a BAZ angle) and arithmetic mean of INC predictions. This is an **untrimmed ensemble mean**, not the event-specific trimmed-ensemble uncertainty calculation discussed elsewhere in the study.

The script creates:

```text
ensemble_test_results_revised/
├── ensemble_statistics_per_model.csv
└── ensemble_statistics_mean.csv
```

With `N_ENSEMBLE = 1`, the ensemble summary describes the single supplied model pair. The script does not export per-event predictions or hypocentral locations.

## Scope and reproducibility notes

This directory covers **CNN BAZ/INC training and held-out testing only**. It does not perform PhaseNet detection, P/S phase association, conversion of BAZ/INC/S–P time to hypocentral coordinates, or the statistical analyses of the earthquake sequence. Those are separate workflow stages.

The event MiniSEED files **are bundled** in `mseed.zip`. After extracting the archive and setting the paths, this folder provides the waveform inputs, training/testing labels, and an example pretrained BAZ/INC model pair needed to run the scripts. Successful execution still depends on a compatible Python/TensorFlow/ObsPy environment. The distributed model pair is **one example iteration**; it is not the complete 50-model ensemble, and this testing script does not reproduce the paper’s event-specific trimmed-ensemble uncertainty calculations.

When adapting the workflow to another station or study region, verify the station-specific waveform naming, channel order, sampling rate, filtering, labels, and model compatibility rather than simply changing the directory path.
