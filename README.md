# 📡 RHEED-ML-Pipeline

### Real-Time RHEED Acquisition, ROI Analytics & Machine Learning Framework

---

## 🧭 Overview

**RHEED-ML-Pipeline** is a high-performance, multi-threaded system for:

* 🎥 Real-time RHEED acquisition (FLIR Blackfly)
* 🎯 Interactive ROI tracking (ellipse / rectangle)
* 📏 Line profile extraction
* 📊 Live intensity monitoring
* 🎬 Hardware-locked high-FPS recording
* 📝 Structured ROI CSV logging
* 🧠 PCA decomposition of RHEED videos
* 📈 K-Means clustering of temporal evolution
* 🖼 Publication-ready visualization outputs

The system supports:

* ✅ FLIR Blackfly cameras via **Spinnaker SDK + PySpin**
* ✅ Dummy camera simulation (for development without hardware)

---

# 🏗 Project Structure

```
RHEED-ML-PIPELINE/
│
├── main.py                  # Real-time RHEED Dashboard
├── camera.py                # Blackfly (PySpin) wrapper
├── dummy_camera.py          # Simulated RHEED feed
├── roi_manager.py           # ROI + Line Profile manager
├── config.py                # Output directory configuration
│
├── run_full.py              # Complete ML processing pipeline
├── pre_processing.py        # Video → H5 preprocessing
├── pca.py                   # PCA decomposition
├── k_means.py               # K-means clustering
├── plot_pca.py              # PCA visualization
├── plot_kmeans.py           # K-means visualization
│
├── check_pyspin.py          # Verifies Spinnaker installation
├── requirements.txt         # Python dependencies
├── install.md               # Detailed installation guide
├── rhedd_run.cmd.bat        # Windows launcher script
│
├── captures/                # Raw recorded videos
├── rheed_videos/            # Structured video recordings
├── roi_data/                # Structured ROI CSV logs
└── results/
    ├── pre_processing/
    ├── pca_plots/
    └── kmeans_plots/
```

---

# 🎥 Real-Time RHEED Dashboard (`main.py`)

A threaded acquisition and visualization system designed for stability and high frame rates.

### Architecture

| Thread            | Purpose                       |
| ----------------- | ----------------------------- |
| Capture Thread    | Frame acquisition + recording |
| Processing Thread | ROI updates + logging         |
| Main Thread       | Rendering + UI interaction    |

Frame queue buffer: **1800 frames**

Recording occurs immediately at acquisition (no processing delay).

---

# 🖱 ROI System

Supports up to **6 ROIs**.

Each ROI stores:

* UUID (unique identifier)
* Shape (ellipse or rectangle)
* Center (cx, cy)
* Radii (rx, ry)
* Mean intensity
* Sum intensity
* Area
* Timestamp history
* Moving average smoothing

### ROI Data Logged To CSV

```
frame_idx
timestamp_s
roi_uuid
roi_display_id
mean_intensity
sum_intensity
area
cx
cy
rx
ry
shape
```

Saved automatically in:

```
roi_data/YYYY/Month/MMDDYY/
```

Auto-flush interval: **1 second**

---

# 📏 Line Profile System

Draw a line across diffraction features to extract intensity profile.

Features:

* Dynamic extraction
* Separate pop-out window
* Grid, axis labels
* Min intensity display
* Elapsed session time

---

# 🎮 Keyboard Controls

## General

| Key | Action                   |
| --- | ------------------------ |
| Q   | Quit                     |
| C   | Toggle gradient colormap |

---

## ROI Controls

| Action                              | Key / Mouse                          |
| ------------------------------------ | ------------------------------------- |
| Draw ROI                            | Left click (drag)                    |
| Select ROI                          | Left click inside ROI                |
| Move ROI                            | Drag inside selected ROI             |
| Resize ROI (mouse)                  | Shift + drag inside ROI              |
| Resize ROI larger                   | Click inside ROI → press `.`         |
| Resize ROI smaller                  | Click inside ROI → press `,`         |
| Deselect ROI                        | Click inside ROI again               |
| Delete nearest ROI                  | Right click                          |
| Switch to ellipse                   | `E`                                  |
| Switch to rectangle                 | `T`                                  |
| Reset all ROIs                      | `R`                                  |

---

## Line Profile

| Action           | Key |
| ---------------- | --- |
| Toggle line mode | `L` |
| Clear line       | `X` |

---

## Recording

| Key | Action               |
| --- | -------------------- |
| M   | Start/Stop recording |

Recording includes:

* Hardware FPS detection
* Recording duration
* Frame count
* Effective FPS calculation
* Encoded video duration

Saved to:

```
rheed_videos/YYYY/Month/MMDDYY/
```

---

## FPS Lock (Hardware)

| Key | FPS    |
| --- | ------ |
| 1   | 10 FPS |
| 2   | 20 FPS |
| 3   | 30 FPS |

Console prints:

```
🔁 FPS changed → Locked: XX | Resulting: XX
```

---

## Camera Controls

| Key       | Function            |
| --------- | ------------------- |
| `[` / `]` | Exposure down / up  |
| `-` / `=` | Gain down / up      |
| `J` / `K` | Gamma down / up     |
| `G`       | Toggle gamma enable |

---

# 📡 Blackfly Camera Support (`camera.py`)

Implements:

* ExposureAuto OFF
* GainAuto OFF
* Gamma enable control
* Stream buffer handling
* Manual buffer count (100)
* FPS locking
* Safe cleanup

If no camera detected:

```
⚠️ No camera found, using dummy feed.
```

---

# 🧪 Dummy Camera (`dummy_camera.py`)

Simulates:

* Moving diffraction spots
* Adjustable exposure
* Adjustable gain (dB scaling)
* Gamma transformation
* Noise + Gaussian blur

Allows full UI testing without hardware.

---

# 🎬 Recording System

Recording characteristics:

* Uses hardware resulting FPS
* Gradient colored frames
* XVID codec
* AVI container
* Recording begins immediately after acquisition

Displays during recording:

* `[REC]`
* Timer
* Frame count

---

# 🧠 Offline ML Pipeline

Run:

```
python run_full.py
```

Pipeline steps:

### 1️⃣ Preprocessing

* Frame sampling
* Blank filtering
* Scar removal
* Background subtraction
* Drift correction
* RSS alignment
* Translation
* Cropping
* HDR simulation

Output:

```
results/pre_processing/video_name.h5
```

---

### 2️⃣ PCA

* Standard scaling
* Eigenvector decomposition
* Explained variance storage

---

### 3️⃣ K-Means

* Clustering of PCA features
* Configurable cluster count
* Multiple runs

---

### 4️⃣ Plot Generation

Saved to:

```
results/pca_plots/
results/kmeans_plots/
```

---

Here is a **clean, minimal installation summary** suitable for your README:

---

# ⚙ Installation

⚠ **Full detailed installation instructions are available in `INSTALL.md`.**
Please follow that guide for complete setup steps.

---

## Quick Start (Summary)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/UDFINCHLab/RHEEDVideo.git
cd RHEEDVideo
```

---

### 2️⃣ Create Virtual Environment (Python 3.10 Required)

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
pip install third_party\spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl
```

---

### 4️⃣ Verify PySpin

```bash
python check_pyspin.py
```

---

### ▶ Run Dashboard

```bash
python main.py
```

---

📌 Requirements:

* Windows 64-bit
* Python 3.10
* Spinnaker SDK 4.2.x (installed before PySpin)

For complete instructions (Spinnaker setup, PATH fixes, troubleshooting, etc.), see **`INSTALL.md`**.


# 🔒 Performance Design

* Multi-threaded architecture
* Frame drop tolerant (processing only)
* Recording never waits on processing
* Hardware FPS lock
* Large queue buffer (1800 frames)

---

# 🎯 Intended Use

Designed for:

* Thin film growth monitoring
* Real-time diffraction analysis
* ML-driven phase detection
* Materials science research
* Lab automation workflows

---

# 📌 Important Notes

* Python 3.10 required
* SpinView must be closed before running
* Spinnaker SDK must be installed before PySpin wheel
* DummyCamera auto-activates if hardware not detected

---

# ✔ System Capabilities

✅ Real-time acquisition
✅ ROI analytics
✅ Line profile extraction
✅ Hardware FPS locking
✅ Structured logging
✅ ML preprocessing
✅ PCA
✅ K-Means clustering
✅ Publication-ready plotting
✅ Dummy simulation mode

---


