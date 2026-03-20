````markdown
# RHEED-ML-PIPELINE – Installation Guide

Repository:  
[https://github.com/UDFINCHLab/RHEEDVideo.git](https://github.com/UDFINCHLab/RHEEDVideo.git)

This guide explains how to clone, set up, verify, and run the **RHEED real-time dashboard**, **HDR dashboard**, and **offline ML pipeline** on a new Windows machine.

---

# System Requirements

This project requires:

- Windows 64-bit
- Python 3.10
- FLIR Spinnaker SDK 4.2.x
- Blackfly camera for real hardware acquisition  
  or
- dummy video mode for development without hardware

---

# 1️⃣ Install FLIR Spinnaker SDK (Required for Real Camera Use)

Install **Spinnaker SDK 4.2.x** for Windows 64-bit.

During installation, ensure the following are included:

- Spinnaker Runtime
- GenICam Runtime
- Camera Drivers (USB / PCIe, depending on your hardware)

After installation:

- restart your computer if prompted

## Important
**SpinView must be closed before running the Python dashboards.**

If SpinView is open, PySpin camera access may fail.

---

# 2️⃣ Clone the GitHub Repository

Open Command Prompt or PowerShell and run:

```bash
git clone https://github.com/UDFINCHLab/RHEEDVideo.git
cd RHEEDVideo
````

You should now see files such as:

* `main.py`
* `hdr_rheed_base.py`
* `camera.py`
* `dummy_camera.py`
* `roi_manager.py`
* `requirements.txt`
* `INSTALL.md`
* `README.md`

You should also see folders such as:

* `ML/`
* `captures/`
* `roi_data/`
* `rheed_videos/`
* `results/`
* `third_party/`

---

# 3️⃣ Install Python 3.10

Install **Python 3.10.x** on Windows.

Then verify:

```bash
py -3.10 --version
```

It must report **Python 3.10.x**.

## Important

Do **not** use Python 3.11, 3.12, or 3.13 for this environment.
The PySpin wheel used by this project must match Python 3.10.

---

# 4️⃣ Create a Virtual Environment

Inside the cloned repository folder:

```bash
py -3.10 -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

After activation, your prompt should begin with something like:

```text
(venv)
```

---

# 5️⃣ Install Python Dependencies

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the main Python dependencies used across the dashboard and ML pipeline, including:

* numpy
* opencv-python
* matplotlib
* pillow
* scikit-learn
* h5py
* scipy
* torch
* tqdm
* pandas

---

# 6️⃣ Install PySpin (Spinnaker Python Wheel)

This project requires a local Spinnaker Python wheel matching:

* Python 3.10
* Windows 64-bit
* Spinnaker 4.2.x

If the repository includes:

```text
third_party\spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl
```

install it using:

```bash
pip install third_party\spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl
```

If the `third_party` folder is missing or the wheel is not present, obtain the correct wheel from your lab distribution or Spinnaker package source.

---

# 7️⃣ Verify PySpin Installation

Run:

```bash
python check_pyspin.py
```

You can also test the import directly:

```bash
python -c "import PySpin; print('PySpin OK')"
```

If you get DLL-related errors, add the following directories to your Windows PATH:

```text
C:\Program Files\FLIR Systems\Spinnaker\bin64
C:\Program Files\FLIR Systems\Spinnaker\bin64\vs2015
```

Then restart your terminal and try again.

---

# 8️⃣ Verify Camera / HDR Capability (Recommended)

Before using HDR mode on a real camera, run:

```bash
python check_hdr_camera_capability.py
```

This checks:

* camera detection
* pixel format
* FPS capability
* exposure range
* gain range
* stream test performance
* device link throughput

This is especially useful before running `hdr_rheed_base.py`.

---

# 9️⃣ Run the Standard Live Dashboard

With `venv` activated:

```bash
python main.py
```

You should see:

```text
🚀 RHEED Dashboard (Enhanced Build)
```

If a real camera is connected and available, the script will initialize the Blackfly camera.

If no real camera is detected, it will automatically fall back to dummy mode.

Typical fallback message:

```text
⚠️ No camera found, using dummy feed.
```

---

# 🔟 Run the HDR Dashboard

To run the HDR live pipeline:

```bash
python hdr_rheed_base.py
```

This starts the HDR dashboard, which performs multi-exposure capture and real-time exposure fusion.

Use this mode when you want HDR live visualization instead of the standard single-exposure feed.

---

# 1️⃣1️⃣ Output Folder Structure

## Video recordings

Recorded videos are automatically saved to:

```text
rheed_videos/YYYY/Month/MMDDYY/
```

Each recording session creates separate folders such as:

```text
raw/
color/
```

## ROI CSV logs

ROI CSV files are saved to:

```text
roi_data/YYYY/Month/MMDDYY/
```

## HDR exposure triplets

In HDR mode, a representative triplet may be saved to:

```text
hdr_exposure_triplet/
hdr_exposure_triplet/raw_sensor/
```

---

# 1️⃣2️⃣ Run the Offline ML Pipeline

After recording a video, place the target video inside:

```text
captures/
```

Then run:

```bash
python ML\run_full.py
```

Outputs are written under:

```text
results/
```

Depending on the pipeline, this includes folders such as:

```text
results/pre_processing/
results/pca_plots/
results/kmeans_plots/
```

---

# 1️⃣3️⃣ Inspect a Video and Suggest a Crop Region

A helper utility is included for quick video inspection:

```bash
python ML\inspect_rheed_video.py
```

This script:

* finds the latest video inside `captures/`
* samples frames
* detects bright regions
* saves debug frames
* suggests a crop region for later processing

Debug output is saved in:

```text
ML/inspect_debug/
```

---

# Optional: Run Using Batch Launcher

If using Windows, you may also launch the standard dashboard with a batch file such as:

```text
RHEED_RUN.cmd.bat
```

This is useful for quickly opening the environment and starting the dashboard.

If you maintain a separate HDR batch launcher, it should point to:

```text
python hdr_rheed_base.py
```

---

# Troubleshooting

## Problem: Camera not detected

* ensure SpinView is closed
* ensure Spinnaker SDK is installed
* check USB / PCIe / cable connection
* confirm the camera is visible in SpinView before closing it

## Problem: PySpin import error

* confirm Python 3.10 is being used
* confirm the wheel matches `cp310`
* confirm Spinnaker SDK is installed before the wheel
* confirm PATH variables include Spinnaker DLL folders if needed

## Problem: DLL load failed

* add Spinnaker `bin64` folders to Windows PATH
* restart terminal after updating PATH

## Problem: Dashboard opens in dummy mode unexpectedly

* verify the camera is connected
* verify no other application is using the camera
* verify SpinView is closed
* run `python check_pyspin.py`

## Problem: HDR mode performance is slow

* reduce display load from other applications
* verify camera throughput settings
* confirm the system is not overloaded
* use the capability test script to inspect supported FPS and exposure limits

---

# ✔ Installation Complete

If all steps are completed correctly, you should be able to:

* run the standard real-time RHEED dashboard
* run the HDR live dashboard
* use ROI drawing and line profile tools
* record raw and color videos
* generate ROI CSV logs
* inspect videos for crop suggestions
* run the offline ML pipeline

---

# Important Notes

* Python 3.10 is required
* Spinnaker SDK must be installed before PySpin
* always activate `venv` before running the project
* SpinView must be closed before launching the dashboards
* `main.py` runs the standard live feed
* `hdr_rheed_base.py` runs the HDR live feed
* dummy mode activates automatically when no real camera is available

```

