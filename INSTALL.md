# RHEED-ML-PIPELINE – Installation Guide

Repository:
[https://github.com/UDFINCHLab/RHEEDVideo.git](https://github.com/UDFINCHLab/RHEEDVideo.git)

This guide explains how to properly clone, set up, and run the RHEED real-time dashboard and ML pipeline on a new Windows machine.

---

# System Requirements

This project requires:

* Windows 64-bit
* Python 3.10 (required)
* FLIR Spinnaker SDK 4.2.x
* Blackfly camera (optional — DummyCamera runs without hardware)

---

# 1️⃣ Install FLIR Spinnaker SDK (REQUIRED for real camera use)

Download and install:

Spinnaker SDK 4.2.x (Windows 64-bit)

During installation, ensure the following are selected:

* Spinnaker Runtime
* GenICam Runtime
* Camera Drivers (USB / PCIe)

After installation:

Restart your computer if prompted.

⚠ Important:
SpinView must be closed before running the Python dashboard.

---

# 2️⃣ Clone the GitHub Repository

Open Command Prompt or PowerShell and run:

```bash
git clone https://github.com/UDFINCHLab/RHEEDVideo.git
cd RHEEDVideo
```

You should now see files like:

* main.py
* camera.py
* dummy_camera.py
* roi_manager.py
* requirements.txt
* INSTALL.md
* RHEED_RUN.cmd.bat
* etc.

---

# 3️⃣ Install Python 3.10

Install Python 3.10 using the Python launcher:

```bash
py install 3.10
```

Verify installation:

```bash
py -3.10 --version
```

It must show Python 3.10.x

⚠ Do NOT use Python 3.11 or 3.12 — PySpin wheel requires 3.10.

---

# 4️⃣ Create Virtual Environment (REQUIRED)

Inside the cloned repository folder:

```bash
py -3.10 -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

After activation, your prompt should begin with:

```
(.venv)
```

---

# 5️⃣ Install Python Dependencies

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

* numpy
* opencv-python
* matplotlib
* scikit-learn
* h5py
* scipy
* torch (optional ML)
* pandas
* tqdm

---

# 6️⃣ Install PySpin (Spinnaker Python Wheel)

⚠ This project requires a local Spinnaker Python wheel.

The wheel must match:

* Python 3.10
* Windows 64-bit
* Spinnaker 4.2.x

If the repository includes:

```
third_party/spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl
```

Install it using:

```bash
pip install third_party\spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl
```

If you do not see a `third_party` folder:
You must download the correct wheel from your lab distribution or FLIR installation package.

---

# 7️⃣ Verify PySpin Installation

Run:

```bash
python check_pyspin.py
```

Expected output:

```
Spinnaker version: X.X.X.X
Number of cameras detected: X
```

OR verify import directly:

```bash
python -c "import PySpin; print('PySpin OK')"
```

If you see DLL errors, add the following to your Windows PATH:

```
C:\Program Files\FLIR Systems\Spinnaker\bin64
C:\Program Files\FLIR Systems\Spinnaker\bin64\vs2015
```

Then restart terminal.

---

# 8️⃣ Running the Dashboard

With `.venv` activated:

```bash
python main.py
```

You should see:

```
🚀 RHEED Dashboard (Enhanced Build)
```

If camera is connected:

```
📸 Camera initialized
🎥 Acquisition started
```

If no camera detected:

```
⚠ No camera found, using dummy feed.
```

The dashboard will still run using DummyCamera.

---

# 9️⃣ Recording Output Structure

Videos are automatically saved to:

```
rheed_videos/YYYY/Month/MMDDYY/
```

ROI CSV logs are saved to:

```
roi_data/YYYY/Month/MMDDYY/
```

---

# 🔟 Running the Offline ML Pipeline

After recording a video:

Place it inside:

```
captures/
```

Then run:

```bash
python run_full.py
```

Outputs will be saved in:

```
results/pre_processing/
results/pca_plots/
results/kmeans_plots/
```

---

# Optional: Run Using Batch Launcher

If using Windows:

Double-click:

```
RHEED_RUN.cmd.bat
```

This automatically:

* Activates the virtual environment
* Runs main.py

---

# Troubleshooting

Problem: Camera not detected
→ Ensure SpinView is closed
→ Ensure Spinnaker SDK installed
→ Check USB connection

Problem: PySpin import error
→ Confirm Python 3.10
→ Confirm wheel matches cp310
→ Confirm PATH variables

Problem: DLL load failed
→ Add Spinnaker bin64 paths to Windows PATH

---

# ✔ Installation Complete

If all steps are followed correctly:

* Real-time RHEED dashboard will run
* Hardware FPS lock will work
* ROI logging will work
* Recording will work
* ML pipeline will execute successfully

---

# Important Notes

* Python 3.10 is mandatory
* Spinnaker SDK must be installed before installing PySpin
* Always activate `.venv` before running code
* SpinView must be closed before running main.py

---
