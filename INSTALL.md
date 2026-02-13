# RHEED-ML-PIPELINE – Installation Guide

This project uses:
- Python 3.10 (required)
- FLIR Spinnaker SDK 4.2.x
- PySpin / spinnaker-python (installed from local wheel)
- PyTorch, OpenCV, NumPy, Matplotlib

Follow the steps below to run this project on a new machine.

---
## FOR INSTALLATION VIA THE COMMAND PROMPT TERMINAL:

### 1. Install FLIR Spinnaker SDK (required; must be installed before installing PySpin) 

Download and install **Spinnaker SDK 4.2.x** for Windows 64-bit.

During installation, select:
- Spinnaker Runtime  
- GenICam Runtime  
- Camera Drivers (USB / PCIe)

Restart the computer if recommended.

---

### 2. From the Command Prompt Terminal:


#### a. Install Python 3.10 (required) built-in Python Launcher

---
py install 3.10


#### b. Verify installation
---
py -3.10 --version

#### c. Create your venv using Python 3.10 in the "rheed-ml-pipeline-main" directory
---
py -3.10 -m venv .venv

#### d. Activate the new virtual environment (after activation, your path should be preceded by "(.venv)" )
---
.venv\Scripts\activate

#### e. Install dependencies
---
pip install -r requirements.txt

#### f. Install the spinnaker_python wheel file (this requires the wheel file to be in the "third_party" subdirectory)
---
pip install third_party\spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl

#### g. Verify PySpin installation 
---
py -c "import PySpin; print('PySpin OK')"


---

### 3. Prepare the Project Folder

Place the project folder anywhere on your system.

Create the following structure:

```

RHEED-ML-PIPELINE/
│
├── camera.py
├── main.py
├── ML_model.py
├── roi_manager.py
├── test_camera.py
├── rheed_gradient_lut.npy
├── results.csv
├── requirements.txt
├── INSTALL.md
│
└── third_party/
└── spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl

````

**Important Notes:**
- The Spinnaker wheel file *must* be inside the `third_party/` directory  
- The wheel must match your Python version: **Python 3.10 (cp310)**  

---

## 4. Create a Virtual Environment

Open PowerShell inside the project folder:

```powershell
python -m venv .venv
````

Activate it:

```powershell
.\.venv\Scripts\activate
```

---

## 5. Install Python Requirements

```powershell
pip install -r requirements.txt
```

---

## 6. Install PySpin / Spinnaker Python Wheel

```powershell
pip install .\third_party\spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl
```

---

## 7. Verify PySpin Installation

Run:

```powershell
python -c "import PySpin; print('PySpin OK')"
```

Expected output:

```
PySpin OK
```

If you see DLL-related errors, add these paths to your Windows PATH environment variable:

```
C:\Program Files\FLIR Systems\Spinnaker\bin64
C:\Program Files\FLIR Systems\Spinnaker\bin64\vs2015
```

Then restart your terminal.

---

## 8. Run the Project

Make sure the `.venv` is activated, then run:

```powershell
python main.py
```

The full RHEED ML pipeline should now run correctly.

---

## ✔ Installation Complete

If you followed all steps, the project will work on any Windows machine with a supported FLIR camera.

```

---

```
