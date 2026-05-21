"""
PySpin installation check — check_pyspin.py

Quick diagnostic script to verify that the PySpin SDK (FLIR Spinnaker)
is correctly installed and can detect connected cameras.

Prints the installed Spinnaker library version and the number of
cameras currently detected on the system, then cleanly releases
all resources.

Usage: Run directly before launching the dashboard to confirm
       the camera SDK is working.
"""
import PySpin
# ── Initialize PySpin system singleton ────────────────────────────────
# GetInstance() returns the global Spinnaker system object.
# Must be released with ReleaseInstance() before the script exits
# to avoid resource leaks.
system = PySpin.System.GetInstance()

version = system.GetLibraryVersion()
print(
    f"Spinnaker version: "
    f"{version.major}.{version.minor}.{version.type}.{version.build}"
)

cams = system.GetCameras()
print("Number of cameras detected:", cams.GetSize())
# ── Release all resources ─────────────────────────────────────────────
# Cameras must be cleared before releasing the system instance.
# Skipping this step causes a PySpin resource leak warning.
cams.Clear()
system.ReleaseInstance()
