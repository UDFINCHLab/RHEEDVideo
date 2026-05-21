"""
HDR camera capability diagnostic — check_hdr_camera_capability.py

Standalone test script to verify that the connected FLIR Blackfly S camera
supports all features required for the HDR RHEED pipeline before running
the full dashboard.

Checks performed:
    - Camera model and serial number
    - Auto-exposure and auto-gain can be disabled
    - Pixel format can be set to Mono8
    - FPS range and maximum sustainable frame rate
    - Exposure range supports HDR triplet (5000 / 15000 / 45000 µs)
    - Gain range supports 17.5 dB
    - 3-second sustained stream test at max FPS
    - Device link USB3 throughput

Usage: Run directly with the camera connected before launching
       hdr_rheed_base.py for the first time.
"""
import PySpin
import numpy as np
import time
import sys
import gc


def print_line(title):
    """Print a formatted section divider with a title for readable test output."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def stream_test(cam, seconds=3.0, gain_db=17.5, exposure_us=12000.0):
    """
    Run a timed live acquisition test to measure sustained FPS and mean intensity.

    Sets manual exposure and gain, acquires frames for the given duration,
    then stops acquisition and returns summary statistics.

    Args:
        cam:          Initialized PySpin camera object
        seconds:      Duration of the stream test in seconds
        gain_db:      Gain to apply during the test (dB)
        exposure_us:  Exposure time to apply during the test (microseconds)

    Returns: (frame_count, sustained_fps, avg_intensity)
    """
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.ExposureTime.SetValue(float(exposure_us))
    cam.Gain.SetValue(gain_db)

    cam.BeginAcquisition()

    start = time.time()
    frame_count = 0
    means = []

    while time.time() - start < seconds:
        image = cam.GetNextImage(1000)

        if image.IsIncomplete():
            image.Release()
            continue

        img = image.GetNDArray()
        means.append(np.mean(img))
        frame_count += 1
        image.Release()

    cam.EndAcquisition()

    avg_intensity = float(np.mean(means)) if means else 0.0
    sustained_fps = frame_count / seconds

    return frame_count, sustained_fps, avg_intensity


def main():
    # ── Initialize PySpin and get first camera ─────────────────────────
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("❌ No camera detected.")
        cam_list.Clear()
        system.ReleaseInstance()
        sys.exit(1)

    cam = cam_list.GetByIndex(0)
    cam.Init()

    try:
    # ── Run capability checks ──────────────────────────────────────────
    # Each section tests one aspect of the camera required for HDR mode.
    # A failure in any section means hdr_rheed_base.py may not work correctly.
        print_line("CAMERA BASIC INFO")
        print("Model:", cam.DeviceModelName.GetValue())
        print("Serial:", cam.DeviceSerialNumber.GetValue())

        print_line("AUTO CONTROL CHECK")
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        print("ExposureAuto: OFF ✅")
        print("GainAuto: OFF ✅")

        print_line("PIXEL FORMAT VALIDATION")
        cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
        pixel_format = cam.PixelFormat.GetCurrentEntry().GetSymbolic()
        print("PixelFormat:", pixel_format)

        print_line("FPS CAPABILITY")

        cam.AcquisitionFrameRateEnable.SetValue(True)

        min_fps = cam.AcquisitionFrameRate.GetMin()
        max_fps = cam.AcquisitionFrameRate.GetMax()

        print(f"FPS Range: {min_fps:.2f} – {max_fps:.2f}")
        print(f"MAX Supported FPS: {max_fps:.2f}")

        # -------- LOCK TO MAX FPS --------
        cam.AcquisitionFrameRate.SetValue(max_fps)
        resulting = cam.AcquisitionResultingFrameRate.GetValue()

        print(f"Locked to MAX FPS → Resulting: {resulting:.2f}")

        print_line("EXPOSURE RANGE")
        exp_min = cam.ExposureTime.GetMin()
        exp_max = cam.ExposureTime.GetMax()
        print(f"Exposure Range: {exp_min:.0f} – {exp_max:.0f} µs")
        print(f"HDR triplet 5000 | 15000 | 45000 µs → {'✅ supported' if exp_max >= 45000 else '⚠️ 45000 µs EXCEEDS max — reduce long exposure'}")

        print_line("GAIN RANGE")
        gain_min = cam.Gain.GetMin()
        gain_max = cam.Gain.GetMax()
        print(f"Gain Range: {gain_min:.2f} – {gain_max:.2f} dB")

        print_line("STREAM TEST @ MAX FPS")

        frames, sustained, mean_int = stream_test(cam, seconds=3.0, gain_db=17.5, exposure_us=12000.0)

        print(f"Frames captured in 3 sec: {frames}")
        print(f"Sustained FPS: {sustained:.2f}")
        print(f"Mean intensity @ gain=17.5dB, exp=12000µs: {mean_int:.2f}")

        print_line("DEVICE LINK THROUGHPUT")
        throughput = cam.DeviceLinkCurrentThroughput.GetValue()
        print("Current Throughput:", throughput)

        print("\n✅ HDR Capability Test Complete.")
        
        
    # ── Clean up all PySpin resources in correct order ─────────────────
    # Camera must be deinitialized before clearing the camera list,
    # and the list must be cleared before releasing the system instance.
    finally:
        try:
            cam.DeInit()
        except:
            pass

        try:
            del cam
        except:
            pass

        cam_list.Clear()
        del cam_list

        try:
            system.ReleaseInstance()
        except Exception as e:
            print("Cleanup warning:", e)

        del system
        gc.collect()


if __name__ == "__main__":
    main()