import PySpin
import numpy as np
import time
import sys


def print_line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_stream(cam, duration_sec=3, gain_value=17.5):
    """Run streaming test for given duration and report sustained FPS."""
    cam.BeginAcquisition()

    cam.Gain.SetValue(gain_value)

    start = time.time()
    frame_count = 0
    means = []

    while time.time() - start < duration_sec:
        image = cam.GetNextImage(1000)

        if not image.IsIncomplete():
            img = image.GetNDArray()
            means.append(np.mean(img))
            frame_count += 1

        image.Release()

    cam.EndAcquisition()

    avg_intensity = np.mean(means) if means else 0
    sustained_fps = frame_count / duration_sec

    print(f"Frames captured in {duration_sec} sec: {frame_count}")
    print(f"Sustained FPS: {sustained_fps:.2f}")
    print(f"Mean intensity @ {gain_value} dB: {avg_intensity:.2f}")


def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("❌ No camera detected.")
        cam_list.Clear()
        system.ReleaseInstance()
        sys.exit(1)

    cam = cam_list.GetByIndex(0)
    cam.Init()

    print_line("CAMERA BASIC INFO")
    print("Model:", cam.DeviceModelName.GetValue())
    print("Serial:", cam.DeviceSerialNumber.GetValue())

    # --------------------------------------------------
    # AUTO CONTROL DISABLE
    # --------------------------------------------------
    print_line("AUTO CONTROL CHECK")

    try:
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        print("ExposureAuto: OFF ✅")
    except Exception as e:
        print("ExposureAuto control failed:", e)

    try:
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        print("GainAuto: OFF ✅")
    except Exception as e:
        print("GainAuto control failed:", e)

    # --------------------------------------------------
    # PIXEL FORMAT
    # --------------------------------------------------
    print_line("PIXEL FORMAT VALIDATION")

    try:
        cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
        pixel_format = cam.PixelFormat.GetCurrentEntry().GetSymbolic()
        print("PixelFormat:", pixel_format)

        if pixel_format == "Mono8":
            print("Mono8 OK ✅")
        else:
            print("⚠ Recommended format is Mono8 for HDR performance.")

    except Exception as e:
        print("PixelFormat setup failed:", e)

    # --------------------------------------------------
    # FPS CAPABILITY
    # --------------------------------------------------
    print_line("FPS CAPABILITY")

    try:
        cam.AcquisitionFrameRateEnable.SetValue(True)

        min_fps = cam.AcquisitionFrameRate.GetMin()
        max_fps = cam.AcquisitionFrameRate.GetMax()

        print(f"FPS Range: {min_fps:.2f} – {max_fps:.2f}")

        # ---- Test 60 FPS ----
        print("\nTesting 60 FPS Lock")
        cam.AcquisitionFrameRate.SetValue(60.0)
        resulting_60 = cam.AcquisitionResultingFrameRate.GetValue()
        print(f"Attempted 60 → Resulting: {resulting_60:.2f}")

        # ---- Test 90 FPS ----
        print("\nTesting 90 FPS Lock")
        target_90 = min(90.0, max_fps)
        cam.AcquisitionFrameRate.SetValue(target_90)
        resulting_90 = cam.AcquisitionResultingFrameRate.GetValue()
        print(f"Attempted {target_90:.2f} → Resulting: {resulting_90:.2f}")

    except Exception as e:
        print("FPS control not supported:", e)

    # --------------------------------------------------
    # GAIN RANGE
    # --------------------------------------------------
    print_line("GAIN RANGE")

    try:
        gain_min = cam.Gain.GetMin()
        gain_max = cam.Gain.GetMax()
        print(f"Gain Range: {gain_min:.2f} – {gain_max:.2f} dB")

        if gain_max < 24:
            print("⚠ Gain range may limit HDR dynamic range.")

    except Exception as e:
        print("Gain range read failed:", e)

    # --------------------------------------------------
    # STREAM TEST @ 60 FPS
    # --------------------------------------------------
    print_line("STREAM TEST @ 60 FPS")

    try:
        cam.AcquisitionFrameRate.SetValue(60.0)
        test_stream(cam, duration_sec=3, gain_value=17.5)
    except Exception as e:
        print("60 FPS stream test failed:", e)

    # --------------------------------------------------
    # STREAM TEST @ 90 FPS
    # --------------------------------------------------
    print_line("STREAM TEST @ 90 FPS")

    try:
        target_90 = min(90.0, cam.AcquisitionFrameRate.GetMax())
        cam.AcquisitionFrameRate.SetValue(target_90)
        test_stream(cam, duration_sec=3, gain_value=17.5)
    except Exception as e:
        print("90 FPS stream test failed:", e)

    # --------------------------------------------------
    # DEVICE LINK THROUGHPUT
    # --------------------------------------------------
    print_line("DEVICE LINK THROUGHPUT")

    try:
        throughput = cam.DeviceLinkCurrentThroughput.GetValue()
        print("Current Throughput:", throughput)

        if throughput < 120000000:
            print("⚠ Throughput may limit 60+ FPS HDR.")

    except Exception as e:
        print("Throughput read failed:", e)

    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()

    print("\n✅ HDR Compatibility test complete.")


if __name__ == "__main__":
    main()