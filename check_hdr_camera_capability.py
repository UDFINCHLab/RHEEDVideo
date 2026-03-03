import PySpin
import numpy as np
import time
import sys
import gc


def print_line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def stream_test(cam, seconds=3.0, gain_db=17.5):
    cam.BeginAcquisition()

    cam.Gain.SetValue(gain_db)

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

        print_line("GAIN RANGE")
        gain_min = cam.Gain.GetMin()
        gain_max = cam.Gain.GetMax()
        print(f"Gain Range: {gain_min:.2f} – {gain_max:.2f} dB")

        print_line("STREAM TEST @ MAX FPS")

        frames, sustained, mean_int = stream_test(cam, seconds=3.0, gain_db=17.5)

        print(f"Frames captured in 3 sec: {frames}")
        print(f"Sustained FPS: {sustained:.2f}")
        print(f"Mean intensity @ 17.5 dB: {mean_int:.2f}")

        print_line("DEVICE LINK THROUGHPUT")
        throughput = cam.DeviceLinkCurrentThroughput.GetValue()
        print("Current Throughput:", throughput)

        print("\n✅ HDR Capability Test Complete.")

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