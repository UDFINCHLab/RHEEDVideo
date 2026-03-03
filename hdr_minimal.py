import cv2
import numpy as np
import threading
import queue
import time
import os
from dummy_camera import DummyCamera
from camera import Camera  # add this at top


# ---------------- HDR SETTINGS ----------------
GAIN_CYCLE = [8.0, 12.0, 24.0]
TARGET_CAMERA_FPS = 60
DISPLAY_WINDOW = "HDR Preview (Dummy)"

# ---------------- GLOBAL STATE ----------------
stop_event = threading.Event()

raw_queue = queue.Queue(maxsize=180)
hdr_queue = queue.Queue(maxsize=60)

merge_mertens = cv2.createMergeMertens()


def apply_gradient(frame, lut="rheed_gradient_lut.npy", strength=0.8):
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if not hasattr(apply_gradient, "lut"):
        try:
            arr = np.load(lut)
        except Exception:
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        x = np.linspace(0, 255, len(arr))
        full = np.zeros((256, 1, 3), np.uint8)
        for c in range(3):
            full[:, 0, c] = np.interp(np.arange(256), x, arr[:, c])
        apply_gradient.lut = full

    color = cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), apply_gradient.lut)

    return cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                           1 - strength, color, strength, 0)


# ---------------- ACQUISITION THREAD ----------------
def capture_loop(cam):
    gain_index = 0

    while not stop_event.is_set():
        # Set gain before grabbing frame
        cam.set_gain_db(GAIN_CYCLE[gain_index])

        frame = cam.get_frame()
        if frame is None:
            continue

        timestamp = time.time()

        try:
            raw_queue.put_nowait((frame, GAIN_CYCLE[gain_index], timestamp))
        except queue.Full:
            pass

        gain_index = (gain_index + 1) % 3


# ---------------- HDR FUSION THREAD ----------------
def hdr_loop():
    buffer = []
    hdr_triplet_count = 0

    while not stop_event.is_set():
        try:
            frame, gain, ts = raw_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Convert to 8-bit if needed
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)

        buffer.append(frame)
        
        if len(buffer) == 3 and hdr_triplet_count > 20 and not hasattr(hdr_loop, "saved_test"):

            os.makedirs("hdr_gain_test", exist_ok=True)

            # ---------- FORCE DELETE OLD FILES ----------
            for f in ["gain_8.png", "gain_12.png", "gain_24.png"]:
                path = os.path.join("hdr_gain_test", f)
                if os.path.exists(path):
                    os.remove(path)

            # ---------- Apply LUT ----------
            g8  = apply_gradient(buffer[0])
            g12 = apply_gradient(buffer[1])
            g24 = apply_gradient(buffer[2])

            # ---------- Save New Files ----------
            cv2.imwrite("hdr_gain_test/gain_8.png",  g8)
            cv2.imwrite("hdr_gain_test/gain_12.png", g12)
            cv2.imwrite("hdr_gain_test/gain_24.png", g24)

            print("✅ Overwritten and refreshed gain frames")
            hdr_loop.saved_test = True

        if len(buffer) == 3:
            hdr_triplet_count += 1
            print("Triplet count:", hdr_triplet_count)
            # Mertens expects list of images
            fused = merge_mertens.process(buffer)

            # Convert float32 [0,1] to uint8
            fused_8 = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX)
            fused_8 = fused_8.astype(np.uint8)

            # Apply gradient LUT (reuse logic)
            color = apply_gradient(fused_8)

            try:
                hdr_queue.put_nowait(color)
            except queue.Full:
                pass

            buffer.clear()
            

def fit_to_window(frame, screen_w, screen_h):
    h, w = frame.shape[:2]

    if screen_w <= 1 or screen_h <= 1 or w <= 0 or h <= 0:
        return frame

    scale = min(screen_w / w, screen_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    y0 = (screen_h - new_h) // 2
    x0 = (screen_w - new_w) // 2

    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas            


# ---------------- MAIN DISPLAY LOOP ----------------
def main():
    print("🚀 Minimal HDR Prototype (Auto Camera Mode)")
    print("Target: 60 FPS acquisition → 20 FPS HDR output")

    # ---------------- Camera Selection ----------------
    try:
        cam = Camera()

        # Enable FPS control if supported
        if hasattr(cam, "cam") and hasattr(cam.cam, "AcquisitionFrameRateEnable"):
            try:
                cam.cam.AcquisitionFrameRateEnable.SetValue(True)

                max_fps = cam.cam.AcquisitionFrameRate.GetMax()
                min_fps = cam.cam.AcquisitionFrameRate.GetMin()

                print(f"📷 Blackfly FPS capability: {min_fps:.2f} – {max_fps:.2f}")

                cam.cam.AcquisitionFrameRate.SetValue(TARGET_CAMERA_FPS)

                resulting = cam.cam.AcquisitionResultingFrameRate.GetValue()

                print(f"🔒 Camera locked to: {TARGET_CAMERA_FPS:.2f} FPS")
                print(f"📊 Resulting hardware FPS: {resulting:.2f}")

            except Exception as e:
                print("⚠ FPS lock not supported:", e)

        cam.start()
        print("🎥 Using REAL Blackfly camera")

    except Exception as e:
        print(f"⚠ Real camera not found. Switching to Dummy. ({e})")
        cam = DummyCamera(width=640, height=480, fps=TARGET_CAMERA_FPS)
        cam.start()
        print("🎥 Using DUMMY camera")

    # ---------------- Threads ----------------
    cap_thread = threading.Thread(target=capture_loop, args=(cam,), daemon=True)
    fusion_thread = threading.Thread(target=hdr_loop, daemon=True)

    cap_thread.start()
    fusion_thread.start()

    cv2.namedWindow("HDR Preview", cv2.WINDOW_NORMAL)

    last_time = time.time()
    frame_counter = 0

    try:
        while True:
            try:
                hdr_frame = hdr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_counter += 1

            now = time.time()
            if now - last_time >= 1.0:
                print(f"HDR FPS: {frame_counter}")
                frame_counter = 0
                last_time = now

            try:
                _, _, win_w, win_h = cv2.getWindowImageRect("HDR Preview")
            except:
                win_w, win_h = hdr_frame.shape[1], hdr_frame.shape[0]

            fitted = fit_to_window(hdr_frame, win_w, win_h)

            cv2.imshow("HDR Preview", fitted)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        stop_event.set()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()