import numpy as np
import cv2
import time


class DummyCamera:
    """
    Simulates a live RHEED camera using a looping video file.
    Exposure / Gain / Gamma are applied in software so the UI behaves like a real camera.
    """

    def __init__(self, width=1920, height=1200, fps=30, video_path="RHEED_video_03-09-26_11-40-38.avi"):
        self.has_hw_control = False

        self.width = width
        self.height = height
        self.fps = fps
        self.frame_idx = 0
        self.running = False

        self.video_path = video_path
        self.cap = None

        # Typical camera-like ranges
        self.exposure_min_us = 10.0
        self.exposure_max_us = 1_000_000.0
        self.exposure_us = 20_000.0

        self.gain_min_db = 0.0
        self.gain_max_db = 24.0
        self.gain_db = 0.0

        self.gamma_min = 0.25
        self.gamma_max = 4.0
        self.gamma = 1.0
        self.gamma_enabled = True

        # baseline exposure reference
        self._exposure_ref_us = self.exposure_us

    # ------------------------------------------------------------
    # START CAMERA
    # ------------------------------------------------------------
    def start(self):

        print("🎥 Dummy camera started (video playback mode).")

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Could not open dummy video: {self.video_path}")

        # read video FPS if available
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps > 1:
            self.fps = video_fps

        self.running = True
        self._next_frame_time = time.perf_counter()

    # ------------------------------------------------------------
    # SOFTWARE CAMERA CONTROLS
    # ------------------------------------------------------------
    def _apply_software_controls(self, frame_u8: np.ndarray) -> np.ndarray:

        img = frame_u8.astype(np.float32)

        # Exposure simulation
        exp_mult = float(self.exposure_us) / float(self._exposure_ref_us) if self._exposure_ref_us > 0 else 1.0

        # Gain simulation (dB → amplitude)
        gain_mult = 10.0 ** (float(self.gain_db) / 20.0)

        img *= (exp_mult * gain_mult)

        img = np.clip(img, 0.0, 255.0)

        # Gamma simulation
        if self.gamma_enabled and abs(self.gamma - 1.0) > 1e-6:
            inv = 1.0 / float(self.gamma)
            img = 255.0 * ((img / 255.0) ** inv)

        img = np.clip(img, 0.0, 255.0).astype(np.uint8)

        return img

    # ------------------------------------------------------------
    # FRAME GENERATION (VIDEO LOOP)
    # ------------------------------------------------------------
    def get_frame(self):

        if not self.running:
            return None

        frame_period = 1.0 / self.fps

        now = time.perf_counter()
        sleep_time = self._next_frame_time - now

        # Only sleep if this is a meaningful wait — skip tiny sleeps
        # so HDR discard frames don't throttle the pipeline
        if sleep_time > 0.002:
            time.sleep(sleep_time)

        self._next_frame_time += frame_period

        ret, frame = self.cap.read()

        # restart video when finished
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if frame is None:
            return None

        # convert to grayscale like real camera
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # use video native resolution
        h, w = frame.shape[:2]
        self.width = w
        self.height = h


        # apply software exposure/gain/gamma
        frame = self._apply_software_controls(frame)

        self.frame_idx += 1

        return frame

    # ------------------------------------------------------------
    # STOP CAMERA
    # ------------------------------------------------------------
    def stop(self):

        print("🛑 Dummy camera stopped.")

        self.running = False

        if self.cap is not None:
            self.cap.release()

    # ------------------------------------------------------------
    # CAMERA API COMPATIBILITY
    # ------------------------------------------------------------
    def get_settings(self):

        return {
            "has_hw_control": False,
            "exposure_us": self.exposure_us,
            "exposure_min_us": self.exposure_min_us,
            "exposure_max_us": self.exposure_max_us,
            "gain_db": self.gain_db,
            "gain_min_db": self.gain_min_db,
            "gain_max_db": self.gain_max_db,
            "gamma": self.gamma,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "gamma_enabled": self.gamma_enabled,
        }

    def set_exposure_us(self, value_us: float):

        self.exposure_us = float(max(self.exposure_min_us, min(self.exposure_max_us, float(value_us))))

        return self.exposure_us

    def set_gain_db(self, value_db: float):

        self.gain_db = float(max(self.gain_min_db, min(self.gain_max_db, float(value_db))))

        return self.gain_db

    def set_gamma_enabled(self, enabled: bool):

        self.gamma_enabled = bool(enabled)

        return self.gamma_enabled

    def set_gamma(self, gamma_value: float):

        self.gamma = float(max(self.gamma_min, min(self.gamma_max, float(gamma_value))))
        self.gamma_enabled = True

        return self.gamma

    def get_fps(self):

        return float(self.fps)