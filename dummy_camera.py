import numpy as np
import cv2
import time


class DummyCamera:
    """
    Simulates a live RHEED-like video feed for testing.
    Also provides Exposure/Gain/Gamma setters (software-applied) so UI can be tested without hardware.
    """
    def __init__(self, width=640, height=480, fps=30):
        self.has_hw_control = False

        self.width = width
        self.height = height
        self.fps = fps
        self.frame_idx = 0
        self.running = False

        # Typical camera-like ranges (Spinnaker-ish)
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

        # Baseline reference for "exposure" brightness scaling on dummy feed
        self._exposure_ref_us = self.exposure_us

    def start(self):
        print("🎥 Dummy camera started (simulated RHEED feed).")
        self.running = True

    def _apply_software_controls(self, frame_u8: np.ndarray) -> np.ndarray:
        # frame_u8 is grayscale uint8

        img = frame_u8.astype(np.float32)

        # Exposure simulation: linear multiplier relative to reference
        exp_mult = float(self.exposure_us) / float(self._exposure_ref_us) if self._exposure_ref_us > 0 else 1.0

        # Gain simulation: dB -> linear amplitude multiplier
        # +6 dB ≈ 2x amplitude => multiplier = 10^(dB/20)
        gain_mult = 10.0 ** (float(self.gain_db) / 20.0)

        img *= (exp_mult * gain_mult)

        # Clamp before gamma
        img = np.clip(img, 0.0, 255.0)

        # Gamma simulation (if enabled)
        if self.gamma_enabled and abs(self.gamma - 1.0) > 1e-6:
            # Standard gamma mapping: out = (in/255)^(1/gamma) * 255
            inv = 1.0 / float(self.gamma)
            img = 255.0 * ((img / 255.0) ** inv)

        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img

    def get_frame(self):
        if not self.running:
            return None

        frame_period = 1.0 / self.fps

        # initialize scheduler
        if not hasattr(self, "_next_frame_time"):
            self._next_frame_time = time.perf_counter()

        # ----- BLOCK until next frame time (like real camera) -----
        now = time.perf_counter()
        sleep_time = self._next_frame_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)

        # schedule next frame
        self._next_frame_time += frame_period

        # ----- generate synthetic frame -----
        h, w = self.height, self.width
        base = np.zeros((h, w), dtype=np.uint8)

        x1 = int(320 + 120 * np.sin(self.frame_idx * 0.05))
        x2 = int(320 + 150 * np.cos(self.frame_idx * 0.04))

        cv2.circle(base, (x1, 180), 35, 180 + int(60 * np.sin(self.frame_idx * 0.1)), -1)
        cv2.circle(base, (x2, 300), 30, 160 + int(70 * np.cos(self.frame_idx * 0.12)), -1)

        noise = np.random.randint(0, 15, (h, w), dtype=np.uint8)
        frame = cv2.add(base, noise)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        frame = self._apply_software_controls(frame)

        self.frame_idx += 1
        return frame
    
    def stop(self):
        print("🛑 Dummy camera stopped.")
        self.running = False

    # ----- same public API as real camera -----
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
