import PySpin
import numpy as np


class Camera:
    """
    Blackfly camera wrapper using PySpin.
    Exposes Exposure/Gain/Gamma controls via GenICam nodes (Spinnaker-style).
    """
    def __init__(self):
        self.has_hw_control = True

        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            # Keep behavior: main.py will catch and fallback to DummyCamera
            self.system.ReleaseInstance()
            raise RuntimeError("❌ No camera detected. Connect Blackfly and ensure SpinView is closed.")

        self.cam = self.cam_list[0]
        self.cam.Init()
        self.nodemap = self.cam.GetNodeMap()

        # Cache ranges + current values (read from camera if available)
        self.exposure_min_us, self.exposure_max_us, self.exposure_us = self._init_exposure()
        self.gain_min_db, self.gain_max_db, self.gain_db = self._init_gain()
        self.gamma_min, self.gamma_max, self.gamma, self.gamma_enabled = self._init_gamma()

        print("📸 Camera initialized (PySpin acquisition + node control ready).")

    # -------------------- generic node helpers --------------------
    def _get_node(self, name):
        try:
            return self.nodemap.GetNode(name)
        except Exception:
            return None

    def _get_float_node(self, name):
        node = PySpin.CFloatPtr(self._get_node(name))
        if node is None or (not PySpin.IsAvailable(node)) or (not PySpin.IsReadable(node)):
            return None
        return node

    def _get_bool_node(self, name):
        node = PySpin.CBooleanPtr(self._get_node(name))
        if node is None or (not PySpin.IsAvailable(node)) or (not PySpin.IsReadable(node)):
            return None
        return node

    def _get_enum_node(self, name):
        node = PySpin.CEnumerationPtr(self._get_node(name))
        if node is None or (not PySpin.IsAvailable(node)) or (not PySpin.IsReadable(node)):
            return None
        return node

    def _set_enum_symbolic(self, enum_name, symbolic):
        enum_node = self._get_enum_node(enum_name)
        if enum_node is None or (not PySpin.IsWritable(enum_node)):
            return False

        entry = enum_node.GetEntryByName(symbolic)
        if entry is None or (not PySpin.IsAvailable(entry)) or (not PySpin.IsReadable(entry)):
            return False

        enum_node.SetIntValue(entry.GetValue())
        return True

    @staticmethod
    def _clamp(v, vmin, vmax):
        return max(vmin, min(vmax, v))

    # -------------------- node init (read ranges/current) --------------------
    def _init_exposure(self):
        # Ensure manual control if possible
        self._set_enum_symbolic("ExposureAuto", "Off")

        node = self._get_float_node("ExposureTime")
        if node is None:
            return 0.0, 0.0, 0.0

        vmin, vmax = float(node.GetMin()), float(node.GetMax())
        val = float(node.GetValue())
        return vmin, vmax, val

    def _init_gain(self):
        self._set_enum_symbolic("GainAuto", "Off")

        node = self._get_float_node("Gain")
        if node is None:
            return 0.0, 0.0, 0.0

        vmin, vmax = float(node.GetMin()), float(node.GetMax())
        val = float(node.GetValue())
        return vmin, vmax, val

    def _init_gamma(self):
        # Some cameras expose GammaEnable (bool) + Gamma (float)
        # Some expose GammaEnabled or no enable at all.
        gamma_node = self._get_float_node("Gamma")

        enable_node = self._get_bool_node("GammaEnable")
        if enable_node is None:
            enable_node = self._get_bool_node("GammaEnabled")

        enabled = False
        if enable_node is not None:
            try:
                enabled = bool(enable_node.GetValue())
            except Exception:
                enabled = False

        if gamma_node is None:
            # Gamma might not be supported depending on pixel format / model
            return 0.0, 0.0, 0.0, enabled

        vmin, vmax = float(gamma_node.GetMin()), float(gamma_node.GetMax())
        val = float(gamma_node.GetValue())
        return vmin, vmax, val, enabled

    # -------------------- acquisition --------------------
    def start(self):
        self.cam.BeginAcquisition()
        print("🎥 Acquisition started.")



    def get_frame(self):
        try:
            image = self.cam.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                return None

            frame = image.GetNDArray()
            image.Release()
            return frame

        except PySpin.SpinnakerException as e:
            print(f"⚠️ Frame grab error: {e}")
            return None

    def stop(self):
        try:
            self.cam.EndAcquisition()
        except Exception:
            pass

        print("🛑 Acquisition stopped.")

        try:
            self.cam.DeInit()
        except Exception:
            pass

        try:
            del self.cam
        except Exception:
            pass

        try:
            self.cam_list.Clear()
        except Exception:
            pass

        try:
            self.system.ReleaseInstance()
        except Exception:
            pass

        print("✅ Camera disconnected and cleaned up.")

    # -------------------- public settings API --------------------
    def get_settings(self):
        return {
            "has_hw_control": True,
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
        self._set_enum_symbolic("ExposureAuto", "Off")
        node = self._get_float_node("ExposureTime")
        if node is None or (not PySpin.IsWritable(node)):
            return self.exposure_us

        v = self._clamp(float(value_us), float(node.GetMin()), float(node.GetMax()))
        node.SetValue(v)
        self.exposure_us = float(node.GetValue())
        return self.exposure_us

    def set_gain_db(self, value_db: float):
        self._set_enum_symbolic("GainAuto", "Off")
        node = self._get_float_node("Gain")
        if node is None or (not PySpin.IsWritable(node)):
            return self.gain_db

        v = self._clamp(float(value_db), float(node.GetMin()), float(node.GetMax()))
        node.SetValue(v)
        self.gain_db = float(node.GetValue())
        return self.gain_db

    def set_gamma_enabled(self, enabled: bool):
        enable_node = self._get_bool_node("GammaEnable")
        if enable_node is None:
            enable_node = self._get_bool_node("GammaEnabled")

        if enable_node is None or (not PySpin.IsWritable(enable_node)):
            # If no enable node exists, treat gamma as always enabled from UI perspective
            self.gamma_enabled = bool(enabled)
            return self.gamma_enabled

        enable_node.SetValue(bool(enabled))
        self.gamma_enabled = bool(enable_node.GetValue())
        return self.gamma_enabled

    def set_gamma(self, gamma_value: float):
        gamma_node = self._get_float_node("Gamma")
        if gamma_node is None or (not PySpin.IsWritable(gamma_node)):
            return self.gamma

        # If there is an enable node, ensure gamma is enabled when user sets it
        self.set_gamma_enabled(True)

        v = self._clamp(float(gamma_value), float(gamma_node.GetMin()), float(gamma_node.GetMax()))
        gamma_node.SetValue(v)
        self.gamma = float(gamma_node.GetValue())
        return self.gamma

    def get_fps(self):
        return float(self.cam.AcquisitionFrameRate())
