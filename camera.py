"""
FLIR Blackfly S camera wrapper — camera.py

Wraps the PySpin (Spinnaker) SDK to provide a clean, simple API
for the RHEED dashboard. Handles camera initialization, stream buffer
configuration, exposure/gain/gamma control via GenICam nodes,
frame acquisition, and clean shutdown.

Raises RuntimeError on init if no camera is detected,
which main.py catches to fall back to DummyCamera.

Dependencies: PySpin (FLIR Spinnaker SDK), NumPy
"""
import PySpin
import numpy as np


class Camera:
    """
    Blackfly camera wrapper using PySpin.
    Exposes Exposure/Gain/Gamma controls via GenICam nodes (Spinnaker-style).
    """
    def __init__(self):
        """
        Initialize the first detected FLIR camera.
        Configures stream buffering to 'OldestFirst' with 100 manual buffers
        to prevent frame drops during HDR exposure cycling.
        Reads and caches exposure, gain, and gamma ranges from the camera hardware.
        Raises RuntimeError if no camera is found.
        """
        self.has_hw_control = True

        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            # Keep behavior: main.py will catch and fallback to DummyCamera
            self.system.ReleaseInstance()
            raise RuntimeError("❌ No camera detected. Connect Blackfly and ensure SpinView is closed.")

        self.cam = self.cam_list[0]
        self.cam.Init()

        # Always set nodemap immediately (do NOT put this inside buffer_count block)
        self.nodemap = self.cam.GetNodeMap()

        # ---- Improve stream buffering to prevent frame drops ----
        s_node_map = self.cam.GetTLStreamNodeMap()

        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsAvailable(handling_mode) and PySpin.IsWritable(handling_mode):
            entry = handling_mode.GetEntryByName("OldestFirst")
            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                handling_mode.SetIntValue(entry.GetValue())

        buffer_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferCountMode"))
        if PySpin.IsAvailable(buffer_mode) and PySpin.IsWritable(buffer_mode):
            entry = buffer_mode.GetEntryByName("Manual")
            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                buffer_mode.SetIntValue(entry.GetValue())

        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode("StreamBufferCountManual"))
        if PySpin.IsAvailable(buffer_count) and PySpin.IsWritable(buffer_count):
            # 100 is okay; for GigE you can try 200 if RAM is fine
            buffer_count.SetValue(100)

        # Cache ranges + current values (read from camera if available)
        self.exposure_min_us, self.exposure_max_us, self.exposure_us = self._init_exposure()
        self.gain_min_db, self.gain_max_db, self.gain_db = self._init_gain()
        self.gamma_min, self.gamma_max, self.gamma, self.gamma_enabled = self._init_gamma()

        print("📸 Camera initialized (PySpin acquisition + node control ready).")

    # -------------------- generic node helpers --------------------
    # ── GenICam node helpers ───────────────────────────────────────────
    # These private methods safely retrieve typed node pointers from the
    # camera's nodemap. Each returns None if the node is unavailable or
    # not readable, so callers can check before attempting to read/write.
    def _get_node(self, name):
        """Return a raw GenICam node by name, or None on failure."""
        try:
            return self.nodemap.GetNode(name)
        except Exception:
            return None

    def _get_float_node(self, name):
        """Return a readable CFloatPtr node by name, or None if unavailable."""
        node = PySpin.CFloatPtr(self._get_node(name))
        if node is None or (not PySpin.IsAvailable(node)) or (not PySpin.IsReadable(node)):
            return None
        return node

    def _get_bool_node(self, name):
        """Return a readable CBooleanPtr node by name, or None if unavailable."""

        node = PySpin.CBooleanPtr(self._get_node(name))
        if node is None or (not PySpin.IsAvailable(node)) or (not PySpin.IsReadable(node)):
            return None
        return node

    def _get_enum_node(self, name):
        """Return a readable CEnumerationPtr node by name, or None if unavailable."""
        node = PySpin.CEnumerationPtr(self._get_node(name))
        if node is None or (not PySpin.IsAvailable(node)) or (not PySpin.IsReadable(node)):
            return None
        return node

    def _set_enum_symbolic(self, enum_name, symbolic):
        """
        Set an enumeration node to the given symbolic value (e.g. 'Off', 'Manual').
        Returns True if successful, False if the node or entry is unavailable.
        """
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
        """Clamp value v to the range [vmin, vmax]."""
        return max(vmin, min(vmax, v))

    # -------------------- node init (read ranges/current) --------------------
    def _init_exposure(self):
        """
        Disable auto-exposure and read the hardware min, max, and current
        exposure time in microseconds from the ExposureTime node.
        Returns (min_us, max_us, current_us).
        """
        # Ensure manual control if possible
        self._set_enum_symbolic("ExposureAuto", "Off")

        node = self._get_float_node("ExposureTime")
        if node is None:
            return 0.0, 0.0, 0.0

        vmin, vmax = float(node.GetMin()), float(node.GetMax())
        val = float(node.GetValue())
        return vmin, vmax, val

    def _init_gain(self):
        """
        Disable auto-gain and read the hardware min, max, and current
        gain in dB from the Gain node.
        Returns (min_db, max_db, current_db).
        """
        self._set_enum_symbolic("GainAuto", "Off")

        node = self._get_float_node("Gain")
        if node is None:
            return 0.0, 0.0, 0.0

        vmin, vmax = float(node.GetMin()), float(node.GetMax())
        val = float(node.GetValue())
        return vmin, vmax, val

    def _init_gamma(self):
        """
        Read the hardware gamma range and current value.
        Handles cameras that expose GammaEnable, GammaEnabled, or neither.
        Returns (min, max, current_value, enabled).
        """
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
        """Begin image acquisition on the camera."""
        self.cam.BeginAcquisition()
        print("🎥 Acquisition started.")



    def get_frame(self):
        """
        Grab and return the next frame as a NumPy array.
        Waits up to 1000 ms for a frame before timing out.
        Returns None if the frame is incomplete or a grab error occurs.
        """
        try:
            image = self.cam.GetNextImage(1000)  # 1000ms timeout
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
        """
        Stop acquisition, deinitialize the camera, and release all
        PySpin resources in the correct order to avoid SDK warnings.
        """
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
        """
        Return cached camera settings as a dict.
        Matches the format of DummyCamera.get_settings() so the dashboard
        can read min/max ranges from either camera type identically.
        """
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
        """
        Set exposure time in microseconds via the ExposureTime GenICam node.
        Disables auto-exposure first, clamps to hardware limits.
        Returns the value actually applied by the camera.
        """
        self._set_enum_symbolic("ExposureAuto", "Off")
        node = self._get_float_node("ExposureTime")
        if node is None or (not PySpin.IsWritable(node)):
            return self.exposure_us

        v = self._clamp(float(value_us), float(node.GetMin()), float(node.GetMax()))
        node.SetValue(v)
        self.exposure_us = float(node.GetValue())
        return self.exposure_us

    def set_gain_db(self, value_db: float):
        """
        Set sensor gain in dB via the Gain GenICam node.
        Disables auto-gain first, clamps to hardware limits.
        Returns the value actually applied by the camera.
        """
        self._set_enum_symbolic("GainAuto", "Off")
        node = self._get_float_node("Gain")
        if node is None or (not PySpin.IsWritable(node)):
            return self.gain_db

        v = self._clamp(float(value_db), float(node.GetMin()), float(node.GetMax()))
        node.SetValue(v)
        self.gain_db = float(node.GetValue())
        return self.gain_db

    def set_gamma_enabled(self, enabled: bool):
        """
        Enable or disable gamma correction via GammaEnable / GammaEnabled node.
        Falls back gracefully if the camera does not expose an enable node.
        Returns the applied boolean value.
        """
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
        """
        Set gamma correction value via the Gamma GenICam node.
        Automatically enables gamma when called.
        Clamps to hardware min/max limits.
        Returns the value actually applied by the camera.
        """
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
        """
        Return the current resulting frame rate from the camera hardware.
        Tries AcquisitionResultingFrameRate first, then AcquisitionFrameRate.
        Returns 0.0 if neither node is available.
        """
        try:
            node = self._get_float_node("AcquisitionResultingFrameRate")
            if node is not None:
                return float(node.GetValue())
            node = self._get_float_node("AcquisitionFrameRate")
            if node is not None:
                return float(node.GetValue())
            return 0.0
        except Exception:
            return 0.0
