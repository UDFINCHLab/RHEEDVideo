import cv2
import numpy as np
from collections import deque
import pandas as pd
import datetime
import uuid
from pathlib import Path



ROI_COLORS = {
    1: (255, 0, 0),       # Blue
    2: (0, 165, 255),     # Orange
    3: (0, 255, 0),       # Green
    4: (255, 0, 255),     # Magenta
    5: (128, 0, 255),     # Purple
    6: (255, 255, 0),     # Cyan
    
}


class ROIManager:
    """Manages two ROIs (ellipse or rectangle) with mouse + keyboard interaction."""
    def __init__(self, max_history=6000):
        self.rois = {}
        self.max_history = max_history
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.active_roi = None
        self.start_point = None
        self.temp_roi = None
        self.roi_saves=0
        
        



        # NEW: shape toggle → "ellipse" or "rect"
        self.shape_mode = "ellipse"

    # ---------- toggle shape ----------
    def toggle_shape(self, mode):
        if mode in ("ellipse", "rect"):
            self.shape_mode = mode
            print(f"✏️ ROI drawing mode → {self.shape_mode}")

    # ---------- helpers ----------
    def _allocate_id(self):
        for rid in range(1, 7):   # Allow only 1–6
            if rid not in self.rois:
                return rid
        print("⚠️ Max 6 ROIs reached.")
        return None



    def _new_roi(self, rid, center, rx, ry, shape=None):
        return {
            "id": rid,
            "uuid": str(uuid.uuid4()),  # ← ADD THIS LINE
            "center": (int(center[0]), int(center[1])),
            "rx": int(max(5, rx)),
            "ry": int(max(5, ry)),
            "shape": shape if shape else self.shape_mode,
            "color": ROI_COLORS[rid],
            "t": deque(maxlen=self.max_history),
            "y": deque(maxlen=self.max_history),
        }

    def reset(self):
        self.rois.clear()
        self.drawing = self.moving = self.resizing = False
        self.active_roi = self.start_point = self.temp_roi = None

    def remove_nearest(self, x, y):
        if not self.rois:
            return
        rid = min(self.rois.keys(),
                  key=lambda k: np.hypot(self.rois[k]["center"][0]-x,
                                         self.rois[k]["center"][1]-y))
        self.rois.pop(rid, None)
        print(f"🗑️ Removed ROI #{rid}")

    # ---------- draw / move / resize ----------
    def start_drawing(self, x, y):
        
        rid = self._allocate_id()
        if rid is None:
            return
        self.drawing = True
        self.start_point = (x, y)

        # NEW: include shape
        self.temp_roi = self._new_roi(rid, (x, y), 1, 1, shape=self.shape_mode)

    def update_drawing(self, x, y):
        if self.drawing and self.temp_roi:
            cx, cy = self.start_point
            self.temp_roi["rx"] = abs(x - cx)
            self.temp_roi["ry"] = abs(y - cy)

    def finish_drawing(self):
        if self.drawing and self.temp_roi:
            rid = self.temp_roi["id"]
            self.rois[rid] = self.temp_roi
            print(f"✅ ROI #{rid} created ({self.temp_roi['shape']})")
        self.drawing = False
        self.temp_roi = None

    def select_roi(self, x, y, shift=False):
        """Select ROI for move or resize. Shift=True → resize."""
        for rid in sorted(self.rois.keys(), reverse=True):
            roi = self.rois.get(rid)
            if roi is None:
                continue
            cx, cy = roi["center"]
            rx, ry = roi["rx"], roi["ry"]

            if roi["shape"] == "ellipse":
                inside = ((x - cx)**2) / (rx**2) + ((y - cy)**2) / (ry**2) <= 1.0
            else:  # rect
                inside = (cx - rx <= x <= cx + rx) and (cy - ry <= y <= cy + ry)

            if inside:
                self.active_roi = rid
                self.start_point = (x, y)
                self.moving = not shift
                self.resizing = shift
                return True
        return False

    def move_selected(self, x, y):
        if self.moving and self.active_roi:
            roi = self.rois[self.active_roi]
            dx, dy = x - self.start_point[0], y - self.start_point[1]
            cx, cy = roi["center"]
            roi["center"] = (cx + dx, cy + dy)
            self.start_point = (x, y)

    def resize_selected(self, delta=None, x=None, y=None):
        """Keyboard delta OR mouse resize (Shift + drag)."""
        if not self.active_roi:
            return
        roi = self.rois[self.active_roi]

        if delta is not None:
            roi["rx"] = max(5, roi["rx"] + delta)
            roi["ry"] = max(5, roi["ry"] + delta)
        elif self.resizing and self.start_point and x is not None and y is not None:
            cx, cy = roi["center"]

            if roi["shape"] == "ellipse":
                roi["rx"] = max(5, abs(x - cx))
                roi["ry"] = max(5, abs(y - cy))

            else:  # rectangle
                roi["rx"] = max(5, abs(x - cx))
                roi["ry"] = max(5, abs(y - cy))


    def release(self):
        self.moving = self.resizing = False
        self.active_roi = None
        self.start_point = None

    # ---------- intensity ----------
    def update_intensities(self, frame_gray, timestamp_s):
        for roi in self.rois.values():
            mask = np.zeros_like(frame_gray, dtype=np.uint8)
            cx, cy = roi["center"]

            if roi["shape"] == "ellipse":
                cv2.ellipse(mask, roi["center"], (roi["rx"], roi["ry"]), 0, 0, 360, 255, -1)

            else:  # rectangle mask
                x1 = cx - roi["rx"]
                y1 = cy - roi["ry"]
                x2 = cx + roi["rx"]
                y2 = cy + roi["ry"]
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            raw_mean = cv2.mean(frame_gray, mask=mask)[0]

            moving_avg_window = 15
            summands = list(roi["y"])[-moving_avg_window:]
            moving_avg = raw_mean
            moving_avg += np.sum(summands)
            moving_avg /= (len(summands) + 1)

            roi["t"].append(timestamp_s)
            roi["y"].append(moving_avg)

            # NEW: store raw + sum + area for logging
            roi["last_raw_mean"] = float(raw_mean)
            roi["last_sum"] = float(np.sum(frame_gray[mask == 255]))
            roi["last_area"] = int(np.sum(mask == 255))

    # ---------- overlay ----------
    def draw_overlays(self, frame_bgr):
        for rid, roi in self.rois.items():
            cx, cy = roi["center"]

            if roi["shape"] == "ellipse":
                cv2.ellipse(frame_bgr, (cx, cy), (roi["rx"], roi["ry"]),
                            0, 0, 360, roi["color"], 1, cv2.LINE_AA)
            else:
                x1 = cx - roi["rx"]
                y1 = cy - roi["ry"]
                x2 = cx + roi["rx"]
                y2 = cy + roi["ry"]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2),
                              roi["color"], 1, cv2.LINE_AA)

            cv2.putText(frame_bgr, str(rid), (cx - 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi["color"], 1, cv2.LINE_AA)

        # temp ROI while drawing
        if self.drawing and self.temp_roi:
            cx, cy = self.temp_roi["center"]
            rx, ry = self.temp_roi["rx"], self.temp_roi["ry"]
            if self.temp_roi["shape"] == "ellipse":
                cv2.ellipse(frame_bgr, (cx, cy), (rx, ry), 0, 0, 360,
                            (200, 200, 255), 1, cv2.LINE_AA)
            else:
                x1 = cx - rx
                y1 = cy - ry
                x2 = cx + rx
                y2 = cy + ry
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2),
                              (200, 200, 255), 1, cv2.LINE_AA)


# ============================================================
# ------------------- Line Profile Manager -------------------
# ============================================================

class LineManager:
    def __init__(self):
        self.draw_mode = False
        self.drawing = False
        self.pt1 = None
        self.pt2 = None
        self.profile = None
        self.elapsed_s = 0.0
        self._win_created = False

    def toggle(self):
        self.draw_mode = not self.draw_mode
        print(f"📏 Line Draw Mode → {'ON' if self.draw_mode else 'OFF'}")
        
    def clear_line(self):
        self.pt1 = None
        self.pt2 = None
        self.profile = None
        self.elapsed_s = 0.0
        try:
            cv2.destroyWindow("RHEED Line Profile")
        except:
            pass
        print("🗑 Line Cleared")

    def start_drawing(self, x, y):
        self.drawing = True
        self.pt1 = (x, y)
        self.pt2 = (x, y)

    def update_drawing(self, x, y):
        if self.drawing:
            self.pt2 = (x, y)

    def finish_drawing(self):
        self.drawing = False

    # Recording window line color (mint like reference)
    def draw_overlay(self, frame):
        if self.pt1 and self.pt2:
            cv2.line(frame, self.pt1, self.pt2, (120, 255, 120), 2, cv2.LINE_AA)

    def extract_profile(self, gray_frame, now_s=None):
        if not self.pt1 or not self.pt2:
            return None

        x1, y1 = self.pt1
        x2, y2 = self.pt2

        length = int(np.hypot(x2 - x1, y2 - y1))
        if length < 2:
            return None

        xs = np.linspace(x1, x2, length).astype(np.int32)
        ys = np.linspace(y1, y2, length).astype(np.int32)

        xs = np.clip(xs, 0, gray_frame.shape[1] - 1)
        ys = np.clip(ys, 0, gray_frame.shape[0] - 1)

        self.profile = gray_frame[ys, xs]

        if now_s is not None:
            self.elapsed_s = float(now_s)

        return self.profile

    # Aspect-fit helper (same logic as dashboard)
    def _fit_to_window(self, frame, screen_w, screen_h):
        h, w = frame.shape[:2]
        if screen_w <= 0 or screen_h <= 0:
            return frame

        scale = min(screen_w / w, screen_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        result = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        x0 = (screen_w - new_w) // 2
        y0 = (screen_h - new_h) // 2
        result[y0:y0 + new_h, x0:x0 + new_w] = resized
        return result

    def render_window(self):
        if self.profile is None or len(self.profile) < 2:
            return

        win_name = "RHEED Line Profile"

        # Create window once
        if not self._win_created:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 950, 520)
            self._win_created = True

        # Get current window size
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(win_name)
        except:
            win_w, win_h = 950, 520

        # ---- INTERNAL BASE RESOLUTION (stable layout) ----
        base_w = 950
        base_h = 520
        chart = np.full((base_h, base_w, 3), 255, np.uint8)

        # Fixed margins for clean layout
        ml, mr, mt, mb = 110, 40, 40, 80
        x0, y0 = ml, mt
        plot_w = base_w - ml - mr
        plot_h = base_h - mt - mb

        # Border
        cv2.rectangle(chart, (x0, y0),
                      (x0 + plot_w, y0 + plot_h),
                      (160, 160, 160), 1)

        y_vals = np.asarray(self.profile, dtype=float)
        x_vals = np.arange(len(y_vals))

        ymin = float(np.min(y_vals))
        ymax = float(np.max(y_vals))
        if abs(ymax - ymin) < 1e-6:
            ymax = ymin + 1.0

        # ---- GRID (darker like reference) ----
        yticks = np.linspace(ymin, ymax, 5)
        for v in yticks:
            yy = y0 + plot_h - int((v - ymin) / (ymax - ymin) * plot_h)
            cv2.line(chart, (x0, yy), (x0 + plot_w, yy),
                     (185, 185, 185), 1)
            cv2.putText(chart, f"{v:.2f}",
                        (50, yy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (60, 60, 60), 1, cv2.LINE_AA)

        xticks = np.linspace(0, len(x_vals) - 1, 5)
        for v in xticks:
            xx = x0 + int(v / (len(x_vals) - 1) * plot_w)
            cv2.line(chart, (xx, y0), (xx, y0 + plot_h),
                     (185, 185, 185), 1)
            cv2.putText(chart, f"{int(v)}",
                        (xx - 15, y0 + plot_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (60, 60, 60), 1, cv2.LINE_AA)

        # ---- Waveform with glow ----
        pts = np.column_stack((
            x0 + (x_vals / (len(x_vals) - 1) * plot_w).astype(int),
            y0 + plot_h - ((y_vals - ymin) / (ymax - ymin) * plot_h).astype(int)
        )).astype(np.int32)

        cv2.polylines(chart, [pts], False, (180, 235, 180), 3, cv2.LINE_AA)
        cv2.polylines(chart, [pts], False, (120, 200, 120), 1, cv2.LINE_AA)

        # ---- Axis labels ----
        cv2.putText(chart, "Position (pixels)",
                    (base_w // 2 - 90, base_h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (40, 40, 40), 1, cv2.LINE_AA)

        label = "Intensity (a.u.)"
        (tw, th), _ = cv2.getTextSize(label,
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        y_img = np.full((th + 8, tw + 8, 3), 255, np.uint8)
        cv2.putText(y_img, label, (4, th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (40, 40, 40), 1, cv2.LINE_AA)
        y_img = cv2.rotate(y_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        chart[base_h // 2 - y_img.shape[0] // 2:
              base_h // 2 - y_img.shape[0] // 2 + y_img.shape[0],
              5:5 + y_img.shape[1]] = y_img

        # ---- Footer ----
        footer_text = (
            f"Data Pt.: {len(y_vals)}    "
            f"Elapsed Time: {self.elapsed_s:.2f}    "
            f"Min: {ymin:.4f}"
        )

        cv2.putText(chart, footer_text,
                    (30, base_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (60, 60, 60), 1, cv2.LINE_AA)

        # ---- Fit to current window (prevents squeezing) ----
        display = self._fit_to_window(chart, win_w, win_h)
        cv2.imshow(win_name, display)
