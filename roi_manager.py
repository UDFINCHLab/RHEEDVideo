import cv2
import numpy as np
from collections import deque
import pandas as pd
import datetime
from pathlib import Path
from config import ROI_OUTPUT_DIR


ROI_COLORS = {1: (255, 0, 0), 2: (0, 165, 255)}  # Blue / Orange

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
        
        self.roi_main_directory = ROI_OUTPUT_DIR
        self.roi_main_directory.mkdir(parents=True, exist_ok=True)



        # NEW: shape toggle → "ellipse" or "rect"
        self.shape_mode = "ellipse"

    # ---------- toggle shape ----------
    def toggle_shape(self, mode):
        if mode in ("ellipse", "rect"):
            self.shape_mode = mode
            print(f"✏️ ROI drawing mode → {self.shape_mode}")

    # ---------- helpers ----------
    def _allocate_id(self):
        for rid in (1, 2):
            if rid not in self.rois:
                return rid
        return None

    def _new_roi(self, rid, center, rx, ry, shape=None):
        return {
            "id": rid,
            "center": (int(center[0]), int(center[1])),
            "rx": int(max(5, rx)),
            "ry": int(max(5, ry)),
            "shape": shape if shape else self.shape_mode,   # NEW
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
        if len(self.rois) >= 2:
            print("⚠️ Max 2 ROIs reached.")
            return
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
        for rid in (2, 1):
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
        elif self.resizing and self.start_point and x is not None:
            cx, cy = roi["center"]
            if roi["shape"] == "ellipse":
                r = int(np.hypot(x - cx, y - cy))
                roi["rx"] = roi["ry"] = max(5, r)
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

            mean_int = cv2.mean(frame_gray, mask=mask)[0]

            #Temporary moving average method for the ROIS
            moving_avg_window=15
            moving_avg=mean_int
            summands=list(roi["y"])[-moving_avg_window:]
            summand_length=len(summands)+1
            moving_avg+=np.sum(summands)
            moving_avg/=(summand_length)

         
            
            roi["t"].append(timestamp_s)
            roi["y"].append(moving_avg)

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

# ------------------- ROI Save to CSV -------------------- #

    def roi_2_csv(self):
        
        df=pd.DataFrame()

        timestamp=datetime.datetime.now()
       
        for osc_curve in self.rois:
            keys=self.rois[osc_curve].keys()
            osc_curve_values={key : self.rois[osc_curve][key] for key in keys & {'t','y'} }
            osc_curve_params={key : self.rois[osc_curve][key] for key in keys - {'t','y'}}
            temp_df=pd.DataFrame.from_dict(osc_curve_values)
            temp_df.insert(0, 'ROI Number', osc_curve)
            df=pd.concat([df,temp_df])
       

        try:
            final_output_path=self.roi_main_directory / f'{timestamp.strftime("%Y")}'/ f'{timestamp.strftime("%B")}'/f'{timestamp.strftime("%m%d%y")}'
            print(f'\n⏳ Attempting to create output directory : \"{final_output_path}\" ...')
            subdirectory_extension=final_output_path.mkdir(parents=True, exist_ok=False)
            print('✅     ... successfully created!\n')
        except FileExistsError:
            print('⚠️     ... directory already exists.\n')

        file_name=final_output_path/f'RHEED_ROI_data_{timestamp.strftime("%m-%d-%y_%H-%M-%S")}.txt'

        if file_name.exists():
            print(f'\n\n⚠️ File : \"{file_name}\" already exists.')
            file_name=final_output_path/f'RHEED_ROI_data_{timestamp.strftime("%m-%d-%y_%H-%M-%S Newer")}.txt'
            print(f'\n✅ Creating file with file name: \"{file_name}\" instead.\n')
        

        with open(file_name, 'w') as output:
            output.write('--- RHEED ROI Data ---\n\n')
            output.write(f'Timestamp: {timestamp}\n')

            output.write(f'\n---DATA START---\n')

        df.to_csv(file_name, mode='a', header=True)

        print(f'\n✅ 📈 RHEED ROI Saved to \"{file_name}\" at {timestamp}!\n')      

