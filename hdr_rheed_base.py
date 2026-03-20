import os
import sys
import cv2
import numpy as np
import time
import datetime
import threading
import queue
from threading import Lock
from camera import Camera
from dummy_camera import DummyCamera
from roi_manager import ROIManager, LineManager
from config import VIDEO_OUTPUT_DIR, ROI_OUTPUT_DIR
from pathlib import Path


HDR_MODE = True  # locked HDR pipeline (no RAW mode)

EXPOSURE_CYCLE = [5000.0, 15000.0, 45000.0]  # microseconds
GAIN_CONSTANT = 17.5  # fixed gain

merge_mertens = cv2.createMergeMertens(
    contrast_weight=0,
    saturation_weight=0,
    exposure_weight=1
)
SAVE_ONE_TRIPLET = True


# ---------------------- aspect ratio helper ----------------------
def fit_to_window(frame, screen_w, screen_h):
    h, w = frame.shape[:2]

    # ---- Guard against invalid window sizes ----
    if screen_w <= 1 or screen_h <= 1 or w <= 0 or h <= 0:
        return frame, 1.0, 0, 0

    # Compute scale
    scale = min(screen_w / w, screen_h / h)

    # Ensure positive dimensions
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # Resize safely
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded canvas
    result = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    y0 = (screen_h - new_h) // 2
    x0 = (screen_w - new_w) // 2

    result[y0:y0 + new_h, x0:x0 + new_w] = resized

    return result, scale, x0, y0


def fit_preserve_aspect(frame, target_w, target_h):
    fh, fw = frame.shape[:2]

    scale = min(target_w / fw, target_h / fh)
    new_w = int(fw * scale)
    new_h = int(fh * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas, scale, x_offset, y_offset



# ---------------------- chart rendering ----------------------
def render_chart(roi, width, height, now_s, title, y_state):
    chart = np.full((height, width, 3), 255, np.uint8)

    ml, mr, mt, mb = 100, 20, 54, 44
    x0, y0, w, h = ml, mt, width - ml - mr, height - mt - mb

    cv2.rectangle(chart, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 1)
    cv2.putText(chart, title, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

    label = "Peak Int. (Intensity)"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y_img = np.full((th + 8, tw + 8, 3), 255, np.uint8)
    cv2.putText(y_img, label, (4, th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    y_img = cv2.rotate(y_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    y_pos = y0 + h // 2 - y_img.shape[0] // 2
    chart[y_pos:y_pos + y_img.shape[0], 20:20 + y_img.shape[1]] = y_img

    if roi is None or len(roi["t"]) < 2:
        cv2.putText(chart, "(Add ROI)", (x0 + 10, y0 + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(chart, "Elapsed Time (s)", (x0 + w - 150, y0 + h + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
        return chart, (0, 0, 0, 0)

    t = np.asarray(roi["t"], dtype=float)
    y = np.asarray(roi["y"], dtype=float)

    if t.size < 2:
        return chart, (0, 0, 0, 0)

    ymin_t, ymax_t = float(y.min()), float(y.max())
    if "ymin" not in y_state:
        y_state["ymin"], y_state["ymax"] = ymin_t, ymax_t

    a = 0.15
    y_state["ymin"] = (1 - a) * y_state["ymin"] + a * ymin_t
    y_state["ymax"] = (1 - a) * y_state["ymax"] + a * ymax_t
    ymin, ymax = y_state["ymin"], y_state["ymax"]

    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0

    # Grid
    yt = np.linspace(ymin, ymax, 5)
    for i, v in enumerate(yt):
        yy = y0 + h - int((v - ymin) / (ymax - ymin) * h + 0.5)
        if i == len(yt) - 1:
            yy = max(y0 + 12, min(y0 + h - 1, yy + 8))
        cv2.line(chart, (x0, yy), (x0 + w, yy), (235, 235, 235), 1)
        cv2.putText(chart, f"{v:.1f}", (52, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 55, 55), 1, cv2.LINE_AA)

    # Time axis
    tx0, tx1 = t[0], t[-1]
    if tx1 <= tx0:
        tx1 = tx0 + 1e-3
    xt = np.linspace(tx0, tx1, 5)

    for v in xt:
        xx = x0 + int((v - tx0) / (tx1 - tx0) * w + 0.5)
        cv2.line(chart, (xx, y0), (xx, y0 + h), (235, 235, 235), 1)
        cv2.putText(chart, f"{v:.1f}", (xx - 22, y0 + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 55, 55), 1, cv2.LINE_AA)

    # waveform
    pts = np.column_stack((
        x0 + ((t - tx0) / (tx1 - tx0) * w).astype(int),
        y0 + h - (((y - ymin) / (ymax - ymin) * h).astype(int))
    )).astype(np.int32)
    cv2.polylines(chart, [pts], False, roi["color"], 1, cv2.LINE_AA)

    cv2.putText(chart, "Elapsed Time (s)", (x0 + w - 150, y0 + h + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)

    return chart, (int(y.size), float(t[-1] - t[0]), float(ymin_t), float(ymax_t))


# ---------------------- gradient ----------------------
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


# ---------------------- main ----------------------
def main():
    global HDR_MODE, GAIN_CONSTANT
    print("🚀 RHEED Dashboard (Enhanced Build)")
    selected_hdr_index = 0
    hdr_step = 500.0
    gain_step_hdr = 0.5
    pending_exposure_cycle = EXPOSURE_CYCLE.copy()
    pending_gain = GAIN_CONSTANT
    roi = ROIManager()
    line_manager = LineManager()
    
    # ---------------- Threaded runtime state ----------------
    stop_event = threading.Event()

    roi_lock = Lock()      # protects roi + line_manager mutations
    frame_lock = Lock()    # protects latest frame buffers
    log_lock = Lock()

    latest_gray = None
    latest_disp = None
    latest_now = 0.0
    latest_frame_idx = 0
    hdr_cycle_dirty = False
    # ---- HDR FPS measurement ----
    hdr_frame_counter = 0
    hdr_fps = 0.0
    hdr_last_time = time.time()
    
    
    
    
    # Buffer of frames between acquisition and processing.
    # 600 frames ≈ 20 seconds at 30 fps. Increase if you have RAM.
    frame_q = queue.Queue(maxsize=1800)


    # camera
    try:
        cam = Camera()

        #Device Link Throughput Limit check and change. May need to be adjusted if your camera/computer communication line has a different speed.
        device_link_TL=cam.cam.DeviceLinkCurrentThroughput() 
        if device_link_TL < 120000000:   
            print(f'⚠️ 📡 Camera\'s Current Device Link Throughput Limit: {device_link_TL}')
            cam.cam.DeviceLinkThroughputLimit.SetValue(120000000)
            print(f'✅ 📡Camera\'s New Device Link Throughput Limit: {cam.cam.DeviceLinkCurrentThroughput()}')
        else:
            print(f'✅ 📡Camera\'s Device Link Throughput Limit: {cam.cam.DeviceLinkCurrentThroughput()}')
            
        cam.start()
        
                # ---------- BLACKFLY FPS INIT ----------
        # ---------- BLACKFLY FPS INIT ----------
        if hasattr(cam, "cam") and hasattr(cam.cam, "AcquisitionFrameRateEnable"):
            try:
                cam.cam.AcquisitionFrameRateEnable.SetValue(True)

                max_fps = cam.cam.AcquisitionFrameRate.GetMax()
                min_fps = cam.cam.AcquisitionFrameRate.GetMin()
                print(f"📷 Blackfly FPS capability: {min_fps:.2f} – {max_fps:.2f}")

                # Lock to MAX (real camera supports ~23.10)
                TARGET_FPS = max_fps
                cam.cam.AcquisitionFrameRate.SetValue(TARGET_FPS)

                resulting = cam.cam.AcquisitionResultingFrameRate.GetValue()
                print(f"🔒 Camera locked to: {TARGET_FPS:.2f} FPS")
                print(f"📊 Resulting hardware FPS: {resulting:.2f}")

            except Exception as e:
                print("⚠️ FPS lock not supported:", e)
        
    except Exception as e:
        print(f"⚠️ No camera found, using dummy feed. ({e})")
        cam = DummyCamera()
        cam.start()

    # ---- camera settings (works for real + dummy) ----
    if hasattr(cam, "get_settings"):
        s = cam.get_settings()
    else:
        s = {}



    exposure_min = float(s.get("exposure_min_us", 10.0))
    exposure_max = float(s.get("exposure_max_us", 1_000_000.0))
    gain_min = float(s.get("gain_min_db", 0.0))
    gain_max = float(s.get("gain_max_db", 30.0))
    gamma_min = float(s.get("gamma_min", 0.25))
    gamma_max = float(s.get("gamma_max", 4.0))

    #Sets default camera values

    exposure_us=float(12000)
    gain_db=float(17.5)
    gamma_enabled=True
    gamma_val=float(1.0)

    if hasattr(cam, "set_exposure_us"):
        exposure_us = cam.set_exposure_us(exposure_us)
    if hasattr(cam, "set_gain_db"):
        gain_db = cam.set_gain_db(gain_db)
    if hasattr(cam, "set_gamma_enabled"):
        gamma_enabled = cam.set_gamma_enabled(True)
    if hasattr(cam, "set_gamma"):
        gamma_val = cam.set_gamma(gamma_val)
                

    


    gradient = True
    strength = 0.7
    frame_idx = 0
    total_logged_rows = 0
    t0 = time.time()
    
    # ---------------------- ROI CSV Logging Setup ----------------------
    session_timestamp = datetime.datetime.now()

    csv_dir = (
        ROI_OUTPUT_DIR
        / f"{session_timestamp.strftime('%Y')}"
        / f"{session_timestamp.strftime('%B')}"
        / f"{session_timestamp.strftime('%m%d%y')}"
    )

    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = csv_dir / f"RHEED_ROI_{session_timestamp.strftime('%m-%d-%y_%H-%M-%S')}.csv"

    log_buffer = []
    last_flush_time = [time.time()]
    flush_interval = 1.0  # seconds

    # Write CSV header once
    with open(csv_filename, "w") as f:
        f.write(
            "frame_idx,Time Stamp,camera_time,video_frame,roi_uuid,roi_display_id,"
            "mean_intensity,sum_intensity,area,cx,cy,rx,ry,shape\n"
        )
        
    # ---------------- Hover state ----------------
    hover_feed_info = None
    
    mouse_x = 0
    mouse_y = 0
    
    popup_mouse_x = 0
    popup_mouse_y = 0
    popup_hover_info = None

    # ML capture state only (process removed)
    ml_capture = False
    ml_writer_raw = None
    ml_writer_color = None
    raw_file = None
    color_file = None
    ml_frame_count = 0
    ml_toggle_request = False
    record_start_time = None
    record_duration = 0
    effective_fps = 0
    encoded_duration = 0
    actual_fps = 0.0
    last_gray_shape = None
    feed_scale = 1.0
    feed_x_offset = 0
    feed_y_offset = 0
    global_scale = 1.0
    global_x_offset = 0
    global_y_offset = 0



    cv2.namedWindow("RHEED Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RHEED Dashboard", 1280, 960)

    roi_monitor_created = False


  

    # mouse callback
    def mouse_cb(event, x, y, flags, _param):
        nonlocal ml_toggle_request, feed_scale, feed_x_offset, feed_y_offset
        nonlocal global_scale, global_x_offset, global_y_offset
        nonlocal mouse_x, mouse_y

        mouse_x = x
        mouse_y = y



        # 1) Reverse global scaling: window -> combined-space
        x_combined = (x - global_x_offset) / global_scale
        y_combined = (y - global_y_offset) / global_scale

        # Ignore clicks outside the combined image area (black padding)
        if x_combined < 0 or y_combined < 0:
            return

        # 2) Ignore clicks inside the left plot panel
        
        # 3) Convert to feed-panel coordinates (right side of combined)
        fx = x_combined
        fy = y_combined

        # 4) Reverse feed aspect fit: feed-canvas -> original disp coords
        fx = (fx - feed_x_offset) / feed_scale
        fy = (fy - feed_y_offset) / feed_scale

        fx = int(fx)
        fy = int(fy)


        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        
        if line_manager.draw_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                line_manager.start_drawing(fx, fy)
            elif event == cv2.EVENT_MOUSEMOVE:
                line_manager.update_drawing(fx, fy)
            elif event == cv2.EVENT_LBUTTONUP:
                line_manager.finish_drawing()
            return         

        with roi_lock:
            # existing ROI code      
            if event == cv2.EVENT_LBUTTONDOWN:
                if not roi.select_roi(fx, fy, shift=shift):
                    roi.start_drawing(fx, fy)

            elif event == cv2.EVENT_MOUSEMOVE:
                if roi.drawing:
                    roi.update_drawing(fx, fy)
                elif roi.moving:
                    roi.move_selected(fx, fy)
                elif roi.resizing:
                    roi.resize_selected(x=fx, y=fy)
                    

            elif event == cv2.EVENT_LBUTTONUP:
                if roi.drawing:
                    roi.finish_drawing()
                roi.release()

            elif event == cv2.EVENT_RBUTTONDOWN:
                roi.remove_nearest(fx, fy)

    cv2.setMouseCallback("RHEED Dashboard", mouse_cb)
    
    def popup_mouse_cb(event, x, y, flags, _param):
        nonlocal popup_mouse_x, popup_mouse_y
        popup_mouse_x = x
        popup_mouse_y = y
        
        
        # ---------------- WORKER THREAD ----------------
    def capture_loop():
        """Acquisition loop supporting RAW and HDR fusion."""
        nonlocal ml_capture, ml_writer_raw, ml_writer_color, ml_frame_count, hdr_cycle_dirty
            

        exposure_index = 0
        hdr_buffer = []
        triplet_counter = 0
        triplet_saved = False
        cycle = EXPOSURE_CYCLE.copy()

        # Track last applied gain
        # Apply initial gain to real camera immediately
        if hasattr(cam, "set_gain_db"):
            cam.set_gain_db(GAIN_CONSTANT)
        last_gain = GAIN_CONSTANT

        while not stop_event.is_set():
            # If user changed HDR exposures, restart the triplet cleanly
            if hdr_cycle_dirty:

                cycle = EXPOSURE_CYCLE.copy()
                hdr_buffer.clear()
                exposure_index = 0

                # Re-apply current gain after cycle reset
                if hasattr(cam, "set_gain_db"):
                    cam.set_gain_db(GAIN_CONSTANT)

                # Flush camera pipeline (remove old exposure frames)
                for _ in range(6):
                    cam.get_frame()

                # Clear queued frames waiting for processing
                try:
                    while True:
                        frame_q.get_nowait()
                except queue.Empty:
                    pass

                hdr_cycle_dirty = False
                continue

            # Cycle HDR exposures
            # ---------- HDR Exposure Capture ----------
            current_exposure = cycle[exposure_index]

            if hasattr(cam, "set_exposure_us"):
                cam.set_exposure_us(current_exposure)

            # Use old anti-strobing exposure-settle logic
            discard_count = 3
            for _ in range(discard_count):
                cam.get_frame()

            frame = cam.get_frame()
            
            if frame is None:
                continue

            now = time.time() - t0

            # convert safely to uint8
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                frame = frame.astype(np.uint8)


            frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            hdr_buffer.append(frame)

            # downscale for HDR fusion speed
            

            exposure_index = (exposure_index + 1) % len(cycle)

            # wait until we have a full HDR triplet
            if len(hdr_buffer) < len(cycle):
                continue

            triplet_counter += 1

            # ---- Fuse ----

            imgs = []

            for img, exp in zip(hdr_buffer, cycle):
                img_f = img.astype(np.float32) / 255.0

                # Old anti-strobing logic: compensate by exposure before fusion
                img_f = img_f / (exp / max(cycle))

                imgs.append(np.clip(img_f, 0, 1))

            # Merge exposures
            fused = merge_mertens.process(imgs)

            # Old anti-strobing logic: normalize each fused triplet to full display range
            fused = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX)
            fused_8 = fused.astype(np.uint8)

            # Old anti-strobing logic: strong temporal smoothing
            if not hasattr(capture_loop, "prev_hdr"):
                capture_loop.prev_hdr = fused_8

            alpha = 0.7

            fused_8 = cv2.addWeighted(
                capture_loop.prev_hdr,
                alpha,
                fused_8,
                1 - alpha,
                0
            )

            capture_loop.prev_hdr = fused_8

            output_frame = fused_8            

            # ---- SAVE ONLY ONE TRIPLET ----
            # ---- SAVE ONLY ONE TRIPLET ----
            if SAVE_ONE_TRIPLET and not triplet_saved and triplet_counter > 10:

                triplet_dir = Path(__file__).resolve().parent / "hdr_exposure_triplet"
                triplet_dir.mkdir(exist_ok=True)

                raw_triplet_dir = triplet_dir / "raw_sensor"
                raw_triplet_dir.mkdir(exist_ok=True)

                e1, e2, e3 = [int(x) for x in cycle]

                # Remove old color files
                for fname in [f"exp_{e1}us.png", f"exp_{e2}us.png",
                               f"exp_{e3}us.png", "fused_hdr.png"]:
                    p = triplet_dir / fname
                    if p.exists():
                        p.unlink()

                # Remove old raw sensor files
                for fname in [f"raw_{e1}us.png", f"raw_{e2}us.png", f"raw_{e3}us.png"]:
                    p = raw_triplet_dir / fname
                    if p.exists():
                        p.unlink()

                # Save true raw sensor frames (grayscale, no LUT)
                cv2.imwrite(str(raw_triplet_dir / f"raw_{e1}us.png"), hdr_buffer[0])
                cv2.imwrite(str(raw_triplet_dir / f"raw_{e2}us.png"), hdr_buffer[1])
                cv2.imwrite(str(raw_triplet_dir / f"raw_{e3}us.png"), hdr_buffer[2])

                # Save color LUT versions
                exp1_color = apply_gradient(hdr_buffer[0], strength=strength)
                exp2_color = apply_gradient(hdr_buffer[1], strength=strength)
                exp3_color = apply_gradient(hdr_buffer[2], strength=strength)

                cv2.imwrite(str(triplet_dir / f"exp_{e1}us.png"), exp1_color)
                cv2.imwrite(str(triplet_dir / f"exp_{e2}us.png"), exp2_color)
                cv2.imwrite(str(triplet_dir / f"exp_{e3}us.png"), exp3_color)

                # Save fused with gradient (COLOR)
                fused_color = apply_gradient(fused_8, strength=strength)
                cv2.imwrite(str(triplet_dir / "fused_hdr.png"), fused_color)

                # Save fused raw grayscale too
                cv2.imwrite(str(raw_triplet_dir / "fused_hdr_raw.png"), fused_8)

                print("✅ Saved ONE HDR triplet")
                print(f"   Color LUT → hdr_exposure_triplet/")
                print(f"   Raw sensor → hdr_exposure_triplet/raw_sensor/")
                triplet_saved = True
                
            hdr_buffer.clear()                

            # ---------------- Recording ----------------
            if ml_capture:
                try:
                    # output_frame is already fused grayscale uint8
                    raw_frame = fused_8.copy()

                    color_frame = apply_gradient(raw_frame, strength=strength)
                    color_frame = color_frame.astype(np.uint8)

                    frame_written = False

                    expected_h, expected_w = last_gray_shape if last_gray_shape else raw_frame.shape[:2]

                    if ml_writer_raw is not None and ml_writer_raw.isOpened():
                        if raw_frame.shape[0] == expected_h and raw_frame.shape[1] == expected_w:
                            ml_writer_raw.write(raw_frame)
                            frame_written = True

                    if ml_writer_color is not None and ml_writer_color.isOpened():
                        if color_frame.shape[0] == expected_h and color_frame.shape[1] == expected_w:
                            ml_writer_color.write(color_frame)
                            frame_written = True

                    if frame_written:
                        ml_frame_count += 1

                except Exception as e:
                    print("Video write warning:", e)

            try:
                frame_q.put_nowait((output_frame, now))
            except queue.Full:
                pass


    def process_loop():
        """All heavier work happens here: ROI, overlays, logging, recording, UI buffers."""
        nonlocal frame_idx, total_logged_rows
        nonlocal latest_gray, latest_disp, latest_now, latest_frame_idx
        nonlocal ml_capture, ml_writer_raw, ml_writer_color, ml_frame_count, last_gray_shape
        nonlocal hdr_frame_counter, hdr_fps, hdr_last_time

        while not stop_event.is_set():
            # Apply gain if user changed it
            try:
                frame, now = frame_q.get(timeout=0.2)
            except queue.Empty:
                continue

            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            last_gray_shape = gray.shape[:2]

            with roi_lock:
                roi.update_intensities(gray, now)

                for rid, r in roi.rois.items():
                    if len(r["t"]) == 0:
                        continue

                    mean_int = r.get("last_raw_mean", 0.0)
                    sum_int = r.get("last_sum", 0.0)
                    area = r.get("last_area", 0)

                    cx, cy = r["center"]
                    rx, ry = r["rx"], r["ry"]
                    shape = r["shape"]

                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    camera_time = f"{now:.6f}"
                    video_frame = ml_frame_count if ml_capture else "NaN"

                    log_buffer.append(
                        f"{frame_idx+1},{timestamp},{camera_time},{video_frame},{r.get('uuid','NA')},{rid},"
                        f"{mean_int:.6f},{sum_int},{area},{cx},{cy},{rx},{ry},{shape}\n"
                    )
                    
                    total_logged_rows += 1

                if line_manager.pt1 and line_manager.pt2:
                    line_manager.extract_profile(gray, now)

            disp = apply_gradient(gray, strength=strength) \
                if gradient else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Draw overlays on local disp copy — no lock needed
            roi.draw_overlays(disp)
            line_manager.draw_overlay(disp)

            with frame_lock:
                frame_idx += 1
                latest_gray = gray
                latest_disp = disp
                latest_now = now
                latest_frame_idx = frame_idx

            # ---- HDR FPS measurement (outside lock) ----
            hdr_frame_counter += 1
            elapsed = time.time() - hdr_last_time
            if elapsed >= 1.0:
                hdr_fps = hdr_frame_counter / elapsed
                hdr_frame_counter = 0
                hdr_last_time = time.time()

            # Flush CSV occasionally
            if time.time() - last_flush_time[0] > flush_interval:
                with log_lock:
                    if log_buffer:
                        with open(csv_filename, "a") as f:
                            f.writelines(log_buffer)
                        log_buffer.clear()
                        last_flush_time[0] = time.time()
            
    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    proc_thread = threading.Thread(target=process_loop, daemon=True)

    cap_thread.start()
    proc_thread.start()
    print("Frame grabbed")        
    
    try:
        while True:
            with frame_lock:
                if latest_disp is None or latest_gray is None:
                    time.sleep(0.005)
                    continue

                disp = latest_disp.copy()
                gray = latest_gray
                now = latest_now
                frame_idx = latest_frame_idx
            
        


                        # -------- Camera settings overlay --------
            # -------- Camera settings overlay --------
            hw = getattr(cam, "has_hw_control", False)

            overlay = [
                f"Live Feed: {'YES' if hw else 'NO (Dummy)'}",
                f"Active HDR:  {int(EXPOSURE_CYCLE[0])} | {int(EXPOSURE_CYCLE[1])} | {int(EXPOSURE_CYCLE[2])} us",
                f"Pending HDR: {int(pending_exposure_cycle[0])} | {int(pending_exposure_cycle[1])} | {int(pending_exposure_cycle[2])} us",
                f"Active Gain: {GAIN_CONSTANT:.2f} dB  |  Pending Gain: {pending_gain:.2f} dB",
                f"Gamma: {'ON' if gamma_enabled else 'OFF'}  {gamma_val:.2f}",
                "Keys: 7/8/9 Exp | +/- Adjust | U Apply | G/F Gain Pending | K/J Gamma | H Toggle"
            ]
                
            y = 22
            for line in overlay:
                cv2.putText(
                    disp, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (230, 230, 230), 2, cv2.LINE_AA
                )
                y += 35



            mode_text = "HDR MODE"
            cv2.putText(disp, mode_text, (disp.shape[1] - 180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
            
            # -------- RECORDING INDICATOR --------
            if ml_capture:
                rec_elapsed = time.time() - record_start_time if record_start_time else 0

                # Red REC text
                cv2.putText(
                    disp,
                    "[REC]",
                    (disp.shape[1] - 180, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                # Timer
                cv2.putText(
                    disp,
                    f"{rec_elapsed:.1f} sec",
                    (disp.shape[1] - 180, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                # Frame counter
                cv2.putText(
                    disp,
                    f"Frames: {ml_frame_count}",
                    (disp.shape[1] - 180, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            

            # footer bar
            # footer bar
            # ---- Get correct hardware FPS display ----
            if hasattr(cam, "cam") and hasattr(cam.cam, "AcquisitionResultingFrameRate"):
                hw_fps_display = cam.cam.AcquisitionResultingFrameRate.GetValue()
            else:
                hw_fps_display = cam.get_fps()

            fps_text = f"Cam FPS: {hw_fps_display:.2f} | HDR FPS: {hdr_fps:.2f}"
            # ---- Compute footer size dynamically ----
            footer_text_sample = "Pixel Int: 000.0 | Cam FPS: 000.00 | HDR FPS: 00.00 | Frame: 0000"

            (text_w, text_h), baseline = cv2.getTextSize(
                footer_text_sample,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                2
            )

            FOOTER_HEIGHT = text_h + baseline + 20

            feed_bar = np.zeros((FOOTER_HEIGHT, disp.shape[1], 3), np.uint8)

            # ---- Draw actual footer text ----
            cv2.putText(
                feed_bar,
                f"Pixel Int: {float(np.mean(gray)):.1f} | {fps_text} | Frame: {frame_idx}",
                (10, text_h + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (230, 230, 230),
                2,
                cv2.LINE_AA
            )

            # Determine available height for feed (excluding footer)
            # Determine available height for feed (excluding footer)
            feed_height = disp.shape[0]
            feed_width = disp.shape[1]

            # Apply aspect-preserving scaling
            disp_fitted, feed_scale, feed_x_offset, feed_y_offset = fit_preserve_aspect(
                disp, feed_width, feed_height
            )

            right_block = np.vstack([disp_fitted, feed_bar])


            combined = right_block
            

            # Get current window size
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect("RHEED Dashboard")
            except:
                win_w, win_h = 1280, 960

            # Render dashboard to match window exactly
            display, global_scale, global_x_offset, global_y_offset = fit_to_window(combined, win_w, win_h)
            
            # ---------------- LIVE HOVER COMPUTATION ----------------

            hover_feed_info = None

            x_combined = (mouse_x - global_x_offset) / global_scale
            y_combined = (mouse_y - global_y_offset) / global_scale

            fx = (x_combined - feed_x_offset) / feed_scale
            fy = (y_combined - feed_y_offset) / feed_scale

            fx = int(fx)
            fy = int(fy)

            with roi_lock:
                roi_items = list(roi.rois.items())

            for rid, r in roi_items:
                cx, cy = r["center"]
                rx, ry = r["rx"], r["ry"]

                inside = False

                if r["shape"] == "ellipse":
                    if rx > 0 and ry > 0:
                        inside = ((fx - cx) ** 2) / (rx ** 2) + ((fy - cy) ** 2) / (ry ** 2) <= 1
                else:
                    inside = (cx - rx <= fx <= cx + rx) and (cy - ry <= fy <= cy + ry)

                if inside:
                    mask = np.zeros_like(gray, dtype=np.uint8)

                    if r["shape"] == "ellipse":
                        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
                    else:
                        cv2.rectangle(mask,
                                    (cx - rx, cy - ry),
                                    (cx + rx, cy + ry),
                                    255, -1)

                    sum_int = int(np.sum(gray[mask == 255]))
                    hover_feed_info = (rid, sum_int, mouse_x, mouse_y)
                    break
                        
            # ---------------- Hover Tooltips ----------------

            
            if hover_feed_info:
                rid, sum_int, mx, my = hover_feed_info
                txt = f"ROI {rid} SUM = {sum_int}"

                (tw, th), _ = cv2.getTextSize(txt,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.45, 1)

                pad = 6
                box_w = tw + pad * 2
                box_h = th + pad * 2

                bx = mx + 12
                by = my - box_h - 8

                H, W = display.shape[:2]

                if bx + box_w > W:
                    bx = W - box_w - 5
                if bx < 0:
                    bx = 5
                if by < 0:
                    by = my + 12
                if by + box_h > H:
                    by = H - box_h - 5

                cv2.rectangle(display,
                            (bx, by),
                            (bx + box_w, by + box_h),
                            (0, 0, 0), -1)

                cv2.putText(display, txt,
                            (bx + pad, by + box_h - pad),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1, cv2.LINE_AA)


            cv2.imshow("RHEED Dashboard", display)
            
            
            with roi_lock:    
                if line_manager.pt1 and line_manager.pt2:
                    line_manager.render_window()


            key = cv2.waitKey(1) & 0xFF
            window_status=cv2.getWindowProperty("RHEED Dashboard", cv2.WND_PROP_VISIBLE)
            
            
            # ----- ROI Monitor Popout -----
            with roi_lock:
                extra_rois = sorted(roi.rois.keys())

            if len(extra_rois) > 0:

                if not roi_monitor_created:
                    cv2.namedWindow("RHEED ROI Monitor", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("RHEED ROI Monitor", 1200, 700)
                    cv2.setMouseCallback("RHEED ROI Monitor", popup_mouse_cb)
                    roi_monitor_created = True

                max_extra = 6
                extra_rois = extra_rois[:max_extra]

                cols = 3
                rows = 2

                popup_w = 1200
                popup_h = 700

                cell_w = popup_w // cols
                cell_h = popup_h // rows

                popup = np.zeros((popup_h, popup_w, 3), dtype=np.uint8)

                for idx, rid in enumerate(extra_rois):
                    r = idx // cols
                    c = idx % cols

                    chart, stats = render_chart(
                        roi.rois.get(rid),
                        cell_w,
                        cell_h - 22,
                        now,
                        f"ROI {rid}",
                        {}
                    )

                    footer = np.full((22, cell_w, 3), 235, np.uint8)
                    text = f"Data Pt:{stats[0]:>4} | Elapsed:{stats[1]:>6.1f}s | Min:{stats[2]:>6.1f} | Max:{stats[3]:>6.1f}"
                    cv2.putText(footer, text, (6, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (60, 60, 60), 1, cv2.LINE_AA)

                    cell = np.vstack([chart, footer])

                    y0 = r * cell_h
                    x0 = c * cell_w

                    popup[y0:y0 + cell_h, x0:x0 + cell_w] = cell

                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect("RHEED ROI Monitor")
                except:
                    win_w, win_h = popup_w, popup_h

                if win_w > 0 and win_h > 0:

                    popup_display, popup_scale, popup_x_offset, popup_y_offset = \
                        fit_to_window(popup, win_w, win_h)

                    # -------- Popup Hover Computation --------
                    popup_hover_info = None

                    mx = (popup_mouse_x - popup_x_offset) / popup_scale
                    my = (popup_mouse_y - popup_y_offset) / popup_scale

                    if 0 <= mx < popup_w and 0 <= my < popup_h:

                        col = int(mx // cell_w)
                        row = int(my // cell_h)
                        idx = row * 3 + col

                        if idx < len(extra_rois):

                            rid = extra_rois[idx]

                            with roi_lock:
                                roi_data = roi.rois.get(rid)

                                if roi_data and len(roi_data["t"]) > 1:

                                    chart_h = cell_h - 22
                                    chart_y0 = row * cell_h
                                    chart_x0 = col * cell_w

                                    ml, mr, mt, mb = 100, 20, 54, 44
                                    plot_w = cell_w - ml - mr
                                    plot_h = chart_h - mt - mb

                                    px = mx - chart_x0 - ml
                                    py = my - chart_y0 - mt

                                    if 0 <= px <= plot_w and 0 <= py <= plot_h:

                                        t_arr = np.array(roi_data["t"])
                                        y_arr = np.array(roi_data["y"])

                                        tx0, tx1 = t_arr[0], t_arr[-1]
                                        frac = px / plot_w
                                        target_time = tx0 + frac * (tx1 - tx0)

                                        idx2 = np.argmin(np.abs(t_arr - target_time))

                                        popup_hover_info = (
                                            rid,
                                            float(t_arr[idx2]),
                                            float(y_arr[idx2]),
                                            popup_mouse_x,
                                            popup_mouse_y
                                        )

                    # -------- Draw Tooltip --------
                    if popup_hover_info:
                        rid, tval, yval, mxw, myw = popup_hover_info
                        txt = f"ROI {rid} | t={tval:.2f}s | I={yval:.2f}"

                        (tw, th), _ = cv2.getTextSize(txt,
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.45, 1)

                        pad = 6
                        box_w = tw + pad * 2
                        box_h = th + pad * 2

                        bx = mxw + 12
                        by = myw - box_h - 8

                        H, W = popup_display.shape[:2]

                        if bx + box_w > W:
                            bx = W - box_w - 5
                        if bx < 0:
                            bx = 5
                        if by < 0:
                            by = myw + 12
                        if by + box_h > H:
                            by = H - box_h - 5

                        cv2.rectangle(popup_display,
                                    (bx, by),
                                    (bx + box_w, by + box_h),
                                    (30,30,30), -1)

                        cv2.putText(popup_display, txt,
                                    (bx + pad, by + box_h - pad),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (255,255,255), 1, cv2.LINE_AA)

                    cv2.imshow("RHEED ROI Monitor", popup_display)

            else:
                if roi_monitor_created:
                    cv2.destroyWindow("RHEED ROI Monitor")
                    roi_monitor_created = False





            # quit
            if key == ord('q') or key == ord('Q') or window_status==0:
                break
                

            # gradient toggle
            elif key == ord('c'):
                gradient = not gradient

            # ROI resizing
            elif key == ord('.'):
                roi.resize_selected(+5)
            elif key == ord(','):
                roi.resize_selected(-5)

            # reset ROIs
            elif key == ord('r'):
                with log_lock:
                    if log_buffer:
                        with open(csv_filename, "a") as f:
                            f.writelines(log_buffer)
                        log_buffer.clear()

                roi.reset()
                

            
            # line toggle
            elif key == ord('l') or key == ord('L'):
                line_manager.toggle()
            
            elif key == ord('x') or key == ord('X'):
                line_manager.clear_line()
    

            # shape toggles
            elif key == ord('e')or  key== ord('E'):
                roi.toggle_shape("ellipse")
            elif key == ord('t') or key==ord('T'):
                roi.toggle_shape("rect")

        
            

            # capture toggle
            elif key == ord('m') or key==ord('M'):
                ml_toggle_request = True
                
            
            # -------- HDR Exposure Editing --------
            if HDR_MODE:

                if key == ord('7'):
                    selected_hdr_index = 0
                    print("Editing HDR Exposure 1")

                elif key == ord('8'):
                    selected_hdr_index = 1
                    print("Editing HDR Exposure 2")

                elif key == ord('9'):
                    selected_hdr_index = 2
                    print("Editing HDR Exposure 3")

                elif key == ord('+') or key == ord('='):
                    pending_exposure_cycle[selected_hdr_index] += hdr_step
                    print("Pending HDR exposures:", pending_exposure_cycle)

                elif key == ord('-'):
                    pending_exposure_cycle[selected_hdr_index] = max(
                        1000.0,
                        pending_exposure_cycle[selected_hdr_index] - hdr_step
                    )
                    print("Pending HDR exposures:", pending_exposure_cycle) 

                elif key == ord('u') or key == ord('U'):
                    EXPOSURE_CYCLE[:] = pending_exposure_cycle
                    GAIN_CONSTANT = pending_gain
                    if hasattr(cam, "set_gain_db"):
                        cam.set_gain_db(GAIN_CONSTANT)
                    hdr_cycle_dirty = True
                    print(f"✅ Applied HDR exposures: {EXPOSURE_CYCLE}")
                    print(f"✅ Applied Gain: {GAIN_CONSTANT:.2f} dB")
                    
                
                # ----- Gain adjustment — pending system (press U to apply) -----
                elif key == ord('g'):
                    pending_gain = min(gain_max, pending_gain + gain_step_hdr)
                    print(f"Pending Gain: {pending_gain:.2f} dB  (press U to apply)")

                elif key == ord('f'):
                    pending_gain = max(gain_min, pending_gain - gain_step_hdr)
                    print(f"Pending Gain: {pending_gain:.2f} dB  (press U to apply)")  


                # -------- Gamma control (same as main.py) --------
                gamma_step = 0.05

                if key == ord('k') or key == ord('K'):  # gamma up
                    gamma_val = min(gamma_max, gamma_val + gamma_step)
                    if hasattr(cam, "set_gamma"):
                        gamma_val = cam.set_gamma(gamma_val)
                    gamma_enabled = True
                    if hasattr(cam, "set_gamma_enabled"):
                        gamma_enabled = cam.set_gamma_enabled(True)
                    print(f"Gamma increased → {gamma_val:.2f}")

                elif key == ord('j') or key == ord('J'):  # gamma down
                    gamma_val = max(gamma_min, gamma_val - gamma_step)
                    if hasattr(cam, "set_gamma"):
                        gamma_val = cam.set_gamma(gamma_val)
                    gamma_enabled = True
                    if hasattr(cam, "set_gamma_enabled"):
                        gamma_enabled = cam.set_gamma_enabled(True)
                    print(f"Gamma decreased → {gamma_val:.2f}")

                elif key == ord('h') or key == ord('H'):  # gamma toggle
                    gamma_enabled = not gamma_enabled
                    if hasattr(cam, "set_gamma_enabled"):
                        gamma_enabled = cam.set_gamma_enabled(gamma_enabled)
                    print(f"Gamma enabled: {gamma_enabled}")
                                
            

            
            
            # handle capture toggle
            if ml_toggle_request:
                ml_toggle_request = False

                if not ml_capture:
                    if last_gray_shape is None:
                        print("⚠️ Cannot start ML capture: no frame yet.")
                    else:
                        timestamp = datetime.datetime.now()

                        final_output_path = (
                            VIDEO_OUTPUT_DIR
                            / f"{timestamp.strftime('%Y')}"
                            / f"{timestamp.strftime('%B')}"
                            / f"{timestamp.strftime('%m%d%y')}"
                        )

                        final_output_path.mkdir(parents=True, exist_ok=True)

                        raw_dir = final_output_path / "raw"
                        color_dir = final_output_path / "color"
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        color_dir.mkdir(parents=True, exist_ok=True)

                        raw_file = raw_dir / f"RHEED_HDR_video_{timestamp.strftime('%m-%d-%y_%H-%M-%S')}_raw.avi"
                        color_file = color_dir / f"RHEED_HDR_video_{timestamp.strftime('%m-%d-%y_%H-%M-%S')}_color.avi"


                        h, w = last_gray_shape
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")

                        actual_fps = max(3.0, hdr_fps)
                        print(f"🎥 Recording using logical FPS: {actual_fps:.2f}")

                        ml_writer_raw = cv2.VideoWriter(str(raw_file), fourcc, actual_fps, (w, h), False)
                        ml_writer_color = cv2.VideoWriter(str(color_file), fourcc, actual_fps, (w, h), True)

                        if not ml_writer_raw.isOpened() or not ml_writer_color.isOpened():
                            print("❌ Failed to open VideoWriter for ML capture.")
                            ml_writer_raw = None
                            ml_writer_color = None
                        else:
                            ml_capture = True
                            ml_frame_count = 0
                            record_start_time = time.time()
                            print(f"📥 ML capture STARTED")
                            print(f"   RAW   → {raw_file}")
                            print(f"   COLOR → {color_file}")
                else:
                    # stop capture (NO ML PROCESSING)
                    if ml_writer_raw is not None:
                        ml_writer_raw.release()

                    if ml_writer_color is not None:
                        ml_writer_color.release()

                        record_duration = time.time() - record_start_time if record_start_time else 0
                        encoded_duration = ml_frame_count / actual_fps if actual_fps > 0 else 0
                        effective_fps = ml_frame_count / record_duration if record_duration > 0 else 0

                        print(f"\n📤 ML capture STOPPED")
                        print(f"   RAW   → {raw_file}")
                        print(f"   COLOR → {color_file}")
                        print(f"   ⏱ Real recording time: {record_duration:.2f} sec")
                        print(f"   🎞 Frames written: {ml_frame_count}")
                        print(f"   📊 Effective FPS: {effective_fps:.2f}")
                        print(f"   📁 Encoded video duration: {encoded_duration:.2f} sec\n")

                    ml_writer_raw = None
                    ml_writer_color = None
                    ml_capture = False
                    ml_frame_count = 0
                    raw_file = None
                    color_file = None
    finally:
        stop_event.set()
        try:
            cap_thread.join(timeout=2.0)
            proc_thread.join(timeout=2.0)
        except:
            pass
        
        if ml_writer_raw is not None:
            ml_writer_raw.release()

        if ml_writer_color is not None:
            ml_writer_color.release()

        if raw_file is not None:
            print(f"\n📤 ML capture STOPPED on exit.")
            print(f"   RAW   → {raw_file}")
            print(f"   COLOR → {color_file}")
            print(f"   Saved {ml_frame_count} frames.")
            
            
        # ---------------------- Final CSV Flush ----------------------
        total_rows_written = 0
        with log_lock:
            if log_buffer:
                with open(csv_filename, "a") as f:
                    f.writelines(log_buffer)
                total_rows_written = len(log_buffer)
                log_buffer.clear()

            print("\n📁 ROI logging session closed.")
            print(f"   ➤ File: {csv_filename.resolve()}")
            print(f"   ➤ Total rows written (entire session): {total_logged_rows}")

            

        

if __name__ == "__main__":
    main()
