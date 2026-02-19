import os
import sys
import cv2
import numpy as np
import time
import datetime
from camera import Camera
from dummy_camera import DummyCamera
from roi_manager import ROIManager, LineManager
from config import VIDEO_OUTPUT_DIR
from pathlib import Path




# ---------------------- aspect ratio helper ----------------------
def fit_to_window(frame, screen_w, screen_h):
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
        arr = np.load(lut)
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
    print("🚀 RHEED Dashboard (Enhanced Build)")
    roi = ROIManager()
    line_manager = LineManager()


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
    show_plot = True
    frame_idx = 0
    t0 = time.time()
    

    # ML capture state only (process removed)
    ml_capture = False
    ml_writer = None
    ml_filename = None
    ml_frame_count = 0
    ml_toggle_request = False
    last_gray_shape = None
    feed_scale = 1.0
    feed_x_offset = 0
    feed_y_offset = 0
    global_scale = 1.0
    global_x_offset = 0
    global_y_offset = 0


    capture_button_rect = [0, 0, 0, 0]

    y_anim_1, y_anim_2 = {}, {}
    cv2.namedWindow("RHEED Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RHEED Dashboard", 1280, 960)

    roi_monitor_created = False


  

    # mouse callback
    def mouse_cb(event, x, y, flags, _param):
        nonlocal ml_toggle_request, feed_scale, feed_x_offset, feed_y_offset, global_scale, global_x_offset, global_y_offset


        if show_plot and event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = capture_button_rect
            if x1 < x < x2 and y1 < y < y2:
                ml_toggle_request = True
                return

        if show_plot and x < panel_w:
            return

        # 1) Reverse global scaling: window -> combined-space
        x_combined = (x - global_x_offset) / global_scale
        y_combined = (y - global_y_offset) / global_scale

        # Ignore clicks outside the combined image area (black padding)
        if x_combined < 0 or y_combined < 0:
            return

        # 2) Ignore clicks inside the left plot panel
        if show_plot and x_combined < panel_w:
            return

        # 3) Convert to feed-panel coordinates (right side of combined)
        fx = x_combined - (panel_w if show_plot else 0)
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

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            now = time.time() - t0
            frame_idx += 1
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            last_gray_shape = gray.shape[:2]
            roi.update_intensities(gray, now)
            if line_manager.pt1 and line_manager.pt2:
                line_manager.extract_profile(gray, now)


            # ML capture only (no ML process)
            if ml_capture and ml_writer is not None:
                colored_frame = apply_gradient(gray, strength=strength)
                ml_writer.write(colored_frame)
                ml_frame_count += 1


            disp = apply_gradient(gray, strength=strength) \
                if gradient else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            roi.draw_overlays(disp)
            line_manager.draw_overlay(disp)


                        # -------- Camera settings overlay --------
            hw = getattr(cam, "has_hw_control", False)
            overlay = [
                f"Live Feed: {'YES' if hw else 'NO (Dummy)'}",
                f"Exposure: {exposure_us:.0f} us",
                f"Gain:     {gain_db:.2f} dB",
                f"Gamma:    {'ON' if gamma_enabled else 'OFF'}  {gamma_val:.2f}",
                "Keys: [ or ] Expos;  - or = Gain;  j or k Gamma;  g Toggle"
            ]

            y = 22
            for line in overlay:
                cv2.putText(
                    disp, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (230, 230, 230), 2, cv2.LINE_AA
                )
                y += 35



            mode_text = "FULL FEED" if not show_plot else "DASHBOARD"
            cv2.putText(disp, mode_text, (disp.shape[1] - 180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

            if show_plot:
                H = disp.shape[0]
                top_h, bot_h = H // 2, H - H // 2
                panel_w=disp.shape[1] // 2 #Used to change the ROI plot width on the main window
                chart1, stats1 = render_chart(roi.rois.get(1), panel_w, top_h, now, "ROI 1", y_anim_1)
                chart2, stats2 = render_chart(roi.rois.get(2), panel_w, bot_h, now, "ROI 2", y_anim_2)

                def make_footer(stats):
                    bar = np.full((22, panel_w, 3), 235, np.uint8)
                    text = f"Data Pt:{stats[0]:>4} | Elapsed:{stats[1]:>6.1f}s | Min:{stats[2]:>6.1f} | Max:{stats[3]:>6.1f}"
                    cv2.putText(bar, text, (6, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)
                    return bar

                left_block = np.vstack([
                    chart1, make_footer(stats1),
                    chart2, make_footer(stats2)
                ])

            else:
                left_block = None

            # footer bar
            # footer bar
            feed_bar = np.zeros((24, disp.shape[1], 3), np.uint8)
            cv2.putText(feed_bar,
                        f"Pixel Int: {float(np.mean(gray)):.1f} | FPS: {np.round(cam.get_fps(), 2)} | Frame: {frame_idx}",
                        (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        .85, (230, 230, 230), 2, cv2.LINE_AA)

            # Determine available height for feed (excluding footer)
            feed_height = disp.shape[0]
            feed_width = disp.shape[1]

            # Apply aspect-preserving scaling
            disp_fitted, feed_scale, feed_x_offset, feed_y_offset = fit_preserve_aspect(
                disp, feed_width, feed_height
            )

            right_block = np.vstack([disp_fitted, feed_bar])


            if show_plot:
                total_h = max(left_block.shape[0], right_block.shape[0])

                if left_block.shape[0] < total_h:
                    left_block = np.vstack([
                        left_block,
                        np.full((total_h - left_block.shape[0], panel_w, 3), 255, np.uint8)
                    ])

                if right_block.shape[0] < total_h:
                    right_block = np.vstack([
                        right_block,
                        np.zeros((total_h - right_block.shape[0], right_block.shape[1], 3), np.uint8)
                    ])

                combined = np.hstack([left_block, right_block])

                # ML CAPTURE button (still used for recording only)
                btn_w, btn_h = 140, 26
                dashboard_x = disp.shape[1] - 180
                dashboard_y = 30

                bx0 = panel_w + dashboard_x - btn_w - 20
                by0 = dashboard_y - 12
                bx1 = bx0 + btn_w
                by1 = by0 + btn_h

                capture_button_rect = [bx0, by0, bx1, by1]

                btn_color = (0, 160, 0) if not ml_capture else (0, 0, 200)
                label = "ML CAPTURE" if not ml_capture else "STOP CAPTURE"

                cv2.rectangle(combined, (bx0, by0), (bx1, by1), btn_color, -1)
                cv2.putText(combined, label, (bx0 + 6, by0 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            else:
                combined = right_block
                capture_button_rect = [0, 0, 0, 0]

            # Get current window size
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect("RHEED Dashboard")
            except:
                win_w, win_h = 1280, 960

            # Render dashboard to match window exactly
            display, global_scale, global_x_offset, global_y_offset = fit_to_window(combined, win_w, win_h)


            cv2.imshow("RHEED Dashboard", display)
            if line_manager.pt1 and line_manager.pt2:
                line_manager.render_window()


            key = cv2.waitKey(1) & 0xFF
            window_status=cv2.getWindowProperty("RHEED Dashboard", cv2.WND_PROP_VISIBLE)
            
            
            # ----- ROI Monitor Popout -----
            extra_rois = sorted(roi.rois.keys())[2:]

            if len(extra_rois) > 0:

                if not roi_monitor_created:
                    cv2.namedWindow("RHEED ROI Monitor", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("RHEED ROI Monitor", 1200, 700)
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

                # Scale like main dashboard
                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect("RHEED ROI Monitor")
                except:
                    win_w, win_h = popup_w, popup_h

                if win_w > 0 and win_h > 0:
                    popup_display, _, _, _ = fit_to_window(popup, win_w, win_h)
                    cv2.imshow("RHEED ROI Monitor", popup_display)
                
                






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
                roi.reset()
                y_anim_1.clear()
                y_anim_2.clear()

            # toggle full view
            elif key in (ord('f'), ord('F')):
                show_plot = not show_plot
            
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

            

                # fullscreen ROI view
            elif key == ord('o') or key== ord('O'):
                roi.roi_2_csv()
            

            # capture toggle
            elif key == ord('m') or key==ord('O'):
                ml_toggle_request = True

                        # -------- Camera controls (Exposure / Gain / Gamma) --------
            exp_step = 1000.0    # microseconds
            gain_step = 0.5      # dB
            gamma_step = 0.05    # unitless

            if key == ord(']'):   # exposure up
                exposure_us = min(exposure_max, exposure_us + exp_step)
                if hasattr(cam, "set_exposure_us"):
                    exposure_us = cam.set_exposure_us(exposure_us)

            elif key == ord('['): # exposure down
                exposure_us = max(exposure_min, exposure_us - exp_step)
                if hasattr(cam, "set_exposure_us"):
                    exposure_us = cam.set_exposure_us(exposure_us)

            elif key == ord('='): # gain up
                gain_db = min(gain_max, gain_db + gain_step)
                if hasattr(cam, "set_gain_db"):
                    gain_db = cam.set_gain_db(gain_db)

            elif key == ord('-'): # gain down
                gain_db = max(gain_min, gain_db - gain_step)
                if hasattr(cam, "set_gain_db"):
                    gain_db = cam.set_gain_db(gain_db)

            elif key == ord('k') or key==ord('K'): # gamma up
                gamma_val = min(gamma_max, gamma_val + gamma_step)
                if hasattr(cam, "set_gamma"):
                    gamma_val = cam.set_gamma(gamma_val)
                gamma_enabled = True
                if hasattr(cam, "set_gamma_enabled"):
                    gamma_enabled = cam.set_gamma_enabled(True)

            elif key == ord('j') or key == ord('J'): # gamma down
                gamma_val = max(gamma_min, gamma_val - gamma_step)
                if hasattr(cam, "set_gamma"):
                    gamma_val = cam.set_gamma(gamma_val)
                gamma_enabled = True
                if hasattr(cam, "set_gamma_enabled"):
                    gamma_enabled = cam.set_gamma_enabled(True)

            elif key == ord('g') or key ==ord('G'): # toggle gamma enable
                gamma_enabled = not gamma_enabled
                if hasattr(cam, "set_gamma_enabled"):
                    gamma_enabled = cam.set_gamma_enabled(gamma_enabled)
    

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

                        ml_filename = final_output_path / f"RHEED_video_{timestamp.strftime('%m-%d-%y_%H-%M-%S')}.mp4"

                        h, w = last_gray_shape
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        ml_writer = cv2.VideoWriter(str(ml_filename), fourcc, 30.0, (w, h), True)

                        if not ml_writer.isOpened():
                            print("❌ Failed to open VideoWriter for ML capture.")
                            ml_writer = None
                        else:
                            ml_capture = True
                            ml_frame_count = 0
                            print(f"📥 ML capture STARTED → {ml_filename}")
                else:
                    # stop capture (NO ML PROCESSING)
                    if ml_writer is not None:
                        ml_writer.release()
                        print(f"📤 ML capture STOPPED. Saved {ml_frame_count} frames → {ml_filename}")

                    ml_writer = None
                    ml_capture = False
                    ml_frame_count = 0

    finally:
        if ml_writer is not None:
            ml_writer.release()
            print(f"📤 ML capture STOPPED on exit. Saved {ml_frame_count} frames → {ml_filename}")

        cam.stop()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete.")

       

        

if __name__ == "__main__":
    main()
