import cv2
import numpy as np
import random
import os
from pathlib import Path


PADDING = 30           # pixels padding around detected ROI
DEBUG_PREFIX = "debug_frame_"


def inspect_video(video_path, sample_frames=10):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n===== VIDEO INFO =====")
    print(f"Video Path     : {video_path}")
    print(f"Frames         : {total_frames}")
    print(f"FPS            : {fps}")
    print(f"Resolution     : {width} x {height}")
    print(f"Pixels/frame   : {width * height}")
    print("======================\n")

    # remove old debug images so they don't accumulate
    for f in Path(".").glob(f"{DEBUG_PREFIX}*.png"):
        f.unlink()

    frame_indices = sorted(random.sample(range(total_frames), sample_frames))

    bounding_boxes = []

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        gray = frame[:, :, 1]  # green channel

        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        _, mask = cv2.threshold(norm, 40, 255, cv2.THRESH_BINARY)

        mask = cv2.GaussianBlur(mask, (9, 9), 0)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        largest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest)

        bounding_boxes.append((x, y, w, h))

        debug = frame.copy()
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 3)

        debug_name = f"{DEBUG_PREFIX}{idx}.png"
        cv2.imwrite(debug_name, debug)

        print(f"Saved debug frame: {debug_name}")

    cap.release()

    if not bounding_boxes:
        print("No bright region detected")
        return

    xs = [b[0] for b in bounding_boxes]
    ys = [b[1] for b in bounding_boxes]
    ws = [b[2] for b in bounding_boxes]
    hs = [b[3] for b in bounding_boxes]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max([x + w for x, w in zip(xs, ws)])
    y_max = max([y + h for y, h in zip(ys, hs)])

    # apply padding
    x_min = max(0, x_min - PADDING)
    y_min = max(0, y_min - PADDING)
    x_max = min(width, x_max + PADDING)
    y_max = min(height, y_max + PADDING)

    print("\n===== SUGGESTED CROP =====")
    print("Use this in your pipeline:\n")

    print(f"CROP = [{y_min}, {y_max}, {x_min}, {x_max}]")

    print("\nInterpretation:")
    print(f"top    = {y_min}")
    print(f"bottom = {y_max}")
    print(f"left   = {x_min}")
    print(f"right  = {x_max}")


def get_latest_video():

    captures_folder = Path("../captures")

    video_extensions = ["*.avi", "*.mp4", "*.mov"]

    videos = []

    for ext in video_extensions:
        videos.extend(captures_folder.glob(ext))

    if not videos:
        raise Exception("No videos found in captures folder")

    latest_video = max(videos, key=os.path.getmtime)

    return latest_video


if __name__ == "__main__":

    latest_video = get_latest_video()

    print("\nLatest video detected:")
    print(latest_video)

    inspect_video(str(latest_video))