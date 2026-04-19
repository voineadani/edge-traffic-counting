import os
import queue
import time
import logging
import cv2
import numpy as np
from pathlib import Path
import sys

# Ensure REPO_ROOT is in path so we can import deepstream_backend
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

import deepstream_backend

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("nvdcf_exporter")

VEHICLE_CLASS_IDS = {2, 3, 5, 7} # car, motorcycle, bus, truck

def export_tracks(video_path, roi_path, output_mot_path, infer_config, max_frames=0):
    """Run DeepStream and save MOT-formatted tracks."""
    logger.info(f"Processing {video_path}...")
    
    # Override the default infer config in deepstream_backend
    # This is a bit hacky but deepstream_backend uses a global constant
    deepstream_backend.INFER_CONFIG = str(REPO_ROOT / infer_config)
    
    result_queue = queue.Queue(maxsize=1000)
    
    # Load ROI mask
    roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    
    # Start the pipeline
    thread, stop_event = deepstream_backend.start_pipeline_thread(
        video_path=str(video_path),
        result_queue=result_queue,
        conf_thres=0.25,
        iou_thres=0.45,
        vehicle_only=True
    )
    
    # Wait for metadata msg
    msg = result_queue.get(timeout=120)
    meta = msg.get("meta", {})
    stream_w, stream_h = 1920, 1080
    if "resolution" in meta:
        parts = meta["resolution"].split("x")
        stream_w, stream_h = int(parts[0]), int(parts[1])
        
    if roi_mask is not None:
        roi_mask = cv2.resize(roi_mask, (stream_w, stream_h), interpolation=cv2.INTER_NEAREST)

    tracks_file = open(output_mot_path, "w")
    
    frames_processed = 0
    start_time = time.time()
    
    try:
        while True:
            if max_frames > 0 and frames_processed >= max_frames:
                break
            try:
                msg = result_queue.get(timeout=5.0)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue
                
            if msg.get("done"):
                break
                
            frame_num = msg["frame_num"] + 1 # 1-based for MOT
            detections = msg["detections"]
            
            for det in detections:
                tid = det["tracker_id"]
                if tid < 0:
                    continue
                
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # ROI filter
                if roi_mask is not None:
                    if not (0 <= cx < stream_w and 0 <= cy < stream_h and roi_mask[cy, cx] > 0):
                        continue
                
                w = x2 - x1
                h = y2 - y1
                conf = det["confidence"]
                # MOT format: frame, id, x, y, w, h, conf, class, -1, -1
                tracks_file.write(f"{frame_num},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")
            
            frames_processed += 1
            if frames_processed % 500 == 0:
                fps = frames_processed / (time.time() - start_time)
                logger.info(f"Processed {frames_processed} frames... ({fps:.1f} fps)")
                
    finally:
        tracks_file.close()
        stop_event.set()
        thread.join()

    logger.info(f"Finished {video_path}. Tracks saved to {output_mot_path}")

if __name__ == "__main__":
    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--condition", choices=["day", "dusk", "night"], default=None,
                        help="Process only this condition (runs all if omitted, each in a subprocess)")
    args = parser.parse_args()

    output_dir = REPO_ROOT / "benchmark" / "nvdcf_tracks"
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = {
        "day": ("7_45.mp4", "roi_masks/7_45_roi.png"),
        "dusk": ("17_30.mp4", "roi_masks/17_30_roi.png"),
        "night": ("18_45.mp4", "roi_masks/18_45_roi.png"),
    }

    if args.condition:
        # Single-condition mode: process one video and exit.
        # This allows the caller to spawn a fresh process per condition,
        # avoiding GStreamer main-loop restart issues.
        conds = {args.condition: videos[args.condition]}
    else:
        # Multi-condition mode: re-invoke ourselves once per condition so that
        # each condition runs in a fresh process with a clean GStreamer state.
        for cond in videos:
            cmd = [sys.executable, __file__,
                   "--condition", cond,
                   "--max-frames", str(args.max_frames)]
            logger.info(f"Spawning subprocess for condition: {cond}")
            ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if ret.returncode != 0:
                logger.error(f"Subprocess failed for condition: {cond} (exit {ret.returncode})")
        sys.exit(0)
        conds = {}  # unreachable, keeps linter happy

    # We use YOLOv8n (R1) as the baseline for the reference tracker comparison.
    for cond, (vid, roi) in conds.items():
        out_path = output_dir / f"R1_30W_{cond}.txt"
        export_tracks(REPO_ROOT / vid, REPO_ROOT / roi, out_path, "config_yolov8n_infer.txt", max_frames=args.max_frames)

        # Also symlink or copy for MAXN if we want to evaluate it too
        maxn_path = output_dir / f"R1_MAXN_{cond}.txt"
        if os.path.exists(maxn_path):
            os.remove(maxn_path)
        os.link(out_path, maxn_path)
