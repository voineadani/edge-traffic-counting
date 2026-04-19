"""
DeepStream backend for traffic analysis.

Uses pyservicemaker (DeepStream 7.1 Service Maker Python API) to build a
hardware-accelerated pipeline:
  nvv4l2decoder (HW decode) → nvinfer TensorRT FP16 (custom YOLO11 parser)
  → nvtracker NvDCF → Python probe

The nvinfer plugin runs inference and calls the C++ custom parser
(libnvdsinfer_custom_impl_yolo11.so) to create NvDsObjectMeta entries.
NVDCF then tracks those objects and assigns stable track IDs.
The Python probe in TrackedResultsCollector reads tracked objects and
pushes per-frame results to a queue consumed by the Streamlit dashboard.
"""

import os
import queue
import threading
import time
import logging

import cv2

logger = logging.getLogger(__name__)

# COCO class names (80 classes)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Vehicle class IDs in COCO
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_CONFIG = os.path.join(BASE_DIR, "config_yolo11n_infer.txt")
TRACKER_LL_CONFIG = "/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
TRACKER_LL_LIB = "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so"


class TrackedResultsCollector:
    """BatchMetadataOperator attached AFTER nvtracker: extracts tracked objects
    and pushes results to a queue."""

    def __init__(self, result_queue, vehicle_only=True):
        self.result_queue = result_queue
        self.vehicle_only = vehicle_only
        self._start_time = time.time()

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            frame_num = frame_meta.frame_number
            detections = []

            for obj in frame_meta.object_items:
                cid = obj.class_id
                if self.vehicle_only and cid not in VEHICLE_CLASS_IDS:
                    continue
                tracker_id = obj.object_id if hasattr(obj, "object_id") else -1
                detections.append({
                    "class_id": cid,
                    "class_name": COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid),
                    "confidence": obj.confidence,
                    "tracker_id": int(tracker_id),
                    "bbox": [
                        obj.rect_params.left,
                        obj.rect_params.top,
                        obj.rect_params.left + obj.rect_params.width,
                        obj.rect_params.top + obj.rect_params.height,
                    ],
                })

            self.result_queue.put({
                "frame_num": frame_num,
                "timestamp": time.time() - self._start_time,
                "detections": detections,
                "done": False,
            })


def _make_bmo(cls_or_instance):
    """Wrap a plain object with handle_metadata into a BatchMetadataOperator subclass."""
    from pyservicemaker import BatchMetadataOperator

    class _Wrapper(BatchMetadataOperator):
        def __init__(self, inner):
            super().__init__()
            self._inner = inner

        def handle_metadata(self, batch_meta):
            self._inner.handle_metadata(batch_meta)

    return _Wrapper(cls_or_instance)


def run_pipeline(video_path, result_queue, conf_thres=0.25, iou_thres=0.45,
                 vehicle_only=True, stop_event=None):
    """Run the DeepStream pipeline in the current thread.

    The nvinfer plugin uses the C++ custom parser (NvDsInferParseCustomYolo11)
    to create NvDsObjectMeta entries from YOLO11 raw tensor output.
    NVDCF then tracks those objects and assigns stable track IDs.

    Args:
        video_path: Path to input video file (H264/H265 mp4).
        result_queue: queue.Queue to push per-frame results.
        conf_thres: Detection confidence threshold (informational; threshold is
                    baked into the C++ parser at compile time as 0.25).
        iou_thres: NMS IoU threshold (informational; baked into parser as 0.45).
        vehicle_only: If True, only output vehicle classes.
        stop_event: threading.Event — set to request early stop.
    """
    from pyservicemaker import Pipeline, Probe

    # Get video dimensions for metadata
    cap = cv2.VideoCapture(video_path)
    stream_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    stream_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Push video metadata
    result_queue.put({
        "meta": {
            "source": video_path,
            "resolution": f"{stream_w}x{stream_h}",
            "video_fps": video_fps,
            "total_frames": total_frames,
        }
    })

    collector = TrackedResultsCollector(result_queue, vehicle_only=vehicle_only)

    logger.info("Building DeepStream pipeline for %s (%dx%d)", video_path, stream_w, stream_h)

    try:
        pipeline = (
            Pipeline("traffic-ds")
            .add("filesrc", "src", {"location": video_path})
            .add("qtdemux", "demux")
            .add("h264parse", "parser")
            .add("nvv4l2decoder", "decoder")
            .add("nvstreammux", "mux", {
                "batch-size": 1,
                "width": stream_w,
                "height": stream_h,
            })
            .add("nvinfer", "infer", {
                "config-file-path": INFER_CONFIG,
            })
            .add("nvtracker", "tracker", {
                "ll-config-file": TRACKER_LL_CONFIG,
                "ll-lib-file": TRACKER_LL_LIB,
                "tracker-width": 640,
                "tracker-height": 384,
            })
            .add("fakesink", "sink")
            .link("src", "demux")
            .link("demux", "parser")
            .link("parser", "decoder")
            .link(("decoder", "mux"), ("", "sink_%u"))
            .link("mux", "infer", "tracker", "sink")
            .attach("tracker", Probe("collector", _make_bmo(collector)))
        )

        logger.info("Starting DeepStream pipeline (first run may take time for TensorRT engine build)")
        pipeline.start()

        if stop_event:
            # Poll for stop while pipeline runs
            while not stop_event.is_set():
                time.sleep(0.1)
            pipeline.stop()
        else:
            pipeline.wait()

    except Exception as e:
        logger.exception("DeepStream pipeline error")
        result_queue.put({"error": str(e), "done": True})
        return

    # Signal completion
    result_queue.put({"frame_num": -1, "detections": [], "done": True})
    logger.info("DeepStream pipeline finished")


def start_pipeline_thread(video_path, result_queue, conf_thres=0.25, iou_thres=0.45,
                           vehicle_only=True):
    """Start the DeepStream pipeline in a background thread.

    Returns (thread, stop_event).
    """
    stop_event = threading.Event()
    t = threading.Thread(
        target=run_pipeline,
        args=(video_path, result_queue, conf_thres, iou_thres, vehicle_only, stop_event),
        daemon=True,
    )
    t.start()
    return t, stop_event
