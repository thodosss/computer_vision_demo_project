"""Optional second stage: YOLOv8n object detection (Ultralytics).

The semantic search pipeline does not depend on this module. Load it only after CLIP has been
released, so that one network at a time occupies the GPU.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("YOLO_OFFLINE", "1")  # no update checks / hub calls at inference time

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from .config import Config  # noqa: E402
from .utils import release_gpu_memory, select_device  # noqa: E402

log = logging.getLogger("ovs.detect")

YOLO_WEIGHTS = "yolov8n.pt"


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels


class ObjectDetector:
    def __init__(self, config: Optional[Config] = None, conf: float = 0.25, imgsz: int = 640,
                 device: Optional[str] = None):
        from ultralytics import YOLO, settings
        settings.update({"sync": False})  # disable Ultralytics anonymous analytics

        self.config = config or Config()
        self.conf, self.imgsz = conf, imgsz
        self.device = select_device(device or self.config.device)
        weights = self.config.models_dir / YOLO_WEIGHTS
        if not weights.exists():
            os.environ["YOLO_OFFLINE"] = "0"  # one-time download of the 6 MB weights
            log.info("Downloading %s to %s (one time)", YOLO_WEIGHTS, weights)
            weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights))
        os.environ["YOLO_OFFLINE"] = "1"
        log.info("Loaded YOLOv8n on %s (fp32, imgsz=%d)", self.device, imgsz)

    def detect(self, image_path: Path) -> Tuple[List[Detection], object]:
        try:
            result = self._predict(image_path)
        except torch.cuda.OutOfMemoryError:
            log.error("CUDA out of memory in YOLOv8n (imgsz=%d); other processes may hold VRAM. "
                      "Retrying on CPU.", self.imgsz)
            release_gpu_memory()
            self.device = "cpu"
            result = self._predict(image_path)
        names = result.names
        dets = [
            Detection(names[int(c)], float(p), tuple(float(v) for v in xyxy))
            for xyxy, p, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
        ]
        return sorted(dets, key=lambda d: -d.confidence), result

    def detect_and_save(self, image_path: Path, out_dir: Optional[Path] = None) -> Tuple[List[Detection], Path]:
        dets, result = self.detect(image_path)
        out_dir = Path(out_dir or self.config.results_dir / "detections")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(image_path).stem}_det.jpg"
        Image.fromarray(result.plot()[..., ::-1]).save(out_path, quality=90)  # plot() returns BGR
        return dets, out_path

    def close(self) -> None:
        self.model = None
        release_gpu_memory()

    def _predict(self, image_path: Path):
        with torch.inference_mode():
            return self.model.predict(str(image_path), conf=self.conf, imgsz=self.imgsz, device=self.device,
                                      verbose=False)[0]  # FP32 is the Ultralytics default
