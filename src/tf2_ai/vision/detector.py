from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class VisionConfig:
    weights_path: str
    confidence_threshold: float
    iou_threshold: float
    infer_every_n_frames: int
    enemy_class_names: Sequence[str]


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    class_id: int
    center: tuple[float, float]


class YoloDetector:
    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._model = YOLO(config.weights_path)
        self._frame_idx = 0
        self._last_detections: List[Detection] = []

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        self._frame_idx += 1
        infer_every = max(1, int(self._config.infer_every_n_frames))
        if (self._frame_idx % infer_every) != 0 and self._last_detections:
            return self._last_detections

        result = self._model.predict(
            source=frame_bgr,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            verbose=False,
            device=0,
        )[0]

        names: Dict[int, str] = result.names
        detections: List[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                class_name = names.get(class_id, str(class_id))
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_name=class_name,
                        class_id=class_id,
                        center=center,
                    )
                )

        self._last_detections = detections
        return detections

    def best_target(self, detections: Sequence[Detection]) -> Optional[Detection]:
        if not detections:
            return None

        enemy_names = {name.lower() for name in self._config.enemy_class_names}
        enemy_detections = [
            det for det in detections if det.class_name.lower() in enemy_names
        ]
        if enemy_detections:
            return max(enemy_detections, key=lambda d: d.confidence)
        return max(detections, key=lambda d: d.confidence)

