from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import mss
import numpy as np


@dataclass(frozen=True)
class CaptureConfig:
    monitor_index: int
    region: Dict[str, int]


class ScreenCapture:
    def __init__(self, config: CaptureConfig) -> None:
        self._monitor_index = config.monitor_index
        self._region = config.region
        self._sct = mss.mss()

    def frame(self) -> Optional[np.ndarray]:
        if self._monitor_index >= len(self._sct.monitors):
            return None

        monitor = self._sct.monitors[self._monitor_index]
        capture_region = {
            "left": self._region.get("left", monitor["left"]),
            "top": self._region.get("top", monitor["top"]),
            "width": self._region.get("width", monitor["width"]),
            "height": self._region.get("height", monitor["height"]),
        }
        raw = self._sct.grab(capture_region)
        bgra = np.array(raw, dtype=np.uint8)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

