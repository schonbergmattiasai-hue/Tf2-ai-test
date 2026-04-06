from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from tf2_ai.capture.screen import CaptureConfig, ScreenCapture
from tf2_ai.control.input_controller import ControlConfig, InputController
from tf2_ai.policy.heavy_fsm import HeavyPolicyFSM, PolicyConfig
from tf2_ai.vision.detector import VisionConfig, YoloDetector


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF2 Heavy vision autopilot (local/private only)")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    capture = ScreenCapture(CaptureConfig(**cfg["capture"]))
    detector = YoloDetector(VisionConfig(**cfg["vision"]))
    controller = InputController(ControlConfig(**cfg["control"]))
    policy = HeavyPolicyFSM(PolicyConfig(**cfg["policy"]))

    runtime_cfg = cfg["runtime"]
    target_fps = max(1, int(runtime_cfg.get("target_fps", 60)))
    loop_sleep_ms = max(0, int(runtime_cfg.get("loop_sleep_ms", 1)))
    frame_time_budget = 1.0 / target_fps

    controller.start_emergency_listener()
    print("TF2 AI started (local/private server use only). Press emergency stop key to exit.")

    try:
        while controller.running:
            loop_started = time.perf_counter()
            frame = capture.frame()
            if frame is None:
                time.sleep(0.1)
                continue

            detections = detector.detect(frame)
            target = detector.best_target(detections)

            h, w = frame.shape[:2]
            action = policy.update(target=target, frame_width=w, frame_height=h)
            controller.set_movement(
                forward=action.move_forward,
                left=action.strafe_left,
                right=action.strafe_right,
            )
            controller.move_mouse_relative(action.look_dx, action.look_dy)
            controller.hold_fire(action.fire)

            elapsed = time.perf_counter() - loop_started
            remaining = frame_time_budget - elapsed
            if remaining > 0:
                time.sleep(remaining)
            elif loop_sleep_ms > 0:
                time.sleep(loop_sleep_ms / 1000.0)
    finally:
        controller.close()
        print("TF2 AI stopped safely.")


if __name__ == "__main__":
    main()

