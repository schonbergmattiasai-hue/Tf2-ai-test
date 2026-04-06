from __future__ import annotations

import argparse
import threading
import time
import tkinter as tk
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


class Tf2AiApp:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._capture = ScreenCapture(CaptureConfig(**cfg["capture"]))
        self._detector = YoloDetector(VisionConfig(**cfg["vision"]))
        self._controller = InputController(ControlConfig(**cfg["control"]))
        self._policy = HeavyPolicyFSM(PolicyConfig(**cfg["policy"]))
        self._runtime_cfg = cfg["runtime"]
        self._target_fps = max(1, int(self._runtime_cfg.get("target_fps", 60)))
        self._loop_sleep_ms = max(0, int(self._runtime_cfg.get("loop_sleep_ms", 1)))
        self._frame_time_budget = 1.0 / self._target_fps
        self._bot_thread: threading.Thread | None = None

        self._root = tk.Tk()
        self._root.title("TF2 AI Controller (Local-Only)")
        self._status_var = tk.StringVar(value="Stopped")
        self._build_ui()

        self._controller.set_toggle_callback(self._on_toggle_hotkey)
        self._controller.start_hotkey_listener()

    def _build_ui(self) -> None:
        frame = tk.Frame(self._root, padx=16, pady=16)
        frame.pack(fill=tk.BOTH, expand=True)

        status_label = tk.Label(frame, textvariable=self._status_var, width=40, anchor="w")
        status_label.pack(pady=(0, 10))

        buttons = tk.Frame(frame)
        buttons.pack(fill=tk.X)

        start_btn = tk.Button(buttons, text="Start", command=self.start_ai, width=12)
        start_btn.pack(side=tk.LEFT, padx=(0, 8))
        stop_btn = tk.Button(buttons, text="Stop", command=self.stop_ai, width=12)
        stop_btn.pack(side=tk.LEFT)

        hotkey_hint = tk.Label(
            frame,
            text=(
                f"{self._cfg['control']['toggle_key'].upper()} toggles Start/Stop, "
                f"{self._cfg['control']['emergency_stop_key'].upper()} exits."
            ),
            anchor="w",
        )
        hotkey_hint.pack(pady=(10, 0))

        self._root.protocol("WM_DELETE_WINDOW", self.shutdown)

    def _on_toggle_hotkey(self) -> None:
        self._status_var.set("Running" if self._controller.active else "Stopped")

    def start_ai(self) -> None:
        if not self._controller.running:
            return
        self._controller.set_active(True)
        self._status_var.set("Running")
        if self._bot_thread is None or not self._bot_thread.is_alive():
            self._bot_thread = threading.Thread(target=self._run_loop, daemon=True)
            self._bot_thread.start()

    def stop_ai(self) -> None:
        self._controller.set_active(False)
        self._status_var.set("Stopped")

    def _run_loop(self) -> None:
        while self._controller.running:
            if not self._controller.active:
                time.sleep(0.01)
                continue

            loop_started = time.perf_counter()
            frame = self._capture.frame()
            if frame is None:
                time.sleep(0.1)
                continue

            detections = self._detector.detect(frame)
            target = self._detector.best_target(detections)
            h, w = frame.shape[:2]
            action = self._policy.update(target=target, frame_width=w, frame_height=h)

            self._controller.set_movement(
                forward=action.move_forward,
                left=action.strafe_left,
                right=action.strafe_right,
            )
            self._controller.move_mouse_relative(action.look_dx, action.look_dy)
            self._controller.hold_fire(action.fire)

            elapsed = time.perf_counter() - loop_started
            remaining = self._frame_time_budget - elapsed
            if remaining > 0:
                time.sleep(remaining)
            elif self._loop_sleep_ms > 0:
                time.sleep(self._loop_sleep_ms / 1000.0)

        self._root.after(0, self._status_var.set, "Stopped")

    def shutdown(self) -> None:
        self._controller.close()
        self._status_var.set("Stopped")
        self._root.after(50, self._root.destroy)

    def run(self) -> None:
        toggle_key = str(self._cfg["control"]["toggle_key"]).upper()
        emergency_key = str(self._cfg["control"]["emergency_stop_key"]).upper()
        print(
            "TF2 AI controller ready (local/private server use only). "
            f"Use Start/Stop buttons or {toggle_key} toggle, {emergency_key} to emergency stop."
        )
        self._root.mainloop()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    app = Tf2AiApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
