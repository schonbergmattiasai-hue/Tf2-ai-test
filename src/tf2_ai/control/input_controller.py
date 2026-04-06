from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import pydirectinput
from pynput import keyboard


@dataclass(frozen=True)
class ControlConfig:
    sensitivity_scale: float
    smoothing: float
    max_pixels_per_step: int
    min_action_interval_ms: int
    emergency_stop_key: str
    keybinds: Dict[str, str]


class EmergencyStop(Exception):
    pass


class InputController:
    def __init__(self, config: ControlConfig) -> None:
        self._config = config
        self._running = True
        self._fire_down = False
        self._pressed_keys: set[str] = set()
        self._last_action_time = 0.0
        self._last_dx = 0.0
        self._last_dy = 0.0
        self._listener: Optional[keyboard.Listener] = None

    @property
    def running(self) -> bool:
        return self._running

    def start_emergency_listener(self) -> None:
        stop_key = self._config.emergency_stop_key.lower()

        def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
            name = None
            if isinstance(key, keyboard.KeyCode) and key.char:
                name = key.char.lower()
            elif isinstance(key, keyboard.Key):
                name = key.name.lower()
            if name == stop_key:
                self.emergency_stop()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def _rate_limited(self) -> bool:
        now = time.perf_counter()
        min_interval = self._config.min_action_interval_ms / 1000.0
        if (now - self._last_action_time) < min_interval:
            return True
        self._last_action_time = now
        return False

    def move_mouse_relative(self, dx: float, dy: float) -> None:
        if self._rate_limited():
            return
        smoothing = max(0.0, min(1.0, self._config.smoothing))
        self._last_dx = self._last_dx * (1.0 - smoothing) + dx * smoothing
        self._last_dy = self._last_dy * (1.0 - smoothing) + dy * smoothing

        scaled_x = self._last_dx * self._config.sensitivity_scale
        scaled_y = self._last_dy * self._config.sensitivity_scale
        max_step = float(self._config.max_pixels_per_step)
        clipped_x = int(max(-max_step, min(max_step, scaled_x)))
        clipped_y = int(max(-max_step, min(max_step, scaled_y)))
        if clipped_x or clipped_y:
            pydirectinput.moveRel(clipped_x, clipped_y, relative=True)

    def hold_fire(self, enabled: bool) -> None:
        fire_key = self._config.keybinds.get("fire", "left")
        if enabled and not self._fire_down:
            pydirectinput.mouseDown(button=fire_key)
            self._fire_down = True
        elif not enabled and self._fire_down:
            pydirectinput.mouseUp(button=fire_key)
            self._fire_down = False

    def set_movement(
        self,
        forward: bool = False,
        backward: bool = False,
        left: bool = False,
        right: bool = False,
    ) -> None:
        desired = set()
        kb = self._config.keybinds
        if forward:
            desired.add(kb.get("forward", "w"))
        if backward:
            desired.add(kb.get("backward", "s"))
        if left:
            desired.add(kb.get("left", "a"))
        if right:
            desired.add(kb.get("right", "d"))

        for key in list(self._pressed_keys - desired):
            pydirectinput.keyUp(key)
            self._pressed_keys.remove(key)
        for key in list(desired - self._pressed_keys):
            pydirectinput.keyDown(key)
            self._pressed_keys.add(key)

    def emergency_stop(self) -> None:
        self._running = False
        self.hold_fire(False)
        self.set_movement(False, False, False, False)
        if self._listener is not None:
            listener = self._listener

            def _stop() -> None:
                try:
                    listener.stop()
                except RuntimeError:
                    pass

            threading.Thread(target=_stop, daemon=True).start()

    def close(self) -> None:
        self.emergency_stop()

