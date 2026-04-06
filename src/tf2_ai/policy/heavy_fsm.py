from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from tf2_ai.vision.detector import Detection


class State(str, Enum):
    ROAM = "ROAM"
    ENGAGE = "ENGAGE"
    RECOVER = "RECOVER"


@dataclass(frozen=True)
class PolicyConfig:
    lost_target_timeout_ms: int
    recover_duration_ms: int
    roam_strafe_interval_ms: int
    roam_scan_interval_ms: int
    roam_scan_pixels: int
    fire_hold_ms: int
    fire_burst_cooldown_ms: int
    aim_tolerance_px: int


@dataclass(frozen=True)
class PolicyAction:
    move_forward: bool = False
    strafe_left: bool = False
    strafe_right: bool = False
    look_dx: float = 0.0
    look_dy: float = 0.0
    fire: bool = False


class HeavyPolicyFSM:
    def __init__(self, config: PolicyConfig) -> None:
        self._config = config
        self._state = State.ROAM
        self._state_started_at = self._now_ms()
        self._last_target_at = self._state_started_at
        self._last_strafe_switch_at = self._state_started_at
        self._last_scan_at = self._state_started_at
        self._strafe_left = True
        self._fire_started_at: Optional[int] = None
        self._last_burst_end_at = self._state_started_at

    @staticmethod
    def _now_ms() -> int:
        return int(time.perf_counter() * 1000)

    @property
    def state(self) -> State:
        return self._state

    def _set_state(self, state: State) -> None:
        self._state = state
        self._state_started_at = self._now_ms()

    def update(
        self,
        target: Optional[Detection],
        frame_width: int,
        frame_height: int,
    ) -> PolicyAction:
        now = self._now_ms()
        has_target = target is not None

        if has_target:
            self._last_target_at = now

        if self._state == State.ROAM and has_target:
            self._set_state(State.ENGAGE)
        elif self._state == State.ENGAGE:
            if (now - self._last_target_at) > self._config.lost_target_timeout_ms:
                self._set_state(State.RECOVER)
                self._fire_started_at = None
                self._last_burst_end_at = now
        elif self._state == State.RECOVER:
            if has_target:
                self._set_state(State.ENGAGE)
            elif (now - self._state_started_at) > self._config.recover_duration_ms:
                self._set_state(State.ROAM)

        if self._state == State.ROAM:
            return self._roam_action(now)
        if self._state == State.ENGAGE:
            return self._engage_action(now, target, frame_width, frame_height)
        return self._recover_action(now)

    def _roam_action(self, now: int) -> PolicyAction:
        if (now - self._last_strafe_switch_at) >= self._config.roam_strafe_interval_ms:
            self._last_strafe_switch_at = now
            self._strafe_left = random.random() < 0.5

        look_dx = 0.0
        if (now - self._last_scan_at) >= self._config.roam_scan_interval_ms:
            self._last_scan_at = now
            look_dx = float(
                self._config.roam_scan_pixels if random.random() < 0.5 else -self._config.roam_scan_pixels
            )

        return PolicyAction(
            move_forward=True,
            strafe_left=self._strafe_left,
            strafe_right=not self._strafe_left,
            look_dx=look_dx,
            fire=False,
        )

    def _engage_action(
        self,
        now: int,
        target: Optional[Detection],
        frame_width: int,
        frame_height: int,
    ) -> PolicyAction:
        if target is None:
            return PolicyAction()

        cx = frame_width / 2.0
        cy = frame_height / 2.0
        tx, ty = target.center
        dx = tx - cx
        dy = ty - cy

        near_crosshair = abs(dx) <= self._config.aim_tolerance_px and abs(dy) <= self._config.aim_tolerance_px
        should_fire = near_crosshair
        if should_fire:
            if self._fire_started_at is None:
                self._fire_started_at = now
            fire_elapsed = now - self._fire_started_at
            if fire_elapsed >= self._config.fire_hold_ms:
                should_fire = False
                self._fire_started_at = None
                self._last_burst_end_at = now
        else:
            self._fire_started_at = None

        if not should_fire and (now - self._last_burst_end_at) < self._config.fire_burst_cooldown_ms:
            should_fire = False

        return PolicyAction(
            move_forward=True,
            look_dx=dx,
            look_dy=dy,
            fire=should_fire,
        )

    def _recover_action(self, now: int) -> PolicyAction:
        if (now - self._last_scan_at) >= max(1, self._config.roam_scan_interval_ms // 2):
            self._last_scan_at = now
            return PolicyAction(
                look_dx=float(self._config.roam_scan_pixels),
                fire=False,
            )
        return PolicyAction(fire=False)

