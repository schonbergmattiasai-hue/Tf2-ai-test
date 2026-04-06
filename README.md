# TF2 Vision Autopilot (Heavy MVP, Local-Only)

> **Important usage restriction**
>
> This project is intended for **offline/local/private server experimentation only**.
> Use with TF2 launched in a compatible private mode (for example `-insecure`).
> Do **not** use on public/VAC-secured matchmaking servers.
> The implementation is **vision + input only** (no memory reading, no process injection).

## What this MVP does

- Captures a configured screen region at high frequency (target ~60 FPS loop).
- Runs YOLO-based object detection every N frames.
- Chooses a target (`enemy` preferred, else highest confidence).
- Executes a simple Heavy finite state machine:
  - `ROAM` (move + occasional strafe/scan),
  - `ENGAGE` (aim + burst fire),
  - `RECOVER` (scan after target loss, then resume roaming).
- Sends normal keyboard/mouse input via Windows-safe user input APIs.
- Includes a simple GUI with **Start** / **Stop** controls.
- Supports **F3** hotkey toggle for start/stop and emergency stop (`F9` by default).

## Repository layout

```text
configs/default.yaml
requirements.txt
src/tf2_ai/
  capture/
  control/
  policy/
  vision/
  main.py
```

## Requirements

- Windows
- Python 3.12
- NVIDIA GPU (recommended for YOLO inference)
- Team Fortress 2 running in a capturable display mode (windowed fullscreen recommended)

## Setup

1. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install CUDA-enabled PyTorch first (pick command from official selector for your CUDA version), then install project requirements:

   ```powershell
   pip install -r requirements.txt
   ```

3. Provide TF2-specific YOLO weights and set the path in `configs/default.yaml`:

   ```yaml
   vision:
     weights_path: "weights/tf2_heavy_enemy.pt"
   ```

> This MVP needs TF2-trained data/weights to produce meaningful detections.

## TF2 launch recommendations (local/private)

- Use private/local server setup only.
- Launch with a compatible private/offline configuration (e.g., `-insecure`).
- Use windowed fullscreen and set `capture.region` to your active game area.

## Calibration tips

- `capture.region`: match your actual game viewport.
- `control.sensitivity_scale`: tune aim speed.
- `control.smoothing`: increase for smoother but slower aim.
- `control.max_pixels_per_step`: clamp sudden aim jumps.
- `vision.infer_every_n_frames`: increase for lower GPU load, decrease for responsiveness.
- `policy.aim_tolerance_px`: only fire when close to crosshair center.

## Run

From repository root:

```powershell
python -m tf2_ai.main --config configs/default.yaml
```

Emergency stop key is configured by `control.emergency_stop_key` (`f9` by default).
Start/Stop toggle key is configured by `control.toggle_key` (`f3` by default).
