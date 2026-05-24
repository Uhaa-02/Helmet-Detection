"""
detect.py — Real-Time Helmet Detection (PC / Webcam)
──────────────────────────────────────────────────────
Runs the trained MobileNetV2 model on live webcam frames.

Features:
  - Preprocessing pipeline (blur, CLAHE) matching training
  - Consecutive-frame voting to reduce false positives
  - Cooldown timer between alerts
  - Violation frame saving
  - FPS counter and confidence overlay

Usage:
    python src/detect.py
    python src/detect.py --source video.mp4   # run on video file
    python src/detect.py --source 0           # webcam index 0

Controls:
    Q — quit
    S — save current frame manually
    T — cycle threshold up (+0.05)
    R — reset threshold to config value
"""

import os
import cv2
import json
import yaml
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


# ─────────────────────────────────────────────
#  Config & Model Loading
# ─────────────────────────────────────────────

def load_config(path="config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_classes(cfg: dict):
    model_path = cfg["paths"]["model_save_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    print(f"  Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    class_path = "models/class_indices.json"
    if Path(class_path).exists():
        with open(class_path) as f:
            class_indices = json.load(f)
    else:
        class_indices = {"helmet": 0, "no_helmet": 1}
        print("  [WARNING] class_indices.json not found, using default.")

    # Reverse: index → name
    idx_to_class = {v: k for k, v in class_indices.items()}
    no_helmet_idx = class_indices.get("no_helmet", 1)
    return model, idx_to_class, no_helmet_idx


# ─────────────────────────────────────────────
#  Preprocessing (mirrors train-time pipeline)
# ─────────────────────────────────────────────

def preprocess_frame(frame_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Apply the same preprocessing used during training.
    Returns normalized (0-1) float32 array ready for model input.
    """
    target = tuple(cfg["model"]["input_size"])
    blur_k = cfg["preprocessing"]["blur_kernel"]
    clahe_clip = cfg["preprocessing"]["clahe_clip_limit"]

    img = cv2.resize(frame_bgr, target)

    # Gaussian blur — reduces noise, same as training
    if blur_k > 1:
        k = blur_k if blur_k % 2 == 1 else blur_k + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    # CLAHE for contrast normalization
    if clahe_clip > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        lab_merged = cv2.merge([clahe.apply(l), a, b])
        img = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    # Normalize to [0, 1] and convert BGR→RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
#  Frame Voting Buffer
# ─────────────────────────────────────────────

class ViolationVoter:
    """
    Requires N consecutive 'no_helmet' predictions before triggering alert.
    This reduces false positives from momentary occlusions.
    """
    def __init__(self, required_consecutive: int, cooldown_sec: float):
        self.required = required_consecutive
        self.cooldown = cooldown_sec
        self.buffer = deque(maxlen=required_consecutive)
        self.last_alert_time = 0.0

    def update(self, is_violation: bool) -> bool:
        """Returns True if alert should be triggered."""
        self.buffer.append(is_violation)

        if len(self.buffer) < self.required:
            return False

        all_violations = all(self.buffer)
        now = time.time()
        in_cooldown = (now - self.last_alert_time) < self.cooldown

        if all_violations and not in_cooldown:
            self.last_alert_time = now
            return True
        return False


# ─────────────────────────────────────────────
#  Drawing Helpers
# ─────────────────────────────────────────────

COLORS = {
    "helmet":     (50, 200, 50),    # green
    "no_helmet":  (30, 30, 220),    # red
    "alert":      (0, 100, 255),    # orange-red
    "fps":        (200, 200, 200),  # grey
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame: np.ndarray, label: str, confidence: float,
                 is_alert: bool, fps: float, threshold: float) -> np.ndarray:
    h, w = frame.shape[:2]

    # Background banner
    banner_color = COLORS["alert"] if is_alert else (30, 30, 30)
    cv2.rectangle(frame, (0, 0), (w, 70), banner_color, -1)
    cv2.rectangle(frame, (0, h - 40), (w, h), (20, 20, 20), -1)

    # Main label
    display_text = "⚠ VIOLATION: NO HELMET" if is_alert else label.upper().replace("_", " ")
    text_color = (255, 255, 255)
    cv2.putText(frame, display_text, (15, 45), FONT, 1.1, text_color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = w - 220, 15, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * confidence)
    bar_color = COLORS.get(label, (150, 150, 150))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    cv2.putText(frame, f"{confidence:.1%}", (bar_x + bar_w + 5, bar_y + 15),
                FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Footer: FPS + threshold
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Threshold: {threshold:.2f}",
                (10, h - 12), FONT, 0.5, COLORS["fps"], 1, cv2.LINE_AA)
    cv2.putText(frame, "Q:Quit  S:Save  T:Threshold+  R:Reset",
                (w // 2 - 180, h - 12), FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


def save_violation_frame(frame: np.ndarray, log_cfg: dict):
    if not log_cfg.get("save_violations"):
        return
    vdir = Path(log_cfg["violation_img_dir"])
    vdir.mkdir(parents=True, exist_ok=True)
    fname = datetime.now().strftime("violation_%Y%m%d_%H%M%S_%f.jpg")
    cv2.imwrite(str(vdir / fname), frame)


# ─────────────────────────────────────────────
#  Main Detection Loop
# ─────────────────────────────────────────────

def run_detection(cfg: dict, source):
    model, idx_to_class, no_helmet_idx = load_model_and_classes(cfg)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

    voter = ViolationVoter(
        required_consecutive=cfg["detection"]["consecutive_frames"],
        cooldown_sec=cfg["detection"]["alert_cooldown_sec"]
    )

    threshold = cfg["detection"]["violation_threshold"]
    default_threshold = threshold
    fps_history = deque(maxlen=30)
    prev_time = time.time()

    print("\n  🎥 Detection started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  End of stream or camera disconnected.")
            break

        # ── Inference ──────────────────────────────────
        preprocessed = preprocess_frame(frame, cfg)
        batch = np.expand_dims(preprocessed, axis=0)
        scores = model.predict(batch, verbose=0)[0]  # shape: (2,)

        confidence = float(scores[no_helmet_idx])
        is_violation_frame = confidence >= threshold
        label = "no_helmet" if is_violation_frame else "helmet"

        # ── Voting ─────────────────────────────────────
        should_alert = voter.update(is_violation_frame)

        if should_alert:
            save_violation_frame(frame, cfg["logging"])
            print(f"  ⚠  ALERT: No helmet detected! Confidence={confidence:.3f}")

        # ── FPS ────────────────────────────────────────
        now = time.time()
        fps_history.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = float(np.mean(fps_history))

        # ── Draw ───────────────────────────────────────
        display = draw_overlay(frame.copy(), label, confidence,
                               should_alert or (is_violation_frame and len(voter.buffer) > 0),
                               fps, threshold)
        cv2.imshow("Helmet Detection System", display)

        # ── Key Handling ───────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"logs/manual_{datetime.now().strftime('%H%M%S')}.jpg", frame)
            print("  📸 Frame saved manually.")
        elif key == ord("t"):
            threshold = min(threshold + 0.05, 0.99)
            print(f"  Threshold → {threshold:.2f}")
        elif key == ord("r"):
            threshold = default_threshold
            print(f"  Threshold reset → {threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Detection stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet Detection — Real-time")
    parser.add_argument("--source", default=None,
                        help="Video source: camera index (0,1) or path to video file")
    args = parser.parse_args()

    cfg = load_config()
    source = int(args.source) if args.source and args.source.isdigit() \
             else (args.source if args.source else cfg["camera"]["device_id"])

    run_detection(cfg, source)
