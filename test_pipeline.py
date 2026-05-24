"""
test_pipeline.py — Unit Tests for Helmet Detection Pipeline
────────────────────────────────────────────────────────────
Tests core preprocessing, voting, and utility functions
without requiring a trained model or GPU.

Run with:
    pytest tests/ -v
"""

import os
import sys
import time
import numpy as np
import pytest
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─────────────────────────────────────────────
#  Config Tests
# ─────────────────────────────────────────────

def test_config_loads():
    import yaml
    cfg_path = Path("config/config.yaml")
    assert cfg_path.exists(), "config/config.yaml must exist"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "model" in cfg
    assert "training" in cfg
    assert "detection" in cfg
    assert "raspberry_pi" in cfg


def test_config_thresholds_valid():
    import yaml
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    t = cfg["detection"]["confidence_threshold"]
    vt = cfg["detection"]["violation_threshold"]
    assert 0 < t < 1, "confidence_threshold must be in (0, 1)"
    assert 0 < vt < 1, "violation_threshold must be in (0, 1)"
    assert vt >= t, "violation_threshold should be >= confidence_threshold"


# ─────────────────────────────────────────────
#  Preprocessing Tests
# ─────────────────────────────────────────────

def make_test_frame(h=480, w=640):
    """Create a random BGR test frame."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_preprocess_output_shape():
    import yaml, cv2
    from preprocess import preprocess_image
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    frame = make_test_frame()
    target = tuple(cfg["model"]["input_size"])
    result = preprocess_image(frame, target, blur_kernel=3, clahe_clip=2.0)
    assert result.shape == (*target, 3), f"Expected {(*target, 3)}, got {result.shape}"


def test_preprocess_output_dtype():
    import yaml
    from preprocess import preprocess_image
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    frame = make_test_frame()
    target = tuple(cfg["model"]["input_size"])
    result = preprocess_image(frame, target)
    assert result.dtype == np.uint8, "Output should be uint8"


def test_preprocess_pixel_range():
    import yaml
    from preprocess import preprocess_image
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    frame = make_test_frame()
    target = tuple(cfg["model"]["input_size"])
    result = preprocess_image(frame, target)
    assert result.min() >= 0, "Pixel values should be >= 0"
    assert result.max() <= 255, "Pixel values should be <= 255"


def test_preprocess_blur_kernel_must_be_odd():
    import yaml
    from preprocess import preprocess_image
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Even kernel should be auto-corrected without crashing
    frame = make_test_frame()
    target = tuple(cfg["model"]["input_size"])
    result = preprocess_image(frame, target, blur_kernel=4)  # even kernel
    assert result is not None


def test_preprocess_with_edge_detection():
    import yaml
    from preprocess import preprocess_image
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    frame = make_test_frame()
    target = tuple(cfg["model"]["input_size"])
    result = preprocess_image(frame, target, use_edge=True)
    assert result.shape == (*target, 3)


# ─────────────────────────────────────────────
#  Voting Buffer Tests
# ─────────────────────────────────────────────

# Import the voter directly to avoid GPIO/camera dependencies
class ViolationVoter:
    def __init__(self, required, cooldown):
        self.required = required
        self.cooldown = cooldown
        self.buffer = deque(maxlen=required)
        self.last_alert = 0.0

    def update(self, is_violation):
        self.buffer.append(is_violation)
        if len(self.buffer) < self.required:
            return False
        if all(self.buffer) and (time.time() - self.last_alert) >= self.cooldown:
            self.last_alert = time.time()
            return True
        return False


def test_voter_no_alert_before_required_frames():
    voter = ViolationVoter(required=3, cooldown=5)
    assert not voter.update(True)
    assert not voter.update(True)  # only 2, need 3


def test_voter_alerts_after_required_frames():
    voter = ViolationVoter(required=3, cooldown=0)
    voter.update(True)
    voter.update(True)
    result = voter.update(True)
    assert result is True


def test_voter_no_alert_with_mixed_frames():
    voter = ViolationVoter(required=3, cooldown=0)
    voter.update(True)
    voter.update(False)  # breaks the streak
    result = voter.update(True)
    assert not result


def test_voter_cooldown_prevents_repeat_alerts():
    voter = ViolationVoter(required=2, cooldown=10)
    voter.update(True)
    voter.update(True)  # triggers alert, starts cooldown

    # Immediate subsequent alert should be blocked
    voter.update(True)
    result = voter.update(True)
    assert not result, "Cooldown should prevent repeated alerts"


def test_voter_alert_resets_after_cooldown():
    voter = ViolationVoter(required=2, cooldown=0.05)
    voter.update(True)
    voter.update(True)  # first alert
    time.sleep(0.1)     # wait out cooldown
    voter.update(True)
    result = voter.update(True)
    assert result, "Should alert again after cooldown expires"


# ─────────────────────────────────────────────
#  Directory Structure Tests
# ─────────────────────────────────────────────

def test_required_dirs_exist():
    required = ["data/raw", "data/processed", "data/augmented",
                "models", "src", "scripts", "config", "logs"]
    for d in required:
        assert Path(d).exists(), f"Required directory missing: {d}"


def test_src_files_exist():
    required_scripts = [
        "src/preprocess.py",
        "src/train.py",
        "src/detect.py",
        "src/detect_pi.py",
        "src/evaluate.py",
        "scripts/convert_tflite.py",
        "config/config.yaml",
    ]
    for f in required_scripts:
        assert Path(f).exists(), f"Required file missing: {f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
