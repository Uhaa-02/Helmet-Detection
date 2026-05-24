"""
convert_tflite.py — Convert Keras Model to TFLite for Raspberry Pi
────────────────────────────────────────────────────────────────────
Converts the trained .h5 model to an optimized TFLite flatbuffer.

Optimizations applied:
  - DEFAULT quantization: float32 → float16 weights (2x size reduction)
  - INT8 quantization (optional): requires representative dataset
    → further 4x reduction, ~2x faster on Pi with NNAPI

Usage:
    python scripts/convert_tflite.py              # float16 (recommended)
    python scripts/convert_tflite.py --int8       # INT8 quantization
    python scripts/convert_tflite.py --benchmark  # benchmark on current machine
"""

import os
import time
import yaml
import json
import argparse
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


def load_config(path="config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_representative_dataset(cfg: dict):
    """
    Generator for INT8 calibration.
    Feeds ~100 real images from the processed dataset through the converter
    so it can calculate optimal quantization ranges per layer.
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    proc_dir = Path(cfg["data"]["processed_dir"])
    input_size = tuple(cfg["model"]["input_size"])
    count = 0

    for cls in cfg["data"]["classes"]:
        cls_dir = proc_dir / cls
        if not cls_dir.exists():
            continue
        for img_path in list(cls_dir.glob("*.jpg"))[:50]:
            img = load_img(str(img_path), target_size=input_size)
            arr = img_to_array(img).astype(np.float32) / 255.0
            yield [np.expand_dims(arr, axis=0)]
            count += 1
            if count >= 100:
                return


def convert_to_tflite(cfg: dict, use_int8: bool = False):
    h5_path = cfg["paths"]["model_save_path"]
    tflite_path = cfg["paths"]["tflite_save_path"]

    if not Path(h5_path).exists():
        raise FileNotFoundError(f"Model not found: {h5_path}. Run train.py first.")

    print(f"\n  Loading model from {h5_path}...")
    model = tf.keras.models.load_model(h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if use_int8:
        print("  Applying INT8 quantization (requires representative dataset)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: get_representative_dataset(cfg)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        print("  Applying float16 optimization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    Path(tflite_path).parent.mkdir(exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    original_mb = Path(h5_path).stat().st_size / 1024 / 1024
    tflite_mb = len(tflite_model) / 1024 / 1024
    print(f"\n  ✅ TFLite model saved → {tflite_path}")
    print(f"     Original .h5:  {original_mb:.2f} MB")
    print(f"     TFLite:        {tflite_mb:.2f} MB")
    print(f"     Compression:   {original_mb / tflite_mb:.1f}x")


def benchmark_tflite(cfg: dict, n_runs: int = 50):
    """Run benchmark to estimate inference latency on current machine."""
    tflite_path = cfg["paths"]["tflite_save_path"]
    input_size = tuple(cfg["model"]["input_size"])

    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine dtype
    dtype = input_details[0]["dtype"]
    dummy = np.random.rand(1, *input_size, 3).astype(
        np.float32 if dtype == np.float32 else np.uint8
    )

    # Warm up
    for _ in range(5):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        times.append((time.perf_counter() - t) * 1000)

    print(f"\n  📊 TFLite Benchmark ({n_runs} runs):")
    print(f"     Mean:   {np.mean(times):.2f} ms")
    print(f"     Median: {np.median(times):.2f} ms")
    print(f"     Min:    {np.min(times):.2f} ms")
    print(f"     Max:    {np.max(times):.2f} ms")
    print(f"\n  ⚡ Expected on Pi 4 (~3x slower): ~{np.mean(times) * 3:.0f} ms")
    print(f"     Target: <200ms per frame ({'✅ OK' if np.mean(times) * 3 < 200 else '⚠ May need INT8'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 quantization (smaller + faster, needs dataset)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark TFLite model after conversion")
    args = parser.parse_args()

    cfg = load_config()
    convert_to_tflite(cfg, use_int8=args.int8)

    if args.benchmark:
        benchmark_tflite(cfg)
