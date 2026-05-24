"""
preprocess.py — Dataset Preparation & Augmentation
────────────────────────────────────────────────────
1. Reads raw images from data/raw/helmet/ and data/raw/no_helmet/
2. Resizes to 224x224, applies CLAHE contrast enhancement
3. Optionally applies Gaussian blur to reduce noise
4. Augments dataset to reach target multiplier
5. Saves processed images to data/processed/ and data/augmented/

Usage:
    python src/preprocess.py
"""

import os
import cv2
import yaml
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, img_to_array, array_to_img, load_img
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(
    img_bgr: np.ndarray,
    target_size: tuple,
    blur_kernel: int = 3,
    clahe_clip: float = 2.0,
    use_edge: bool = False
) -> np.ndarray:
    """
    Apply preprocessing pipeline to a single image.

    Steps:
      1. Resize to target_size
      2. Gaussian blur (noise reduction → fewer false positives)
      3. CLAHE contrast enhancement (handles varying lighting)
      4. Optional Canny edge overlay

    Returns: RGB image as numpy array (uint8, 0-255)
    """
    # Resize
    img = cv2.resize(img_bgr, target_size)

    # Gaussian blur — reduces high-frequency noise
    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    # CLAHE on L channel of LAB — improves contrast in dark/bright conditions
    if clahe_clip > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_merged = cv2.merge([l_enhanced, a, b])
        img = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    # Optional edge detection overlay
    if use_edge:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.8, edges_colored, 0.2, 0)

    # Convert to RGB for Keras
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process_dataset(cfg: dict):
    """Resize and preprocess all raw images → data/processed/"""
    raw_dir = Path(cfg["data"]["raw_dir"])
    proc_dir = Path(cfg["data"]["processed_dir"])
    target_size = tuple(cfg["model"]["input_size"])
    blur_k = cfg["preprocessing"]["blur_kernel"]
    clahe_clip = cfg["preprocessing"]["clahe_clip_limit"]
    use_edge = cfg["preprocessing"]["use_edge_detection"]

    classes = cfg["data"]["classes"]
    total = 0

    for cls in classes:
        in_dir = raw_dir / cls
        out_dir = proc_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        if not in_dir.exists():
            print(f"  [WARNING] {in_dir} not found. Skipping.")
            continue

        image_files = list(in_dir.glob("*.[jp][pn][eg]*")) + list(in_dir.glob("*.jpeg"))
        print(f"\n  Processing '{cls}': {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"  {cls}"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"    [SKIP] Could not read {img_path.name}")
                continue

            processed = preprocess_image(img_bgr, target_size, blur_k, clahe_clip, use_edge)
            out_path = out_dir / img_path.name
            # Save as RGB (cv2 expects BGR, so convert back)
            cv2.imwrite(str(out_path), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            total += 1

    print(f"\n  ✅ Preprocessed {total} images → {proc_dir}")
    return total


def augment_dataset(cfg: dict):
    """
    Augment processed images to increase dataset size.
    Uses ImageDataGenerator for realistic augmentations.
    """
    proc_dir = Path(cfg["data"]["processed_dir"])
    aug_dir = Path(cfg["data"]["augmented_dir"])
    multiplier = cfg["augmentation"]["augment_multiplier"]
    aug_cfg = cfg["augmentation"]

    datagen = ImageDataGenerator(
        rotation_range=aug_cfg["rotation_range"],
        width_shift_range=aug_cfg["width_shift"],
        height_shift_range=aug_cfg["height_shift"],
        shear_range=aug_cfg["shear_range"],
        zoom_range=aug_cfg["zoom_range"],
        horizontal_flip=aug_cfg["horizontal_flip"],
        brightness_range=aug_cfg["brightness_range"],
        fill_mode="nearest"
    )

    classes = cfg["data"]["classes"]
    total = 0

    for cls in classes:
        in_dir = proc_dir / cls
        out_dir = aug_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        # First copy originals
        for f in in_dir.glob("*.[jp][pn][eg]*"):
            shutil.copy(f, out_dir / f.name)

        image_files = list(in_dir.glob("*.[jp][pn][eg]*"))
        print(f"\n  Augmenting '{cls}': {len(image_files)} originals × {multiplier}")

        for img_path in tqdm(image_files, desc=f"  {cls}"):
            img = load_img(str(img_path))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            prefix = img_path.stem
            count = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=str(out_dir),
                                       save_prefix=f"aug_{prefix}", save_format="jpg"):
                count += 1
                total += 1
                if count >= multiplier:
                    break

    print(f"\n  ✅ Augmented dataset: {total} new images → {aug_dir}")


def print_dataset_stats(cfg: dict):
    """Print class distribution for processed and augmented datasets."""
    for label, base_dir in [("Processed", cfg["data"]["processed_dir"]),
                             ("Augmented", cfg["data"]["augmented_dir"])]:
        base = Path(base_dir)
        if not base.exists():
            continue
        print(f"\n  📊 {label} dataset:")
        total = 0
        for cls in cfg["data"]["classes"]:
            count = len(list((base / cls).glob("*.[jp][pn][eg]*"))) if (base / cls).exists() else 0
            print(f"     {cls}: {count} images")
            total += count
        print(f"     TOTAL: {total}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Helmet Detection — Data Preprocessing Pipeline")
    print("=" * 55)

    cfg = load_config()

    print("\n[1/2] Preprocessing raw images...")
    process_dataset(cfg)

    print("\n[2/2] Augmenting dataset...")
    augment_dataset(cfg)

    print_dataset_stats(cfg)

    print("\n✅ Preprocessing complete. Run train.py next.")
