"""
download_dataset.py — Dataset Setup Helper
───────────────────────────────────────────
Helps you set up the dataset in two ways:

Option A: Organize existing images you have collected
Option B: Download sample dataset from Roboflow (requires API key)

Recommended public datasets for helmet detection:
  1. Hard Hat Workers Dataset — Roboflow Universe
     https://universe.roboflow.com/joseph-nelson/hard-hat-workers
  2. Safety Helmet Detection — Kaggle
     https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection

Usage:
    python scripts/download_dataset.py --organize /path/to/your/images
    python scripts/download_dataset.py --stats        # check dataset balance
"""

import os
import shutil
import argparse
import random
from pathlib import Path


def organize_images(source_dir: str, split: float = 0.8):
    """
    Organize images from a flat directory into helmet/no_helmet structure.
    Expects filenames to contain 'helmet' or 'nohelmet'/'no_helmet'.
    """
    src = Path(source_dir)
    raw = Path("data/raw")
    (raw / "helmet").mkdir(parents=True, exist_ok=True)
    (raw / "no_helmet").mkdir(parents=True, exist_ok=True)

    all_images = list(src.glob("*.[jp][pn][eg]*")) + list(src.glob("*.jpeg"))
    print(f"Found {len(all_images)} images in {source_dir}")

    helmet_count, no_helmet_count, unclassified = 0, 0, 0

    for img_path in all_images:
        name_lower = img_path.stem.lower()
        if "no_helmet" in name_lower or "nohelmet" in name_lower or "without" in name_lower:
            dest = raw / "no_helmet" / img_path.name
            no_helmet_count += 1
        elif "helmet" in name_lower or "with_helmet" in name_lower or "wearing" in name_lower:
            dest = raw / "helmet" / img_path.name
            helmet_count += 1
        else:
            # Ask user
            unclassified += 1
            print(f"  Unclassified: {img_path.name} — skipping")
            continue
        shutil.copy2(img_path, dest)

    print(f"\n  ✅ Organized:")
    print(f"     helmet:    {helmet_count}")
    print(f"     no_helmet: {no_helmet_count}")
    print(f"     skipped:   {unclassified}")
    print(f"\n  📁 Images are in data/raw/")


def check_dataset_stats():
    """Print class counts and balance ratio."""
    raw = Path("data/raw")
    for split in ["raw", "processed", "augmented"]:
        base = Path(f"data/{split}")
        if not base.exists():
            continue
        print(f"\n  📊 data/{split}/")
        totals = {}
        for cls_dir in sorted(base.iterdir()):
            if cls_dir.is_dir():
                imgs = list(cls_dir.glob("*.[jp][pn][eg]*"))
                totals[cls_dir.name] = len(imgs)
                print(f"     {cls_dir.name:15s}: {len(imgs):5d} images")
        if len(totals) == 2:
            vals = list(totals.values())
            ratio = max(vals) / max(min(vals), 1)
            balance = "✅ Balanced" if ratio < 1.5 else "⚠ Imbalanced"
            print(f"     Balance ratio: {ratio:.2f}  {balance}")


def create_sample_structure():
    """Create the expected directory structure with instructions."""
    for folder in ["data/raw/helmet", "data/raw/no_helmet"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    readme = """# Dataset Instructions

Place your images here:

  data/raw/helmet/      ← Images of people WEARING helmets
  data/raw/no_helmet/   ← Images of people WITHOUT helmets

Recommended: 250+ images per class minimum.
More = better. 500+ total is ideal for ~92% accuracy.

## Free Dataset Sources:
1. Roboflow Universe — Hard Hat Workers Dataset
   https://universe.roboflow.com/joseph-nelson/hard-hat-workers
   (Export as JPG classification format)

2. Kaggle — Hard Hat Detection
   https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection

3. Google Images / Open Images Dataset
   Search: "construction worker helmet" and "construction worker no helmet"

## Tips:
- Vary lighting conditions (indoor, outdoor, shadows)
- Include different helmet colors
- Include partial occlusions
- Include different angles (front, side, top-down)
- Mix genders, ethnicities for robustness
"""
    with open("data/raw/README.md", "w") as f:
        f.write(readme)
    print("  ✅ Created data/raw/ structure with README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--organize", metavar="DIR",
                        help="Organize images from a flat directory")
    parser.add_argument("--stats", action="store_true",
                        help="Print dataset statistics")
    parser.add_argument("--setup", action="store_true",
                        help="Create dataset directory structure")
    args = parser.parse_args()

    if args.organize:
        organize_images(args.organize)
    elif args.stats:
        check_dataset_stats()
    elif args.setup:
        create_sample_structure()
    else:
        create_sample_structure()
        check_dataset_stats()
        print("\n  Next step: Add images to data/raw/helmet/ and data/raw/no_helmet/")
        print("  Then run: python src/preprocess.py")
