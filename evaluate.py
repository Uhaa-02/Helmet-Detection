"""
evaluate.py — Model Evaluation & Metrics
─────────────────────────────────────────
Generates:
  - Confusion matrix
  - Classification report (precision, recall, F1)
  - ROC curve
  - Top-K error analysis

Usage:
    python src/evaluate.py
    python src/evaluate.py --threshold 0.75   # test different thresholds
"""

import os
import json
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)


def load_config(path="config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_class_indices(path="models/class_indices.json") -> dict:
    with open(path) as f:
        return json.load(f)


def plot_confusion_matrix(cm, class_names, save_path, threshold=None):
    """Plot normalized + raw confusion matrix side by side."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title_suffix = f" (threshold={threshold:.2f})" if threshold else ""

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        [f"Confusion Matrix{title_suffix}", f"Normalized{title_suffix}"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, ax=ax, annot_kws={"size": 14})
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_roc_curve(y_true, y_scores, class_names, save_path):
    """Plot ROC curve for each class."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved → {save_path}")


def plot_threshold_analysis(y_true, y_scores, no_helmet_idx, save_path):
    """
    Plot how precision, recall, and F1 change with threshold.
    Helps find the sweet spot that reduces false positives by 30%.
    """
    thresholds = np.arange(0.3, 0.99, 0.01)
    precisions, recalls, f1s, fp_rates = [], [], [], []

    y_binary = (y_true == no_helmet_idx).astype(int)
    scores = y_scores[:, no_helmet_idx]

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (y_binary == 1))
        fp = np.sum((preds == 1) & (y_binary == 0))
        fn = np.sum((preds == 0) & (y_binary == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr  = fp / max(np.sum(y_binary == 0), 1)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        fp_rates.append(fpr)

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, label="Precision", color="#2196F3", linewidth=2)
    ax.plot(thresholds, recalls,    label="Recall",    color="#FF5722", linewidth=2)
    ax.plot(thresholds, f1s,        label="F1 Score",  color="#4CAF50", linewidth=2)
    ax.plot(thresholds, fp_rates,   label="FP Rate",   color="#9C27B0", linewidth=2,
            linestyle="--")
    ax.axvline(x=best_threshold, color="gray", linestyle=":", alpha=0.8,
               label=f"Best F1 @ {best_threshold:.2f}")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold vs. Metrics (no_helmet class)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Threshold analysis saved → {save_path}")
    print(f"  Recommended threshold: {best_threshold:.2f} (F1 = {f1s[best_idx]:.4f})")
    return best_threshold


def evaluate(cfg: dict, threshold: float = None):
    if threshold is None:
        threshold = cfg["detection"]["violation_threshold"]

    model_path = cfg["paths"]["model_save_path"]
    aug_dir = cfg["data"]["augmented_dir"]
    input_size = tuple(cfg["model"]["input_size"])
    batch_size = cfg["training"]["batch_size"]
    val_split = cfg["training"]["validation_split"]

    print(f"\n  Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    class_indices = load_class_indices()
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    no_helmet_idx = class_indices.get("no_helmet", 1)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=val_split)
    val_gen = val_datagen.flow_from_directory(
        aug_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    print(f"  Running inference on {val_gen.samples} validation samples...")
    y_scores = model.predict(val_gen, verbose=1)  # shape: (N, 2)

    # Apply threshold to no_helmet class
    y_pred = np.where(y_scores[:, no_helmet_idx] >= threshold, no_helmet_idx,
                      1 - no_helmet_idx)
    y_true = val_gen.classes

    # Classification report
    print(f"\n  📊 Classification Report (threshold={threshold:.2f}):")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    Path("logs").mkdir(exist_ok=True)
    plot_confusion_matrix(cm, class_names, "logs/confusion_matrix.png", threshold)
    plot_roc_curve(y_true, y_scores, class_names, "logs/roc_curve.png")
    best_t = plot_threshold_analysis(y_true, y_scores, no_helmet_idx,
                                     "logs/threshold_analysis.png")

    # Summary stats
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    print(f"\n  Summary:")
    print(f"    Accuracy:      {accuracy:.4f}")
    print(f"    True Positives (helmet correctly identified):     {tp}")
    print(f"    False Positives (incorrect violation alerts):     {fp}")
    print(f"    Recommended threshold for config.yaml:            {best_t:.2f}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override detection threshold (0-1)")
    args = parser.parse_args()

    cfg = load_config()
    evaluate(cfg, threshold=args.threshold)
