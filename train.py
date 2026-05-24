"""
train.py — MobileNetV2 Transfer Learning for Helmet Detection
──────────────────────────────────────────────────────────────
Strategy:
  Phase 1 — Feature Extraction:
    Freeze all MobileNetV2 base layers, train only the new
    classification head (5-10 epochs). Fast convergence.

  Phase 2 — Fine-Tuning:
    Unfreeze last N layers of base model, train end-to-end
    at very low LR. Adapts ImageNet features to helmet domain.

Why MobileNetV2?
  - 3.4M params vs YOLOv8's 11M+ — runs on Pi 4 at ~5-8 FPS
  - Depthwise separable convolutions → 8x fewer ops than VGG
  - Pre-trained on ImageNet → excellent feature extractor

Usage:
    python src/train.py
"""

import os
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight


def load_config(path="config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> keras.Model:
    """
    Build MobileNetV2 + custom head.

    Architecture:
      MobileNetV2 (frozen) → GlobalAveragePooling → BatchNorm
      → Dense(256, relu) → Dropout(0.4)
      → Dense(128, relu) → Dropout(0.3)
      → Dense(2, softmax)

    BatchNorm + Dropout combo is crucial for ~500 image datasets
    to prevent overfitting.
    """
    input_size = tuple(cfg["model"]["input_size"]) + (3,)

    base_model = MobileNetV2(
        input_shape=input_size,
        include_top=False,
        weights=cfg["model"]["pretrained_weights"]
    )
    base_model.trainable = False  # Phase 1: freeze all

    inputs = keras.Input(shape=input_size)

    # Use training=False so BatchNorm layers in base stay frozen
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(cfg["model"]["num_classes"], activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def unfreeze_top_layers(model: keras.Model, base_model, n_layers: int, lr: float):
    """Phase 2: Unfreeze last n_layers of base model for fine-tuning."""
    base_model.trainable = True

    # Freeze everything except the last n_layers
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"  Fine-tuning: {trainable} trainable layers (last {n_layers} of base unfrozen)")

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )


def get_data_generators(cfg: dict):
    """Create training and validation generators from augmented data."""
    aug_dir = cfg["data"]["augmented_dir"]
    input_size = tuple(cfg["model"]["input_size"])
    batch_size = cfg["training"]["batch_size"]
    val_split = cfg["training"]["validation_split"]

    # Normalization only for validation — augmentation already done offline
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
        # Light online augmentation as extra regularization
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.1
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split
    )

    train_gen = train_datagen.flow_from_directory(
        aug_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        aug_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42
    )

    print(f"\n  Classes: {train_gen.class_indices}")
    print(f"  Training samples:   {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")

    return train_gen, val_gen


def compute_class_weights_from_gen(train_gen) -> dict:
    """Compute class weights to handle class imbalance."""
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weight_dict = dict(zip(classes, weights))
    print(f"  Class weights: {class_weight_dict}")
    return class_weight_dict


def plot_history(history_p1, history_p2, save_path: str):
    """Plot training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Concatenate phase 1 and phase 2
    def concat(key):
        p1 = history_p1.history.get(key, [])
        p2 = history_p2.history.get(key, [])
        return p1 + p2

    epochs_total = len(concat("accuracy"))
    phase2_start = len(history_p1.history["accuracy"])

    for ax, (metric, val_metric, title) in zip(
        axes,
        [("accuracy", "val_accuracy", "Accuracy"),
         ("loss", "val_loss", "Loss")]
    ):
        ax.plot(concat(metric), label="Train", color="#2196F3", linewidth=2)
        ax.plot(concat(val_metric), label="Validation", color="#FF5722",
                linewidth=2, linestyle="--")
        ax.axvline(x=phase2_start, color="gray", linestyle=":", alpha=0.7,
                   label="Fine-tune start")
        ax.set_title(f"Training {title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved → {save_path}")


def train(cfg: dict):
    print("\n" + "=" * 55)
    print("  Phase 1: Feature Extraction (frozen base)")
    print("=" * 55)

    model, base_model = build_model(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg["training"]["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    model.summary()

    train_gen, val_gen = get_data_generators(cfg)
    class_weights = compute_class_weights_from_gen(train_gen)

    save_path = cfg["paths"]["model_save_path"]
    Path(save_path).parent.mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    callbacks_p1 = [
        EarlyStopping(monitor="val_accuracy", patience=cfg["training"]["early_stopping_patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=cfg["training"]["reduce_lr_factor"],
                          patience=cfg["training"]["reduce_lr_patience"], verbose=1, min_lr=1e-7),
        ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True,
                        verbose=1),
        TensorBoard(log_dir="logs/tensorboard", histogram_freq=0)
    ]

    history_p1 = model.fit(
        train_gen,
        epochs=cfg["training"]["epochs"],
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        verbose=1
    )

    print("\n" + "=" * 55)
    print("  Phase 2: Fine-Tuning (unfreezing top layers)")
    print("=" * 55)

    unfreeze_top_layers(
        model, base_model,
        n_layers=cfg["model"]["fine_tune_layers"],
        lr=cfg["training"]["fine_tune_learning_rate"]
    )

    callbacks_p2 = [
        EarlyStopping(monitor="val_accuracy", patience=cfg["training"]["early_stopping_patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=cfg["training"]["reduce_lr_factor"],
                          patience=cfg["training"]["reduce_lr_patience"], verbose=1, min_lr=1e-8),
        ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history_p2 = model.fit(
        train_gen,
        epochs=cfg["training"]["epochs"],
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        verbose=1
    )

    # Save training history
    history_combined = {
        "phase1": {k: [float(v) for v in vals]
                   for k, vals in history_p1.history.items()},
        "phase2": {k: [float(v) for v in vals]
                   for k, vals in history_p2.history.items()}
    }
    hist_path = cfg["paths"]["training_history_path"]
    with open(hist_path, "w") as f:
        json.dump(history_combined, f, indent=2)

    # Save class indices
    class_indices = train_gen.class_indices
    with open("models/class_indices.json", "w") as f:
        json.dump(class_indices, f)

    # Plot training curves
    plot_history(history_p1, history_p2, "logs/training_curves.png")

    # Final validation accuracy
    val_results = model.evaluate(val_gen, verbose=0)
    print(f"\n  ✅ Final Validation Accuracy: {val_results[1]:.4f}")
    print(f"  ✅ Model saved → {save_path}")

    return model


if __name__ == "__main__":
    print("=" * 55)
    print("  Helmet Detection — Model Training")
    print("=" * 55)

    cfg = load_config()

    # Check GPU
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  GPUs available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train(cfg)
    print("\n✅ Training complete. Run evaluate.py to see metrics.")
