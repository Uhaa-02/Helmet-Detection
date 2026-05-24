# 🪖 Helmet Detection System

A real-time computer vision pipeline to detect helmet violations using MobileNetV2 + OpenCV, deployable on Raspberry Pi with IoT alert integration.

## Architecture

```
Camera Feed → Preprocessing → MobileNetV2 Classifier → Threshold Filter → Alert Engine
                                                              ↓
                                                    Logs + GPIO/Buzzer (Pi)
```

## Project Structure

```
helmet_detection/
├── data/
│   ├── raw/              # Original images (helmet/ and no_helmet/ folders)
│   ├── processed/        # Resized & normalized images
│   └── augmented/        # Augmented training data
├── models/               # Saved .h5 / .tflite models
├── src/
│   ├── train.py          # Model training (MobileNetV2 transfer learning)
│   ├── detect.py         # Real-time webcam detection (PC)
│   ├── detect_pi.py      # Raspberry Pi detection + GPIO alerts
│   ├── preprocess.py     # Dataset preparation & augmentation
│   └── evaluate.py       # Model evaluation & confusion matrix
├── scripts/
│   ├── download_dataset.py   # Helper to organize dataset
│   └── convert_tflite.py     # Convert model to TFLite for Pi
├── config/
│   └── config.yaml       # All tunable parameters
├── tests/
│   └── test_pipeline.py  # Unit tests
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```
data/raw/
  helmet/       ← ~250+ images of people with helmets
  no_helmet/    ← ~250+ images of people without helmets
```

### 3. Preprocess & Augment
```bash
python src/preprocess.py
```

### 4. Train Model
```bash
python src/train.py
```

### 5. Real-time Detection (PC/Webcam)
```bash
python src/detect.py
```

### 6. Deploy on Raspberry Pi
```bash
python scripts/convert_tflite.py        # Convert model first
python src/detect_pi.py                 # Run on Pi with GPIO alerts
```

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | ≥ 90% | ~92% |
| False Positive Reduction | 30% | Via threshold tuning |
| Alert Latency (Pi) | ≤ 200ms | GPIO trigger |

## Config

Edit `config/config.yaml` to tune thresholds, image size, alert pins, etc.
