from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("soundsense_processed_v4_10windows")
MODEL_DIR = Path("soundsense_padded_model_v5_recording_balanced")

MODEL_PATH = MODEL_DIR / "soundsense_v5_recording_balanced.keras"
NORMALIZATION_PATH = MODEL_DIR / "normalization_stats.npz"
CLASSES_PATH = MODEL_DIR / "class_names.json"

BATCH_SIZE = 64

# ============================================================
# VERIFY FILES
# ============================================================

for path in (
    MODEL_PATH,
    NORMALIZATION_PATH,
    CLASSES_PATH,
    DATA_DIR / "X_validation.npy",
    DATA_DIR / "y_validation.npy",
    DATA_DIR / "recording_ids_validation.npy",
):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("Loading saved V5 model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH) as f:
    classes = json.load(f)

num_classes = len(classes)

# Load the exact statistics saved by the training script.
with np.load(NORMALIZATION_PATH) as stats:
    mean = float(stats["mean"])
    std = float(stats["std"])

if not np.isfinite(mean):
    raise ValueError(f"Invalid normalization mean: {mean}")

if not np.isfinite(std) or std <= 1e-8:
    raise ValueError(f"Invalid normalization std: {std}")

print(f"Classes: {num_classes}")
print(f"Normalization mean: {mean:.6f}")
print(f"Normalization std:  {std:.6f}")

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation arrays...")

X_val = np.load(
    DATA_DIR / "X_validation.npy",
    mmap_mode="r",
)
y_val = np.load(DATA_DIR / "y_validation.npy")
ids_val = np.load(
    DATA_DIR / "recording_ids_validation.npy",
    allow_pickle=False,
)

if not (len(X_val) == len(y_val) == len(ids_val)):
    raise ValueError("Validation array lengths do not match.")

if X_val.shape[1:] != tuple(model.input_shape[1:]):
    raise ValueError(
        f"Feature shape mismatch: data={X_val.shape[1:]}, "
        f"model={model.input_shape[1:]}"
    )

if model.output_shape[-1] != num_classes:
    raise ValueError(
        f"Model outputs {model.output_shape[-1]} classes, "
        f"but class_names.json contains {num_classes}."
    )

if set(np.unique(y_val)) - set(range(num_classes)):
    raise ValueError("Unexpected class index in validation labels.")

print(f"Validation windows: {len(y_val):,}")
print(f"Validation recordings: {len(set(ids_val.tolist())):,}")

# ============================================================
# PREDICT ALL VALIDATION WINDOWS
# ============================================================

print("\nPredicting validation windows...")

probability_batches = []

for start in range(0, len(y_val), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(y_val))

    xb = np.asarray(X_val[start:end], dtype=np.float32)
    xb = (xb - mean) / std

    probabilities = model.predict_on_batch(xb)
    probability_batches.append(np.asarray(probabilities))

probabilities = np.concatenate(probability_batches, axis=0)

if len(probabilities) != len(y_val):
    raise ValueError("Prediction count does not match validation labels.")

if not np.all(np.isfinite(probabilities)):
    raise ValueError("Model generated non-finite predictions.")

print(f"Predictions generated: {len(probabilities):,}")

# ============================================================
# AGGREGATE WINDOW PROBABILITIES BY RECORDING
# ============================================================

print("\nAggregating predictions by recording...")

grouped_probabilities = defaultdict(list)
grouped_labels = {}

for probability, label, recording_id in zip(
    probabilities,
    y_val,
    ids_val,
):
    recording_id = str(recording_id)
    label = int(label)

    grouped_probabilities[recording_id].append(probability)

    if recording_id in grouped_labels:
        if grouped_labels[recording_id] != label:
            raise ValueError(
                f"Inconsistent labels for recording {recording_id}"
            )
    else:
        grouped_labels[recording_id] = label

recording_names = sorted(grouped_probabilities)

y_true = np.asarray(
    [grouped_labels[name] for name in recording_names],
    dtype=np.int32,
)

y_prob = np.asarray(
    [
        np.mean(grouped_probabilities[name], axis=0)
        for name in recording_names
    ],
    dtype=np.float32,
)

y_pred = np.argmax(y_prob, axis=1)
confidence = np.max(y_prob, axis=1)

# ============================================================
# RECORDING-LEVEL METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    average="weighted",
    zero_division=0,
)

print("\n========== V5 VALIDATION RESULTS ==========")
print(f"Recording count:         {len(recording_names)}")
print(f"Recording-level accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Recording-level macro F1:  {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
print(f"Recording-level weighted F1: {weighted_f1:.4f} ({weighted_f1 * 100:.2f}%)")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

print("\nPer-class validation report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=classes,
        zero_division=0,
    )
)

# ============================================================
# SAVE REPORTS
# ============================================================

report_path = MODEL_DIR / "validation_recording_report_v5.json"

with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
)

matrix_path = MODEL_DIR / "validation_recording_confusion_matrix_v5.csv"

np.savetxt(
    matrix_path,
    matrix,
    delimiter=",",
    fmt="%d",
)

predictions_path = MODEL_DIR / "validation_recording_predictions_v5.csv"

with open(predictions_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "recording_id",
        "actual_class",
        "predicted_class",
        "confidence",
        "correct",
    ])

    for name, actual, predicted, score in zip(
        recording_names,
        y_true,
        y_pred,
        confidence,
    ):
        writer.writerow([
            name,
            classes[int(actual)],
            classes[int(predicted)],
            float(score),
            bool(actual == predicted),
        ])

summary = {
    "model": str(MODEL_PATH),
    "evaluation_split": "validation",
    "evaluation_unit": "recording",
    "windows": int(len(y_val)),
    "recordings": int(len(recording_names)),
    "classes": num_classes,
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "normalization_mean": mean,
    "normalization_std": std,
    "checkpoint_selection_warning": (
        "The model was trained without validation loss being available "
        "to the callbacks. This evaluation does not establish that the "
        "saved model is the best-generalizing checkpoint."
    ),
}

summary_path = MODEL_DIR / "validation_summary_v5.json"

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved evaluation artifacts:")
print(f"  {summary_path}")
print(f"  {report_path}")
print(f"  {matrix_path}")
print(f"  {predictions_path}")

print("\nV5 validation evaluation completed.")
print("The test set has not been accessed.")
