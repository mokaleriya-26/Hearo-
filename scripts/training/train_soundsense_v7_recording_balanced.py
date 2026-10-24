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

DATA_DIR = Path("soundsense_processed_v6_20class")
MODEL_DIR = Path("soundsense_padded_model_v7_recording_balanced")

SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
CHUNK_SIZE = 256

# Prevent extreme sample weights.
MIN_WEIGHT = 0.25
MAX_WEIGHT = 4.0

MODEL_DIR.mkdir(parents=True, exist_ok=False)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================

def load_split(split):
    X = np.load(DATA_DIR / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA_DIR / f"y_{split}.npy", mmap_mode="r")
    ids = np.load(
        DATA_DIR / f"recording_ids_{split}.npy",
        allow_pickle=False,
    )

    if not (len(X) == len(y) == len(ids)):
        raise ValueError(f"Array length mismatch for {split}")

    return X, y, ids


print("Loading SoundSense V7 dataset...")

X_train, y_train, ids_train = load_split("train")
X_val, y_val, ids_val = load_split("validation")

with open(DATA_DIR / "class_names.json") as f:
    CLASSES = json.load(f)

NUM_CLASSES = len(CLASSES)

if NUM_CLASSES != 20:
    raise ValueError(f"Expected 20 classes, found {NUM_CLASSES}")

if X_train.shape[1:] != X_val.shape[1:]:
    raise ValueError("Training and validation feature shapes differ.")

if set(np.unique(y_train)) != set(range(NUM_CLASSES)):
    raise ValueError("Training labels do not cover all 20 classes.")

if set(np.unique(y_val)) - set(range(NUM_CLASSES)):
    raise ValueError("Unexpected validation labels.")

# Check that every recording has a consistent label.
def validate_recording_labels(y, ids, split):
    labels = {}

    for label, rid in zip(y, ids):
        rid = str(rid)
        label = int(label)

        if rid in labels and labels[rid] != label:
            raise ValueError(
                f"Inconsistent label in {split}: recording {rid}"
            )

        labels[rid] = label

    return labels


train_recording_labels = validate_recording_labels(
    y_train, ids_train, "train"
)

validate_recording_labels(y_val, ids_val, "validation")

print("Training windows:", len(y_train))
print("Training recordings:", len(train_recording_labels))
print("Validation windows:", len(y_val))
print("Validation recordings:", len(np.unique(ids_val)))
print("Number of classes:", NUM_CLASSES)

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(CLASSES, f, indent=2)

# ============================================================
# NORMALIZATION — TRAINING DATA ONLY
# Same method as V6.
# ============================================================

print("\nCalculating training-only normalization statistics...")

total_sum = 0.0
total_squared_sum = 0.0
total_elements = 0

for start in range(0, len(X_train), CHUNK_SIZE):
    chunk = np.asarray(
        X_train[start:start + CHUNK_SIZE],
        dtype=np.float64,
    )

    total_sum += np.sum(chunk, dtype=np.float64)
    total_squared_sum += np.sum(chunk * chunk, dtype=np.float64)
    total_elements += chunk.size

mean = total_sum / total_elements
variance = total_squared_sum / total_elements - mean * mean
std = float(np.sqrt(max(variance, 0.0)))

if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-8:
    raise ValueError(f"Invalid normalization statistics: {mean}, {std}")

np.savez(
    MODEL_DIR / "normalization_stats.npz",
    mean=np.float32(mean),
    std=np.float32(std),
)

print(f"Training mean: {mean:.6f}")
print(f"Training std:  {std:.6f}")

# ============================================================
# RECORDING-AWARE SAMPLE WEIGHTS
#
# Each recording's total weight is divided across its windows.
# Each class receives a factor inversely proportional to its
# unique training recording count.
#
# Weights are normalized to mean 1 and clipped to avoid
# excessive influence from rare classes.
# ============================================================

print("\nCalculating recording-aware sample weights...")

recording_window_counts = defaultdict(int)

for rid in ids_train:
    recording_window_counts[str(rid)] += 1

class_recording_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for rid, label in train_recording_labels.items():
    class_recording_counts[label] += 1

if np.any(class_recording_counts == 0):
    raise ValueError("At least one class has no training recordings.")

# Inverse-square-root class weighting is intentionally smoother
# than full inverse-frequency weighting.
class_factors = 1.0 / np.sqrt(class_recording_counts.astype(np.float64))

# Normalize class factors to mean 1 across classes.
class_factors /= np.mean(class_factors)

sample_weights = np.empty(len(y_train), dtype=np.float32)

for i, (label, rid) in enumerate(zip(y_train, ids_train)):
    rid = str(rid)

    # Divide the recording's contribution across its windows.
    sample_weights[i] = (
        class_factors[int(label)]
        / recording_window_counts[rid]
    )

# Normalize to mean 1, then clip and normalize again.
sample_weights /= np.mean(sample_weights)
sample_weights = np.clip(
    sample_weights,
    MIN_WEIGHT,
    MAX_WEIGHT,
)
sample_weights /= np.mean(sample_weights)

if not np.all(np.isfinite(sample_weights)):
    raise ValueError("Non-finite sample weights detected.")

print("\nRecording-aware class factors:")
for i, name in enumerate(CLASSES):
    print(
        f"{name:22} "
        f"recordings={class_recording_counts[i]:6,d} "
        f"class_factor={class_factors[i]:.4f}"
    )

print(f"\nSample-weight minimum: {sample_weights.min():.4f}")
print(f"Sample-weight maximum: {sample_weights.max():.4f}")
print(f"Sample-weight mean:    {sample_weights.mean():.4f}")

# ============================================================
# DATASETS
# ============================================================

def make_dataset(X, y, training=False, weights=None):
    n = len(y)

    if weights is not None and len(weights) != n:
        raise ValueError("Sample-weight length mismatch.")

    def generator():
        rng = np.random.default_rng(
            None if training else SEED
        )

        while True:
            order = np.arange(n)

            if training:
                rng.shuffle(order)

            for start in range(0, n, BATCH_SIZE):
                idx = order[start:start + BATCH_SIZE]

                xb = np.asarray(X[idx], dtype=np.float32)
                xb = (
                    xb - np.float32(mean)
                ) / np.float32(std)

                yb = np.asarray(y[idx], dtype=np.int32)

                if weights is not None:
                    wb = np.asarray(weights[idx], dtype=np.float32)
                    yield xb, yb, wb
                else:
                    yield xb, yb

    x_signature = tf.TensorSpec(
        shape=(None,) + tuple(X.shape[1:]),
        dtype=tf.float32,
    )
    y_signature = tf.TensorSpec(
        shape=(None,),
        dtype=tf.int32,
    )

    if weights is not None:
        signature = (
            x_signature,
            y_signature,
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )
    else:
        signature = (x_signature, y_signature)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=signature,
    )

    return ds.prefetch(1)


train_ds = make_dataset(
    X_train,
    y_train,
    training=True,
    weights=sample_weights,
)

val_ds = make_dataset(
    X_val,
    y_val,
    training=False,
)

STEPS_PER_EPOCH = (
    len(y_train) + BATCH_SIZE - 1
) // BATCH_SIZE

VALIDATION_STEPS = (
    len(y_val) + BATCH_SIZE - 1
) // BATCH_SIZE

# ============================================================
# CNN — SAME ARCHITECTURE AS V6
# ============================================================

def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        16, (5, 3), padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


model = build_model(X_train.shape[1:], NUM_CLASSES)
model.summary()

# ============================================================
# CALLBACKS
# ============================================================

best_model_path = MODEL_DIR / "best_model_v7_recording_balanced.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(MODEL_DIR / "training_history_v7.csv")
    ),
]

# ============================================================
# TRAIN
# ============================================================

print("\nStarting SoundSense V7 recording-aware training...")
print("V6 files will not be modified.")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1,
)

if not history.history.get("val_loss"):
    raise RuntimeError("No validation loss was recorded.")

if not best_model_path.exists():
    raise RuntimeError("Best validation checkpoint was not created.")

best_epoch = int(
    np.argmin(history.history["val_loss"]) + 1
)
best_val_loss = float(min(history.history["val_loss"]))

print("\nLoading best validation-loss checkpoint...")
best_model = tf.keras.models.load_model(best_model_path)

final_model_path = MODEL_DIR / "soundsense_v7_recording_balanced.keras"
best_model.save(final_model_path)

# ============================================================
# RECORDING-LEVEL VALIDATION
# Average probabilities across windows from each recording.
# ============================================================

print("\nPredicting validation windows...")

all_probs = []

for start in range(0, len(X_val), BATCH_SIZE):
    xb = np.asarray(
        X_val[start:start + BATCH_SIZE],
        dtype=np.float32,
    )
    xb = (xb - np.float32(mean)) / np.float32(std)

    probs = best_model(xb, training=False).numpy()
    all_probs.append(probs)

window_probs = np.concatenate(all_probs, axis=0)

if len(window_probs) != len(y_val):
    raise RuntimeError("Validation prediction count mismatch.")

recording_probs = defaultdict(list)
recording_labels = {}

for i, recording_id in enumerate(ids_val):
    rid = str(recording_id)
    recording_probs[rid].append(window_probs[i])

    label = int(y_val[i])

    if rid in recording_labels and recording_labels[rid] != label:
        raise ValueError(f"Inconsistent validation label: {rid}")

    recording_labels[rid] = label

recording_ids = sorted(recording_probs)

y_true = np.asarray(
    [recording_labels[rid] for rid in recording_ids],
    dtype=np.int32,
)

y_pred = np.asarray(
    [
        np.mean(recording_probs[rid], axis=0).argmax()
        for rid in recording_ids
    ],
    dtype=np.int32,
)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)
weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
)

summary = {
    "model": "SoundSense V7 recording-aware",
    "classes": NUM_CLASSES,
    "validation_recordings": len(recording_ids),
    "validation_windows": len(y_val),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "test_set_accessed": False,
    "weighting": "inverse_sqrt_recording_count_divided_by_windows",
    "weight_clip": [MIN_WEIGHT, MAX_WEIGHT],
}

with open(MODEL_DIR / "validation_summary_v7.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(MODEL_DIR / "validation_recording_report_v7.json", "w") as f:
    json.dump(report, f, indent=2)

with open(MODEL_DIR / "validation_recording_confusion_matrix_v7.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["actual/predicted"] + CLASSES)

    for name, row in zip(CLASSES, cm):
        writer.writerow([name] + row.tolist())

with open(MODEL_DIR / "validation_recording_predictions_v7.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["recording_id", "actual", "predicted"])

    for rid, actual, predicted in zip(recording_ids, y_true, y_pred):
        writer.writerow([
            rid,
            CLASSES[int(actual)],
            CLASSES[int(predicted)],
        ])

print("\n========== V7 VALIDATION RESULTS ==========")
print(f"Validation recordings:    {len(recording_ids)}")
print(f"Recording-level accuracy: {accuracy:.4f} ({accuracy:.2%})")
print(f"Recording-level macro F1: {macro_f1:.4f} ({macro_f1:.2%})")
print(f"Recording-level weighted F1: {weighted_f1:.4f} ({weighted_f1:.2%})")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")

print("\nPer-class validation report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASSES,
        zero_division=0,
    )
)

print("\nSaved V7 artifacts to:", MODEL_DIR.resolve())
print("V6 artifacts remain unchanged.")
print("Test set has not been accessed.")
print("V7 training and validation completed.")
