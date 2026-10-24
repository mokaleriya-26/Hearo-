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
MODEL_DIR = Path("soundsense_padded_model_v8_recording_f1")

SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
CHUNK_SIZE = 256

# Moderate recording-aware weighting.
# The inverse-square-root class factor is normalized to mean 1.
# Recording contribution is divided among its windows.
MIN_WEIGHT = 0.25
MAX_WEIGHT = 4.0

# Model selection is based on recording-level validation macro F1.
PATIENCE = 15
LR_PATIENCE = 5

np.random.seed(SEED)
tf.random.set_seed(SEED)

if MODEL_DIR.exists():
    raise FileExistsError(
        f"{MODEL_DIR} already exists. "
        "Nothing was overwritten. Choose a new output directory."
    )

MODEL_DIR.mkdir(parents=True)

# ============================================================
# LOAD DATA
# ============================================================

def load_split(split):
    X = np.load(
        DATA_DIR / f"X_{split}.npy",
        mmap_mode="r",
    )
    y = np.load(
        DATA_DIR / f"y_{split}.npy",
        mmap_mode="r",
    )
    ids = np.load(
        DATA_DIR / f"recording_ids_{split}.npy",
        allow_pickle=False,
    )

    if not (len(X) == len(y) == len(ids)):
        raise ValueError(f"Array length mismatch for {split}")

    return X, y, ids


print("Loading SoundSense V8 data...")

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

def validate_recordings(y, ids, split):
    labels = {}

    for label, rid in zip(y, ids):
        rid = str(rid)
        label = int(label)

        if rid in labels and labels[rid] != label:
            raise ValueError(
                f"Inconsistent labels for recording {rid} in {split}"
            )

        labels[rid] = label

    return labels


train_recording_labels = validate_recordings(
    y_train, ids_train, "train"
)
validate_recordings(y_val, ids_val, "validation")

# Do not allow the same recording to cross the train/validation boundary.
train_ids = set(train_recording_labels)
val_ids = set(map(str, ids_val))

overlap = train_ids.intersection(val_ids)

if overlap:
    raise ValueError(
        f"Train/validation recording leakage detected: "
        f"{len(overlap)} overlapping IDs"
    )

print("Training windows:", len(y_train))
print("Training recordings:", len(train_recording_labels))
print("Validation windows:", len(y_val))
print("Validation recordings:", len(val_ids))
print("Classes:", NUM_CLASSES)

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(CLASSES, f, indent=2)

# ============================================================
# NORMALIZATION — TRAINING DATA ONLY
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
# ============================================================

print("\nCalculating sample weights...")

recording_window_counts = defaultdict(int)

for rid in ids_train:
    recording_window_counts[str(rid)] += 1

class_recording_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for _, label in train_recording_labels.items():
    class_recording_counts[label] += 1

if np.any(class_recording_counts == 0):
    raise ValueError("At least one class has no training recordings.")

# Moderated inverse-square-root class weighting.
class_factors = 1.0 / np.sqrt(class_recording_counts.astype(np.float64))
class_factors /= np.mean(class_factors)

sample_weights = np.empty(len(y_train), dtype=np.float32)

for i, (label, rid) in enumerate(zip(y_train, ids_train)):
    rid = str(rid)

    sample_weights[i] = (
        class_factors[int(label)]
        / recording_window_counts[rid]
    )

# Normalize, clip extreme values, and normalize again.
sample_weights /= np.mean(sample_weights)
sample_weights = np.clip(
    sample_weights,
    MIN_WEIGHT,
    MAX_WEIGHT,
)
sample_weights /= np.mean(sample_weights)

if not np.all(np.isfinite(sample_weights)):
    raise ValueError("Invalid sample weights.")

np.save(
    MODEL_DIR / "training_sample_weights_v8.npy",
    sample_weights,
)

print(f"Weight minimum: {sample_weights.min():.4f}")
print(f"Weight maximum: {sample_weights.max():.4f}")
print(f"Weight mean:    {sample_weights.mean():.4f}")

# ============================================================
# DATA GENERATORS
# ============================================================

def make_dataset(X, y, training=False, weights=None):
    n = len(y)

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

                xb = np.asarray(
                    X[idx],
                    dtype=np.float32,
                )
                xb = (
                    xb - np.float32(mean)
                ) / np.float32(std)

                yb = np.asarray(
                    y[idx],
                    dtype=np.int32,
                )

                if weights is None:
                    yield xb, yb
                else:
                    wb = np.asarray(
                        weights[idx],
                        dtype=np.float32,
                    )
                    yield xb, yb, wb

    x_signature = tf.TensorSpec(
        shape=(None,) + tuple(X.shape[1:]),
        dtype=tf.float32,
    )

    y_signature = tf.TensorSpec(
        shape=(None,),
        dtype=tf.int32,
    )

    if weights is None:
        signature = (x_signature, y_signature)
    else:
        signature = (
            x_signature,
            y_signature,
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=signature,
    ).prefetch(1)


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
# CNN — SAME ARCHITECTURE AS V6/V7
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
# RECORDING-LEVEL EVALUATION
# ============================================================

def evaluate_recordings(model, X, y, ids, split_name):
    all_probs = []

    for start in range(0, len(X), BATCH_SIZE):
        xb = np.asarray(
            X[start:start + BATCH_SIZE],
            dtype=np.float32,
        )
        xb = (
            xb - np.float32(mean)
        ) / np.float32(std)

        probs = model(xb, training=False).numpy()
        all_probs.append(probs)

    window_probs = np.concatenate(all_probs, axis=0)

    if len(window_probs) != len(y):
        raise RuntimeError(
            f"{split_name}: prediction count mismatch"
        )

    recording_probs = defaultdict(list)
    recording_labels = {}

    for i, rid_value in enumerate(ids):
        rid = str(rid_value)
        label = int(y[i])

        recording_probs[rid].append(window_probs[i])

        if rid in recording_labels and recording_labels[rid] != label:
            raise ValueError(f"Inconsistent label for recording {rid}")

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

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )

    return {
        "recording_ids": recording_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================
# CALLBACK: SELECT CHECKPOINT BY RECORDING-LEVEL MACRO F1
# ============================================================

class RecordingMacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_macro_f1 = -1.0
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.history = []

        self.best_model_path = (
            MODEL_DIR / "best_model_v8_recording_macro_f1.keras"
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}

        metrics = evaluate_recordings(
            self.model,
            X_val,
            y_val,
            ids_val,
            "validation",
        )

        current_f1 = metrics["macro_f1"]
        logs["val_recording_macro_f1"] = current_f1
        logs["val_recording_accuracy"] = metrics["accuracy"]
        logs["val_recording_weighted_f1"] = metrics["weighted_f1"]

        row = {
            "epoch": epoch + 1,
            "val_loss": float(logs.get("val_loss", float("nan"))),
            "val_window_accuracy": float(
                logs.get("val_accuracy", float("nan"))
            ),
            "val_recording_accuracy": metrics["accuracy"],
            "val_recording_macro_f1": current_f1,
            "val_recording_weighted_f1": metrics["weighted_f1"],
        }

        self.history.append(row)

        print(
            f"\nRecording-level validation — "
            f"accuracy: {metrics['accuracy']:.4f}, "
            f"macro F1: {current_f1:.4f}, "
            f"weighted F1: {metrics['weighted_f1']:.4f}"
        )

        # Save the first epoch, then save only when macro F1 improves.
        if current_f1 > self.best_macro_f1 + 1e-6:
            self.best_macro_f1 = current_f1
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            self.wait = 0

            self.model.save(self.best_model_path)

            print(
                f"Saved new best recording-level macro F1: "
                f"{current_f1:.4f}"
            )
        else:
            self.wait += 1

        if self.wait >= PATIENCE:
            print(
                f"\nEarly stopping: recording-level macro F1 "
                f"did not improve for {PATIENCE} epochs."
            )
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

        with open(
            MODEL_DIR / "recording_level_history_v8.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "val_loss",
                    "val_window_accuracy",
                    "val_recording_accuracy",
                    "val_recording_macro_f1",
                    "val_recording_weighted_f1",
                ],
            )
            writer.writeheader()
            writer.writerows(self.history)

        with open(
            MODEL_DIR / "checkpoint_selection_v8.json",
            "w",
        ) as f:
            json.dump(
                {
                    "selection_metric": "recording_level_macro_f1",
                    "best_epoch": self.best_epoch,
                    "best_recording_macro_f1": self.best_macro_f1,
                    "epochs_recorded": len(self.history),
                },
                f,
                indent=2,
            )


# ============================================================
# CALLBACKS AND TRAINING
# ============================================================

recording_f1_callback = RecordingMacroF1Callback()

callbacks = [
    recording_f1_callback,
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(MODEL_DIR / "training_history_v8.csv")
    ),
]

print("\nStarting SoundSense V8 training...")
print("Checkpoint selection: recording-level macro F1")
print("V6 and V7 files will not be modified.")
print("Test data will not be accessed.")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1,
)

if recording_f1_callback.best_weights is None:
    raise RuntimeError("No valid V8 checkpoint was selected.")

# Load the checkpoint actually saved at the best recording-level macro F1.
best_model = tf.keras.models.load_model(
    recording_f1_callback.best_model_path
)

final_model_path = MODEL_DIR / "soundsense_v8_recording_f1.keras"
best_model.save(final_model_path)

# ============================================================
# FINAL RECORDING-LEVEL VALIDATION
# ============================================================

print("\nFinal evaluation of selected V8 checkpoint...")

result = evaluate_recordings(
    best_model,
    X_val,
    y_val,
    ids_val,
    "validation",
)

y_true = result["y_true"]
y_pred = result["y_pred"]
recording_ids = result["recording_ids"]

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
    "model": "SoundSense V8 recording-level macro F1",
    "classes": NUM_CLASSES,
    "validation_recordings": len(recording_ids),
    "validation_windows": len(y_val),
    "accuracy": result["accuracy"],
    "macro_f1": result["macro_f1"],
    "weighted_f1": result["weighted_f1"],
    "best_epoch": recording_f1_callback.best_epoch,
    "best_recording_macro_f1": recording_f1_callback.best_macro_f1,
    "test_set_accessed": False,
    "checkpoint_selection": "recording_level_macro_f1",
}

with open(MODEL_DIR / "validation_summary_v8.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(MODEL_DIR / "validation_recording_report_v8.json", "w") as f:
    json.dump(report, f, indent=2)

with open(
    MODEL_DIR / "validation_recording_confusion_matrix_v8.csv",
    "w",
) as f:
    writer = csv.writer(f)
    writer.writerow(["actual/predicted"] + CLASSES)

    for name, row in zip(CLASSES, cm):
        writer.writerow([name] + row.tolist())

with open(
    MODEL_DIR / "validation_recording_predictions_v8.csv",
    "w",
) as f:
    writer = csv.writer(f)
    writer.writerow(["recording_id", "actual", "predicted"])

    for rid, actual, predicted in zip(recording_ids, y_true, y_pred):
        writer.writerow([
            rid,
            CLASSES[int(actual)],
            CLASSES[int(predicted)],
        ])

print("\n========== V8 VALIDATION RESULTS ==========")
print(f"Validation recordings:    {len(recording_ids)}")
print(f"Recording-level accuracy: {result['accuracy']:.4f}")
print(f"Recording-level macro F1: {result['macro_f1']:.4f}")
print(f"Recording-level weighted F1: {result['weighted_f1']:.4f}")
print(f"Best epoch: {recording_f1_callback.best_epoch}")
print(f"Best macro F1: {recording_f1_callback.best_macro_f1:.4f}")

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

print("\nV8 artifacts saved to:", MODEL_DIR.resolve())
print("V6 and V7 files remain unchanged.")
print("Test set has not been accessed.")
print("SoundSense V8 training and validation completed.")
