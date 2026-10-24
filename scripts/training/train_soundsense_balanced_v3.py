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

SEED = 42
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.0003

DATA_DIR = Path("soundsense_padded_processed_2s")
MODEL_DIR = Path("soundsense_padded_model_balanced_v3")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
        raise ValueError(f"Array length mismatch: {split}")

    return X, y, ids


X_train, y_train, ids_train = load_split("train")
X_val, y_val, ids_val = load_split("validation")

with open(DATA_DIR / "metadata.json") as f:
    metadata = json.load(f)

CLASSES = metadata["classes"]
NUM_CLASSES = len(CLASSES)

if NUM_CLASSES != 24:
    raise ValueError(f"Expected 24 classes, got {NUM_CLASSES}")

with np.load("soundsense_padded_model_2s/normalization_stats.npz") as stats:
    MEAN = float(stats["mean"])
    STD = float(stats["std"])

if not np.isfinite(STD) or STD <= 0:
    raise ValueError(f"Invalid normalization std: {STD}")

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Classes:", NUM_CLASSES)
print("Normalization:", MEAN, STD)

# ============================================================
# GROUP TRAINING WINDOWS BY RECORDING
# ============================================================

recording_windows = defaultdict(list)
recording_labels = {}

for index, (recording_id, label) in enumerate(zip(ids_train, y_train)):
    recording_id = str(recording_id)
    label = int(label)

    if recording_id in recording_labels:
        if recording_labels[recording_id] != label:
            raise ValueError(f"Inconsistent labels: {recording_id}")

    recording_labels[recording_id] = label
    recording_windows[recording_id].append(index)

class_recordings = defaultdict(list)

for recording_id, label in recording_labels.items():
    class_recordings[label].append(recording_id)

for label in range(NUM_CLASSES):
    if not class_recordings[label]:
        raise ValueError(f"No training recordings for {CLASSES[label]}")

print("\nTraining recordings per class:")
for label, name in enumerate(CLASSES):
    print(f"{name:20s}: {len(class_recordings[label]):6d}")

# ============================================================
# NORMALIZED BATCH GENERATOR
#
# Each batch samples classes evenly, then selects a recording
# and one of its windows. Different windows can be selected
# over training epochs.
#
# No additional class_weight is used because the sampling
# itself balances the classes.
# ============================================================

STEPS_PER_EPOCH = max(
    1,
    int(np.ceil(len(y_train) / BATCH_SIZE)),
)

def training_generator():
    rng = np.random.default_rng(SEED)

    while True:
        batch_x = []
        batch_y = []

        # Shuffle class order each batch.
        class_order = rng.permutation(NUM_CLASSES)

        for slot in range(BATCH_SIZE):
            label = int(class_order[slot % NUM_CLASSES])
            recording_id = rng.choice(class_recordings[label])
            available = recording_windows[recording_id]
            window_index = int(rng.choice(available))

            x = np.asarray(X_train[window_index], dtype=np.float32)
            x = (x - MEAN) / STD

            batch_x.append(x)
            batch_y.append(label)

        yield (
            np.stack(batch_x).astype(np.float32),
            np.asarray(batch_y, dtype=np.int32),
        )


train_ds = tf.data.Dataset.from_generator(
    training_generator,
    output_signature=(
        tf.TensorSpec(
            shape=(BATCH_SIZE,) + tuple(X_train.shape[1:]),
            dtype=tf.float32,
        ),
        tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
    ),
).prefetch(1)

# ============================================================
# VALIDATION PIPELINE
# Uses existing five-window processed validation data.
# Recording-level metrics are calculated after every epoch.
# ============================================================

def make_validation_dataset():
    def generator():
        for start in range(0, len(y_val), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(y_val))

            xb = np.asarray(X_val[start:end], dtype=np.float32)
            xb = (xb - MEAN) / STD
            yb = np.asarray(y_val[start:end], dtype=np.int32)

            yield xb, yb

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(None,) + tuple(X_val.shape[1:]),
                dtype=tf.float32,
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    ).prefetch(1)


val_ds = make_validation_dataset()

# ============================================================
# SAME CNN ARCHITECTURE AS ORIGINAL
# ============================================================

def build_model():
    inputs = tf.keras.Input(shape=X_train.shape[1:])

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
        NUM_CLASSES, activation="softmax"
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


model = build_model()
model.summary()

# ============================================================
# RECORDING-LEVEL VALIDATION
# ============================================================

def aggregate_recordings(probabilities, labels, recording_ids):
    grouped_probabilities = defaultdict(list)
    grouped_labels = {}

    for probability, label, recording_id in zip(
        probabilities, labels, recording_ids
    ):
        recording_id = str(recording_id)
        label = int(label)

        if recording_id in grouped_labels:
            if grouped_labels[recording_id] != label:
                raise ValueError(f"Inconsistent label: {recording_id}")
        else:
            grouped_labels[recording_id] = label

        grouped_probabilities[recording_id].append(probability)

    names = sorted(grouped_probabilities)

    y_true = np.asarray(
        [grouped_labels[name] for name in names],
        dtype=np.int32,
    )
    y_prob = np.asarray([
        np.mean(grouped_probabilities[name], axis=0)
        for name in names
    ], dtype=np.float32)

    return names, y_true, y_prob, np.argmax(y_prob, axis=1)


class RecordingLevelValidation(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_macro_f1 = -1.0
        self.best_epoch = 0
        self.history = []
        self.best_path = MODEL_DIR / "best_model_balanced_v3.keras"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        probabilities = self.model.predict(val_ds, verbose=0)

        names, y_true, y_prob, y_pred = aggregate_recordings(
            probabilities, y_val, ids_val
        )

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        logs["val_recording_accuracy"] = float(accuracy)
        logs["val_recording_macro_f1"] = float(macro_f1)
        logs["val_recording_weighted_f1"] = float(weighted_f1)

        row = {
            "epoch": epoch + 1,
            "val_recording_accuracy": float(accuracy),
            "val_recording_macro_f1": float(macro_f1),
            "val_recording_weighted_f1": float(weighted_f1),
            "val_loss": float(logs.get("val_loss", np.nan)),
        }
        self.history.append(row)

        print(
            f"\nRecording-level validation — "
            f"epoch {epoch + 1}: "
            f"accuracy={accuracy:.4f}, "
            f"macro_f1={macro_f1:.4f}, "
            f"weighted_f1={weighted_f1:.4f}"
        )

        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = float(macro_f1)
            self.best_epoch = epoch + 1
            self.model.save(self.best_path)
            print(f"Saved new best model: {self.best_path}")

        with open(MODEL_DIR / "validation_history_v3.json", "w") as f:
            json.dump(self.history, f, indent=2)


recording_validation = RecordingLevelValidation()

callbacks = [
    recording_validation,
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=False,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(MODEL_DIR / "training_history_v3.csv")
    ),
]

# ============================================================
# TRAIN
# ============================================================

print("\nStarting SoundSense Balanced V3 training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=callbacks,
    verbose=1,
)

if not recording_validation.best_path.exists():
    raise RuntimeError("No best model checkpoint was saved.")

best_model = tf.keras.models.load_model(
    recording_validation.best_path
)

best_model.save(MODEL_DIR / "soundsense_balanced_v3.keras")

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(CLASSES, f, indent=2)

# Save normalization for future deployment.
np.savez(
    MODEL_DIR / "normalization_stats.npz",
    mean=np.float32(MEAN),
    std=np.float32(STD),
)

# ============================================================
# FINAL VALIDATION REPORT
# Uses the existing capped-window validation set.
# Expanded-window evaluation is a separate subsequent step.
# ============================================================

print("\nCreating final recording-level validation report...")

probabilities = best_model.predict(val_ds, verbose=1)

names, y_true, y_prob, y_pred = aggregate_recordings(
    probabilities, y_val, ids_val
)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0,
)

with open(MODEL_DIR / "validation_recording_report_v3.json", "w") as f:
    json.dump(report, f, indent=2)

np.savetxt(
    MODEL_DIR / "validation_confusion_matrix_v3.csv",
    confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES)),
    delimiter=",",
    fmt="%d",
)

with open(MODEL_DIR / "validation_predictions_v3.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "recording_id",
        "actual_class",
        "predicted_class",
        "confidence",
        "correct",
    ])

    for name, actual, predicted, probs in zip(
        names, y_true, y_pred, y_prob
    ):
        writer.writerow([
            name,
            CLASSES[int(actual)],
            CLASSES[int(predicted)],
            float(np.max(probs)),
            bool(actual == predicted),
        ])

summary = {
    "experiment": "SoundSense Balanced Training V3",
    "best_epoch": recording_validation.best_epoch,
    "best_validation_macro_f1_during_training":
        recording_validation.best_macro_f1,
    "validation_recordings": len(names),
    "validation_accuracy": float(accuracy_score(y_true, y_pred)),
    "validation_macro_f1": float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    ),
    "validation_weighted_f1": float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ),
    "normalization_mean": MEAN,
    "normalization_std": STD,
}

with open(MODEL_DIR / "evaluation_summary_v3.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nV3 TRAINING COMPLETE")
print(json.dumps(summary, indent=2))
print("\nModel directory:", MODEL_DIR.resolve())
print("Original model was not modified.")
print("Test set was not used.")
