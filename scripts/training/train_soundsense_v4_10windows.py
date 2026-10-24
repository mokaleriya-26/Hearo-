from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
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
MODEL_DIR = Path("soundsense_padded_model_v4_10windows")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# LOAD DATA USING MEMORY MAPPING
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


print("Loading padded SoundSense dataset...")

X_train, y_train, ids_train = load_split("train")
X_val, y_val, ids_val = load_split("validation")

with open(DATA_DIR / "metadata.json") as f:
    metadata = json.load(f)

CLASSES = metadata["classes"]
NUM_CLASSES = len(CLASSES)

print("\nDataset shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Number of classes:", NUM_CLASSES)

if X_train.shape[1:] != X_val.shape[1:]:
    raise ValueError("Training and validation feature shapes differ.")

if set(np.unique(y_train)) != set(range(NUM_CLASSES)):
    raise ValueError("One or more classes are missing from training.")

if set(np.unique(y_val)) - set(range(NUM_CLASSES)):
    raise ValueError("Unexpected validation class index.")

# ============================================================
# NORMALIZATION
# Calculate statistics from training data only.
# Use chunked processing to avoid loading the full dataset.
# ============================================================

print("\nCalculating training normalization statistics...")

total_sum = 0.0
total_squared_sum = 0.0
total_elements = 0

CHUNK_SIZE = 256

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

if not np.isfinite(std) or std < 1e-8:
    raise ValueError(f"Invalid standard deviation: {std}")

np.savez(
    MODEL_DIR / "normalization_stats.npz",
    mean=np.float32(mean),
    std=np.float32(std),
)

print(f"Feature mean: {mean:.6f}")
print(f"Feature std:  {std:.6f}")

# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=np.asarray(y_train),
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("\nClass weights:")
for i, name in enumerate(CLASSES):
    print(f"{name:20} {class_weights[i]:.4f}")

# ============================================================
# BATCHED DATA PIPELINE
# Normalize on the fly and keep memory usage manageable.
# ============================================================

def make_dataset(X, y, training=False):
    indices = np.arange(len(y))

    if training:
        rng = np.random.default_rng(SEED)
        rng.shuffle(indices)

    def generator():
        order = indices.copy()

        if training:
            rng = np.random.default_rng()
            rng.shuffle(order)

        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]
            xb = np.asarray(X[batch_indices], dtype=np.float32)
            xb = (xb - mean) / std
            yb = np.asarray(y[batch_indices], dtype=np.int32)
            yield xb, yb

    output_signature = (
        tf.TensorSpec(
            shape=(None,) + tuple(X.shape[1:]),
            dtype=tf.float32,
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature,
    )

    return dataset.prefetch(1)


train_ds = make_dataset(X_train, y_train, training=True)
val_ds = make_dataset(X_val, y_val)

# ============================================================
# CNN MODEL — SAME ARCHITECTURE AS PREVIOUS EXPERIMENT
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
        num_classes, activation="softmax"
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

best_model_path = MODEL_DIR / "best_model_v4_10windows.keras"

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
        str(MODEL_DIR / "training_history_v4_10windows.csv")
    ),
]

# ============================================================
# TRAIN
# ============================================================

print("\nStarting SoundSense V4 training (10-window cap)...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=(len(y_train) + BATCH_SIZE - 1) // BATCH_SIZE,
    validation_steps=(len(y_val) + BATCH_SIZE - 1) // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# Load and save the best validation-loss checkpoint.
print("\nLoading best model checkpoint...")
best_model = tf.keras.models.load_model(best_model_path)
best_model.save(MODEL_DIR / "soundsense_v4_10windows.keras")

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(CLASSES, f, indent=2)

# ============================================================
# RECORDING-LEVEL AGGREGATION
# ============================================================

def aggregate_recordings(probabilities, labels, recording_ids):
    grouped_probabilities = defaultdict(list)
    grouped_labels = {}

    for probability, label, recording_id in zip(
        probabilities, labels, recording_ids
    ):
        recording_id = str(recording_id)
        grouped_probabilities[recording_id].append(probability)

        if recording_id in grouped_labels:
            if grouped_labels[recording_id] != int(label):
                raise ValueError(
                    f"Inconsistent labels for {recording_id}"
                )
        else:
            grouped_labels[recording_id] = int(label)

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

    return recording_names, y_true, y_prob, y_pred


def evaluate_split(split, X, y, ids):
    print(f"\nEvaluating {split.upper()} set...")

    dataset = make_dataset(X, y)
    probabilities = best_model.predict(dataset, verbose=1)

    names, y_true, y_prob, y_pred = aggregate_recordings(
        probabilities,
        y,
        ids,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"\n{split.upper()} recording-level accuracy: {accuracy:.4f}")
    print(f"{split.upper()} recording-level macro F1: {macro_f1:.4f}")
    print(f"{split.upper()} recording-level weighted F1: {weighted_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    with open(
        MODEL_DIR / f"{split}_recording_report_v4_10windows.json",
        "w",
    ) as f:
        json.dump(report, f, indent=2)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
    )

    np.savetxt(
        MODEL_DIR / f"{split}_recording_confusion_matrix_v4_10windows.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    with open(
        MODEL_DIR / f"{split}_recording_predictions_v4_10windows.csv",
        "w",
        newline="",
    ) as f:
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

    return {
        "split": split,
        "recordings": len(names),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


# Evaluate validation only.
# Keep the test set untouched until a final candidate is selected.
results = []
results.append(evaluate_split("validation", X_val, y_val, ids_val))

with open(MODEL_DIR / "validation_summary_v4_10windows.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nTRAINING AND VALIDATION COMPLETE")
print("Model directory:", MODEL_DIR.resolve())
print("Validation summary:")
print(json.dumps(results, indent=2))
