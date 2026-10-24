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

DATA_DIR = Path("soundsense_padded_processed_2s")
MODEL_DIR = Path("soundsense_padded_model_balanced_v2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.0005

MEAN = -23.255867
STD = 94.155594

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
    raise ValueError(f"Expected 24 classes, found {NUM_CLASSES}")

if not (
    X_train.shape[1:] == X_val.shape[1:]
):
    raise ValueError("Training and validation feature shapes differ.")

if set(np.unique(y_train)) != set(range(NUM_CLASSES)):
    raise ValueError("Training classes are missing or invalid.")

if set(np.unique(y_val)) - set(range(NUM_CLASSES)):
    raise ValueError("Unexpected validation class index.")

print("Training windows:", len(y_train))
print("Validation windows:", len(y_val))
print("Classes:", NUM_CLASSES)
print("Using fixed normalization:", MEAN, STD)

# ============================================================
# GROUP WINDOWS BY RECORDING
# ============================================================

def group_recordings(y, ids):
    groups = defaultdict(list)
    labels = {}

    for index, (label, recording_id) in enumerate(zip(y, ids)):
        recording_id = str(recording_id)
        label = int(label)

        if recording_id in labels and labels[recording_id] != label:
            raise ValueError(
                f"Inconsistent labels for recording {recording_id}"
            )

        labels[recording_id] = label
        groups[recording_id].append(index)

    recording_names = sorted(groups)

    return (
        recording_names,
        [np.asarray(groups[name], dtype=np.int32) for name in recording_names],
        np.asarray([labels[name] for name in recording_names], dtype=np.int32),
    )


train_names, train_groups, train_recording_labels = group_recordings(
    y_train, ids_train
)

val_names, val_groups, val_recording_labels = group_recordings(
    y_val, ids_val
)

print("Unique training recordings:", len(train_names))
print("Unique validation recordings:", len(val_names))

# Use recording counts, not window counts, to determine class balance.
recording_counts = np.bincount(
    train_recording_labels,
    minlength=NUM_CLASSES,
)

if np.any(recording_counts == 0):
    raise ValueError("A class has no training recordings.")

print("\nTraining recording counts:")
for index, name in enumerate(CLASSES):
    print(f"{name:20s}: {recording_counts[index]}")

# ============================================================
# RECORDING-LEVEL BALANCED BATCH GENERATOR
# Each selected recording contributes one randomly selected window.
# This reduces overrepresentation from recordings with many windows.
# ============================================================

class_indices = [
    np.flatnonzero(train_recording_labels == class_index)
    for class_index in range(NUM_CLASSES)
]

def balanced_generator():
    rng = np.random.default_rng(SEED)

    while True:
        # Choose a balanced sequence of recording classes each epoch.
        chosen_classes = np.tile(
            np.arange(NUM_CLASSES),
            int(np.ceil(len(train_names) / NUM_CLASSES)),
        )[:len(train_names)]

        rng.shuffle(chosen_classes)

        for start in range(0, len(chosen_classes), BATCH_SIZE):
            batch_classes = chosen_classes[start:start + BATCH_SIZE]

            batch_X = []
            batch_y = []

            for class_index in batch_classes:
                # Select a recording uniformly within the chosen class.
                recording_position = rng.choice(
                    class_indices[int(class_index)]
                )

                window_indices = train_groups[int(recording_position)]

                # Randomize the chosen window for this training example.
                window_index = int(rng.choice(window_indices))

                x = np.asarray(X_train[window_index], dtype=np.float32)
                x = (x - MEAN) / STD

                batch_X.append(x)
                batch_y.append(int(class_index))

            yield (
                np.stack(batch_X).astype(np.float32),
                np.asarray(batch_y, dtype=np.int32),
            )


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


train_ds = tf.data.Dataset.from_generator(
    balanced_generator,
    output_signature=(
        tf.TensorSpec(
            shape=(None,) + tuple(X_train.shape[1:]),
            dtype=tf.float32,
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    ),
).prefetch(1)

val_ds = make_validation_dataset()

# ============================================================
# CNN ARCHITECTURE
# Preserve the existing architecture for a controlled experiment.
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
# RECORDING-LEVEL VALIDATION CALLBACK
# Save the best checkpoint by recording-level macro F1.
# ============================================================

class RecordingValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_f1 = -1.0
        self.best_path = MODEL_DIR / "best_model_balanced_v2.keras"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}

        probabilities = self.model.predict(val_ds, verbose=0)

        grouped = defaultdict(list)
        true_labels = {}

        for probability, label, recording_id in zip(
            probabilities, y_val, ids_val
        ):
            recording_id = str(recording_id)
            grouped[recording_id].append(probability)

            if recording_id in true_labels:
                if true_labels[recording_id] != int(label):
                    raise ValueError(
                        f"Inconsistent validation labels: {recording_id}"
                    )
            else:
                true_labels[recording_id] = int(label)

        names = sorted(grouped)

        y_true = np.asarray(
            [true_labels[name] for name in names],
            dtype=np.int32,
        )

        y_prob = np.asarray(
            [np.mean(grouped[name], axis=0) for name in names],
            dtype=np.float32,
        )

        y_pred = np.argmax(y_prob, axis=1)

        val_macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )

        val_recording_accuracy = accuracy_score(y_true, y_pred)

        logs["val_recording_macro_f1"] = float(val_macro_f1)
        logs["val_recording_accuracy"] = float(val_recording_accuracy)

        print(
            f"\nRecording validation — "
            f"accuracy: {val_recording_accuracy:.4f}, "
            f"macro F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > self.best_f1:
            self.best_f1 = val_macro_f1
            self.model.save(self.best_path)
            print(f"Saved improved checkpoint: {self.best_path}")

    def on_train_end(self, logs=None):
        print("Best recording-level validation macro F1:", self.best_f1)


recording_callback = RecordingValidationCallback()

callbacks = [
    recording_callback,
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=False,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(MODEL_DIR / "training_history_balanced_v2.csv")
    ),
]

# ============================================================
# TRAIN
# ============================================================

steps_per_epoch = int(np.ceil(len(train_names) / BATCH_SIZE))

print("\nStarting recording-balanced training...")
print("Steps per epoch:", steps_per_epoch)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=int(np.ceil(len(y_val) / BATCH_SIZE)),
    callbacks=callbacks,
    verbose=1,
)

best_path = MODEL_DIR / "best_model_balanced_v2.keras"

if not best_path.exists():
    raise RuntimeError("No best checkpoint was saved.")

best_model = tf.keras.models.load_model(best_path)
best_model.save(MODEL_DIR / "soundsense_balanced_v2.keras")

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(CLASSES, f, indent=2)

# ============================================================
# FINAL VALIDATION EVALUATION
# Do not inspect the test set during model selection.
# ============================================================

def evaluate_validation():
    probabilities = best_model.predict(val_ds, verbose=1)

    grouped = defaultdict(list)
    labels = {}

    for probability, label, recording_id in zip(
        probabilities, y_val, ids_val
    ):
        recording_id = str(recording_id)
        grouped[recording_id].append(probability)

        if recording_id in labels and labels[recording_id] != int(label):
            raise ValueError(f"Inconsistent labels: {recording_id}")

        labels[recording_id] = int(label)

    names = sorted(grouped)

    y_true = np.asarray([labels[name] for name in names], dtype=np.int32)
    y_prob = np.asarray(
        [np.mean(grouped[name], axis=0) for name in names],
        dtype=np.float32,
    )
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    np.savetxt(
        MODEL_DIR / "validation_confusion_matrix.csv",
        confusion_matrix(
            y_true, y_pred, labels=np.arange(NUM_CLASSES)
        ),
        delimiter=",",
        fmt="%d",
    )

    with open(MODEL_DIR / "validation_recording_report.json", "w") as f:
        json.dump(report, f, indent=2)

    results = {
        "recordings": len(names),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(NUM_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
    }

    with open(MODEL_DIR / "validation_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nFINAL VALIDATION RESULTS")
    print(json.dumps(results, indent=2))

    print("\nPer-class results:")
    for name in CLASSES:
        row = report[name]
        print(
            f"{name:20s} "
            f"precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} "
            f"F1={row['f1-score']:.3f} "
            f"support={int(row['support'])}"
        )


evaluate_validation()

print("\nEXPERIMENT COMPLETE")
print("Original model was not overwritten.")
print("Experimental model directory:", MODEL_DIR.resolve())