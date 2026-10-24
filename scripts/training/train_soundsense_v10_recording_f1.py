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
# SOUNDSENSE V10: CONTROLLED MFCC AUGMENTATION EXPERIMENT
# ============================================================

DATA_DIR = Path("soundsense_processed_v9")
MODEL_DIR = Path("soundsense_padded_model_v10_recording_f1")

SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
CHUNK_SIZE = 256

MIN_WEIGHT = 0.25
MAX_WEIGHT = 4.0

PATIENCE = 15
LR_PATIENCE = 5

# Mild MFCC augmentation. Training batches only.
AUGMENTATION_ENABLED = True
MAX_TIME_SHIFT_FRAMES = 8
MAX_TIME_MASK_FRAMES = 8
MAX_COEFFICIENT_MASK_WIDTH = 2
FEATURE_NOISE_STD = 0.03

np.random.seed(SEED)
tf.random.set_seed(SEED)

if MODEL_DIR.exists():
    raise FileExistsError(
        f"{MODEL_DIR} already exists. Nothing was overwritten."
    )

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Missing dataset: {DATA_DIR}")


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_split(split):
    X = np.load(DATA_DIR / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA_DIR / f"y_{split}.npy", mmap_mode="r")
    ids = np.load(
        DATA_DIR / f"recording_ids_{split}.npy",
        allow_pickle=False,
    )

    if not (len(X) == len(y) == len(ids)):
        raise ValueError(f"Array length mismatch in {split}")

    return X, y, ids


print("Loading V9 processed training and validation data...")

X_train, y_train, ids_train = load_split("train")
X_val, y_val, ids_val = load_split("validation")

with open(DATA_DIR / "class_names.json") as f:
    CLASSES = json.load(f)

NUM_CLASSES = len(CLASSES)

if NUM_CLASSES != 20 or len(set(CLASSES)) != NUM_CLASSES:
    raise ValueError("Expected 20 unique sound classes.")

if X_train.shape[1:] != (201, 13, 1):
    raise ValueError(f"Unexpected training shape: {X_train.shape}")

if X_val.shape[1:] != X_train.shape[1:]:
    raise ValueError("Training and validation shapes differ.")

if set(np.unique(y_train)) != set(range(NUM_CLASSES)):
    raise ValueError("Training labels do not cover all classes.")

if set(np.unique(y_val)) != set(range(NUM_CLASSES)):
    raise ValueError("Validation labels do not cover all classes.")


def validate_recordings(y, ids, split):
    labels = {}

    for label_value, rid_value in zip(y, ids):
        rid = str(rid_value)
        label = int(label_value)

        if rid in labels and labels[rid] != label:
            raise ValueError(
                f"Inconsistent labels for recording {rid} in {split}"
            )

        labels[rid] = label

    return labels


train_recording_labels = validate_recordings(
    y_train, ids_train, "train"
)

val_recording_labels = validate_recordings(
    y_val, ids_val, "validation"
)

overlap = set(train_recording_labels).intersection(
    set(val_recording_labels)
)

if overlap:
    raise ValueError(
        f"Train/validation recording ID overlap: {len(overlap)}"
    )

print("Training windows:", len(y_train))
print("Training recordings:", len(train_recording_labels))
print("Validation windows:", len(y_val))
print("Validation recordings:", len(val_recording_labels))
print("Number of classes:", NUM_CLASSES)

MODEL_DIR.mkdir(parents=True)

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(CLASSES, f, indent=2)


# ============================================================
# NORMALIZATION: TRAINING DATA ONLY
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

variance = (
    total_squared_sum / total_elements
    - mean * mean
)

std = float(np.sqrt(max(variance, 0.0)))

if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-8:
    raise ValueError("Invalid normalization statistics.")

np.savez(
    MODEL_DIR / "normalization_stats.npz",
    mean=np.float32(mean),
    std=np.float32(std),
)

print(f"Training mean: {mean:.6f}")
print(f"Training std:  {std:.6f}")


# ============================================================
# RECORDING-AWARE CLASS WEIGHTS: SAME METHOD AS V9
# ============================================================

print("\nCalculating recording-aware sample weights...")

recording_window_counts = defaultdict(int)

for rid_value in ids_train:
    recording_window_counts[str(rid_value)] += 1

class_recording_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for label in train_recording_labels.values():
    class_recording_counts[label] += 1

if np.any(class_recording_counts == 0):
    raise ValueError("At least one class has no training recordings.")

class_factors = 1.0 / np.sqrt(
    class_recording_counts.astype(np.float64)
)

class_factors /= np.mean(class_factors)

sample_weights = np.empty(len(y_train), dtype=np.float32)

for i, (label, rid_value) in enumerate(zip(y_train, ids_train)):
    rid = str(rid_value)

    sample_weights[i] = (
        class_factors[int(label)]
        / recording_window_counts[rid]
    )

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
    MODEL_DIR / "training_sample_weights_v10.npy",
    sample_weights,
)

np.save(
    MODEL_DIR / "class_recording_counts_train.npy",
    class_recording_counts,
)

print("Weight minimum:", float(sample_weights.min()))
print("Weight maximum:", float(sample_weights.max()))
print("Weight mean:", float(sample_weights.mean()))


# ============================================================
# TRAINING-ONLY MFCC AUGMENTATION
# ============================================================

def augment_batch(x_batch, rng):
    """
    x_batch is a copy of normalized MFCC features.
    Validation data never passes through this function.
    """
    x_batch = np.array(x_batch, dtype=np.float32, copy=True)

    batch_size, time_steps, coefficients, channels = x_batch.shape

    for i in range(batch_size):

        # Small temporal shift with zero-fill; no wraparound.
        shift = int(
            rng.integers(
                -MAX_TIME_SHIFT_FRAMES,
                MAX_TIME_SHIFT_FRAMES + 1,
            )
        )

        if shift > 0:
            x_batch[i, shift:, :, :] = x_batch[i, :-shift, :, :]
            x_batch[i, :shift, :, :] = 0.0

        elif shift < 0:
            amount = -shift
            x_batch[i, :-amount, :, :] = x_batch[i, amount:, :, :]
            x_batch[i, -amount:, :, :] = 0.0

        # Small additive noise in normalized feature space.
        if FEATURE_NOISE_STD > 0:
            noise = rng.normal(
                0.0,
                FEATURE_NOISE_STD,
                size=x_batch[i].shape,
            ).astype(np.float32)

            x_batch[i] += noise

        # Random time masking.
        width = int(
            rng.integers(0, MAX_TIME_MASK_FRAMES + 1)
        )

        if 0 < width < time_steps:
            start = int(
                rng.integers(0, time_steps - width + 1)
            )

            x_batch[i, start:start + width, :, :] = 0.0

        # Random masking of adjacent MFCC coefficients.
        width = int(
            rng.integers(
                0,
                MAX_COEFFICIENT_MASK_WIDTH + 1,
            )
        )

        if 0 < width < coefficients:
            start = int(
                rng.integers(0, coefficients - width + 1)
            )

            x_batch[i, :, start:start + width, :] = 0.0

    return x_batch


# ============================================================
# DATA GENERATORS
# ============================================================

def make_dataset(X, y, training=False, weights=None):
    n = len(y)

    def generator():
        rng = (
            np.random.default_rng()
            if training
            else np.random.default_rng(SEED)
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

                if training and AUGMENTATION_ENABLED:
                    xb = augment_batch(xb, rng)

                yb = np.asarray(y[idx], dtype=np.int32)

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
            tf.TensorSpec(
                shape=(None,),
                dtype=tf.float32,
            ),
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
# CNN: SAME ARCHITECTURE AS V9
# ============================================================

def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        16,
        (5, 3),
        padding="same",
        activation="relu",
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(x)

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

        all_probs.append(
            model(xb, training=False).numpy()
        )

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

        if (
            rid in recording_labels
            and recording_labels[rid] != label
        ):
            raise ValueError(
                f"Inconsistent label for recording {rid}"
            )

        recording_labels[rid] = label

    recording_ids = sorted(recording_probs)

    y_true = np.asarray(
        [
            recording_labels[rid]
            for rid in recording_ids
        ],
        dtype=np.int32,
    )

    y_pred = np.asarray(
        [
            np.mean(
                recording_probs[rid],
                axis=0,
            ).argmax()
            for rid in recording_ids
        ],
        dtype=np.int32,
    )

    return {
        "recording_ids": recording_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
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
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================
# CHECKPOINT SELECTION: RECORDING-LEVEL MACRO F1
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
            MODEL_DIR
            / "best_model_v10_recording_macro_f1.keras"
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

        self.history.append({
            "epoch": epoch + 1,
            "val_loss": float(
                logs.get("val_loss", float("nan"))
            ),
            "val_window_accuracy": float(
                logs.get("val_accuracy", float("nan"))
            ),
            "val_recording_accuracy": metrics["accuracy"],
            "val_recording_macro_f1": current_f1,
            "val_recording_weighted_f1": metrics["weighted_f1"],
        })

        print(
            "\nRecording-level validation — "
            f"accuracy: {metrics['accuracy']:.4f}, "
            f"macro F1: {current_f1:.4f}, "
            f"weighted F1: {metrics['weighted_f1']:.4f}"
        )

        if current_f1 > self.best_macro_f1 + 1e-6:
            self.best_macro_f1 = current_f1
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            self.wait = 0

            self.model.save(self.best_model_path)

            print(
                "Saved best recording-level macro F1: "
                f"{current_f1:.4f}"
            )

        else:
            self.wait += 1

        if self.wait >= PATIENCE:
            print(
                "Early stopping: recording-level macro F1 "
                f"did not improve for {PATIENCE} epochs."
            )

            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

        with open(
            MODEL_DIR / "recording_level_history_v10.csv",
            "w",
            newline="",
        ) as f:
            fields = [
                "epoch",
                "val_loss",
                "val_window_accuracy",
                "val_recording_accuracy",
                "val_recording_macro_f1",
                "val_recording_weighted_f1",
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(self.history)

        with open(
            MODEL_DIR / "checkpoint_selection_v10.json",
            "w",
        ) as f:
            json.dump({
                "selection_metric": "recording_level_macro_f1",
                "best_epoch": self.best_epoch,
                "best_recording_macro_f1": self.best_macro_f1,
                "epochs_recorded": len(self.history),
            }, f, indent=2)


# ============================================================
# TRAINING CONFIGURATION AND RUN
# ============================================================

config = {
    "model_version": "V10",
    "input_data": str(DATA_DIR),
    "output_directory": str(MODEL_DIR),
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs_max": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "architecture": "same CNN as V9",
    "class_weighting": (
        "same inverse-square-root recording-aware weights as V9"
    ),
    "normalization": (
        "mean and standard deviation from training features only"
    ),
    "recording_prediction": (
        "mean window probabilities, then argmax"
    ),
    "checkpoint_selection": "recording-level macro F1",
    "augmentation": {
        "enabled": AUGMENTATION_ENABLED,
        "training_only": True,
        "max_time_shift_frames": MAX_TIME_SHIFT_FRAMES,
        "max_time_mask_frames": MAX_TIME_MASK_FRAMES,
        "max_coefficient_mask_width": MAX_COEFFICIENT_MASK_WIDTH,
        "normalized_feature_noise_std": FEATURE_NOISE_STD,
    },
    "test_set_accessed": False,
    "note": (
        "V10 uses the existing V9 split. Repeated validation tuning "
        "can overfit that split. Original test set is not accessed."
    ),
}

with open(
    MODEL_DIR / "training_config_v10.json",
    "w",
) as f:
    json.dump(config, f, indent=2)

model = build_model(
    X_train.shape[1:],
    NUM_CLASSES,
)

model.summary()

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
        str(MODEL_DIR / "training_history_v10.csv")
    ),
]

print("\nStarting SoundSense V10 training...")
print("Augmentation is enabled for training batches only.")
print("Original test set will not be accessed.")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1,
)

if recording_f1_callback.best_weights is None:
    raise RuntimeError("No valid V10 checkpoint was selected.")

best_model = tf.keras.models.load_model(
    recording_f1_callback.best_model_path
)

best_model.save(
    MODEL_DIR / "soundsense_v10_recording_f1.keras"
)


# ============================================================
# FINAL VALIDATION REPORT
# ============================================================

print("\nFinal evaluation of selected V10 checkpoint...")

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
    "model": "SoundSense V10 recording-level macro F1",
    "classes": NUM_CLASSES,
    "validation_recordings": len(recording_ids),
    "validation_windows": len(y_val),
    "accuracy": result["accuracy"],
    "macro_f1": result["macro_f1"],
    "weighted_f1": result["weighted_f1"],
    "best_epoch": recording_f1_callback.best_epoch,
    "best_recording_macro_f1": (
        recording_f1_callback.best_macro_f1
    ),
    "test_set_accessed": False,
    "checkpoint_selection": "recording_level_macro_f1",
    "augmentation": (
        "mild MFCC feature augmentation on training batches only"
    ),
    "validation_note": (
        "V9 validation is used for V10 selection. Repeated experiments "
        "may overfit this split. Original test set was not accessed."
    ),
}

with open(
    MODEL_DIR / "validation_summary_v10.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

with open(
    MODEL_DIR / "validation_recording_report_v10.json",
    "w",
) as f:
    json.dump(report, f, indent=2)

with open(
    MODEL_DIR / "validation_recording_confusion_matrix_v10.csv",
    "w",
    newline="",
) as f:
    writer = csv.writer(f)
    writer.writerow(["actual/predicted"] + CLASSES)

    for name, row in zip(CLASSES, cm):
        writer.writerow([name] + row.tolist())

with open(
    MODEL_DIR / "validation_recording_predictions_v10.csv",
    "w",
    newline="",
) as f:
    writer = csv.writer(f)
    writer.writerow(["recording_id", "actual", "predicted"])

    for rid, actual, predicted in zip(
        recording_ids,
        y_true,
        y_pred,
    ):
        writer.writerow([
            rid,
            CLASSES[int(actual)],
            CLASSES[int(predicted)],
        ])

print("\n========== V10 VALIDATION RESULTS ==========")
print("Validation recordings:", len(recording_ids))
print(f"Recording-level accuracy:    {result['accuracy']:.4f}")
print(f"Recording-level macro F1:    {result['macro_f1']:.4f}")
print(f"Recording-level weighted F1: {result['weighted_f1']:.4f}")
print("Best epoch:", recording_f1_callback.best_epoch)

print("\nPer-class validation report:")
print(classification_report(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASSES,
    zero_division=0,
))

print("\nV10 artifacts saved to:", MODEL_DIR.resolve())
print("V8 and V9 artifact directories remain unchanged.")
print("Original test set has not been accessed.")
print("SoundSense V10 training and validation completed.")
