from pathlib import Path
from collections import defaultdict
import json
import csv

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("SoundSense_Final_Dataset")
MODEL_DIR = Path("soundsense_padded_model_balanced_v3")
OUTPUT_DIR = Path("soundsense_all_windows_evaluation_v3")

SAMPLE_RATE = 16000
DURATION_SECONDS = 2
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SECONDS

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 32

# Keep the original model and normalization unchanged.
MODEL_PATH = MODEL_DIR / "best_model_balanced_v3.keras"
NORMALIZATION_PATH = MODEL_DIR / "normalization_stats.npz"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD ORIGINAL MODEL AND NORMALIZATION
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not NORMALIZATION_PATH.exists():
    raise FileNotFoundError(
        f"Normalization file not found: {NORMALIZATION_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

with np.load(NORMALIZATION_PATH) as stats:
    mean = float(stats["mean"])
    std = float(stats["std"])

if not np.isfinite(std) or std <= 0:
    raise ValueError(f"Invalid normalization standard deviation: {std}")

with open("soundsense_padded_processed_2s/class_names.json") as f:
    classes = json.load(f)

class_to_index = {name: i for i, name in enumerate(classes)}

print("Loaded original model:", MODEL_PATH)
print("Classes:", len(classes))
print("Normalization mean:", mean)
print("Normalization std:", std)

# ============================================================
# WINDOW EXTRACTION
# Unlike the original preprocessing, do not cap long recordings
# at five windows.
# ============================================================

def extract_windows(audio):
    if len(audio) == 0:
        return []

    if len(audio) < TARGET_SAMPLES:
        padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        padded[:len(audio)] = audio
        return [padded]

    starts = list(
        range(0, len(audio) - TARGET_SAMPLES + 1, SAMPLE_RATE)
    )

    final_start = len(audio) - TARGET_SAMPLES

    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    return [
        audio[start:start + TARGET_SAMPLES].astype(np.float32)
        for start in starts
    ]


def extract_mfcc(window):
    mfcc = librosa.feature.mfcc(
        y=window,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return mfcc.T[..., np.newaxis].astype(np.float32)


# ============================================================
# EVALUATION
# Stream windows in batches to avoid excessive memory use.
# ============================================================

def evaluate_split(split):
    split_dir = DATASET_DIR / split

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {split_dir}")

    grouped_probabilities = defaultdict(list)
    grouped_labels = {}
    original_window_counts = defaultdict(int)

    batch_features = []
    batch_metadata = []

    total_recordings = 0
    total_windows = 0
    errors = []

    def predict_batch():
        nonlocal batch_features, batch_metadata, total_windows

        if not batch_features:
            return

        X = np.stack(batch_features).astype(np.float32)
        X = (X - mean) / std

        probabilities = model.predict(X, verbose=0)

        for meta, probability in zip(batch_metadata, probabilities):
            recording_id, label, window_index = meta
            grouped_probabilities[recording_id].append(probability)
            total_windows += 1

        batch_features.clear()
        batch_metadata.clear()

    print(f"\nProcessing {split.upper()} recordings...")

    for class_name in classes:
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        label = class_to_index[class_name]

        for path in sorted(class_dir.rglob("*.wav")):
            recording_id = str(path.relative_to(split_dir))

            try:
                audio, _ = librosa.load(
                    path,
                    sr=SAMPLE_RATE,
                    mono=True,
                )

                if len(audio) == 0:
                    errors.append(f"{recording_id}: empty recording")
                    continue

                windows = extract_windows(audio)

                if not windows:
                    errors.append(f"{recording_id}: no windows")
                    continue

                total_recordings += 1
                grouped_labels[recording_id] = label

                # Read the original preprocessing window count
                # for a direct comparison.
                original_window_counts[recording_id] = min(
                    len(windows), 5
                )

                for window_index, window in enumerate(windows):
                    batch_features.append(extract_mfcc(window))
                    batch_metadata.append(
                        (recording_id, label, window_index)
                    )

                    if len(batch_features) >= 64:
                        predict_batch()

            except Exception as exc:
                errors.append(f"{recording_id}: {exc}")

    predict_batch()

    if not grouped_probabilities:
        raise RuntimeError(f"No usable recordings in {split}")

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

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    with open(
        OUTPUT_DIR / f"{split}_all_windows_report.json", "w"
    ) as f:
        json.dump(report, f, indent=2)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
    )

    np.savetxt(
        OUTPUT_DIR / f"{split}_all_windows_confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    with open(
        OUTPUT_DIR / f"{split}_all_windows_predictions.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "recording_id",
            "actual_class",
            "predicted_class",
            "confidence",
            "windows_evaluated",
            "original_windows",
            "correct",
        ])

        for name, actual, predicted, probabilities in zip(
            recording_names, y_true, y_pred, y_prob
        ):
            writer.writerow([
                name,
                classes[int(actual)],
                classes[int(predicted)],
                float(np.max(probabilities)),
                len(grouped_probabilities[name]),
                original_window_counts[name],
                bool(actual == predicted),
            ])

    result = {
        "split": split,
        "recordings_evaluated": len(recording_names),
        "windows_evaluated": total_windows,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "processing_errors": len(errors),
    }

    print("\nRESULTS")
    print("-" * 50)
    print(json.dumps(result, indent=2))

    if errors:
        error_path = OUTPUT_DIR / f"{split}_processing_errors.txt"
        error_path.write_text("\n".join(errors))
        print(f"Processing errors saved to: {error_path}")

    return result


# ============================================================
# RUN VALIDATION FIRST
# Test evaluation is optional and only for final comparison.
# ============================================================

results = []

results.append(evaluate_split("validation"))

print("\nValidation evaluation complete.")
print("Results saved to:", OUTPUT_DIR.resolve())

answer = input(
    "\nRun the expanded-window test evaluation now? "
    "Type YES to continue: "
).strip()

if answer == "YES":
    results.append(evaluate_split("test"))

with open(OUTPUT_DIR / "evaluation_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nEVALUATION COMPLETE")
print("Original model and processed dataset were not modified.")
print("Output directory:", OUTPUT_DIR.resolve())
