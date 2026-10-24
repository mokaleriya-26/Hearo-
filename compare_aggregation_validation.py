from pathlib import Path
from collections import defaultdict
import json

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

DATASET_DIR = Path("SoundSense_Final_Dataset")
MODEL_DIR = Path("soundsense_padded_model_2s")
OUTPUT_DIR = Path("soundsense_aggregation_experiment")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
TARGET_SAMPLES = 32000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 32
BATCH_SIZE = 64

model = tf.keras.models.load_model(
    MODEL_DIR / "best_model_padded_2s.keras"
)

with np.load(MODEL_DIR / "normalization_stats.npz") as stats:
    mean = float(stats["mean"])
    std = float(stats["std"])

with open("soundsense_padded_processed_2s/class_names.json") as f:
    classes = json.load(f)

class_to_index = {name: i for i, name in enumerate(classes)}


def extract_windows(audio):
    if len(audio) == 0:
        return []

    if len(audio) < TARGET_SAMPLES:
        padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        padded[:len(audio)] = audio
        return [padded]

    starts = list(range(
        0,
        len(audio) - TARGET_SAMPLES + 1,
        SAMPLE_RATE,
    ))

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


def evaluate_aggregation(split="validation"):
    split_dir = DATASET_DIR / split

    recordings = {}
    batch_features = []
    batch_metadata = []
    total_windows = 0
    errors = []

    def predict_batch():
        nonlocal total_windows

        if not batch_features:
            return

        X = np.stack(batch_features).astype(np.float32)
        X = (X - mean) / std
        probabilities = model.predict(X, verbose=0)

        for metadata, probability in zip(batch_metadata, probabilities):
            recording_id, label = metadata
            recordings[recording_id]["probabilities"].append(probability)

        total_windows += len(batch_features)
        batch_features.clear()
        batch_metadata.clear()

    print(f"Processing {split} recordings...")

    for class_name in classes:
        class_dir = split_dir / class_name

        for path in sorted(class_dir.rglob("*.wav")):
            recording_id = str(path.relative_to(split_dir))

            try:
                audio, _ = librosa.load(
                    path,
                    sr=SAMPLE_RATE,
                    mono=True,
                )

                windows = extract_windows(audio)

                if not windows:
                    errors.append(f"{recording_id}: no usable windows")
                    continue

                recordings[recording_id] = {
                    "actual": class_to_index[class_name],
                    "probabilities": [],
                }

                for window in windows:
                    batch_features.append(extract_mfcc(window))
                    batch_metadata.append(
                        (recording_id, class_to_index[class_name])
                    )

                    if len(batch_features) >= BATCH_SIZE:
                        predict_batch()

            except Exception as exc:
                errors.append(f"{recording_id}: {exc}")
                recordings.pop(recording_id, None)

    predict_batch()

    if not recordings:
        raise RuntimeError("No recordings were successfully evaluated.")

    y_true = []
    prediction_vectors = {
        "mean_probability": [],
        "maximum_confidence_window": [],
        "top_three_mean": [],
    }

    for recording_id in sorted(recordings):
        record = recordings[recording_id]
        probabilities = np.asarray(
            record["probabilities"],
            dtype=np.float32,
        )

        if len(probabilities) == 0:
            continue

        y_true.append(record["actual"])

        # Existing approach: average all window probabilities.
        prediction_vectors["mean_probability"].append(
            probabilities.mean(axis=0)
        )

        # Experimental approach: use the single most confident window.
        best_index = np.argmax(probabilities.max(axis=1))
        prediction_vectors["maximum_confidence_window"].append(
            probabilities[best_index]
        )

        # Experimental approach: average the three most confident windows.
        top_count = min(3, len(probabilities))
        top_indices = np.argsort(
            probabilities.max(axis=1)
        )[-top_count:]

        prediction_vectors["top_three_mean"].append(
            probabilities[top_indices].mean(axis=0)
        )

    y_true = np.asarray(y_true)
    rows = []

    for method, vectors in prediction_vectors.items():
        y_prob = np.asarray(vectors)
        y_pred = y_prob.argmax(axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        rows.append({
            "method": method,
            "recordings": len(y_true),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })

        report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(classes)),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )

        with open(
            OUTPUT_DIR / f"{method}_validation_report.json",
            "w",
        ) as f:
            json.dump(report, f, indent=2)

        predictions = pd.DataFrame({
            "actual_class": [classes[i] for i in y_true],
            "predicted_class": [classes[i] for i in y_pred],
            "correct": y_true == y_pred,
        })

        predictions.to_csv(
            OUTPUT_DIR / f"{method}_validation_predictions.csv",
            index=False,
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        OUTPUT_DIR / "validation_aggregation_comparison.csv",
        index=False,
    )

    print("\nVALIDATION AGGREGATION RESULTS")
    print("=" * 65)
    print(results.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print(f"\nWindows evaluated: {total_windows}")
    print(f"Recordings evaluated: {len(y_true)}")
    print(f"Processing errors: {len(errors)}")

    if errors:
        print("First processing errors:")
        for error in errors[:10]:
            print(" ", error)

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")
    print("No model, training data, or original evaluation files modified.")


if __name__ == "__main__":
    evaluate_aggregation("validation")
