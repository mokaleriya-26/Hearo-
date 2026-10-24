import os
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.cwd()
TEST_DIR = ROOT / "SoundSense_Merged_20Class" / "test"

V8_DIR = ROOT / "soundsense_padded_model_v8_recording_f1"
V9_DIR = ROOT / "soundsense_padded_model_v9_recording_f1"

OUT_DIR = ROOT / "soundsense_test_evaluation_v8_v9"

SAMPLE_RATE = 16000
DURATION_SECONDS = 2
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
WINDOW_STRIDE = SAMPLE_RATE
MAX_WINDOWS = 10

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 32

BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ============================================================
# SAFETY CHECKS
# ============================================================

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

if OUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUT_DIR}\n"
        "This script will not overwrite previous evaluation results. "
        "Inspect or rename the existing directory before running again."
    )

for model_dir in (V8_DIR, V9_DIR):
    required = [
        model_dir / f"soundsense_{model_dir.name.split('_')[3]}_recording_f1.keras",
        model_dir / "normalization_stats.npz",
        model_dir / "class_names.json",
    ]

    # Check the actual expected filenames explicitly below.
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

v8_model_path = V8_DIR / "soundsense_v8_recording_f1.keras"
v9_model_path = V9_DIR / "soundsense_v9_recording_f1.keras"

for path in (
    v8_model_path,
    v9_model_path,
    V8_DIR / "normalization_stats.npz",
    V9_DIR / "normalization_stats.npz",
    V8_DIR / "class_names.json",
    V9_DIR / "class_names.json",
):
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")

with open(V8_DIR / "class_names.json", encoding="utf-8") as f:
    classes_v8 = json.load(f)

with open(V9_DIR / "class_names.json", encoding="utf-8") as f:
    classes_v9 = json.load(f)

# Accommodate either a direct list or a dictionary containing class names.
if isinstance(classes_v8, dict):
    classes_v8 = classes_v8.get("class_names", classes_v8.get("classes"))

if isinstance(classes_v9, dict):
    classes_v9 = classes_v9.get("class_names", classes_v9.get("classes"))

if not isinstance(classes_v8, list) or not isinstance(classes_v9, list):
    raise ValueError("Could not read class-name lists from model artifacts.")

if classes_v8 != classes_v9:
    raise ValueError(
        "V8 and V9 class orders differ. Stopping to avoid invalid comparison.\n"
        f"V8: {classes_v8}\nV9: {classes_v9}"
    )

CLASSES = classes_v8
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

print("=" * 70)
print("SOUNDSENSE V8 vs V9 — ORIGINAL TEST SET EVALUATION")
print("=" * 70)
print(f"Test directory: {TEST_DIR}")
print(f"Classes ({len(CLASSES)}): {CLASSES}")
print("Test recordings will not be used for training or tuning.")
print()

# ============================================================
# DISCOVER TEST RECORDINGS
# ============================================================

recordings = []

for class_name in CLASSES:
    class_dir = TEST_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Expected test class folder missing: {class_dir}"
        )

    for path in sorted(class_dir.rglob("*.wav")):
        recordings.append({
            "path": path,
            "class_name": class_name,
            "label": CLASS_TO_INDEX[class_name],
            "recording_id": path.relative_to(TEST_DIR).as_posix(),
        })

if not recordings:
    raise RuntimeError(f"No WAV files found under {TEST_DIR}")

# Verify the expected test-set size before evaluating.
if len(recordings) != 2788:
    raise RuntimeError(
        f"Expected 2788 test WAV files, found {len(recordings)}. "
        "Stopping so we do not evaluate an unexpected split."
    )

print(f"Test recordings found: {len(recordings)}")

# ============================================================
# LOAD MODELS AND THEIR OWN NORMALIZATION STATISTICS
# ============================================================

print("\nLoading V8 model...")
model_v8 = tf.keras.models.load_model(v8_model_path, compile=False)

print("Loading V9 model...")
model_v9 = tf.keras.models.load_model(v9_model_path, compile=False)

def load_stats(path):
    with np.load(path) as data:
        keys = list(data.files)
        print(f"Normalization file {path.parent.name}: {keys}")

        # Support the common naming variants used by preprocessing scripts.
        mean_key = next(
            (k for k in ("mean", "feature_mean", "X_mean", "train_mean")
             if k in data.files),
            None
        )
        std_key = next(
            (k for k in ("std", "feature_std", "X_std", "train_std")
             if k in data.files),
            None
        )

        if mean_key is None or std_key is None:
            if len(keys) == 2:
                mean_key, std_key = keys
            else:
                raise ValueError(
                    f"Cannot identify mean/std arrays in {path}; keys={keys}"
                )

        mean = np.asarray(data[mean_key], dtype=np.float32)
        std = np.asarray(data[std_key], dtype=np.float32)

    std = np.where(std < 1e-6, 1.0, std)
    return mean, std

mean_v8, std_v8 = load_stats(V8_DIR / "normalization_stats.npz")
mean_v9, std_v9 = load_stats(V9_DIR / "normalization_stats.npz")

print(f"V8 input shape: {model_v8.input_shape}")
print(f"V9 input shape: {model_v9.input_shape}")

expected_shape = (None, 201, 13, 1)

for name, model in (("V8", model_v8), ("V9", model_v9)):
    if len(model.input_shape) != 4 or tuple(model.input_shape[1:]) != expected_shape[1:]:
        raise ValueError(
            f"{name} input shape {model.input_shape} does not match "
            f"expected (None, 201, 13, 1)."
        )

# ============================================================
# AUDIO WINDOW EXTRACTION — SAME V6/V9 PREPROCESSING SETTINGS
# ============================================================

def get_window_starts(audio_length):
    if audio_length <= TARGET_SAMPLES:
        return [0]

    starts = list(
        range(0, audio_length - TARGET_SAMPLES + 1, WINDOW_STRIDE)
    )
    final_start = audio_length - TARGET_SAMPLES

    if starts[-1] != final_start:
        starts.append(final_start)

    if len(starts) > MAX_WINDOWS:
        chosen = np.linspace(
            0, len(starts) - 1, MAX_WINDOWS
        ).round().astype(int)
        starts = [starts[i] for i in chosen]

    return starts


def extract_recording_features(path):
    audio, _ = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    features = []

    for start in get_window_starts(len(audio)):
        window = audio[start:start + TARGET_SAMPLES]

        if len(window) < TARGET_SAMPLES:
            window = np.pad(
                window,
                (0, TARGET_SAMPLES - len(window))
            )

        mfcc = librosa.feature.mfcc(
            y=window.astype(np.float32),
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        feature = mfcc.T[..., np.newaxis].astype(np.float32)

        if feature.shape != (201, 13, 1):
            raise ValueError(
                f"Unexpected feature shape {feature.shape} for {path}"
            )

        features.append(feature)

    return np.stack(features, axis=0)


# ============================================================
# EVALUATE RECORDINGS
# ============================================================

def normalize_features(features, mean, std):
    return ((features - mean) / std).astype(np.float32)


def predict_one_model(model, features, mean, std):
    normalized = normalize_features(features, mean, std)
    outputs = []

    for start in range(0, len(normalized), BATCH_SIZE):
        batch = normalized[start:start + BATCH_SIZE]
        probabilities = model.predict(batch, verbose=0)
        outputs.append(np.asarray(probabilities, dtype=np.float32))

    window_probabilities = np.concatenate(outputs, axis=0)

    if window_probabilities.shape[1] != len(CLASSES):
        raise ValueError(
            f"Model output has {window_probabilities.shape[1]} classes; "
            f"expected {len(CLASSES)}."
        )

    # Match the recording-level evaluation used for V8/V9:
    # average window probabilities for each recording.
    return window_probabilities.mean(axis=0)


y_true = []
pred_v8 = []
pred_v9 = []
rows = []
errors = []

start_time = time.time()

for i, rec in enumerate(recordings, start=1):
    try:
        features = extract_recording_features(rec["path"])

        probabilities_v8 = predict_one_model(
            model_v8, features, mean_v8, std_v8
        )
        probabilities_v9 = predict_one_model(
            model_v9, features, mean_v9, std_v9
        )

        label = rec["label"]
        p8 = int(np.argmax(probabilities_v8))
        p9 = int(np.argmax(probabilities_v9))

        y_true.append(label)
        pred_v8.append(p8)
        pred_v9.append(p9)

        rows.append({
            "recording_id": rec["recording_id"],
            "true_class": rec["class_name"],
            "v8_predicted_class": CLASSES[p8],
            "v8_correct": bool(p8 == label),
            "v9_predicted_class": CLASSES[p9],
            "v9_correct": bool(p9 == label),
            "v8_confidence": float(np.max(probabilities_v8)),
            "v9_confidence": float(np.max(probabilities_v9)),
            "window_count": int(len(features)),
        })

    except Exception as exc:
        errors.append({
            "recording_id": rec["recording_id"],
            "error": repr(exc),
        })
        print(f"\nERROR processing {rec['recording_id']}: {exc}")

    if i % 100 == 0 or i == len(recordings):
        elapsed = time.time() - start_time
        print(
            f"Processed {i}/{len(recordings)} recordings | "
            f"successful={len(y_true)} | errors={len(errors)} | "
            f"elapsed={elapsed/60:.1f} min",
            flush=True
        )

if errors:
    print(f"\nWARNING: {len(errors)} recordings failed.")
    print("Metrics will be calculated only if every test recording succeeded.")

if len(y_true) != len(recordings):
    raise RuntimeError(
        f"Evaluation incomplete: {len(y_true)}/{len(recordings)} recordings "
        f"succeeded. No final metrics will be reported. "
        f"Inspect errors before deciding whether to retry."
    )

# ============================================================
# METRICS AND OUTPUTS
# ============================================================

OUT_DIR.mkdir(parents=True, exist_ok=False)

y_true = np.asarray(y_true)
pred_v8 = np.asarray(pred_v8)
pred_v9 = np.asarray(pred_v9)

def summarize_model(name, predictions):
    return {
        "model": name,
        "test_recordings": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(
                y_true, predictions,
                labels=np.arange(len(CLASSES)),
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, predictions,
                labels=np.arange(len(CLASSES)),
                average="weighted",
                zero_division=0
            )
        ),
    }


summary = [
    summarize_model("V8", pred_v8),
    summarize_model("V9", pred_v9),
]

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

for item in summary:
    print(
        f"{item['model']}: "
        f"Accuracy={item['accuracy']:.4f} ({item['accuracy']*100:.2f}%), "
        f"Macro F1={item['macro_f1']:.4f} "
        f"({item['macro_f1']*100:.2f}%), "
        f"Weighted F1={item['weighted_f1']:.4f} "
        f"({item['weighted_f1']*100:.2f}%)"
    )

report_v8 = classification_report(
    y_true,
    pred_v8,
    labels=np.arange(len(CLASSES)),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0
)

report_v9 = classification_report(
    y_true,
    pred_v9,
    labels=np.arange(len(CLASSES)),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report_v8).transpose().to_csv(
    OUT_DIR / "v8_test_classification_report.csv"
)
pd.DataFrame(report_v9).transpose().to_csv(
    OUT_DIR / "v9_test_classification_report.csv"
)

pd.DataFrame(
    confusion_matrix(y_true, pred_v8, labels=np.arange(len(CLASSES))),
    index=CLASSES,
    columns=CLASSES
).to_csv(OUT_DIR / "v8_test_confusion_matrix.csv")

pd.DataFrame(
    confusion_matrix(y_true, pred_v9, labels=np.arange(len(CLASSES))),
    index=CLASSES,
    columns=CLASSES
).to_csv(OUT_DIR / "v9_test_confusion_matrix.csv")

pd.DataFrame(rows).to_csv(
    OUT_DIR / "test_recording_predictions_v8_v9.csv",
    index=False
)

pd.DataFrame(summary).to_csv(
    OUT_DIR / "test_metrics_summary.csv",
    index=False
)

with open(OUT_DIR / "test_metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "test_set": str(TEST_DIR),
        "test_recordings": len(recordings),
        "successful_recordings": len(y_true),
        "test_set_used_for_training_or_tuning": False,
        "note": (
            "The original test split was used for final evaluation only. "
            "Do not tune models using these results."
        ),
        "metrics": summary,
    }, f, indent=2)

if errors:
    pd.DataFrame(errors).to_csv(
        OUT_DIR / "processing_errors.csv",
        index=False
    )

print(f"\nResults saved to: {OUT_DIR}")
print("Files:")
for path in sorted(OUT_DIR.iterdir()):
    print(f"  {path.name}")

print("\nIMPORTANT: Keep this test set reserved for final reporting.")
