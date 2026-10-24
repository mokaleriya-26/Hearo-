from pathlib import Path
import json

import librosa
import numpy as np
import soundfile as sf

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = Path("SoundSense_Final_Dataset")
OUTPUT_DIR = Path("soundsense_processed_v4_10windows")

SAMPLE_RATE = 16000
DURATION_SECONDS = 2
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SECONDS

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 32

SPLITS = ["train", "validation", "test"]

# Limit windows from long recordings to avoid excessive
# dominance by very long recordings.
MAX_WINDOWS_PER_RECORDING = 10

# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_windows(audio):
    """
    Return 2-second windows.

    Short recordings are zero-padded.
    Longer recordings are divided into windows with
    a 1-second stride. At most five windows are retained.
    """
    if len(audio) == 0:
        return []

    if len(audio) < TARGET_SAMPLES:
        padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        padded[:len(audio)] = audio
        return [padded]

    starts = list(range(
        0,
        len(audio) - TARGET_SAMPLES + 1,
        SAMPLE_RATE
    ))

    final_start = len(audio) - TARGET_SAMPLES

    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    if len(starts) > MAX_WINDOWS_PER_RECORDING:
        positions = np.linspace(
            0,
            len(starts) - 1,
            MAX_WINDOWS_PER_RECORDING
        ).round().astype(int)

        starts = [starts[i] for i in positions]

    return [
        audio[start:start + TARGET_SAMPLES].astype(np.float32)
        for start in starts
    ]


def extract_mfcc(window):
    """
    Produce an MFCC feature image using the existing
    experiment's configuration.
    """
    mfcc = librosa.feature.mfcc(
        y=window,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    return mfcc.T[..., np.newaxis].astype(np.float32)


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR.resolve()}"
        )

    classes = sorted(
        p.name for p in (DATASET_DIR / "train").iterdir()
        if p.is_dir()
    )

    if len(classes) != 24:
        raise ValueError(
            f"Expected 24 training classes, found {len(classes)}: {classes}"
        )

    class_to_index = {
        name: index for index, name in enumerate(classes)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump(classes, f, indent=2)

    metadata = {
        "dataset": str(DATASET_DIR),
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "target_samples": TARGET_SAMPLES,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "max_windows_per_recording": MAX_WINDOWS_PER_RECORDING,
        "short_recording_policy": "zero_pad_to_two_seconds",
        "long_recording_stride_seconds": 1,
        "classes": classes,
        "class_to_index": class_to_index,
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for split in SPLITS:
        split_dir = DATASET_DIR / split

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}"
            )

        X_values = []
        y_values = []
        recording_ids = []
        window_indices = []

        scanned = 0
        skipped = 0
        errors = 0
        padded_recordings = 0
        total_windows = 0

        print(f"\nProcessing {split.upper()}...")

        for class_name in classes:
            class_dir = split_dir / class_name

            if not class_dir.is_dir():
                print(f"WARNING: Missing class folder: {class_dir}")
                continue

            for path in sorted(class_dir.rglob("*.wav")):
                scanned += 1

                try:
                    audio, _ = librosa.load(
                        path,
                        sr=SAMPLE_RATE,
                        mono=True
                    )

                    if len(audio) == 0:
                        skipped += 1
                        continue

                    if len(audio) < TARGET_SAMPLES:
                        padded_recordings += 1

                    windows = extract_windows(audio)

                    if not windows:
                        skipped += 1
                        continue

                    # IDs are relative to the split directory.
                    recording_id = str(path.relative_to(split_dir))

                    for window_index, window in enumerate(windows):
                        features = extract_mfcc(window)

                        X_values.append(features)
                        y_values.append(class_to_index[class_name])
                        recording_ids.append(recording_id)
                        window_indices.append(window_index)

                    total_windows += len(windows)

                except Exception as exc:
                    errors += 1
                    print(f"ERROR: {path}: {exc}")

                if scanned % 1000 == 0:
                    print(
                        f"  Scanned {scanned:,} files; "
                        f"generated {total_windows:,} windows"
                    )

        if not X_values:
            raise RuntimeError(f"No features generated for {split}")

        X = np.stack(X_values).astype(np.float32)
        y = np.asarray(y_values, dtype=np.int32)
        ids = np.asarray(recording_ids, dtype=str)
        indices = np.asarray(window_indices, dtype=np.int32)

        np.save(OUTPUT_DIR / f"X_{split}.npy", X)
        np.save(OUTPUT_DIR / f"y_{split}.npy", y)
        np.save(OUTPUT_DIR / f"recording_ids_{split}.npy", ids)
        np.save(OUTPUT_DIR / f"window_indices_{split}.npy", indices)

        split_summary = {
            "split": split,
            "files_scanned": scanned,
            "windows_generated": int(len(X)),
            "padded_recordings": padded_recordings,
            "skipped_files": skipped,
            "errors": errors,
            "feature_shape": list(X.shape),
            "class_counts_by_windows": {
                name: int(np.sum(y == class_to_index[name]))
                for name in classes
            },
        }

        with open(OUTPUT_DIR / f"summary_{split}.json", "w") as f:
            json.dump(split_summary, f, indent=2)

        print(f"\n{split.upper()} SUMMARY")
        print("-" * 50)
        print(f"Files scanned:       {scanned:,}")
        print(f"Windows generated:   {len(X):,}")
        print(f"Short files padded:  {padded_recordings:,}")
        print(f"Files skipped:       {skipped:,}")
        print(f"Processing errors:   {errors:,}")
        print(f"Feature shape:       {X.shape}")

        del X, y, ids, indices, X_values, y_values
        del recording_ids, window_indices

    print("\nPREPROCESSING COMPLETE")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("Existing model and processed dataset were not modified.")


if __name__ == "__main__":
    main()
