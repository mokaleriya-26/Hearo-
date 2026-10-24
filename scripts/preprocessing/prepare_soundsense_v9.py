from pathlib import Path
import hashlib
import json

import librosa
import numpy as np

DATASET_DIR = Path("SoundSense_V9_Dataset")
OUTPUT_DIR = Path("soundsense_processed_v9")

SAMPLE_RATE = 16000
DURATION_SECONDS = 2
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SECONDS

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 32

MAX_WINDOWS_PER_RECORDING = 10
SPLITS = ["train", "validation"]


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

    if len(starts) > MAX_WINDOWS_PER_RECORDING:
        positions = np.linspace(
            0,
            len(starts) - 1,
            MAX_WINDOWS_PER_RECORDING,
        ).round().astype(int)

        starts = [starts[i] for i in positions]

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


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"V9 dataset not found: {DATASET_DIR.resolve()}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR.resolve()}\n"
            "Nothing was overwritten. Inspect or rename it before retrying."
        )

    with open(DATASET_DIR / "split_manifest.json") as f:
        split_manifest = json.load(f)

    classes = split_manifest["classes"]

    if len(classes) != 20 or len(set(classes)) != 20:
        raise ValueError("Expected exactly 20 unique classes.")

    class_to_index = {
        name: index for index, name in enumerate(classes)
    }

    for split in SPLITS:
        split_dir = DATASET_DIR / split

        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split: {split_dir}")

        actual_classes = {
            p.name for p in split_dir.iterdir() if p.is_dir()
        }

        if actual_classes != set(classes):
            raise ValueError(
                f"Class folders do not match manifest for {split}."
            )

    # Check that the exact same audio hash does not cross partitions.
    # The dataset creation script already groups exact duplicates.
    print("Checking V9 train/validation duplicate hashes...")

    hashes_by_split = {}

    for split in SPLITS:
        split_dir = DATASET_DIR / split
        hashes = set()

        for path in split_dir.rglob("*.wav"):
            digest = sha256(path)

            if digest in hashes:
                # Duplicates within the same split are allowed and documented.
                pass

            hashes.add(digest)

        hashes_by_split[split] = hashes

    cross_split_duplicates = (
        hashes_by_split["train"] & hashes_by_split["validation"]
    )

    if cross_split_duplicates:
        raise RuntimeError(
            f"Found {len(cross_split_duplicates)} exact duplicate hashes "
            "across train and validation. Aborting preprocessing."
        )

    print("No byte-identical audio crosses train/validation.")

    OUTPUT_DIR.mkdir(parents=True)

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
        "processed_splits": SPLITS,
        "test_split_processed": False,
        "split_manifest": str(DATASET_DIR / "split_manifest.json"),
    }

    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump(classes, f, indent=2)

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for split in SPLITS:
        split_dir = DATASET_DIR / split

        X_values = []
        y_values = []
        recording_ids = []
        window_indices = []

        scanned = 0
        skipped = 0
        errors = 0
        padded_recordings = 0

        print(f"\nProcessing {split.upper()}...")

        for class_name in classes:
            class_dir = split_dir / class_name

            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() != ".wav":
                    continue

                scanned += 1

                try:
                    audio, _ = librosa.load(
                        path,
                        sr=SAMPLE_RATE,
                        mono=True,
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

                    recording_id = str(path.relative_to(split_dir))

                    for window_index, window in enumerate(windows):
                        X_values.append(extract_mfcc(window))
                        y_values.append(class_to_index[class_name])
                        recording_ids.append(recording_id)
                        window_indices.append(window_index)

                except Exception as exc:
                    errors += 1
                    print(f"ERROR processing {path}: {exc}")

                if scanned % 1000 == 0:
                    print(
                        f"  Scanned {scanned:,}; "
                        f"generated {len(X_values):,} windows"
                    )

        if not X_values:
            raise RuntimeError(f"No features generated for {split}.")

        X = np.stack(X_values).astype(np.float32)
        y = np.asarray(y_values, dtype=np.int32)
        ids = np.asarray(recording_ids, dtype=str)
        indices = np.asarray(window_indices, dtype=np.int32)

        np.save(OUTPUT_DIR / f"X_{split}.npy", X)
        np.save(OUTPUT_DIR / f"y_{split}.npy", y)
        np.save(OUTPUT_DIR / f"recording_ids_{split}.npy", ids)
        np.save(OUTPUT_DIR / f"window_indices_{split}.npy", indices)

        summary = {
            "split": split,
            "files_scanned": scanned,
            "windows_generated": len(X),
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
            json.dump(summary, f, indent=2)

        print(f"\n{split.upper()} completed")
        print("Recordings scanned:", scanned)
        print("Windows generated:", len(X))
        print("Feature shape:", X.shape)
        print("Padded recordings:", padded_recordings)
        print("Skipped:", skipped)
        print("Errors:", errors)

        if errors:
            raise RuntimeError(
                f"{split} had {errors} extraction errors. "
                "Review the errors before training."
            )

    print("\nV9 PREPROCESSING COMPLETED")
    print("Output:", OUTPUT_DIR.resolve())
    print("Classes:", len(classes))
    print("Test split has not been processed.")


if __name__ == "__main__":
    main()
