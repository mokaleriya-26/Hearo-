from pathlib import Path
from collections import defaultdict
import hashlib
import json
import shutil

import numpy as np
import soundfile as sf

SOURCE = Path("SoundSense_Merged_20Class")
DEST = Path("SoundSense_V9_Dataset")

SEED = 42
VALIDATION_FRACTION = 0.15

CLASSES = [
    "bell_sounds", "car_horn", "cat", "cheering", "clapping",
    "clock", "coughing", "crying_sobbing", "dog", "door_knock",
    "doorbell", "explosive_sounds", "footsteps", "general_alarm",
    "glass_breaking", "laughter", "phone_ringtone", "screaming",
    "siren", "transport_sounds",
]

DURATION_BANDS = [
    (0, 2, "under_2s"),
    (2, 5, "2_to_5s"),
    (5, 10, "5_to_10s"),
    (10, float("inf"), "10s_or_more"),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_duration(path):
    return float(sf.info(str(path)).duration)


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    if DEST.exists():
        raise FileExistsError(
            f"{DEST} already exists. Nothing was overwritten. "
            "Inspect it before retrying."
        )

    for split in ("train", "validation", "test"):
        folder = SOURCE / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing source split: {folder}")

        actual = {p.name for p in folder.iterdir() if p.is_dir()}
        if actual != set(CLASSES):
            raise ValueError(
                f"Class mismatch in {split}: "
                f"missing={sorted(set(CLASSES)-actual)}, "
                f"unexpected={sorted(actual-set(CLASSES))}"
            )

    # Read only existing training and validation recordings.
    # The test split is deliberately not opened or inspected.
    records = []
    for old_split in ("train", "validation"):
        for class_name in CLASSES:
            folder = SOURCE / old_split / class_name

            for path in sorted(folder.rglob("*.wav")):
                records.append({
                    "source_split": old_split,
                    "class_name": class_name,
                    "source_path": str(path),
                    "filename": path.name,
                    "duration_seconds": get_duration(path),
                    "sha256": sha256(path),
                })

    print(f"Scanned training + validation recordings: {len(records)}")

    # Group byte-identical files so they cannot cross partitions.
    hash_groups = defaultdict(list)
    for record in records:
        hash_groups[record["sha256"]].append(record)

    duplicate_groups = [
        group for group in hash_groups.values() if len(group) > 1
    ]

    print(f"Exact duplicate groups: {len(duplicate_groups)}")
    for i, group in enumerate(duplicate_groups, 1):
        print(f"\nDuplicate group {i}:")
        for record in group:
            print(" ", record["source_path"])

    # Fail safely if identical bytes occur under different class labels.
    for group in duplicate_groups:
        labels = {r["class_name"] for r in group}
        if len(labels) > 1:
            raise RuntimeError(
                "Identical audio appears under different class labels. "
                "Resolve the label conflict before creating V9:\n"
                + "\n".join(r["source_path"] for r in group)
            )

    # Assign groups within each class and duration band.
    # Identical recordings are always assigned together.
    grouped_by_stratum = defaultdict(list)

    for group in hash_groups.values():
        representative = group[0]
        class_name = representative["class_name"]
        d = representative["duration_seconds"]

        band = next(
            name for low, high, name in DURATION_BANDS
            if low <= d < high
        )

        # Keep the entire duplicate group together.
        grouped_by_stratum[(class_name, band)].append(group)

    rng = np.random.default_rng(SEED)
    assignments = []

    for (class_name, band), groups in sorted(grouped_by_stratum.items()):
        order = rng.permutation(len(groups))
        shuffled_groups = [groups[i] for i in order]

        # Split by groups, not individual files.
        if len(shuffled_groups) >= 2:
            n_val = max(
                1,
                int(round(len(shuffled_groups) * VALIDATION_FRACTION))
            )
            n_val = min(n_val, len(shuffled_groups) - 1)
            validation_group_ids = set(range(n_val))
        else:
            validation_group_ids = set()

        for i, group in enumerate(shuffled_groups):
            new_split = (
                "validation" if i in validation_group_ids else "train"
            )

            for record in group:
                record["new_split"] = new_split
                record["duration_band"] = band
                assignments.append(record)

    # Require each class to have both train and validation recordings.
    for class_name in CLASSES:
        for split in ("train", "validation"):
            count = sum(
                r["class_name"] == class_name and r["new_split"] == split
                for r in assignments
            )
            if count == 0:
                raise RuntimeError(
                    f"{class_name} has no recordings in {split}. "
                    "No dataset has been created."
                )

    # Verify exact hashes do not cross the new partition boundary.
    hash_split = {}
    for record in assignments:
        digest = record["sha256"]
        split = record["new_split"]

        if digest in hash_split and hash_split[digest] != split:
            raise RuntimeError(
                "Duplicate-hash leakage detected; refusing to continue."
            )

        hash_split[digest] = split

    # Only now create the destination.
    try:
        for split in ("train", "validation"):
            for class_name in CLASSES:
                (DEST / split / class_name).mkdir(
                    parents=True, exist_ok=True
                )

        for record in assignments:
            source_path = Path(record["source_path"])
            target = (
                DEST / record["new_split"] /
                record["class_name"] / source_path.name
            )

            if target.exists():
                raise FileExistsError(f"Filename collision: {target}")

            shutil.copy2(source_path, target)

        manifest = {
            "dataset_name": "SoundSense_V9_Dataset",
            "seed": SEED,
            "validation_fraction": VALIDATION_FRACTION,
            "classes": CLASSES,
            "source_dataset": str(SOURCE),
            "split_method": (
                "Grouped exact-hash assignment within class and duration "
                "bands. Identical files remain in the same partition."
            ),
            "test_set_copied_or_inspected": False,
            "limitations": [
                "Exact hashes detect byte-identical files, not all related "
                "recordings, re-encodings, or crops.",
                "Filename-based source independence is not guaranteed.",
                "V8 has already trained on some recordings used in this "
                "new split, so V9 validation is for V9 model selection, "
                "not a fair V8-versus-V9 comparison.",
                "Use the untouched test set for the final comparison once "
                "model selection is complete."
            ],
            "duplicate_groups": [
                [r["source_path"] for r in group]
                for group in duplicate_groups
            ],
            "records": assignments,
        }

        with (DEST / "split_manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

        print("\n========== V9 DATASET SUMMARY ==========")

        for split in ("train", "validation"):
            print(f"\n{split.upper()}")
            total = 0

            for class_name in CLASSES:
                files = sorted(
                    (DEST / split / class_name).glob("*.wav")
                )
                durations = [get_duration(p) for p in files]
                total += len(files)

                median = (
                    float(np.median(durations))
                    if durations else float("nan")
                )

                print(
                    f"{class_name:22s} "
                    f"{len(files):6d} recordings | "
                    f"median duration {median:6.2f}s"
                )

            print("Total recordings:", total)

        print("\nV9 dataset created:", DEST.resolve())
        print("Original dataset unchanged.")
        print("V8 checkpoint unchanged.")
        print("Test split untouched.")
        print("Manifest saved to:", DEST / "split_manifest.json")

    except Exception:
        print(
            "\nDataset creation failed. Inspect the destination before "
            "retrying; the original source remains unchanged."
        )
        raise


if __name__ == "__main__":
    main()
