from pathlib import Path
from collections import defaultdict
import hashlib
import json

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path.cwd()
DATASET = ROOT / "SoundSense_V9_Dataset"
OUT = ROOT / "soundsense_v9_data_audit"

SPLITS = ["train", "validation"]
BANDS = [
    (0, 2, "<2s"),
    (2, 5, "2-5s"),
    (5, 10, "5-10s"),
    (10, float("inf"), "10s+"),
]

if not DATASET.is_dir():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

if OUT.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUT}\n"
        "Rename or inspect it before running this audit again."
    )

manifest_path = DATASET / "split_manifest.json"

with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)

classes = manifest["classes"]

rows = []
hash_locations = defaultdict(list)
errors = []

print("=" * 75)
print("SOUNDSENSE V9 TRAINING-DATA AUDIT")
print("=" * 75)

for split in SPLITS:
    for class_name in classes:
        folder = DATASET / split / class_name

        if not folder.is_dir():
            print(f"WARNING: Missing directory: {folder}")
            continue

        files = sorted(folder.rglob("*.wav"))

        for i, path in enumerate(files, start=1):
            try:
                info = sf.info(str(path))
                duration = float(info.duration)

                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hash_locations[digest].append({
                    "split": split,
                    "class": class_name,
                    "path": str(path.relative_to(ROOT)),
                })

                band = next(
                    label for low, high, label in BANDS
                    if low <= duration < high
                )

                rows.append({
                    "split": split,
                    "class_name": class_name,
                    "filename": path.name,
                    "relative_path": str(path.relative_to(ROOT)),
                    "duration_seconds": duration,
                    "duration_band": band,
                    "sample_rate": int(info.samplerate),
                    "channels": int(info.channels),
                    "frames": int(info.frames),
                    "sha256": digest,
                })

            except Exception as exc:
                errors.append({
                    "path": str(path.relative_to(ROOT)),
                    "error": repr(exc),
                })

        print(f"{split}/{class_name}: {len(files)} WAV files")

if not rows:
    raise RuntimeError("No readable WAV files found.")

df = pd.DataFrame(rows)

OUT.mkdir(parents=True)

# Recording counts and duration statistics by class and split.
summary = (
    df.groupby(["split", "class_name"])
    .agg(
        recordings=("relative_path", "count"),
        mean_duration=("duration_seconds", "mean"),
        median_duration=("duration_seconds", "median"),
        min_duration=("duration_seconds", "min"),
        max_duration=("duration_seconds", "max"),
    )
    .reset_index()
)

summary.to_csv(OUT / "class_duration_summary.csv", index=False)

duration_counts = (
    df.groupby(["split", "class_name", "duration_band"])
    .size()
    .rename("recordings")
    .reset_index()
)

duration_counts.to_csv(
    OUT / "duration_band_counts.csv",
    index=False
)

df.to_csv(OUT / "recording_inventory.csv", index=False)

# Exact duplicate groups across training and validation.
duplicate_groups = []

for digest, locations in hash_locations.items():
    if len(locations) > 1:
        duplicate_groups.append({
            "sha256": digest,
            "count": len(locations),
            "locations": locations,
            "splits": sorted(set(x["split"] for x in locations)),
            "classes": sorted(set(x["class"] for x in locations)),
        })

with open(OUT / "exact_duplicate_groups.json", "w", encoding="utf-8") as f:
    json.dump(duplicate_groups, f, indent=2)

if errors:
    pd.DataFrame(errors).to_csv(
        OUT / "file_read_errors.csv",
        index=False
    )

print("\n" + "=" * 75)
print("RECORDING COUNTS BY CLASS")
print("=" * 75)

counts = pd.crosstab(df["class_name"], df["split"])

for split in SPLITS:
    if split not in counts.columns:
        counts[split] = 0

counts = counts[SPLITS]
counts["train_to_validation_ratio"] = (
    counts["train"] / counts["validation"].replace(0, np.nan)
)

print(counts.to_string())

print("\n" + "=" * 75)
print("OVERALL DURATION DISTRIBUTION")
print("=" * 75)

for split in SPLITS:
    subset = df[df["split"] == split]
    print(f"\n{split.upper()}: {len(subset)} recordings")

    print(
        subset["duration_band"]
        .value_counts()
        .reindex([b[2] for b in BANDS], fill_value=0)
        .to_string()
    )

    print(f"Mean duration: {subset['duration_seconds'].mean():.2f}s")
    print(f"Median duration: {subset['duration_seconds'].median():.2f}s")

print("\n" + "=" * 75)
print("DUPLICATE AND FILE-READ AUDIT")
print("=" * 75)

print(f"Exact duplicate groups: {len(duplicate_groups)}")
print(f"File-read errors: {len(errors)}")

cross_split_duplicates = [
    group for group in duplicate_groups
    if len(group["splits"]) > 1
]

cross_class_duplicates = [
    group for group in duplicate_groups
    if len(group["classes"]) > 1
]

print(f"Duplicate groups crossing train/validation: {len(cross_split_duplicates)}")
print(f"Duplicate groups appearing under multiple classes: {len(cross_class_duplicates)}")

print(f"\nAudit files saved to: {OUT}")
for path in sorted(OUT.iterdir()):
    print(" ", path.name)

print("\nNote: exact hashes do not detect re-encoded audio, crops, or related recordings.")
