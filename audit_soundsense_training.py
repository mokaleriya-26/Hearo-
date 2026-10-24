import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("soundsense_padded_processed_2s")

with open(DATA_DIR / "class_names.json") as f:
    classes = json.load(f)

results = []

for split in ["train", "validation", "test"]:
    y = np.load(DATA_DIR / f"y_{split}.npy", mmap_mode="r")
    ids = np.load(
        DATA_DIR / f"recording_ids_{split}.npy",
        allow_pickle=False,
    )

    window_indices_path = DATA_DIR / f"window_indices_{split}.npy"
    window_indices = np.load(window_indices_path, mmap_mode="r")

    if not (len(y) == len(ids) == len(window_indices)):
        raise ValueError(f"Array lengths do not match for {split}")

    df = pd.DataFrame({
        "class_index": np.asarray(y),
        "recording_id": ids.astype(str),
        "window_index": np.asarray(window_indices),
    })

    # Each recording should belong to one class.
    for recording_id, group in df.groupby("recording_id"):
        if group["class_index"].nunique() != 1:
            raise ValueError(
                f"Recording appears under multiple classes: {recording_id}"
            )

    per_class = []

    for class_index, class_name in enumerate(classes):
        subset = df[df["class_index"] == class_index]

        per_class.append({
            "class": class_name,
            "recordings": subset["recording_id"].nunique(),
            "windows": len(subset),
            "windows_per_recording": (
                len(subset) / subset["recording_id"].nunique()
                if subset["recording_id"].nunique()
                else 0
            ),
        })

    results.append((split, pd.DataFrame(per_class)))

for split, report in results:
    print("\n" + "=" * 78)
    print(f"{split.upper()} — RECORDINGS VS PROCESSED WINDOWS")
    print("=" * 78)
    print(
        report.to_string(
            index=False,
            formatters={
                "windows_per_recording": lambda x: f"{x:.2f}"
            }
        )
    )

output = Path("soundsense_experiments")
output.mkdir(exist_ok=True)

for split, report in results:
    path = output / f"recording_window_audit_{split}.csv"
    report.to_csv(path, index=False)
    print(f"Saved: {path}")

print("\nAudit complete. No dataset or model files were modified.")
