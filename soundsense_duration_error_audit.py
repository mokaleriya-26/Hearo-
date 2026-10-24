from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path("SoundSense_Final_Dataset")

PREDICTION_DIRS = {
    "Original": Path("soundsense_all_windows_evaluation"),
    "Balanced_V3": Path("soundsense_all_windows_evaluation_v3"),
}

OUTPUT_DIR = Path("soundsense_experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

def locate_recording(split, recording_id):
    """
    recording_id is expected to be relative to its split directory,
    for example aircraft/aircraft_validation_0001.wav.
    """
    matches = list(ROOT.glob(f"{split}/**/{recording_id}"))

    if len(matches) == 1:
        return matches[0]

    # Fall back to matching the relative path beneath the split folder.
    candidate = ROOT / split / recording_id
    if candidate.is_file():
        return candidate

    return None

def get_duration(path):
    try:
        info = sf.info(str(path))
        return info.duration
    except Exception:
        return np.nan

for split in ["validation", "test"]:
    print("\n" + "=" * 75)
    print(f"{split.upper()} — RECORDING DURATION AUDIT")
    print("=" * 75)

    model_frames = []

    for model_name, folder in PREDICTION_DIRS.items():
        path = folder / f"{split}_all_windows_predictions.csv"
        df = pd.read_csv(path)

        durations = []
        missing = 0

        for recording_id in df["recording_id"]:
            recording_path = locate_recording(split, str(recording_id))

            if recording_path is None:
                durations.append(np.nan)
                missing += 1
            else:
                durations.append(get_duration(recording_path))

        df["duration_seconds"] = durations
        df["model"] = model_name

        print(
            f"\n{model_name}: {len(df)} predictions; "
            f"{missing} recordings not found"
        )

        df = df.dropna(subset=["duration_seconds"]).copy()

        if df.empty:
            print("No durations could be matched.")
            continue

        # Duration bins are diagnostic only, not training categories.
        df["duration_group"] = pd.cut(
            df["duration_seconds"],
            bins=[0, 2, 5, 10, np.inf],
            labels=["<=2s", "2-5s", "5-10s", ">10s"],
            include_lowest=True,
        )

        summary = (
            df.groupby("duration_group", observed=False)
            .agg(
                recordings=("correct", "size"),
                accuracy=("correct", "mean"),
                mean_duration=("duration_seconds", "mean"),
                mean_windows=("windows_evaluated", "mean"),
            )
            .reset_index()
        )

        print("\nAccuracy by recording duration:")
        print(summary.to_string(
            index=False,
            formatters={
                "accuracy": lambda x: f"{x:.3f}",
                "mean_duration": lambda x: f"{x:.2f}",
                "mean_windows": lambda x: f"{x:.2f}",
            },
        ))

        df.to_csv(
            OUTPUT_DIR / f"{split}_{model_name.lower()}_duration_details.csv",
            index=False,
        )

        model_frames.append(df)

    if len(model_frames) == 2:
        original = model_frames[0]
        v3 = model_frames[1]

        original_summary = original.groupby(
            "duration_group", observed=False
        )["correct"].mean()

        v3_summary = v3.groupby(
            "duration_group", observed=False
        )["correct"].mean()

        comparison = pd.DataFrame({
            "original_accuracy": original_summary,
            "v3_accuracy": v3_summary,
        })

        comparison["difference_v3_minus_original"] = (
            comparison["v3_accuracy"] - comparison["original_accuracy"]
        )

        print("\nModel comparison by duration:")
        print(comparison.to_string(
            float_format=lambda x: f"{x:.3f}"
        ))

print("\nDuration audit complete.")
print("No model or dataset files were modified.")
