import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path("soundsense_padded_processed_2s")

with open(ROOT / "class_names.json") as f:
    CLASS_NAMES = json.load(f)

FOLDERS = {
    "Original": Path("soundsense_all_windows_evaluation"),
    "Balanced V3": Path("soundsense_all_windows_evaluation_v3"),
}

for split in ["validation", "test"]:
    print("\n" + "=" * 80)
    print(f"{split.upper()} — CONFUSION MATRIX RECONSTRUCTION")
    print("=" * 80)

    matrices = {}

    for model_name, folder in FOLDERS.items():
        pred_path = folder / f"{split}_all_windows_predictions.csv"
        df = pd.read_csv(pred_path)

        required = {"actual_class", "predicted_class"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing required columns in {pred_path}")

        if df[["actual_class", "predicted_class"]].isna().any().any():
            raise ValueError(f"Missing class labels in {pred_path}")

        unknown = (
            set(df["actual_class"]) | set(df["predicted_class"])
        ) - set(CLASS_NAMES)

        if unknown:
            raise ValueError(f"Unknown class labels in {pred_path}: {unknown}")

        matrix = confusion_matrix(
            df["actual_class"],
            df["predicted_class"],
            labels=CLASS_NAMES,
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )
        matrix_df.index.name = "actual_class"
        matrix_df.columns.name = "predicted_class"

        output_path = folder / f"{split}_all_windows_confusion_matrix_fixed.csv"
        matrix_df.to_csv(output_path)
        matrices[model_name] = matrix_df

        print(f"\n{model_name}")
        print("Recordings:", len(df))
        print("Correct:", int((df["actual_class"] == df["predicted_class"]).sum()))
        print("Accuracy:", round(
            (df["actual_class"] == df["predicted_class"]).mean(), 4
        ))
        print("Matrix shape:", matrix_df.shape)
        print("Saved:", output_path)

        errors = []
        for actual in CLASS_NAMES:
            for predicted in CLASS_NAMES:
                if actual != predicted:
                    count = int(matrix_df.loc[actual, predicted])
                    if count:
                        errors.append((actual, predicted, count))

        print("\nTop 10 classification errors:")
        for actual, predicted, count in sorted(
            errors, key=lambda x: x[2], reverse=True
        )[:10]:
            print(f"  {actual} -> {predicted}: {count}")

    # Compare class-level error counts between models.
    original = matrices["Original"]
    v3 = matrices["Balanced V3"]

    original_errors = original.copy()
    v3_errors = v3.copy()

    for class_name in CLASS_NAMES:
        original_errors.loc[class_name, class_name] = 0
        v3_errors.loc[class_name, class_name] = 0

    changes = []
    for actual in CLASS_NAMES:
        for predicted in CLASS_NAMES:
            if actual != predicted:
                old = int(original_errors.loc[actual, predicted])
                new = int(v3_errors.loc[actual, predicted])
                if old or new:
                    changes.append((actual, predicted, old, new, new - old))

    print("\nLargest increases in individual confusion counts in V3:")
    for actual, predicted, old, new, change in sorted(
        changes, key=lambda x: x[4], reverse=True
    )[:10]:
        print(
            f"  {actual} -> {predicted}: "
            f"original={old}, V3={new}, change={change:+d}"
        )

print("\nDone. Original model files and prediction files were not modified.")
