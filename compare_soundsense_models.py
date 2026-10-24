import json
from pathlib import Path

import pandas as pd

BASELINE = Path("soundsense_all_windows_evaluation")
V3 = Path("soundsense_all_windows_evaluation_v3")

def load_report(folder, split):
    path = folder / f"{split}_all_windows_report.json"
    with open(path, "r") as f:
        return json.load(f)

def get_class_metrics(report):
    # sklearn classification reports contain one entry per class.
    excluded = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    rows = []

    for name, values in report.items():
        if name in excluded or not isinstance(values, dict):
            continue

        if "precision" in values and "recall" in values and "f1-score" in values:
            rows.append({
                "class": name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": values.get("support", 0),
            })

    return pd.DataFrame(rows)

for split in ["validation", "test"]:
    print("\n" + "=" * 85)
    print(f"{split.upper()}: ORIGINAL MODEL VS BALANCED V3")
    print("=" * 85)

    original = load_report(BASELINE, split)
    v3 = load_report(V3, split)

    print("\nOriginal summary:")
    print(json.dumps(original.get("summary", original), indent=2)[:1800])

    print("\nV3 summary:")
    print(json.dumps(v3.get("summary", v3), indent=2)[:1800])

    a = get_class_metrics(original)
    b = get_class_metrics(v3)

    if a.empty or b.empty:
        print("\nCould not extract per-class metrics automatically.")
        print("Original report keys:", list(original.keys()))
        print("V3 report keys:", list(v3.keys()))
        continue

    comparison = a.merge(b, on="class", suffixes=("_original", "_v3"))

    comparison["f1_change"] = comparison["f1_v3"] - comparison["f1_original"]
    comparison["recall_change"] = comparison["recall_v3"] - comparison["recall_original"]

    print("\nPer-class comparison (F1 and recall are proportions):")
    print(
        comparison.sort_values("f1_change")[
            [
                "class",
                "support_original",
                "f1_original",
                "f1_v3",
                "f1_change",
                "recall_original",
                "recall_v3",
                "recall_change",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nClasses with the largest F1 drops in V3:")
    print(
        comparison.nsmallest(8, "f1_change")[
            ["class", "f1_original", "f1_v3", "f1_change"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nClasses where V3 improved F1 the most:")
    print(
        comparison.nlargest(8, "f1_change")[
            ["class", "f1_original", "f1_v3", "f1_change"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    # Compare confusion matrices and find the most frequent mistakes.
    cm_a = pd.read_csv(BASELINE / f"{split}_all_windows_confusion_matrix.csv", index_col=0)
    cm_b = pd.read_csv(V3 / f"{split}_all_windows_confusion_matrix.csv", index_col=0)

    for label, cm in [("ORIGINAL", cm_a), ("V3", cm_b)]:
        cm = cm.apply(pd.to_numeric, errors="coerce").fillna(0)
        errors = []

        for true_class in cm.index:
            for predicted_class in cm.columns:
                count = cm.loc[true_class, predicted_class]
                if true_class != predicted_class and count > 0:
                    errors.append((true_class, predicted_class, int(count)))

        print(f"\nTop 10 {label} confusions:")
        for true_class, predicted_class, count in sorted(
            errors, key=lambda x: x[2], reverse=True
        )[:10]:
            print(f"  {true_class} -> {predicted_class}: {count}")

print("\nComparison complete.")
