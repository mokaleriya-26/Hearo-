from pathlib import Path
import json
import shutil
import sys

SOURCE = Path("SoundSense_Final_Dataset")
DESTINATION = Path("SoundSense_Merged_20Class")

MERGE_MAP = {
    "explosion": "explosive_sounds",
    "fireworks": "explosive_sounds",

    "rail_transport": "transport_sounds",
    "road_vehicle": "transport_sounds",
    "aircraft": "transport_sounds",

    "bicycle_bell": "bell_sounds",
    "church_bells": "bell_sounds",
}

SPLITS = ("train", "validation", "test")

EXPECTED_CLASSES = {
    "bell_sounds",
    "car_horn",
    "cat",
    "cheering",
    "clapping",
    "clock",
    "coughing",
    "crying_sobbing",
    "dog",
    "door_knock",
    "doorbell",
    "explosive_sounds",
    "footsteps",
    "general_alarm",
    "glass_breaking",
    "laughter",
    "phone_ringtone",
    "screaming",
    "siren",
    "transport_sounds",
}


def fail(message):
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main():
    if not SOURCE.is_dir():
        fail(f"Source dataset not found: {SOURCE}")

    if DESTINATION.exists():
        fail(
            f"Destination already exists: {DESTINATION}\n"
            "Nothing was overwritten. Inspect or rename that folder "
            "before rerunning this script."
        )

    # Validate the original dataset before copying anything.
    for split in SPLITS:
        split_dir = SOURCE / split

        if not split_dir.is_dir():
            fail(f"Missing source split: {split_dir}")

        classes = {
            p.name for p in split_dir.iterdir() if p.is_dir()
        }

        if len(classes) != 24:
            fail(
                f"{split} contains {len(classes)} class folders, "
                "but 24 were expected."
            )

        if not classes.issuperset(MERGE_MAP):
            fail(f"{split} is missing one or more merge-source classes.")

        mapped_classes = {
            MERGE_MAP.get(name, name) for name in classes
        }

        if mapped_classes != EXPECTED_CLASSES:
            fail(
                f"Unexpected resulting classes in {split}.\n"
                f"Expected: {sorted(EXPECTED_CLASSES)}\n"
                f"Found: {sorted(mapped_classes)}"
            )

    print("Validation passed. Creating merged dataset...")
    print("Original dataset will not be modified.")

    try:
        for split in SPLITS:
            source_split = SOURCE / split
            destination_split = DESTINATION / split
            destination_split.mkdir(parents=True, exist_ok=True)

            for source_class_dir in sorted(source_split.iterdir()):
                if not source_class_dir.is_dir():
                    continue

                target_class = MERGE_MAP.get(
                    source_class_dir.name,
                    source_class_dir.name,
                )
                target_dir = destination_split / target_class
                target_dir.mkdir(parents=True, exist_ok=True)

                for source_file in source_class_dir.rglob("*"):
                    if not source_file.is_file():
                        continue

                    relative_path = source_file.relative_to(
                        source_class_dir
                    )

                    # Prefix filenames with their original class so
                    # identical filenames from different classes cannot
                    # overwrite one another after merging.
                    prefix = source_class_dir.name
                    destination_file = (
                        target_dir
                        / f"{prefix}__{relative_path.as_posix().replace('/', '__')}"
                    )

                    if destination_file.exists():
                        fail(
                            "Filename collision detected: "
                            f"{destination_file}"
                        )

                    shutil.copy2(source_file, destination_file)

            print(f"Created split: {split}")

        class_names = sorted(EXPECTED_CLASSES)

        manifest = {
            "dataset_name": "SoundSense_Merged_20Class",
            "num_classes": len(class_names),
            "class_names": class_names,
            "splits": list(SPLITS),
            "merge_map": MERGE_MAP,
            "original_dataset": str(SOURCE),
            "source_files_copied": True,
        }

        with (DESTINATION / "class_mapping.json").open("w") as f:
            json.dump(manifest, f, indent=2)

        print("\n========== MERGED DATASET SUMMARY ==========")

        for split in SPLITS:
            split_dir = DESTINATION / split
            classes = sorted(
                p.name for p in split_dir.iterdir() if p.is_dir()
            )
            total_files = sum(
                1 for p in split_dir.rglob("*")
                if p.is_file() and p.suffix.lower() == ".wav"
            )

            print(f"\n{split.upper()}")
            print(f"Classes: {len(classes)}")
            print(f"WAV files: {total_files}")

            for name in classes:
                count = sum(
                    1 for p in (split_dir / name).rglob("*")
                    if p.is_file() and p.suffix.lower() == ".wav"
                )
                print(f"  {name}: {count}")

        print(f"\nMapping saved to: {DESTINATION / 'class_mapping.json'}")
        print(f"Merged dataset created at: {DESTINATION.resolve()}")
        print("Original dataset was not modified.")

    except Exception:
        print(
            "\nDataset creation failed. The original dataset remains "
            "unchanged. Inspect the destination folder before retrying.",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
