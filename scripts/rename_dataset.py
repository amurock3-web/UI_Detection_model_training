"""Rename raw screenshots + labels to stable ssNNN ids, preserving pairing.

Source is already-extracted data/raw (the original zip is gone), so this
copies rather than unzips. Originals are never touched — see CLAUDE.md.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
SRC_IMAGES = ROOT / "data" / "raw" / "images"
SRC_LABELS = ROOT / "data" / "raw" / "labels"
OUT = ROOT / "data" / "renamed"

images_out = OUT / "images"
labels_out = OUT / "labels"
images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

images = sorted(p for p in SRC_IMAGES.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
# classes.txt has no matching image, so it is never picked up here
labels = {p.stem: p for p in SRC_LABELS.glob("*.txt")}

print(f"Images found: {len(images)}")
print(f"Labels found: {len(labels)}")

count = 0
missing = []
for image in images:
    label = labels.get(image.stem)
    if label is None:
        missing.append(image.name)
        continue

    count += 1
    new_name = f"ss{count:03d}"
    new_image = images_out / f"{new_name}{image.suffix.lower()}"
    new_label = labels_out / f"{new_name}.txt"
    shutil.copy2(image, new_image)
    shutil.copy2(label, new_label)
    print(f"{count:03d}: {image.name}  ->  {new_image.name} + {new_label.name}")

# classes.txt travels with the labels — the remap in Phase 1 needs it
classes = SRC_LABELS / "classes.txt"
if classes.exists():
    shutil.copy2(classes, OUT / "classes_22.txt")

print("\n" + "=" * 50)
print("DATASET RENAME COMPLETE")
print("=" * 50)
print(f"Paired screenshots : {count}")
print(f"Missing annotations: {len(missing)}")
for name in missing:
    print(" -", name)
print(f"\nOutput: {OUT}")

assert count == len(images) - len(missing), "pairing lost a file"
