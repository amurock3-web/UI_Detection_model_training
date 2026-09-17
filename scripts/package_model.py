"""Package a finished run into the versioned bundle ui_comparision consumes.

    python scripts/package_model.py --run runs/detect/ui16_20260917_071603 --version v1

Writes models/ui16_<version>/ with best.pt, model_card.json, classes_16.txt and
gt_detections.json - the contract in docs/INTERFACE.md. Run
scripts/verify_bundle.py on the result before handing it over.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import re
import shutil
import sys

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
CLASSES_16 = (CONFIG / "classes_16.txt").read_text().split()
BUNDLE_YAML = CONFIG / "bundle.yaml"
DATA_YAML = CONFIG / "ui.yaml"
LABELS_16 = ROOT / "data" / "labels_16"
IMAGES = ROOT / "data" / "renamed" / "images"
IMG_EXT = (".png", ".jpg", ".jpeg")


def to_detection(cls_id, x1, y1, x2, y2, confidence, W, H):
    """One detection in the INTERFACE.md shape: [x, y, w, h] pixels, top-left.

    verify_bundle.py checks real model output against this shape field by field.
    """
    x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
    y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
    return {
        "cls_id": int(cls_id),
        "cls_name": CLASSES_16[int(cls_id)],
        "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
        "confidence": round(float(confidence), 4),
    }


def split_info(data_yaml):
    """(train, val, seed) from the header split_dataset.py writes."""
    m = re.search(r"(\d+) train / (\d+) val, seed (\d+)",
                  data_yaml.read_text().splitlines()[0])
    if not m:
        sys.exit(f"{data_yaml}: no split header - rerun scripts/split_dataset.py")
    return tuple(int(g) for g in m.groups())


def gt_detections():
    """Every hand-labelled box, keyed by image filename, replayed as detections."""
    images = {p.stem: p for p in IMAGES.iterdir() if p.suffix.lower() in IMG_EXT}
    labels = sorted(LABELS_16.glob("*.txt"))
    if not labels:
        sys.exit(f"{LABELS_16} is empty - run scripts/remap_labels.py first")
    missing = [l.stem for l in labels if l.stem not in images]
    if missing:
        sys.exit(f"labels with no image in {IMAGES}: {missing}")

    out, counts = {}, Counter()
    for lbl in labels:
        img = images[lbl.stem]
        with Image.open(img) as im:
            W, H = im.size
        dets = []
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            c, xc, yc, w, h = line.split()
            c = int(c)
            if not 0 <= c < len(CLASSES_16):
                sys.exit(f"{lbl}: class id {c} outside 0..{len(CLASSES_16) - 1}")
            xc, yc, w, h = (float(v) for v in (xc, yc, w, h))
            dets.append(to_detection(c, (xc - w / 2) * W, (yc - h / 2) * H,
                                     (xc + w / 2) * W, (yc + h / 2) * H, 1.0, W, H))
            counts[CLASSES_16[c]] += 1
        out[img.name] = {"image_width": W, "image_height": H, "detections": dets}
    return out, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="runs/detect/<run>")
    ap.add_argument("--version", required=True, help="e.g. v1, v1.1")
    ap.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    args = ap.parse_args()

    weights = args.run / "weights" / "best.pt"
    meta_path = args.run / "run_meta.json"
    for p in (weights, meta_path, DATA_YAML):
        if not p.exists():
            sys.exit(f"missing: {p}")
    meta = json.loads(meta_path.read_text())
    if meta.get("smoke"):
        sys.exit(f"{args.run} is a smoke run - never publish it")

    out = ROOT / "models" / f"ui16_{args.version}"
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists - bump the version, or pass --force")
    out.mkdir(parents=True, exist_ok=True)

    bundle = yaml.safe_load(BUNDLE_YAML.read_text())
    n_train, n_val, seed = split_info(DATA_YAML)
    if seed != meta["split_seed"]:
        sys.exit(f"split seed {seed} in {DATA_YAML} != {meta['split_seed']} the run used")

    gt, counts = gt_detections()
    low = [c for c in CLASSES_16 if counts[c] < bundle["low_confidence_min_instances"]]

    card = {
        "version": out.name,
        "trained": meta["date_utc"][:10],
        "detector_repo_sha": meta["git_sha"],
        "source_run": meta["run"],
        "base_model": meta["model"],
        "classes": CLASSES_16,
        "imgsz": meta["imgsz"],
        "letterbox": bundle["letterbox"],
        "conf_default": bundle["conf_default"],
        "iou_default": bundle["iou_default"],
        "train_images": n_train,
        "val_images": n_val,
        "split_seed": seed,
        "augmentation": meta["augmentation_overrides"],
        "val_map50": meta["metrics"]["map50"],
        "val_map50_95": meta["metrics"]["map50_95"],
        "per_class_map50": {c: meta["per_class"].get(c, {}).get("map50")
                            for c in CLASSES_16},
        "instances_per_class": {c: counts[c] for c in CLASSES_16},
        "low_confidence_classes": low,
    }

    shutil.copy2(weights, out / "best.pt")
    (out / "classes_16.txt").write_text("\n".join(CLASSES_16) + "\n", newline="\n")
    (out / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", newline="\n")
    (out / "gt_detections.json").write_text(json.dumps(gt, indent=1) + "\n", newline="\n")

    total = sum(counts.values())
    print(f"bundle     : {out}")
    print(f"source run : {meta['run']}  ({meta['git_sha']})")
    print(f"model      : {meta['model']}  imgsz {meta['imgsz']}  "
          f"mAP50 {card['val_map50']}  mAP50-95 {card['val_map50_95']}")
    print(f"split      : {n_train} train / {n_val} val, seed {seed}")
    print(f"gt         : {len(gt)} images, {total} boxes")
    print(f"low-conf   : {low}")
    print(f"\nnext: python scripts/verify_bundle.py {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
