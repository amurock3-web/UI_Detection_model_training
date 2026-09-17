"""Handoff gate for a model bundle. Exits non-zero on any mismatch.

    python scripts/verify_bundle.py models/ui16_v1
    python scripts/verify_bundle.py models/ui16_v1 --image path/to/screen.png

Self-contained on purpose: it reads only the bundle folder, so ui_comparision
can run it too. Needs ultralytics and pillow.

Checks: all four files exist; model.names equals classes_16.txt in order; the
model card agrees and has every INTERFACE.md field; every ground-truth
detection is well formed; and real inference converts to exactly the
gt_detections.json shape.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image

FILES = ("best.pt", "model_card.json", "classes_16.txt", "gt_detections.json")
CARD_FIELDS = ("version", "trained", "detector_repo_sha", "base_model", "classes",
               "imgsz", "letterbox", "conf_default", "iou_default", "train_images",
               "val_images", "split_seed", "augmentation", "per_class_map50",
               "low_confidence_classes")
IMAGE_FIELDS = {"image_width", "image_height", "detections"}
DET_FIELDS = {"cls_id", "cls_name", "bbox", "confidence"}
FALLBACK_IMAGES = Path(__file__).resolve().parent.parent / "data" / "renamed" / "images"


def check_detection(d, names, W, H, where):
    """Problems with one detection dict, as strings."""
    if set(d) != DET_FIELDS:
        return [f"{where}: fields {sorted(d)} != {sorted(DET_FIELDS)}"]
    out = []
    if not isinstance(d["cls_id"], int) or not 0 <= d["cls_id"] < len(names):
        out.append(f"{where}: cls_id {d['cls_id']!r} out of range")
    elif d["cls_name"] != names[d["cls_id"]]:
        out.append(f"{where}: cls_id {d['cls_id']} is {names[d['cls_id']]}, "
                   f"but cls_name says {d['cls_name']}")
    b = d["bbox"]
    if len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
        out.append(f"{where}: bbox {b} is not [x, y, w, h]")
    elif b[2] < 0 or b[3] < 0 or b[0] < 0 or b[1] < 0 \
            or b[0] + b[2] > W + 0.5 or b[1] + b[3] > H + 0.5:
        out.append(f"{where}: bbox {b} outside the {W}x{H} image")
    if not isinstance(d["confidence"], float) or not 0.0 <= d["confidence"] <= 1.0:
        out.append(f"{where}: confidence {d['confidence']!r} not a float in [0, 1]")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--image", type=Path, help="image for the inference check")
    args = ap.parse_args()
    b = args.bundle
    problems = []

    missing = [f for f in FILES if not (b / f).is_file()]
    if missing:
        sys.exit(f"FAIL {b}: missing {missing}")

    names = (b / "classes_16.txt").read_text().split()
    card = json.loads((b / "model_card.json").read_text())
    gt = json.loads((b / "gt_detections.json").read_text())

    # 1. taxonomy - the check that matters most
    if len(names) != 16 or len(set(names)) != 16:
        problems.append(f"classes_16.txt has {len(names)} lines "
                        f"({len(set(names))} unique), expected 16 unique")
    from ultralytics import YOLO
    model = YOLO(str(b / "best.pt"))
    model_names = [model.names[i] for i in range(len(model.names))]
    if model_names != names:
        problems.append(f"taxonomy mismatch:\n  model.names    {model_names}\n"
                        f"  classes_16.txt {names}")

    # 2. model card
    absent = [f for f in CARD_FIELDS if f not in card]
    if absent:
        problems.append(f"model_card.json missing fields {absent}")
    if card.get("classes") != names:
        problems.append("model_card.json classes differ from classes_16.txt")
    if card.get("version") != b.name:
        problems.append(f"model_card.json version {card.get('version')!r} "
                        f"!= folder name {b.name!r}")
    unknown = set(card.get("low_confidence_classes", [])) - set(names)
    if unknown:
        problems.append(f"low_confidence_classes not in taxonomy: {sorted(unknown)}")

    # 3. ground-truth stub
    n_boxes = 0
    for img, entry in gt.items():
        if set(entry) != IMAGE_FIELDS:
            problems.append(f"gt {img}: fields {sorted(entry)} != {sorted(IMAGE_FIELDS)}")
            continue
        for i, d in enumerate(entry["detections"]):
            problems += check_detection(d, names, entry["image_width"],
                                        entry["image_height"], f"gt {img}[{i}]")
        n_boxes += len(entry["detections"])

    # 4. real inference, converted to the same shape
    image = args.image
    if image is None and gt:
        first = FALLBACK_IMAGES / next(iter(gt))
        image = first if first.exists() else None
    if image is None:
        print("note: no --image given and no local dataset - using a blank image, "
              "so only the output shape is checked, not real detections")
        W, H = 1080, 2400
        source = np.full((H, W, 3), 128, dtype=np.uint8)
    else:
        with Image.open(image) as im:
            W, H = im.size
        source = str(image)

    res = model.predict(source, imgsz=card.get("imgsz"), conf=card.get("conf_default"),
                        iou=card.get("iou_default"), verbose=False)[0]
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    if res.orig_shape != (H, W):
        problems.append(f"inference saw {res.orig_shape}, expected {(H, W)}")
    dets = []
    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        x1, x2 = max(0.0, min(float(x1), W)), max(0.0, min(float(x2), W))
        y1, y2 = max(0.0, min(float(y1), H)), max(0.0, min(float(y2), H))
        dets.append({"cls_id": int(c), "cls_name": model.names[int(c)],
                     "bbox": [round(x1, 1), round(y1, 1),
                              round(x2 - x1, 1), round(y2 - y1, 1)],
                     "confidence": round(float(p), 4)})
    for i, d in enumerate(dets):
        problems += check_detection(d, names, W, H, f"inference[{i}]")

    print(f"bundle    : {b}")
    print(f"card      : {card.get('version')}  {card.get('base_model')}  "
          f"imgsz {card.get('imgsz')}  from {card.get('source_run', '?')}")
    print(f"classes   : {len(names)}, model agrees: {model_names == names}")
    print(f"gt        : {len(gt)} images, {n_boxes} boxes")
    print(f"inference : {image or 'blank image'} -> {len(dets)} detections")
    for d in dets[:3]:
        print(f"            {d}")

    if problems:
        print(f"\nFAIL - {len(problems)} problem(s):")
        for p in problems[:30]:
            print("  -", p)
        sys.exit(1)
    print("\nPASS - bundle is safe to hand over")


if __name__ == "__main__":
    main()
