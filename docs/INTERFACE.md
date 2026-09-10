# INTERFACE.md — contract with `ui_comparision`

This file is the boundary between the two repos. Everything crossing it is
listed here. If something changes, bump the bundle version and edit this file in
the same commit.

## What crosses the boundary

Exactly one thing: a versioned model bundle directory.

```
models/ui16_v{N}/
├── best.pt              # ultralytics YOLO weights
├── model_card.json      # everything needed to use best.pt correctly
├── classes_16.txt       # 16 lines, one class name each, in binding order
└── gt_detections.json   # hand labels replayed as detections (stub detector)
```

Nothing else. The comparison repo never imports code from this repo, never reads
the raw dataset, and never trains.

## classes_16.txt

Sixteen lines, in this exact order. The order is the contract — the model emits
integer ids, and this file is the only thing that gives them meaning.

```
content_card
tab_item
overlay_badge
section_title
button
cast_card
icon
metadata_text
navigation_bar
celebrity_card
title
description
search_bar
rank_number
text_input
logo
```

Reordering this list silently corrupts every downstream comparison. If the
taxonomy ever changes, it is a **major** version bump and both repos change
together.

## model_card.json

```json
{
  "version": "ui16_v1",
  "trained": "2026-09-11",
  "detector_repo_sha": "abc1234",
  "base_model": "yolov8s.pt",
  "classes": ["content_card", "tab_item", "..."],
  "imgsz": 1280,
  "letterbox": true,
  "conf_default": 0.25,
  "iou_default": 0.45,
  "train_images": 83,
  "val_images": 20,
  "split_seed": 42,
  "augmentation": { "fliplr": 0.0, "hsv_h": 0.0, "mosaic": 0.3 },
  "per_class_map50": {
    "content_card": 0.87,
    "rank_number": 0.11
  },
  "low_confidence_classes": ["rank_number", "text_input", "logo"]
}
```

Two fields the consumer must actually act on:

- **`imgsz` and `letterbox`** — inference preprocessing has to match training
  preprocessing. The consumer reads these from the card rather than hardcoding
  them. Training at 1280 and inferring at 960 degrades accuracy silently.
- **`low_confidence_classes`** — classes with too few training examples to
  trust. The consumer routes these to its VLM fallback instead of believing the
  detector.

## gt_detections.json

Hand-labeled ground truth, shaped exactly like real detector output, so the
comparison repo can build and debug its whole pipeline before this repo has
finished training anything.

```json
{
  "ss012.png": {
    "image_width": 1080,
    "image_height": 2400,
    "detections": [
      {
        "cls_id": 0,
        "cls_name": "content_card",
        "bbox": [120, 840, 300, 450],
        "confidence": 1.0
      }
    ]
  }
}
```

`bbox` is `[x, y, width, height]` in pixels, top-left origin — matching the
comparison repo's coordinate convention, not YOLO's normalized centre format.
Convert here, once, rather than in both repos.

`confidence` is always `1.0` for ground truth. That is the point: the stub is a
perfect detector, so any bug the consumer hits while using it is definitively a
pipeline bug rather than a detection error.

## Consumer-side contract

The comparison repo implements two classes behind one interface:

```python
class DetectorEngine(Protocol):
    def detect(self, image_path: Path) -> list[UIElement]: ...

class GTStubDetector:   # reads gt_detections.json
    ...

class YOLODetector:     # loads best.pt, reads settings from model_card.json
    ...
```

Swapping between them is one line of config. The score difference between stub
and real model is a directly useful number: it is exactly how much detector
error costs the end-to-end pipeline.

## Mandatory validation on load

The consumer must assert taxonomy agreement at startup and fail loudly:

```python
expected = Path("classes_16.txt").read_text().split()
actual = [model.names[i] for i in range(len(model.names))]
assert actual == expected, f"taxonomy mismatch: {actual} != {expected}"
```

A silent class-order mismatch produces plausible-looking but wrong comparison
reports, which is the worst failure mode available to this system. Crash at load
instead.

`scripts/verify_bundle.py` in this repo performs the same check and is the
handoff gate — a bundle that does not pass it is not published.

## Versioning

- **Patch** (`v1` → `v1.1`): retrained on the same taxonomy and split. Consumer
  needs no changes.
- **Minor**: new training data or changed `imgsz`. Consumer re-reads the card;
  no code change.
- **Major**: taxonomy changed. Both repos change together, and the comparison
  repo's stored UI JSON from earlier runs is no longer comparable.
