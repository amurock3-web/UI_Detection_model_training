# Integration guide — connecting this model to `ui_comparision`

This is the step-by-step plan for plugging the detector into the comparison
project. File formats are defined in `docs/INTERFACE.md`; this file is the
"how".

## The short version

1. This repo builds `models/ui16_v1/`.
2. Copy that folder into the comparison repo.
3. The comparison repo checks it, then uses it through one `DetectorEngine`
   interface.
4. Start with the ground-truth stub. Switch to the real model later with one
   config line.

## Step 1 — build and check the bundle (this repo)

```bash
python scripts/package_model.py --run runs/detect/<run> --version v1
python scripts/verify_bundle.py models/ui16_v1      # must exit 0
```

Do not hand over a bundle that fails `verify_bundle.py`.

## Step 2 — move the bundle across

Copy the whole folder, not just `best.pt`:

```
ui_comparision/models/ui16_v1/
├── best.pt
├── model_card.json
├── classes_16.txt
└── gt_detections.json
```

`best.pt` is not in git (`*.pt` is ignored), so it is copied by hand or from
Drive. The other three files are committed here and can be diffed.

## Step 3 — environment in the comparison repo

The comparison repo runs PaddleOCR on `paddlepaddle-gpu 3.2.0`. Running
`ultralytics` (which brings PyTorch) in the same environment can break it.
Pick one and write it down in that repo:

- **Recommended:** run YOLO in its own process/container and pass detections
  as JSON (the same shape as `gt_detections.json`). Paddle never sees torch.
- Only if tested: install `ultralytics` alongside Paddle, then re-run the
  PaddleOCR smoke test before trusting it.

## Step 4 — the detector interface (comparison repo)

```python
class DetectorEngine(Protocol):
    def detect(self, image_path: Path) -> list[UIElement]: ...

class GTStubDetector:   # reads gt_detections.json — a perfect detector
    ...

class YOLODetector:     # loads best.pt, reads imgsz/conf/iou from model_card.json
    ...
```

Rules for `YOLODetector`:

- Read `imgsz`, `letterbox`, `conf_default`, `iou_default` from
  `model_card.json`. Never hard-code them.
- Convert YOLO output to `bbox = [x, y, width, height]` in pixels, top-left
  origin — the same shape as the stub.
- Put both `cls_id` and `cls_name` on every detection.

## Step 5 — fail loudly on load

At startup, before any comparison:

```python
expected = Path("classes_16.txt").read_text().split()
actual = [model.names[i] for i in range(len(model.names))]
assert actual == expected, f"taxonomy mismatch: {actual} != {expected}"
```

A wrong class order gives reports that look fine but are wrong. Crashing is
better.

## Step 6 — low-confidence classes

`model_card.json` lists `low_confidence_classes`
(`rank_number`, `text_input`, `logo`). These have fewer than 10 training
examples each. The comparison repo should not trust the detector for these and
should check them with its VLM fallback instead.

## Step 7 — roll-out order

| Stage | Detector | What it proves |
|---|---|---|
| 1 | `GTStubDetector` | OCR, fusion, matching, and diff work — any bug is a pipeline bug |
| 2 | `YOLODetector` on the same 103 images | How much real detection error costs end to end (stub score − real score) |
| 3 | `YOLODetector` on new, unlabeled screenshots | Real use |

Stage 1 can start **before** training is finished — it only needs
`gt_detections.json`.

## Step 8 — updating the model later

| Change | Version bump | Comparison repo must |
|---|---|---|
| Retrained, same classes and split | patch (`v1` → `v1.1`) | Swap the folder. Nothing else. |
| New data or new `imgsz` | minor | Swap the folder. It reads the card, so no code change. |
| Class list changed | **major** | Change both repos together. Old stored results are no longer comparable. |

When the contract changes, update `docs/INTERFACE.md` in the same commit.

## Integration checklist

- [ ] `verify_bundle.py` exits 0 in this repo
- [ ] Bundle folder copied complete (4 files)
- [ ] Comparison repo environment decision written down (separate process or shared)
- [ ] PaddleOCR still works after that decision
- [ ] `GTStubDetector` runs the full pipeline on the 103 images
- [ ] `YOLODetector` passes the class-order check on load
- [ ] `YOLODetector` settings come from `model_card.json`
- [ ] Low-confidence classes routed to the VLM fallback
- [ ] Stub vs real score recorded
