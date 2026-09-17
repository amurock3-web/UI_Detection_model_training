# Status

_Last updated: 2026-09-17_

## Phases

| # | Phase | State | Commit |
|---|---|---|---|
| 0 | Scaffold | ✅ Done | |
| 1 | Label audit | ✅ Done | |
| 2 | Remap 22 → 16 | ✅ Done | |
| 3 | Visual check | ✅ Done | |
| 4 | Dataset split | ✅ Done | |
| 5 | Smoke run | ✅ Done | `4d49440` |
| 6 | Full training on Colab | ✅ Done (first run) | `04de2a4` |
| 7 | Evaluation | ✅ Done | `f8ba182` |
| 8 | Error analysis | ✅ Done | `af7a5a7` |
| — | 35 label corrections | ✅ Done | `0a6e1d2` |
| — | Retrain with corrected labels | ✅ Done — `ui16_20260917_071603`, evaluated | `6bad3fd` |
| 9 | Package bundle | ✅ Done — `models/ui16_v1/`, verified | this commit |
| 10 | Improve | ⬜ Only if needed | |

## First full run — `ui16_20260910_142740`

yolov8s, imgsz 1280, batch 8, seed 42, T4. Stopped early at epoch 92 of 200
(patience 50). Labels **before** the 35 corrections.

**Overall: mAP@0.5 = 0.858, mAP@0.5:0.95 = 0.681** (20 val images)

| Class | mAP@0.5 | Precision | Recall | Note |
|---|---|---|---|---|
| content_card | 0.93 | 0.89 | 0.91 | |
| tab_item | 0.96 | 0.87 | 0.96 | |
| overlay_badge | 0.85 | 0.88 | 0.90 | |
| section_title | 0.87 | 0.82 | 0.82 | |
| button | 0.73 | 0.90 | **0.49** | weakest real class |
| cast_card | 1.00 | 0.97 | 1.00 | |
| icon | 0.69 | 0.87 | 0.77 | |
| metadata_text | 0.71 | 0.68 | 0.69 | |
| navigation_bar | 1.00 | 1.00 | 0.99 | |
| celebrity_card | 1.00 | 0.70 | 1.00 | few examples |
| title | 0.80 | 0.75 | 0.80 | few examples |
| description | 0.75 | 0.73 | 1.00 | few examples |
| search_bar | 1.00 | 0.81 | 1.00 | few examples |
| rank_number | 0.50 | 0.85 | 0.50 | too few to trust |
| text_input | 1.00 | 1.00 | 0.97 | too few to trust |
| logo | 1.00 | 0.90 | 1.00 | too few to trust |

## What error analysis found (Phase 8)

At conf 0.25: 361 correct, 47 false positives, 25 missed, 15 wrong class,
2 bad boxes.

Most common mix-ups:

- `button` predicted as `tab_item` (6) — detail-page action rows look exactly
  like tabs
- `section_title` predicted as `content_card` (3) — the "titles" were really
  posters

**Conclusion:** most remaining error came from inconsistent labels, not from
the model being too small. That led to the 35 label corrections instead of a
bigger model.

## Second run — `ui16_20260917_071603` (corrected labels)

Same settings as the first run. Colab T4, 2026-09-17. Stopped at epoch 98,
best epoch 48. Trained and scored **with** the 35 label corrections. Three
corrected images (ss002, ss011, ss036) are in val, so the answer key changed
too — the comparison is close, not exact.

**Overall: mAP@0.5 = 0.877 (+0.019), mAP@0.5:0.95 = 0.703 (+0.022)**

| Class | mAP@0.5 first → new | Precision first → new | Recall first → new |
|---|---|---|---|
| content_card | 0.93 → 0.93 | 0.89 → 0.89 | 0.91 → 0.92 |
| tab_item | 0.96 → **0.99** | 0.87 → **0.97** | 0.96 → 0.99 |
| overlay_badge | 0.85 → 0.85 | 0.88 → 0.88 | 0.90 → 0.88 |
| section_title | 0.87 → **0.96** | 0.82 → **0.97** | 0.82 → 0.84 |
| button | 0.73 → 0.72 | 0.90 → 0.95 | 0.49 → **0.58** |
| cast_card | 1.00 → 1.00 | 0.97 → 0.96 | 1.00 → 1.00 |
| icon | 0.69 → 0.70 | 0.87 → 0.77 | 0.77 → 0.76 |
| metadata_text | 0.71 → 0.72 | 0.68 → 0.72 | 0.69 → 0.77 |
| navigation_bar | 1.00 → 1.00 | 1.00 → 1.00 | 0.99 → 0.99 |
| celebrity_card | 1.00 → 1.00 | 0.70 → 0.35 | 1.00 → 1.00 |
| title | 0.80 → 0.80 | 0.75 → 1.00 | 0.80 → 0.70 |
| description | 0.75 → **0.91** | 0.73 → 0.74 | 1.00 → 1.00 |
| search_bar | 1.00 → 1.00 | 0.81 → 0.86 | 1.00 → 1.00 |
| rank_number | 0.50 → 0.62 | 0.85 → 1.00 | 0.50 → 0.00 |
| text_input | 1.00 → 0.86 | 1.00 → 0.49 | 0.97 → 1.00 |
| logo | 1.00 → 1.00 | 0.90 → 0.91 | 1.00 → 1.00 |

Swings in the last rows (celebrity_card precision, rank_number, text_input)
come from 1–4 val instances each. Noise, not signal.

**Error analysis (conf 0.25):**

| | First run | Second run |
|---|---|---|
| Correct | 361 | 367 |
| Missed | 25 | 22 |
| Wrong class | 15 | **10** |
| Bad box | 2 | 4 |
| False positive | 47 | 55 |

- `button` → `tab_item` mix-up dropped from 6 to 1. The label fix worked.
- False positives went up, mostly `content_card` (14) and `metadata_text`
  (13). Looking at the crops, many are things the labels never mark: the
  status-bar clock, actor names under celebrity circles, cards cut off at the
  screen edge. The model is often right; the labels are incomplete.
- Worst image is ss074 (Help/settings list): rows labelled `button`, model
  finds none. That screen type is rare in training.
- `ss012` has one duplicate label (ultralytics removes it automatically).

**Conclusion:** the label fixes helped as predicted. Remaining error is mostly
unlabeled text and rare screen types, not model size. Good enough to package.

## Bundle `ui16_v1`

Built from `ui16_20260917_071603`:

```bash
python scripts/package_model.py --run runs/detect/ui16_20260917_071603 --version v1
python scripts/verify_bundle.py models/ui16_v1      # PASS, exit 0
pytest tests/ -q                                    # 15 passed (incl. swapped-class check)
```

- `gt_detections.json`: 103 images, 2186 boxes (corrected labels)
- `low_confidence_classes`: rank_number, text_input, logo
- `best.pt` is not in git. Copy it by hand from `models/ui16_v1/` (or
  Drive `MyDrive/ui16_runs`).

## Next steps

1. Hand off to `ui_comparision` (see `docs/INTEGRATION.md`): copy the whole
   `models/ui16_v1/` folder, start with the GT stub, then the real model.
2. Later, optional: label the status-bar clock and names consistently (or
   decide they are never labelled), and add a few more settings/list screens.

## Open risks

| Risk | Impact | Plan |
|---|---|---|
| 20 val images give noisy scores | A small change can look big | Use `evaluate.py --kfold 5` before deciding |
| Torch + Paddle in one environment | Breaks OCR in the comparison repo | Run YOLO in a separate process (see INTEGRATION.md) |
| Very rare classes | Unreliable detections | VLM fallback in the comparison repo |
| Colab session loss | Lost training run | `runs/` on Drive (cell 6) |
