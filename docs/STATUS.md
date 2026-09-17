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
| — | Retrain with corrected labels | ⏳ Started on Colab, results not brought back yet | `06909bc` |
| 9 | Package bundle | ⬜ Not started | |
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

## Next steps

1. Get the corrected-label run from Colab (download zip or
   `MyDrive/ui16_runs`) into `runs/detect/`.
2. Run `evaluate.py` and `error_analysis.py` on it. Compare with the table
   above. Expect `button` and `section_title` to improve.
3. Decide: good enough → Phase 9. Not good enough → Phase 10, one change at a
   time.
4. Phase 9: write `package_model.py` and `verify_bundle.py`, build
   `models/ui16_v1/`.
5. Hand off to `ui_comparision` (see `docs/INTEGRATION.md`).

## Open risks

| Risk | Impact | Plan |
|---|---|---|
| 20 val images give noisy scores | A small change can look big | Use `evaluate.py --kfold 5` before deciding |
| Torch + Paddle in one environment | Breaks OCR in the comparison repo | Run YOLO in a separate process (see INTEGRATION.md) |
| Very rare classes | Unreliable detections | VLM fallback in the comparison repo |
| Colab session loss | Lost training run | `runs/` on Drive (cell 6) |
