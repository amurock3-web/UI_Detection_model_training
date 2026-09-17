# Decisions

Short record of the choices that shape this project. Newest at the bottom.
To change one, add a new entry that replaces it — don't edit the old one.

---

### D1 — Separate repo for the detector
**Decision:** Training lives here, comparison lives in `ui_comparision`.
**Why:** PyTorch and PaddlePaddle break each other in one CUDA 12.6
environment; training and inference run on different machines and at different
rates; client images stay in one place.
**Cost:** Two repos to keep in step, so the contract (`docs/INTERFACE.md`) must
be strict.

### D2 — Hand over a bundle, not a `.pt` file
**Decision:** Ship `models/ui16_vN/` with weights, model card, class list, and
ground-truth detections.
**Why:** A loose `.pt` doesn't say which image size, thresholds, or class order
it expects. Getting any of those wrong fails silently.

### D3 — 16 classes, fixed order
**Decision:** Merge 22 Label Studio classes into 16. Order never changes
without a major version bump.
**Why:** Several original classes look the same and had too few examples to
learn separately. The model outputs numbers; the order is what gives them
meaning.

### D4 — Keep `cast_card` and `celebrity_card` separate
**Why:** They appear on different screens and mean different things in a
comparison report.

### D5 — Remap by name, fail on unknown names
**Why:** Label Studio's class order is not ours. Mapping by index, or skipping
unknown names, would drop or mislabel boxes without anyone noticing.

### D6 — UI-specific augmentation
**Decision:** No flips, no rotation, no perspective, no hue shift, low mosaic,
keep some scale change. Settings in `config/train.yaml`.
**Why:** Screenshots are not photos. A mirrored nav bar or a recoloured button
is a screen that can't exist, and colour is real signal.

### D7 — yolov8s at imgsz 1280
**Why:** UI text and badges are small; resolution helps more than model size.
yolov8s at 1280, batch 8 fits a T4. `yolo11s` is the first thing to compare in
Phase 10.

### D8 — Class-aware split, fixed seed 42, plus 5-fold
**Why:** With 103 images, a random split can put every example of a rare class
on one side. A fixed split keeps runs comparable; folds give a steadier score.

### D9 — Ground-truth stub detector
**Decision:** Ship `gt_detections.json` in the same shape as real detections.
**Why:** The comparison repo can build its whole pipeline before training is
done, and a bug found with a perfect detector must be a pipeline bug.

### D10 — Don't chase the rarest classes
**Decision:** No tuning aimed at `rank_number`, `text_input`, `logo`.
**Why:** 4–6 examples each; any score is noise. The comparison repo checks them
with a VLM.

### D11 — Diagnose before tuning
**Decision:** Phase 8 error analysis comes before any hyperparameter change.
Phase 10 changes one thing per run.
**Why:** Otherwise we can't tell which change helped.

### D12 — Fix labels through an overlay file (2026-09-10)
**Decision:** 35 corrections in `config/label_corrections.yaml`, applied after
the remap. Original label files untouched.
**Why:** Error analysis showed the model was being marked wrong for correct
answers (posters labelled as section titles, action rows labelled as buttons).
The overlay keeps originals intact, makes every fix reviewable in one diff,
and is reproducible.
**Note:** The detail-page action rows (ss002, ss011, ss043) were labelled
`tab_item` by the project owner's choice, because they look the same as the
bottom nav items.

### D13 — Train on Colab, stream output live (2026-09-10)
**Why:** Colab held all output until the process ended, so a 15–30 minute run
looked frozen. The notebook now prints each line as it arrives.

### D14 — Run YOLO apart from Paddle in the comparison repo (proposed)
**Decision:** Run the detector in its own process or container, passing JSON.
**Why:** Same reason as D1. Confirm when the handoff happens.
