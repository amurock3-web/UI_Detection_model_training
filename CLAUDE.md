# CLAUDE.md — UI_Detection_model_training

## What this repo is

Trains a YOLO object detector that finds UI elements in OTT streaming app
screenshots and classifies them into a locked 16-class taxonomy.

This repo does **one thing**: produce a versioned, verified model bundle.

It does **not** do OCR, fusion, comparison, or reporting. Those live in the
separate `ui_comparision` repo, which consumes the bundle this repo publishes.
If a task here starts drifting toward comparison logic, stop — wrong repo.

## Why it is separate

- `ultralytics` pulls PyTorch; the comparison repo runs `paddlepaddle-gpu 3.2.0`.
  Torch and Paddle in one CUDA 12.6 environment break each other. Keeping them
  apart protects a working PaddleOCR install.
- Training runs on Colab (T4, 16 GB). Inference runs on a local GTX 1070 (8 GB).
- Training happens a handful of times. Inference happens constantly.
- The labeled client screenshots live only here, so there is exactly one place
  that can leak them.

## Data — read this before touching git

The 103 labeled screenshots are **client data**.

- They are never committed. `.gitignore` blocks them; verify `git status` before
  every commit.
- They live on disk outside the repo (or in an ignored `data/` folder) and are
  supplied to Colab separately, not via the repo.
- Original 22-class Label Studio label files are never re-annotated in place.
  The remap writes new files; the originals stay untouched.

## The taxonomy — locked, 16 classes, this exact order

```
0  content_card
1  tab_item
2  overlay_badge
3  section_title
4  button
5  cast_card
6  icon
7  metadata_text
8  navigation_bar
9  celebrity_card
10 title
11 description
12 search_bar
13 rank_number
14 text_input
15 logo
```

Order is the contract with the consuming repo. Do not reorder. Ever.

Merges from the original 22-class set:
- `nav_item` + `category_tab` + `detail_tab` + `carousel_filter` → `tab_item`
- `badge` + `episode_badge` + `duration` → `overlay_badge`
- `movie_title` + `series_title` → `title`

`cast_card` and `celebrity_card` stay **separate** — different screens,
different diagnostics. Do not merge them.

## Known data limits — do not tune against these

Approximate instance counts across the 103 images:

```
content_card 657   tab_item 459   overlay_badge 302   section_title 193
button 105   cast_card 95   icon 94   metadata_text 88   navigation_bar 71
celebrity_card 47   title 28   description 21   search_bar 13
logo 6   rank_number 4   text_input 4
```

`rank_number`, `text_input`, and `logo` have single-digit counts. Any metric on
those three is directional, not reliable. They will look bad. That is expected,
it is a data problem not a model problem, and the consuming pipeline covers them
with a VLM fallback. Do not spend effort chasing them.

## Augmentation rules — UI is not photography

Default YOLO augmentation assumes natural images. Several defaults are actively
harmful here:

```
fliplr=0.0    # mirroring puts a left-side nav bar on the right — wrong label
flipud=0.0    # UI is never upside down
hsv_h=0.0     # colour is real diagnostic signal; don't teach invariance to it
hsv_s=0.2     # mild only
degrees=0.0   # UI is axis-aligned
perspective=0.0
mosaic=0.3    # mosaic invents layouts that cannot occur; keep low
scale=0.3     # some scale variation is good — real devices differ in resolution
```

Every augmentation choice gets a one-line comment explaining why.

## Environment

Training: Google Colab, T4 (16 GB VRAM).
Local verification: GTX 1070 (8 GB) inside the `ui_compare_v2` container.

If a job OOMs on the 1070, step down in this order: `batch` → `imgsz` →
`freeze=10` (head-only training).

## The deliverable

A model bundle, not a loose `.pt` file:

```
models/ui16_v{N}/
├── best.pt
├── model_card.json      # classes in order, imgsz, letterbox, defaults, per-class mAP
├── classes_16.txt
└── gt_detections.json   # hand labels replayed as detections, for the consumer's stub
```

`gt_detections.json` matters: the comparison repo uses it as a perfect-accuracy
stub detector so it can build and debug fusion/matching/diff without waiting for
this repo, and so pipeline bugs stay distinguishable from detection errors.

Full contract: `docs/INTERFACE.md`. If the contract changes, bump the version
and update that file in the same commit.

## Working conventions

- **Always give the exact run command** alongside any code.
- When several changes accumulate, output the **complete revised file**, not a diff.
- Thresholds and training config live in `config/*.yaml`. No magic numbers.
- Every training run records: model, imgsz, batch, epochs, augmentation overrides,
  dataset split seed, git sha, date, and resulting per-class mAP.
- Commit after every phase. Check `git status` for images first, every time.

## Phase discipline

Work one phase at a time. At the end of each phase, stop and report:
1. Files created or changed
2. The exact command to run it
3. Actual output from running it
4. Whether the acceptance criterion passed

Do not start the next phase until told to.
