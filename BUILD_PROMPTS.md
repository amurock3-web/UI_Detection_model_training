# Build prompts — UI_Detection_model_training

Paste **one phase at a time** into Claude Code. `CLAUDE.md` is at the repo root
and loads automatically, so these stay short.

After each phase: run the command, check the acceptance criterion, commit.

---

## Phase 0 — Scaffold

```
Set up the repo skeleton.

Create:
- src/, scripts/, config/, docs/, notebooks/, models/, data/ (data/ empty and
  gitignored)
- .gitignore blocking data/, ds/, runs/, models/**/*.pt, *.png, *.jpg, *.jpeg,
  __pycache__/, .venv/, .ipynb_checkpoints/
- requirements.txt: ultralytics, opencv-python-headless, pyyaml, pandas,
  matplotlib, tqdm — pinned to major versions
- config/train.yaml holding the augmentation overrides from CLAUDE.md, each with
  its one-line justification comment

Do not write training logic yet.

Acceptance: `git status` with images present in data/ shows nothing staged, and
`git check-ignore -v data/somefile.png` confirms the rule that blocks it.
```

---

## Phase 1 — Label audit

**Do this before the remap. You cannot fix labels you haven't looked at.**

```
Write scripts/audit_labels.py. It reads the original 22-class YOLO label files
plus classes.txt and reports:

- number of images, number of label files, and any mismatch between them
- images with zero labels
- per-class instance counts, sorted descending
- any class id in the label files that is out of range for classes.txt
- boxes with impossible geometry: width or height <= 0, or coordinates outside
  [0,1]
- suspiciously tiny boxes (< 0.5% of image area) — list them, do not delete
- duplicate boxes (IoU > 0.95, same class) — list them, do not delete

Print a summary table. Change nothing on disk.

Acceptance: the report runs clean, or it names specific files I need to fix.
Show me the output before doing anything else.
```

---

## Phase 2 — Remap 22 → 16

```
Write scripts/remap_labels.py.

Read the original 22-class labels plus classes.txt, map to the 16-class taxonomy
in CLAUDE.md, and write to data/labels_16/ plus config/classes_16.txt.

Rules:
- Map BY CLASS NAME, not by index. If a name in classes.txt is missing from the
  mapping table, raise and stop — never guess or skip silently.
- The output class order must exactly match the numbered list in CLAUDE.md.
- Original label files are read-only. Never modify them.
- Print a before/after per-class count table so the merges are visible
  (tab_item should equal nav_item + category_tab + detail_tab + carousel_filter,
  and so on).

Also write tests/test_remap.py checking a few known name→id mappings and that
every output id lands in 0..15.

Acceptance: `pytest tests/test_remap.py -v` passes, and the printed merge
arithmetic adds up.
```

---

## Phase 3 — Visual verification

**Skip this and you may burn a training run on a broken remap.**

```
Write scripts/draw_labels.py: given an image and its remapped label file, draw
the boxes with class names and save an annotated copy to data/preview/.

Add --sample N to pick N random images, seeded.

Then run it on 10 images covering different screen types (home, detail, search)
so I can visually confirm:
- content_card boxes sit on movie posters
- tab_item covers what used to be nav_item AND category_tab AND detail_tab
- overlay_badge covers "Free" ribbons, episode numbers, and durations
- cast_card and celebrity_card did not get swapped

Acceptance: I look at the 10 previews and confirm the remap is visually correct.
Stop and wait for my confirmation.
```

---

## Phase 4 — Dataset split

```
Write scripts/split_dataset.py.

Default: 83 train / 20 val, seed 42, producing ds/images/{train,val},
ds/labels/{train,val}, and config/ui.yaml pointing at them.

Important: with only 103 images, a random split can put all 4 rank_number
examples in train and none in val (or vice versa). Make the split
class-aware — try to keep at least one instance of each rare class on both
sides, and if that's impossible for a class, print a loud warning naming it.

Print per-class instance counts for train and val side by side.

Also add --kfold N mode producing N fold configs, for Phase 7.

Acceptance: printed counts show no class entirely absent from val without a
warning explaining why.
```

---

## Phase 5 — Smoke training run

**A short cheap run whose only job is to prove the plumbing works.**

```
Write scripts/train.py wrapping ultralytics, reading config/train.yaml for the
augmentation overrides.

Then run a deliberately small smoke test: yolov8n, imgsz=640, epochs=20,
batch=4. This is not meant to be good — it is meant to prove the dataset loads,
the classes are right, and training completes.

Report: did it finish, what does the loss curve look like, and does
runs/*/labels.jpg show a sensible class distribution.

Acceptance: run completes and ultralytics' own class-distribution plot matches
the counts from Phase 4.
```

---

## Phase 6 — Full training on Colab

```
Write notebooks/train_colab.ipynb.

It must:
1. Check GPU with nvidia-smi and print which one it got
2. pip install requirements
3. Clone this repo
4. Expect the dataset to be supplied separately — mounted or uploaded by me,
   never pulled from git. Include a clear cell explaining where to put it and
   a check that fails loudly if it is missing.
5. Train: yolov8s.pt, imgsz=1280, epochs=200, batch=8, patience=50, with the
   augmentation overrides from config/train.yaml
6. IMMEDIATELY after training, zip and download runs/ — Colab kills sessions
   without warning. Do not leave this to a later cell.

Add a commented alternative cell for yolo11s.pt so I can compare architectures
later without rewriting the notebook.

Acceptance: notebook runs top to bottom on a fresh Colab T4 and lands a
best.pt in my downloads.
```

---

## Phase 7 — Evaluation

```
Write scripts/evaluate.py taking a runs directory and producing:

1. A per-class table: instances, precision, recall, mAP@0.5, mAP@0.5:0.95 —
   sorted by instance count descending, with a "LOW-N, directional only" marker
   on any class under 20 instances.
2. The confusion matrix, saved as an image, with a note on which class pairs
   are being confused most.
3. The 10 worst validation images by loss, with predictions and ground truth
   drawn side by side, so I can see what is actually failing.

Then optionally support --kfold to average mAP across the folds from Phase 4.
With 20 validation images, a single split gives a noisy number; 5-fold gives one
I can trust. Report both the mean and the spread across folds.

Acceptance: I get the table, the matrix, and the failure images. I will judge
whether the model is good enough.
```

---

## Phase 8 — Error analysis

```
Before touching hyperparameters, tell me what is actually wrong.

Write scripts/error_analysis.py classifying every validation error as:
- missed detection (ground truth with no matching prediction)
- false positive (prediction with no matching ground truth)
- localization error (right class, IoU between 0.1 and 0.5)
- classification error (good IoU, wrong class)

Report counts per category per class, and for the top 3 problem classes show me
example crops.

Then give me your read: is the remaining error a model capacity problem (train
bigger / longer), a data problem (need more labels of class X), or a labeling
problem (the ground truth itself is inconsistent)?

Do not start fixing anything. Diagnose first.

Acceptance: a written diagnosis with evidence, and a recommendation I can
accept or reject.
```

---

## Phase 9 — Package the bundle

```
Write scripts/package_model.py producing the versioned bundle described in
CLAUDE.md and docs/INTERFACE.md:

models/ui16_v{N}/
├── best.pt
├── model_card.json
├── classes_16.txt
└── gt_detections.json

model_card.json fields: version, trained date, this repo's git sha, classes in
exact order, imgsz, letterbox flag, conf_default, iou_default, per-class mAP@0.5,
training config used, dataset split seed, and total training images.

gt_detections.json: every hand-labeled box from all 103 images, keyed by image
filename, in the SAME shape the real detector outputs — so the comparison repo
can swap between the ground-truth stub and the real model with one line. Include
the class name alongside the id so a mismatch is human-visible.

Also write scripts/verify_bundle.py, which the consuming repo will run: load
best.pt, assert model.names matches classes_16.txt exactly and in order, run
inference on one image, and confirm output shape matches what INTERFACE.md
promises. Exit non-zero on any mismatch.

Acceptance: `python scripts/verify_bundle.py models/ui16_v1` exits 0, and
manually reordering a line in classes_16.txt makes it exit non-zero.
```

---

## Phase 10 — Improve (only if Phase 8 says to)

```
Based on the Phase 8 diagnosis, run ONE experiment at a time and record each in
a results table:

Candidates, in the order I would try them:
1. yolo11s vs yolov8s at identical settings
2. imgsz 1280 vs 1536 (UI text is small; resolution often matters more than
   model size here)
3. yolov8m if VRAM allows on Colab
4. Longer training with a lower final learning rate
5. More labeled data for the specific classes Phase 8 named

Every run appends a row to results.csv with the full config and the resulting
mAP. Never change two things at once.

Acceptance: results.csv shows a clear before/after and I can see which change
paid for itself.
```

---

## Notes on driving Claude Code

- One phase per session where possible. Long sessions drift.
- If it jumps ahead a phase, stop it and re-paste the phase prompt.
- Phases 1, 3, and 8 end with "show me and wait." Hold that line — those are the
  three points where proceeding on a wrong assumption costs the most.
- After each phase: `git add -A && git commit -m "phase N: ..."`.
  Check `git status` for images first, every time.
- If an acceptance criterion fails twice, stop the patching and ask it to
  explain what it thinks is wrong before it writes more code.
