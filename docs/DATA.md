# Data

## Rules (read first)

1. The 103 labeled screenshots are **client data**. They are never committed.
2. Check `git status` before every commit. No images, no `data/`, no `ds/`, no
   `runs/`.
3. The original Label Studio label files are never edited. Fixes go in
   `config/label_corrections.yaml`.
4. Data reaches Colab by upload or Drive, never through git.

Quick check that the ignore rules work:

```bash
git status --short
git check-ignore -v data/renamed/images/ss012.png   # must print a rule
```

## Folders

```
data/
├── raw/            original export: 103 images + 22-class labels (read-only)
├── renamed/        copies with stable ids ss001..ss103 + classes_22.txt
├── renamed.zip     the same, zipped for upload to Colab
├── labels_16/      16-class labels written by scripts/remap_labels.py
└── preview/        annotated images from scripts/draw_labels.py
ds/                 train/val split written by scripts/split_dataset.py
```

All of these are ignored by git.

## Flow

| Step | Script | In | Out |
|---|---|---|---|
| Rename | `rename_dataset.py` | `data/raw/` | `data/renamed/` |
| Audit | `audit_labels.py` | `data/renamed/` | a report only |
| Remap + fix | `remap_labels.py` | `data/renamed/`, `config/label_corrections.yaml` | `data/labels_16/`, `config/classes_16.txt` |
| Look | `draw_labels.py` | `data/labels_16/` | `data/preview/` |
| Split | `split_dataset.py` | `data/labels_16/` | `ds/`, `config/ui.yaml`, `config/folds/` |

## Taxonomy — 16 classes, locked order

The order is the contract with `ui_comparision`. Never reorder.
Full list and merge rules: `CLAUDE.md`.

Merges from the original 22 classes:

- `nav_item` + `category_tab` + `detail_tab` + `carousel_filter` → `tab_item`
- `badge` + `episode_badge` + `duration` → `overlay_badge`
- `movie_title` + `series_title` → `title`

`cast_card` and `celebrity_card` stay separate.

The remap maps **by name**. An unknown class name stops the script.

## Label corrections

`config/label_corrections.yaml` holds 35 fixes, each checked by eye on the
screenshot:

- 20× `section_title` → `content_card`, 1× → `icon` (they were posters or
  tiles, not headings)
- 10× `button` → `tab_item` (detail-page action rows in ss002, ss011, ss043)
- 4× `button` → `overlay_badge` ("Free" ribbons in ss021)

The loader fails on an unknown image, a bad line number, an unknown class, or
a duplicate key.

To add a fix: add a line `ssNNN: {line_number: new_class}`, re-run the remap,
re-run `draw_labels.py` on that image, and look at it.

## Split

- 83 train / 20 val, seed 42, class-aware (tries to keep rare classes on both
  sides and warns when it can't).
- 5-fold configs in `config/folds/` for more stable scores.
- The split must stay the same between runs, or scores can't be compared.
  The Colab notebook checks this ("Split matches the local run exactly").

## Known limits — do not tune against these

| Class | Approx. instances |
|---|---|
| content_card | 657 |
| tab_item | 459 |
| overlay_badge | 302 |
| section_title | 193 |
| button | 105 |
| cast_card | 95 |
| icon | 94 |
| metadata_text | 88 |
| navigation_bar | 71 |
| celebrity_card | 47 |
| title | 28 |
| description | 21 |
| search_bar | 13 |
| logo | 6 |
| rank_number | 4 |
| text_input | 4 |

(Counts are before the 35 corrections.) `rank_number`, `text_input`, and `logo`
are too small to score reliably. That is a data problem. The comparison repo
covers them with a VLM fallback.

## Getting more data (Phase 10, only if needed)

1. Label new screenshots in Label Studio with the **same 22-class project**.
2. Put them in `data/raw/`, then re-run the whole flow above.
3. Adding data changes the split, so bump the minor version and don't compare
   scores directly with older runs.
