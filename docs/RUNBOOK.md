# Runbook — every command, in order

Run from the repo root. Local shell is PowerShell or Git Bash on Windows.

## 0. Setup (once)

```bash
python -m venv .venv
.venv\Scripts\activate          # Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
```

## 1. Prepare data

```bash
python scripts/rename_dataset.py          # data/raw -> data/renamed
python scripts/audit_labels.py            # report only
python scripts/remap_labels.py            # 22 -> 16 + corrections -> data/labels_16
pytest tests/test_remap.py -v             # must pass
python scripts/draw_labels.py --sample 10 # look at data/preview/ before training
python scripts/split_dataset.py           # 83/20, seed 42 -> ds/, config/ui.yaml
python scripts/split_dataset.py --kfold 5 # config/folds/
```

## 2. Smoke run (local, cheap)

```bash
python scripts/train.py --smoke
```

Proves the data loads and training finishes. The score does not matter.

## 3. Full training (Colab)

Beginner-friendly, click-by-click version: `docs/COLAB_TRAINING.md`.

1. Open `notebooks/train_colab.ipynb` in Colab.
2. Menu **Runtime → Change runtime type → T4 GPU**.
3. Upload `data/renamed.zip` when the dataset cell asks.
4. Run all cells top to bottom.
5. The training cell zips `runs/` and downloads it. Output also goes to
   Drive at `MyDrive/ui16_runs` if cell 6 was run.
6. Unzip the run into `runs/detect/<run_name>/` on this machine.

Same thing without the notebook (any machine with a GPU):

```bash
python scripts/train.py                   # uses config/train.yaml
```

## 4. Evaluate

```bash
python scripts/evaluate.py runs/detect/<run_name>
python scripts/evaluate.py runs/detect/<run_name> --kfold 5    # optional
python scripts/error_analysis.py runs/detect/<run_name>
```

Output goes to `runs/detect/<run_name>/eval/`.

## 5. Package and verify (Phase 9)

```bash
python scripts/package_model.py --run runs/detect/<run_name> --version v1
python scripts/verify_bundle.py models/ui16_v1       # must exit 0
```

## 6. Commit

```bash
git status --short          # no images, no data/, ds/, runs/
git add <files>
git commit -m "..."
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Out of GPU memory (1070) | Lower `batch`, then `imgsz`, then use `freeze=10` |
| Colab shows no output while training | Already fixed — the notebook streams output. Pull the latest notebook. |
| Colab session died | Weights are in `MyDrive/ui16_runs` if cell 6 ran |
| Remap stops with "unknown class" | A class name in `classes_22.txt` is missing from the mapping — add it to the mapping, don't skip it |
| Split doesn't match local | Seed or data changed. Check `data/renamed/` is identical on both sides |
| `verify_bundle.py` fails on class order | Do not ship. Rebuild the bundle from the right run |
