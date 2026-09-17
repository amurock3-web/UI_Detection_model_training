# Requirements — what we need to finish

## Hardware

| Need | Have | Used for |
|---|---|---|
| Training GPU, ≥ 16 GB | Google Colab Tesla T4 (15.6 GB usable) | Full training at imgsz 1280 |
| Local GPU | GTX 1070, 8 GB | Evaluation, packaging, verification, inference |
| Storage | Google Drive `MyDrive/ui16_runs` | Keeps weights if Colab disconnects |

## Software

This repo (`requirements.txt`):

- Python 3.11 locally (Colab currently runs 3.13)
- `ultralytics >=8.3,<9` (Colab used 8.4.146), `torch >=2.2`
- `opencv-python-headless`, `pyyaml`, `pandas`, `matplotlib`, `tqdm`, `pytest`

Never add `paddlepaddle` or `ollama` here. They belong to `ui_comparision`.

Comparison repo (for reference): `paddlepaddle-gpu 3.2.0`, CUDA 12.6,
container `ui_compare_v2`.

## Access

- [ ] GitHub repo `amurock3-web/UI_Detection_model_training` — Colab opens the notebook from it
- [ ] Google account with Colab and Drive
- [ ] Local copy of the 103 labeled screenshots (`data/raw/`)
- [ ] Label Studio project with the original 22-class setup (only if we add data)
- [ ] Access to the `ui_comparision` repo for the handoff

## Inputs from people

| Input | From | Needed for |
|---|---|---|
| Confirm label fixes look right | Project owner | Done for the 35 current fixes |
| Download of the corrected-label Colab run | Project owner | Evaluating the retrain |
| "Good enough" call on scores | Project owner | Moving to Phase 9 |
| More labeled screenshots | Project owner / labeling | Phase 10, only if error analysis asks for it |

## Deliverables

| Deliverable | Status |
|---|---|
| Data prep scripts (rename, audit, remap, draw, split) | Done |
| Training script + Colab notebook | Done |
| Evaluation + error analysis scripts | Done |
| `scripts/package_model.py` | To do (Phase 9) |
| `scripts/verify_bundle.py` | To do (Phase 9) |
| `models/ui16_v1/` bundle | To do (Phase 9) |
| `docs/INTERFACE.md` contract | Done |
| Results log `results.csv` | To do (Phase 10, only if we run experiments) |

## Definition of done

- [ ] `python scripts/verify_bundle.py models/ui16_v1` exits 0
- [ ] Swapping two lines in `classes_16.txt` makes it exit non-zero
- [ ] `model_card.json` has every field listed in `docs/INTERFACE.md`
- [ ] `gt_detections.json` covers all 103 images
- [ ] The run behind the bundle has `run_meta.json` (model, imgsz, batch,
      epochs, augmentation, split seed, git sha, date, per-class mAP)
- [ ] No client images in git history
- [ ] `ui_comparision` loads the bundle and passes its class-order check
