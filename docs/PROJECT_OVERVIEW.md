# Project overview — why this exists

## The problem

We need to check whether an OTT streaming app looks the way it should: the
**client** build compared against the **production** build, screen by screen.
Doing that by eye across many screens and devices is slow, and people miss
things.

A pixel diff does not help either. Two screenshots from different devices never
match pixel for pixel, even when nothing is wrong. We need to compare **what is
on the screen** (a row of posters, a tab bar, a "Free" badge, a title), not the
raw pixels.

## The idea

Turn each screenshot into a list of UI elements, then compare the lists.

```
screenshot ──► detect UI elements ──► read text (OCR) ──► build a structure
                                                               │
production structure  ◄──── match & compare ────►  client structure
                                   │
                              difference report
```

The first box, **"detect UI elements"**, is this repo. Everything after it
lives in the `ui_comparision` repo.

This follows the detect → graph → match → diff approach from Moradi et al.
(Tricentis), `graph_paper.pdf`.

## What this repo does

It trains a YOLO object detector. Given a screenshot, the model returns boxes,
and each box is one of 16 fixed UI classes (`content_card`, `tab_item`,
`button`, ... see `CLAUDE.md`).

The output of this repo is **one thing**: a versioned model bundle,
`models/ui16_v{N}/`. The contract for it is `docs/INTERFACE.md`.

## What this repo does not do

No OCR, no text matching, no comparison, no reports. If a task needs those, it
belongs in `ui_comparision`.

## Why two repos and not one

| Reason | Detail |
|---|---|
| Libraries fight | This repo needs PyTorch (ultralytics). The comparison repo needs `paddlepaddle-gpu 3.2.0` for OCR. Both in one CUDA 12.6 environment break each other. |
| Different machines | Training runs on Colab (T4, 16 GB). Inference runs locally on a GTX 1070 (8 GB). |
| Different rhythm | We train a few times. We run comparisons all the time. |
| Client data | The labeled screenshots live only here, so there is only one place they can leak from. |

## Success looks like

1. `models/ui16_v1/` exists and `python scripts/verify_bundle.py models/ui16_v1`
   exits 0.
2. `ui_comparision` loads that bundle, checks the class order, and runs its
   full pipeline with it.
3. The comparison repo can switch between the real model and the ground-truth
   stub (`gt_detections.json`) with one config line.
4. Every trained model can be traced back to its config, data split, and git
   commit.

## Where to read next

| File | What it answers |
|---|---|
| `docs/ARCHITECTURE.md` | How the pieces fit, inside this repo and across both repos |
| `docs/INTEGRATION.md` | How to plug the bundle into `ui_comparision`, step by step |
| `docs/INTERFACE.md` | The exact file formats that cross the boundary |
| `docs/DATA.md` | Where the data lives, how it flows, and the rules around it |
| `docs/RUNBOOK.md` | Every command, in order |
| `docs/COLAB_TRAINING.md` | How to train on Colab, click by click |
| `docs/REQUIREMENTS.md` | Hardware, software, access, and inputs we need |
| `docs/STATUS.md` | What is done, current results, what is left |
| `docs/DECISIONS.md` | Why things are the way they are |
| `BUILD_PROMPTS.md` | The phase-by-phase build prompts |
