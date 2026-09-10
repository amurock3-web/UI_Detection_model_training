"""Classify every validation error, so a fix targets the real cause.

Four kinds, which imply different remedies:
  missed         ground truth with nothing predicted over it      -> data/capacity
  false_positive prediction with no ground truth under it         -> data/capacity
  localization   right class, box in roughly the right place      -> capacity
                 but IoU below 0.5
  classification good box, wrong class                            -> labeling

Diagnose before tuning: a labeling problem cannot be fixed by a bigger model,
and a capacity problem cannot be fixed by relabeling.

    python scripts/error_analysis.py runs/detect/ui16_20260910_142740
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw_labels import font

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
CLASSES_16 = (CONFIG / "classes_16.txt").read_text().split()

CONF = 0.25          # the operating point a consumer would actually use
IOU_HIT = 0.5        # at or above this, the box is in the right place
IOU_NEAR = 0.1       # between NEAR and HIT, the box is close but sloppy
CROPS_PER_CLASS = 6

GT_COLOUR, PRED_COLOUR = "#00FF66", "#FF3B30"


def iou_matrix(a, b):
    """IoU of every box in a (N,4) against every box in b (M,4), xyxy pixels."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def load_gt(label_path, W, H):
    """YOLO normalized xywh -> (classes, xyxy pixels)."""
    classes, boxes = [], []
    if not label_path.exists():
        return np.array(classes, dtype=int), np.zeros((0, 4))
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = line.split()
        xc, yc, w, h = (float(v) for v in (xc, yc, w, h))
        classes.append(int(cls))
        boxes.append([(xc - w/2)*W, (yc - h/2)*H, (xc + w/2)*W, (yc + h/2)*H])
    return np.array(classes, dtype=int), np.array(boxes).reshape(-1, 4)


def classify_image(gt_cls, gt_box, pr_cls, pr_box, pr_conf):
    """Greedy highest-IoU matching, then bucket whatever is left over.

    Predictions are consumed in confidence order so the strongest detection
    claims a ground-truth box first - the same convention the mAP calculation
    uses, which keeps this consistent with the Phase 7 numbers.
    """
    ious = iou_matrix(pr_box, gt_box)
    gt_taken = np.zeros(len(gt_box), dtype=bool)
    pr_taken = np.zeros(len(pr_box), dtype=bool)
    events = []

    # 1. correct detections: right class, box in the right place
    for p in np.argsort(-pr_conf):
        if not len(gt_box):
            break
        candidates = [g for g in range(len(gt_box))
                      if not gt_taken[g] and gt_cls[g] == pr_cls[p]
                      and ious[p, g] >= IOU_HIT]
        if candidates:
            g = max(candidates, key=lambda g: ious[p, g])
            gt_taken[g], pr_taken[p] = True, True
            events.append(("correct", int(pr_cls[p]), int(gt_cls[g]), p, g,
                           float(ious[p, g])))

    # 2. good box, wrong label -> labeling signal
    for p in np.argsort(-pr_conf):
        if pr_taken[p] or not len(gt_box):
            continue
        candidates = [g for g in range(len(gt_box))
                      if not gt_taken[g] and ious[p, g] >= IOU_HIT]
        if candidates:
            g = max(candidates, key=lambda g: ious[p, g])
            gt_taken[g], pr_taken[p] = True, True
            events.append(("classification", int(pr_cls[p]), int(gt_cls[g]), p, g,
                           float(ious[p, g])))

    # 3. right label, sloppy box -> capacity signal
    for p in np.argsort(-pr_conf):
        if pr_taken[p] or not len(gt_box):
            continue
        candidates = [g for g in range(len(gt_box))
                      if not gt_taken[g] and gt_cls[g] == pr_cls[p]
                      and IOU_NEAR <= ious[p, g] < IOU_HIT]
        if candidates:
            g = max(candidates, key=lambda g: ious[p, g])
            gt_taken[g], pr_taken[p] = True, True
            events.append(("localization", int(pr_cls[p]), int(gt_cls[g]), p, g,
                           float(ious[p, g])))

    for g in range(len(gt_box)):
        if not gt_taken[g]:
            events.append(("missed", None, int(gt_cls[g]), None, g, 0.0))
    for p in range(len(pr_box)):
        if not pr_taken[p]:
            events.append(("false_positive", int(pr_cls[p]), None, p, None, 0.0))
    return events


def crop(image, boxes, title, pad=90, max_w=760):
    """Tight crop around the boxes involved, with them drawn on top."""
    W, H = image.size
    xs = [c for b, _ in boxes for c in (b[0], b[2])]
    ys = [c for b, _ in boxes for c in (b[1], b[3])]
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(W, max(xs) + pad), min(H, max(ys) + pad)
    if x2 - x1 < 40 or y2 - y1 < 40:
        x2, y2 = min(W, x1 + 200), min(H, y1 + 200)

    piece = image.crop((x1, y1, x2, y2)).convert("RGB")
    draw = ImageDraw.Draw(piece)
    for box, colour in boxes:
        draw.rectangle((box[0]-x1, box[1]-y1, box[2]-x1, box[3]-y1),
                       outline=colour, width=3)

    tag = font(15)
    bh = draw.textbbox((0, 0), title, font=tag)[3] + 8
    out = Image.new("RGB", (piece.width, piece.height + bh), "black")
    ImageDraw.Draw(out).text((5, 3), title, fill="white", font=tag)
    out.paste(piece, (0, bh))
    if out.width > max_w:
        out = out.resize((max_w, round(max_w * out.height / out.width)), Image.LANCZOS)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--data", type=Path, default=CONFIG / "ui.yaml")
    ap.add_argument("--conf", type=float, default=CONF)
    args = ap.parse_args()

    weights = args.run / "weights" / "best.pt"
    if not weights.exists():
        sys.exit(f"{weights} not found")

    import yaml
    cfg = yaml.safe_load(args.data.read_text())
    val_dir = Path(cfg["path"]) / cfg["val"]
    images = sorted(p for p in val_dir.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

    meta = args.run / "run_meta.json"
    imgsz = json.loads(meta.read_text()).get("imgsz", 1280) if meta.exists() else 1280

    from ultralytics import YOLO
    model = YOLO(str(weights))
    print(f"run  : {args.run}")
    print(f"val  : {len(images)} images   conf {args.conf}   imgsz {imgsz}")
    print(f"rules: hit IoU>={IOU_HIT}, near IoU>={IOU_NEAR}\n")

    KINDS = ["correct", "missed", "false_positive", "localization", "classification"]
    per_class = defaultdict(Counter)
    swaps = Counter()
    examples = defaultdict(list)

    for path in images:
        image = Image.open(path)
        W, H = image.size
        gt_cls, gt_box = load_gt(
            Path(str(path).replace("images", "labels")).with_suffix(".txt"), W, H)
        res = model.predict(str(path), imgsz=imgsz, conf=args.conf, verbose=False)[0]
        pr_cls = res.boxes.cls.cpu().numpy().astype(int)
        pr_box = res.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        pr_conf = res.boxes.conf.cpu().numpy()

        for kind, pc, gc, pi, gi, iou in classify_image(
                gt_cls, gt_box, pr_cls, pr_box, pr_conf):
            owner = CLASSES_16[gc if gc is not None else pc]
            per_class[owner][kind] += 1
            if kind == "classification":
                swaps[(CLASSES_16[pc], CLASSES_16[gc])] += 1
            if kind != "correct":
                examples[owner].append({
                    "kind": kind, "image": path.name, "path": str(path), "iou": iou,
                    "pred": None if pi is None else
                            [CLASSES_16[pc], pr_box[pi].tolist(), float(pr_conf[pi])],
                    "gt": None if gi is None else [CLASSES_16[gc], gt_box[gi].tolist()],
                })

    print(f"{'class':<18}{'correct':>8}{'missed':>8}{'falsepos':>9}"
          f"{'localiz':>9}{'classif':>9}{'errors':>8}")
    print("-" * 70)
    ranked = sorted(per_class.items(),
                    key=lambda kv: -sum(v for k, v in kv[1].items() if k != "correct"))
    totals = Counter()
    for name, counts in ranked:
        errs = sum(v for k, v in counts.items() if k != "correct")
        totals.update(counts)
        print(f"{name:<18}{counts['correct']:>8}{counts['missed']:>8}"
              f"{counts['false_positive']:>9}{counts['localization']:>9}"
              f"{counts['classification']:>9}{errs:>8}")
    print("-" * 70)
    print(f"{'TOTAL':<18}{totals['correct']:>8}{totals['missed']:>8}"
          f"{totals['false_positive']:>9}{totals['localization']:>9}"
          f"{totals['classification']:>9}"
          f"{sum(v for k, v in totals.items() if k != 'correct'):>8}")

    print("\nWrong-label swaps (predicted <- actual), good box in both cases:")
    for (pred, true), n in swaps.most_common(10):
        print(f"   {n:>3}x  {pred:<18} <- {true}")

    out_dir = args.run / "eval" / "errors"
    out_dir.mkdir(parents=True, exist_ok=True)
    top3 = [name for name, _ in ranked[:3]]
    print(f"\nExample crops for the top 3 problem classes: {', '.join(top3)}")
    written = 0
    for name in top3:
        picked = sorted(examples[name], key=lambda e: e["kind"])[:CROPS_PER_CLASS]
        for i, ex in enumerate(picked, 1):
            boxes = []
            if ex["gt"]:
                boxes.append((ex["gt"][1], GT_COLOUR))
            if ex["pred"]:
                boxes.append((ex["pred"][1], PRED_COLOUR))
            if not boxes:
                continue
            bits = [ex["kind"], ex["image"]]
            if ex["gt"]:
                bits.append(f"GT={ex['gt'][0]}")
            if ex["pred"]:
                bits.append(f"PRED={ex['pred'][0]} {ex['pred'][2]:.2f}")
            if ex["iou"]:
                bits.append(f"IoU={ex['iou']:.2f}")
            piece = crop(Image.open(ex["path"]), boxes, "  ".join(bits))
            piece.save(out_dir / f"{name}_{i:02d}_{ex['kind']}.png")
            written += 1
    print(f"Wrote {written} crops to {out_dir}   (green = ground truth, red = prediction)")

    summary = {
        "run": str(args.run), "conf": args.conf,
        "iou_hit": IOU_HIT, "iou_near": IOU_NEAR,
        "totals": dict(totals),
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "swaps": [{"predicted": p, "actual": t, "n": n} for (p, t), n in swaps.most_common()],
    }
    dest = args.run / "eval" / "error_summary.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
