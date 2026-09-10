"""Evaluate a finished training run: per-class table, confusion pairs, failures.

    python scripts/evaluate.py runs/detect/ui16_20260910_142740
    python scripts/evaluate.py runs/detect/ui16_20260910_142740 --kfold 5

Writes everything to <run>/eval/. The per-class table marks any class under 20
instances as directional - with 2 val instances a single hit moves the score by
tens of points, so those rows are not evidence about the model.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import shutil
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw_labels import COLORS, font  # same palette as the Phase 3 previews

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
DATA_YAML = CONFIG / "ui.yaml"
CLASSES_16 = (CONFIG / "classes_16.txt").read_text().split()
LOW_N = 20          # below this, a per-class metric is noise, not signal
CONF = 0.25


def val_metrics(model, data_yaml, imgsz, conf=0.001):
    """Run ultralytics validation and return its metrics object.

    conf=0.001 is correct for mAP - the metric integrates over the whole
    precision/recall curve and needs the low-confidence tail. It is wrong for
    the confusion matrix, which then counts thousands of junk detections no
    one would ever act on, so that is computed again at the operating point.
    """
    return model.val(data=str(data_yaml), split="val", imgsz=imgsz,
                     conf=conf, plots=True, verbose=False)


def instance_counts(data_yaml):
    """Val-split instance count per class name."""
    import yaml
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    val = Path(cfg["path"]) / cfg["val"]
    label_dir = Path(str(val).replace("images", "labels"))
    counts = Counter()
    if label_dir.is_dir():
        files = label_dir.glob("*.txt")
    else:                                    # val is a .txt list of image paths
        files = (Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
                 for p in [Path(l) for l in val.read_text().split()])
    for f in files:
        if Path(f).exists():
            counts.update(CLASSES_16[int(l.split()[0])]
                          for l in Path(f).read_text().splitlines() if l.strip())
    return counts


def per_class_rows(metrics, counts):
    """One row per class, sorted by instance count descending."""
    names = getattr(metrics, "names", None) or {}
    box = metrics.box
    maps = list(box.maps)
    index = list(box.ap_class_index)
    seen = {}
    for i, cls in enumerate(index):
        cls = int(cls)
        seen[names.get(cls, str(cls))] = (
            float(box.p[i]), float(box.r[i]), float(box.ap50[i]), float(maps[cls]))

    rows = []
    for name in CLASSES_16:
        p, r, ap50, ap = seen.get(name, (0.0, 0.0, 0.0, 0.0))
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({"class": name, "instances": counts.get(name, 0),
                     "precision": p, "recall": r, "f1": f1,
                     "map50": ap50, "map50_95": ap,
                     "low_n": counts.get(name, 0) < LOW_N})
    rows.sort(key=lambda row: -row["instances"])
    return rows


def print_table(rows):
    print(f"\n{'class':<18}{'inst':>6}{'P':>7}{'R':>7}{'F1':>7}"
          f"{'mAP50':>9}{'mAP50-95':>10}   note")
    print("-" * 82)
    for row in rows:
        note = "LOW-N, directional only" if row["low_n"] else ""
        print(f"{row['class']:<18}{row['instances']:>6}{row['precision']:>7.2f}"
              f"{row['recall']:>7.2f}{row['f1']:>7.2f}{row['map50']:>9.4f}"
              f"{row['map50_95']:>10.4f}   {note}")
    print("-" * 82)
    solid = [r for r in rows if not r["low_n"]]
    if solid:
        print(f"{'mean (>=%d inst)' % LOW_N:<18}{sum(r['instances'] for r in solid):>6}"
              f"{'':>7}{'':>7}{'':>7}"
              f"{np.mean([r['map50'] for r in solid]):>9.4f}"
              f"{np.mean([r['map50_95'] for r in solid]):>10.4f}")


def confusion_pairs(metrics, top=10):
    """Most-confused (predicted, true) pairs from ultralytics' matrix.

    matrix[i][j] counts detections of class i whose ground truth was class j.
    Index nc is the background row/column: [nc][j] is a miss, [i][nc] a
    spurious box.
    """
    cm = getattr(metrics, "confusion_matrix", None)
    if cm is None:
        return [], [], []
    matrix = np.asarray(cm.matrix)
    nc = len(CLASSES_16)
    swaps, missed, spurious = [], [], []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            n = float(matrix[i][j])
            if n <= 0 or i == j:
                continue
            if i < nc and j < nc:
                swaps.append((n, CLASSES_16[i], CLASSES_16[j]))
            elif i >= nc and j < nc:
                missed.append((n, CLASSES_16[j]))
            elif i < nc and j >= nc:
                spurious.append((n, CLASSES_16[i]))
    swaps.sort(reverse=True)
    missed.sort(reverse=True)
    spurious.sort(reverse=True)
    return swaps[:top], missed[:top], spurious[:top]


def per_image_loss(model, data_yaml, imgsz):
    """True training loss for each val image, one image per batch."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    data = check_det_dataset(str(data_yaml))
    args = get_cfg(DEFAULT_CFG)
    args.imgsz = imgsz
    ds = YOLODataset(img_path=data["val"], imgsz=imgsz, batch_size=1,
                     augment=False, hyp=args, rect=False, data=data, task="detect")

    net = model.model.float()
    net.args = args                    # the criterion reads box/cls/dfl gains here
    net.criterion = net.init_criterion()
    net.train()                        # loss is only computed in train mode

    out = []
    with torch.no_grad():
        for i in range(len(ds)):
            batch = YOLODataset.collate_fn([ds[i]])
            batch["img"] = batch["img"].float() / 255
            loss, items = net(batch)
            parts = {k: round(float(v), 3) for k, v in items.items()} \
                if isinstance(items, dict) else {}
            out.append({"image": Path(ds.im_files[i]).name,
                        "path": ds.im_files[i],
                        "loss": round(float(loss.sum()), 4), **parts})
    net.eval()
    out.sort(key=lambda d: -d["loss"])
    return out


def draw_boxes(image, boxes, title, scale=0.5):
    """boxes: list of (cls_id, x1, y1, x2, y2) in pixels of the ORIGINAL image."""
    image = image.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    width = max(2, round(image.width / 400))
    tag = font(max(14, round(image.width / 55)))

    for cls, x1, y1, x2, y2 in sorted(boxes, key=lambda b: -((b[3]-b[1])*(b[4]-b[2]))):
        colour = COLORS[cls % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=width)
        text = CLASSES_16[cls]
        tw, th = draw.textbbox((0, 0), text, font=tag)[2:]
        ty = y1 - th - 4 if y1 - th - 4 > 0 else y1 + 2
        draw.rectangle((x1, ty, x1 + tw + 6, ty + th + 4), fill=colour)
        draw.text((x1 + 3, ty + 2), text, fill="black", font=tag)

    banner = font(max(20, round(image.width / 28)))
    bh = draw.textbbox((0, 0), title, font=banner)[3] + 12
    strip = Image.new("RGB", (image.width, bh), "black")
    ImageDraw.Draw(strip).text((8, 4), title, fill="white", font=banner)
    stacked = Image.new("RGB", (image.width, image.height + bh))
    stacked.paste(strip, (0, 0))
    stacked.paste(image, (0, bh))
    if scale != 1:
        stacked = stacked.resize((round(stacked.width * scale),
                                  round(stacked.height * scale)), Image.LANCZOS)
    return stacked


def failure_images(model, worst, out_dir, imgsz, top=10):
    """Ground truth beside prediction for the worst images, so failures are visible."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rank, item in enumerate(worst[:top], 1):
        path = Path(item["path"])
        image = Image.open(path).convert("RGB")
        W, H = image.size

        label = Path(str(path).replace("images", "labels")).with_suffix(".txt")
        gt = []
        if label.exists():
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                cls, xc, yc, w, h = line.split()
                xc, yc, w, h = (float(v) for v in (xc, yc, w, h))
                gt.append((int(cls), (xc - w/2)*W, (yc - h/2)*H,
                           (xc + w/2)*W, (yc + h/2)*H))

        result = model.predict(str(path), imgsz=imgsz, conf=CONF, verbose=False)[0]
        pred = [(int(c), *map(float, xyxy)) for c, xyxy in
                zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist())]

        left = draw_boxes(image, gt, f"{rank}. {path.name}  GROUND TRUTH  ({len(gt)} boxes)")
        right = draw_boxes(image, pred,
                           f"loss {item['loss']:.2f}  PREDICTED  ({len(pred)} boxes)")
        pair = Image.new("RGB", (left.width + right.width + 12,
                                 max(left.height, right.height)), "white")
        pair.paste(left, (0, 0))
        pair.paste(right, (left.width + 12, 0))
        dest = out_dir / f"{rank:02d}_{path.stem}_gt_vs_pred.png"
        pair.save(dest)
        written.append({"rank": rank, "image": path.name, "loss": item["loss"],
                        "gt_boxes": len(gt), "pred_boxes": len(pred),
                        "file": dest.name})
    return written


def run_kfold(model, n, imgsz):
    """mAP per fold, mean and spread.

    A fold is only honest if it is scored by a model that never saw its val
    images. One model trained on the Phase 4 split has seen most of every
    fold, so those rows are marked LEAKY and excluded from the mean.
    """
    rows = []
    for i in range(n):
        fold_yaml = CONFIG / "folds" / f"ui_fold{i}.yaml"
        if not fold_yaml.exists():
            sys.exit(f"{fold_yaml} missing - run: python scripts/split_dataset.py --kfold {n}")
        candidates = sorted((ROOT / "runs" / "detect").glob(f"*fold{i}*"))
        fold_model, leaky = model, True
        if candidates:
            from ultralytics import YOLO
            weights = candidates[-1] / "weights" / "best.pt"
            if weights.exists():
                fold_model, leaky = YOLO(str(weights)), False
        metrics = val_metrics(fold_model, fold_yaml, imgsz)
        rows.append({"fold": i, "map50": round(float(metrics.box.map50), 4),
                     "map50_95": round(float(metrics.box.map), 4), "leaky": leaky})

    print(f"\n{'fold':>5}{'mAP50':>10}{'mAP50-95':>11}   status")
    print("-" * 46)
    for row in rows:
        print(f"{row['fold']:>5}{row['map50']:>10.4f}{row['map50_95']:>11.4f}"
              f"   {'LEAKY - model trained on these' if row['leaky'] else 'clean'}")

    clean = [r for r in rows if not r["leaky"]]
    if clean:
        m50 = [r["map50"] for r in clean]
        m = [r["map50_95"] for r in clean]
        print(f"\nmean over {len(clean)} clean folds: "
              f"mAP50 {np.mean(m50):.4f} +/- {np.std(m50):.4f}   "
              f"mAP50-95 {np.mean(m):.4f} +/- {np.std(m):.4f}")
    else:
        bang = "!" * 70
        print(f"\n{bang}\nEVERY FOLD IS LEAKY. One model trained on the Phase 4 split has\n"
              f"already seen most images in every fold's val set, so these numbers\n"
              f"are optimistic and no mean over them is trustworthy.\n\n"
              f"Real {len(rows)}-fold CV needs one model trained per fold:\n"
              f"  for i in 0 1 2 3 4; do\n"
              f"    python scripts/train.py --data config/folds/ui_fold$i.yaml "
              f"--name ui16_fold$i\n  done\n{bang}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="a run dir under runs/detect/")
    ap.add_argument("--data", type=Path, default=DATA_YAML)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--kfold", type=int, help="also score the Phase 4 folds")
    ap.add_argument("--top", type=int, default=10, help="failure images to draw")
    args = ap.parse_args()

    weights = args.run / "weights" / "best.pt"
    if not weights.exists():
        sys.exit(f"{weights} not found")

    imgsz = args.imgsz
    meta_path = args.run / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if imgsz is None:
        imgsz = meta.get("imgsz", 1280)

    from ultralytics import YOLO
    model = YOLO(str(weights))
    out_dir = args.run / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run    : {args.run}")
    print(f"weights: {weights.name}   imgsz {imgsz}   conf {CONF} (failure images)")

    metrics = val_metrics(model, args.data, imgsz)
    counts = instance_counts(args.data)
    rows = per_class_rows(metrics, counts)
    print_table(rows)

    # Second pass at the operating point, so the matrix reflects detections a
    # consumer would actually receive rather than the low-confidence tail.
    print(f"\nrebuilding confusion matrix at conf={CONF} (the operating point) ...")
    op_metrics = val_metrics(model, args.data, imgsz, conf=CONF)
    swaps, missed, spurious = confusion_pairs(op_metrics)
    print(f"\nMost-confused class pairs at conf={CONF} (predicted <- actually):")
    if swaps:
        for n, pred, true in swaps:
            print(f"   {int(n):>4}x  {pred:<18} <- {true}")
    else:
        print("   none - no class is being mistaken for another")
    print("\nMissed entirely (ground truth with no detection):")
    for n, name in missed:
        print(f"   {int(n):>4}x  {name}")
    print("\nSpurious (detection with no ground truth):")
    for n, name in spurious:
        print(f"   {int(n):>4}x  {name}")

    save_dir = Path(getattr(op_metrics, "save_dir", "") or "")
    for plot in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = save_dir / plot
        if src.exists():
            shutil.copy2(src, out_dir / plot)

    print("\ncomputing per-image loss ...")
    worst = per_image_loss(model, args.data, imgsz)
    print(f"\n{'rank':>4}  {'image':<12}{'loss':>9}   components")
    for i, item in enumerate(worst[:args.top], 1):
        parts = "  ".join(f"{k.replace('_loss',''):>3} {v:.2f}"
                          for k, v in item.items() if k.endswith("_loss"))
        print(f"{i:>4}  {item['image']:<12}{item['loss']:>9.3f}   {parts}")

    written = failure_images(model, worst, out_dir / "worst", imgsz, args.top)
    print(f"\nWrote {len(written)} side-by-side failure images to {out_dir / 'worst'}")

    folds = run_kfold(model, args.kfold, imgsz) if args.kfold else None

    summary = {
        "run": str(args.run),
        "imgsz": imgsz,
        "conf_for_matrix": CONF,
        "overall": {"map50": round(float(metrics.box.map50), 4),
                    "map50_95": round(float(metrics.box.map), 4)},
        "per_class": rows,
        "confused_pairs": [{"n": int(n), "predicted": p, "actual": t} for n, p, t in swaps],
        "missed": [{"n": int(n), "class": c} for n, c in missed],
        "spurious": [{"n": int(n), "class": c} for n, c in spurious],
        "worst_images": written,
        "per_image_loss": worst,
        "kfold": folds,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
