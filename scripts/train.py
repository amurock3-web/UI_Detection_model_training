"""Train the 16-class UI detector. Thin wrapper over ultralytics.

All defaults live in config/train.yaml so the augmentation reasoning stays in
one reviewed place; this file only merges that config, applies CLI overrides,
and records what CLAUDE.md requires every run to record.

    python scripts/train.py                                  # full run, config defaults
    python scripts/train.py --smoke                          # cheap plumbing check
    python scripts/train.py --model yolov8n.pt --imgsz 640 --epochs 20 --batch 4
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
TRAIN_YAML = CONFIG / "train.yaml"
DATA_YAML = CONFIG / "ui.yaml"

# Cheap, deliberately bad settings whose only job is to prove the plumbing.
SMOKE = {"model": "yolov8n.pt", "imgsz": 640, "epochs": 20, "batch": 4,
         "patience": 0}


def git_sha():
    """Commit the run came from, marked dirty if the tree has changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def split_seed(data_yaml):
    """Recover the split seed from the header split_dataset.py wrote."""
    match = re.search(r"seed (\d+)", data_yaml.read_text().splitlines()[0])
    return int(match.group(1)) if match else None


def load_config():
    """train.yaml -> flat ultralytics kwargs (it takes augmentation inline)."""
    cfg = yaml.safe_load(TRAIN_YAML.read_text())
    aug = cfg.pop("augmentation", {}) or {}
    overlap = set(aug) & set(cfg)
    if overlap:
        sys.exit(f"{TRAIN_YAML}: {sorted(overlap)} set both at top level and "
                 f"under augmentation - remove one so the run is unambiguous")
    cfg.update(aug)
    return cfg, aug


def resolve_device(want):
    """Fall back to CPU rather than dying, but never do it silently."""
    import torch
    if want in ("cpu", None) or torch.cuda.is_available():
        return want if want is not None else "cpu"
    print(f"\n{'!' * 62}\nWARNING: config asks for device {want!r} but "
          f"torch reports no CUDA\n({torch.__version__}). Falling back to CPU - "
          f"this will be slow. A real\ntraining run belongs on the Colab T4, "
          f"not here.\n{'!' * 62}\n")
    return "cpu"


def per_class_map(metrics):
    """class name -> mAP50-95, mAP50. CLAUDE.md requires this on every run.

    Every value here is a numpy array, so no plain truthiness ('x or []') --
    that raises ValueError on anything longer than one element.
    """
    out = {}
    names = getattr(metrics, "names", None) or {}
    box = metrics.box
    maps = list(getattr(box, "maps", None) if getattr(box, "maps", None) is not None else [])
    index = getattr(box, "ap_class_index", None)
    index = [] if index is None else list(index)
    for i, cls in enumerate(index):
        cls = int(cls)
        out[names.get(cls, str(cls))] = {
            "map50_95": round(float(maps[cls]), 4) if cls < len(maps) else None,
            "map50": round(float(box.ap50[i]), 4),
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="cheap plumbing check: yolov8n, 640px, 20 epochs, batch 4")
    ap.add_argument("--data", type=Path, default=DATA_YAML)
    ap.add_argument("--model")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device")
    ap.add_argument("--name", help="run name under runs/detect/")
    args = ap.parse_args()

    if not args.data.exists():
        sys.exit(f"{args.data} missing - run scripts/split_dataset.py first")

    cfg, aug = load_config()
    if args.smoke:
        cfg.update(SMOKE)
    for key in ("model", "imgsz", "epochs", "batch", "device"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    model_name = cfg.pop("model")
    cfg["device"] = resolve_device(cfg.get("device"))
    cfg["seed"] = cfg.get("seed", 42)
    name = args.name or ("smoke" if args.smoke else
                         f"ui16_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"model  : {model_name}")
    print(f"data   : {args.data}")
    print(f"imgsz  : {cfg['imgsz']}   batch: {cfg['batch']}   "
          f"epochs: {cfg['epochs']}   device: {cfg['device']}")
    print(f"run    : runs/detect/{name}\n")

    from ultralytics import YOLO
    model = YOLO(model_name)
    model.train(data=str(args.data), name=name, **cfg)

    save_dir = Path(model.trainer.save_dir)
    metrics = model.val(data=str(args.data), split="val", device=cfg["device"])

    meta = {
        "run": name,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "model": model_name,
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "device": str(cfg["device"]),
        "seed": cfg["seed"],
        "data_yaml": str(args.data.relative_to(ROOT)),
        "split_seed": split_seed(args.data),
        "augmentation_overrides": aug,
        "smoke": args.smoke,
        "metrics": {
            "map50_95": round(float(metrics.box.map), 4),
            "map50": round(float(metrics.box.map50), 4),
        },
        "per_class": per_class_map(metrics),
    }
    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nmAP50-95 {meta['metrics']['map50_95']}   "
          f"mAP50 {meta['metrics']['map50']}")
    print(f"Wrote {save_dir / 'run_meta.json'}")
    print(f"Artifacts in {save_dir}")


if __name__ == "__main__":
    main()
