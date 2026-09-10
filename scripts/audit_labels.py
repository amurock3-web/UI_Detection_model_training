"""Audit the original 22-class YOLO labels. Reads only — changes nothing.

Run this before the 22->16 remap. Coordinates are normalized, so w*h is
already the fraction of image area: no image decoding needed.
"""
from collections import Counter
from pathlib import Path
import itertools
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "renamed"
IMAGES, LABELS = DATA / "images", DATA / "labels"
CLASSES = DATA / "classes_22.txt"

TINY_AREA = 0.005      # 0.5% of image area
DUP_IOU = 0.95
EPS = 1e-6             # boxes sit exactly on the frame edge; xc+-w/2 drifts ~1e-16
IMG_EXT = (".png", ".jpg", ".jpeg")


def iou(a, b):
    """IoU of two (xc, yc, w, h) normalized boxes."""
    boxes = []
    for xc, yc, w, h in (a, b):
        boxes.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = boxes
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


classes = [c for c in CLASSES.read_text().splitlines() if c.strip()]
images = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in IMG_EXT)
labels = sorted(LABELS.glob("*.txt"))

image_stems = {p.stem for p in images}
label_stems = {p.stem for p in labels}

counts = Counter()
empty, bad_id, bad_geom, tiny, dupes, malformed = [], [], [], [], [], []

for path in labels:
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            malformed.append(f"{path.name}:{lineno}  {len(parts)} fields: {line!r}")
            continue
        try:
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            malformed.append(f"{path.name}:{lineno}  unparseable: {line!r}")
            continue

        rows.append((lineno, cid, xc, yc, w, h))
        counts[cid] += 1

        if not 0 <= cid < len(classes):
            bad_id.append(f"{path.name}:{lineno}  class id {cid} (valid 0-{len(classes)-1})")

        if w <= 0 or h <= 0:
            bad_geom.append(f"{path.name}:{lineno}  w={w:.6f} h={h:.6f} (non-positive)")
        # a box may be clipped at the frame edge; flag only what leaves [0,1]
        elif not (-EPS <= xc - w / 2 and xc + w / 2 <= 1 + EPS
                  and -EPS <= yc - h / 2 and yc + h / 2 <= 1 + EPS):
            bad_geom.append(
                f"{path.name}:{lineno}  extends outside [0,1]: "
                f"x {xc-w/2:.4f}..{xc+w/2:.4f}  y {yc-h/2:.4f}..{yc+h/2:.4f}")

        if 0 < w * h < TINY_AREA:
            name = classes[cid] if 0 <= cid < len(classes) else f"?{cid}"
            tiny.append((name, f"{path.name}:{lineno}", w * h))

    if not rows:
        empty.append(path.name)

    for (la, ca, *ba), (lb, cb, *bb) in itertools.combinations(rows, 2):
        if ca == cb and iou(ba, bb) > DUP_IOU:
            name = classes[ca] if 0 <= ca < len(classes) else f"?{ca}"
            dupes.append(f"{path.name}:{la}+{lb}  {name}  IoU {iou(ba, bb):.3f}")


def section(title, items, limit=40):
    print(f"\n{title}: {len(items)}")
    for item in items[:limit]:
        print("  ", item)
    if len(items) > limit:
        print(f"   ... and {len(items) - limit} more")


print("=" * 62)
print("LABEL AUDIT — original 22-class set")
print("=" * 62)
print(f"classes.txt entries : {len(classes)}")
print(f"images              : {len(images)}")
print(f"label files         : {len(labels)}")
print(f"total instances     : {sum(counts.values())}")

section("Images with no label file", sorted(image_stems - label_stems))
section("Label files with no image", sorted(label_stems - image_stems))
section("Images with zero labels", empty)
section("Malformed lines", malformed)
section("Class ids out of range", bad_id)
section("Impossible geometry", bad_geom)
print("")
print(f"Tiny boxes (< 0.5% of frame): {len(tiny)} of {sum(counts.values())}"
      f" ({len(tiny)/max(sum(counts.values()),1)*100:.0f}%)")
tiny_by_class = Counter(name for name, _, _ in tiny)
for name, n in tiny_by_class.most_common():
    smallest = min((a, loc) for cn, loc, a in tiny if cn == name)
    print(f"   {name:<18}{n:>5}   smallest {smallest[0]*100:.3f}% at {smallest[1]}")
section("Duplicate boxes (IoU > 0.95, same class)", dupes)

print("\n" + "=" * 62)
print(f"{'id':>3}  {'class':<18}{'count':>7}")
print("-" * 62)
for cid, n in counts.most_common():
    name = classes[cid] if 0 <= cid < len(classes) else "OUT OF RANGE"
    print(f"{cid:>3}  {name:<18}{n:>7}")
unused = [f"{i}:{c}" for i, c in enumerate(classes) if i not in counts]
if unused:
    print(f"\nclasses with zero instances: {', '.join(unused)}")

blocking = (image_stems ^ label_stems) or malformed or bad_id or bad_geom
print("\n" + "=" * 62)
print("RESULT:", "FIX NEEDED — see sections above" if blocking else "CLEAN")
print("Tiny and duplicate boxes are listed for review, not errors.")
print("=" * 62)
sys.exit(1 if blocking else 0)
