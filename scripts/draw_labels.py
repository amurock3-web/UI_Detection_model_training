"""Draw remapped 16-class boxes onto screenshots for eyeball verification.

Written for the Phase 3 check: the remap arithmetic can add up perfectly and
still be wrong if a name was mapped to the wrong bucket. Only looking at
pixels catches that.

    python scripts/draw_labels.py --sample 10
    python scripts/draw_labels.py ss011 ss088
"""
from collections import Counter
from pathlib import Path
import argparse
import random
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "renamed" / "images"
LABELS = ROOT / "data" / "labels_16"
OUT = ROOT / "data" / "preview"

CLASSES_16 = (ROOT / "config" / "classes_16.txt").read_text().split()

# Distinct at a glance matters most for the classes Phase 3 checks:
# content_card / tab_item / overlay_badge / cast_card / celebrity_card.
COLORS = [
    "#FF3B30",  # 0  content_card    red
    "#32D74B",  # 1  tab_item        green
    "#FFD60A",  # 2  overlay_badge   yellow
    "#0A84FF",  # 3  section_title   blue
    "#FF9F0A",  # 4  button          orange
    "#BF5AF2",  # 5  cast_card       purple
    "#64D2FF",  # 6  icon            cyan
    "#FF6482",  # 7  metadata_text   pink
    "#FFFFFF",  # 8  navigation_bar  white
    "#00C7BE",  # 9  celebrity_card  teal
    "#FFB3FF",  # 10 title           light magenta
    "#A2845E",  # 11 description     brown
    "#5E5CE6",  # 12 search_bar      indigo
    "#D0FF00",  # 13 rank_number     chartreuse
    "#8E8E93",  # 14 text_input      grey
    "#FF00FF",  # 15 logo            magenta
]


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:                       # no Arial off Windows
        return ImageFont.load_default(size)


def draw_one(stem, out_dir):
    matches = list(IMAGES.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"no image for {stem} in {IMAGES}")
    image = Image.open(matches[0]).convert("RGB")
    W, H = image.size
    draw = ImageDraw.Draw(image, "RGBA")

    scale = max(W, H) / 1200          # tuned on 1080x2400; keeps text legible elsewhere
    width = max(2, round(2 * scale))
    label_font = font(max(11, round(13 * scale)))

    boxes = []
    for lineno, line in enumerate(
            (LABELS / f"{stem}.txt").read_text().splitlines(), 1):
        if not line.strip():
            continue
        cid, xc, yc, w, h = line.split()
        cid = int(cid)
        if not 0 <= cid < len(CLASSES_16):
            raise ValueError(f"{stem}.txt:{lineno} class id {cid} outside 0..15")
        xc, yc, w, h = (float(v) for v in (xc, yc, w, h))
        boxes.append((w * h, cid,
                      (xc - w / 2) * W, (yc - h / 2) * H,
                      (xc + w / 2) * W, (yc + h / 2) * H))

    # biggest first, so a badge sitting on a poster stays visible on top
    boxes.sort(key=lambda b: -b[0])
    for _, cid, x1, y1, x2, y2 in boxes:
        colour = COLORS[cid]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=width)
        text = f"{cid} {CLASSES_16[cid]}"
        tw, th = draw.textbbox((0, 0), text, font=label_font)[2:]
        pad = max(1, round(2 * scale))
        ty = y1 - th - 2 * pad
        if ty < 0:                     # box hugs the top edge; put the tag inside
            ty = y1 + pad
        draw.rectangle((x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad), fill=colour)
        draw.text((x1 + pad, ty + pad), text, fill="black", font=label_font)

    counts = Counter(CLASSES_16[b[1]] for b in boxes)
    legend(draw, counts, scale)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_preview.png"
    image.save(out_path)
    return out_path, counts


def legend(draw, counts, scale):
    """Per-image class tally, so merge coverage is readable without a terminal."""
    f = font(max(12, round(15 * scale)))
    rows = [f"{CLASSES_16.index(n)} {n} x{c}" for n, c in
            sorted(counts.items(), key=lambda kv: CLASSES_16.index(kv[0]))]
    if not rows:
        return
    pad = round(6 * scale)
    line_h = draw.textbbox((0, 0), "Ag", font=f)[3] + round(3 * scale)
    box_w = max(draw.textbbox((0, 0), r, font=f)[2] for r in rows) + 4 * pad + line_h
    draw.rectangle((0, 0, box_w, pad * 2 + line_h * len(rows)), fill=(0, 0, 0, 205))
    for i, row in enumerate(rows):
        y = pad + i * line_h
        cid = int(row.split()[0])
        draw.rectangle((pad, y + round(2 * scale), pad + line_h - round(6 * scale),
                        y + line_h - round(4 * scale)), fill=COLORS[cid])
        draw.text((pad * 2 + line_h - round(6 * scale), y), row, fill="white", font=f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stems", nargs="*", help="image stems, e.g. ss011 (or paths)")
    ap.add_argument("--sample", type=int, help="draw N random images instead")
    ap.add_argument("--seed", type=int, default=0, help="sample seed (default 0)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.sample:
        pool = sorted(p.stem for p in LABELS.glob("*.txt"))
        if args.sample > len(pool):
            sys.exit(f"--sample {args.sample} exceeds {len(pool)} labelled images")
        stems = sorted(random.Random(args.seed).sample(pool, args.sample))
    elif args.stems:
        stems = [Path(s).stem for s in args.stems]
    else:
        sys.exit("give image stems or --sample N")

    total = Counter()
    for stem in stems:
        out_path, counts = draw_one(stem, args.out)
        total.update(counts)
        summary = "  ".join(f"{n} x{c}" for n, c in
                            sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"{out_path.name:<22}{sum(counts.values()):>4} boxes   {summary}")

    print(f"\n{len(stems)} previews -> {args.out}")
    print(f"{'class':<18}{'total':>6}")
    print("-" * 24)
    for name in CLASSES_16:
        if total[name]:
            print(f"{name:<18}{total[name]:>6}")
    missing = [n for n in CLASSES_16 if not total[n]]
    if missing:
        print(f"\nnot present in this selection: {', '.join(missing)}")


if __name__ == "__main__":
    main()
