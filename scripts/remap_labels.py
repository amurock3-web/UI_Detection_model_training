"""Remap the original 22-class labels onto the locked 16-class taxonomy.

Mapping is BY NAME, never by index: classes.txt order is Label Studio's, not
ours. An unknown name is a hard error — a silent skip would drop annotations.
Reads data/renamed/, writes data/labels_16/. Originals are never touched.
"""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "renamed"
OUT = ROOT / "data" / "labels_16"
CLASSES_OUT = ROOT / "config" / "classes_16.txt"
CORRECTIONS = ROOT / "config" / "label_corrections.yaml"

# The contract with the consuming repo — see CLAUDE.md. Do not reorder.
CLASSES_16 = [
    "content_card", "tab_item", "overlay_badge", "section_title",
    "button", "cast_card", "icon", "metadata_text",
    "navigation_bar", "celebrity_card", "title", "description",
    "search_bar", "rank_number", "text_input", "logo",
]

# 22 original names -> 16-class name. cast_card and celebrity_card stay split.
MAPPING = {
    "content_card": "content_card",
    "nav_item": "tab_item",
    "category_tab": "tab_item",
    "detail_tab": "tab_item",
    "carousel_filter": "tab_item",
    "badge": "overlay_badge",
    "episode_badge": "overlay_badge",
    "duration": "overlay_badge",
    "section_title": "section_title",
    "button": "button",
    "cast_card": "cast_card",
    "icon": "icon",
    "metadata_text": "metadata_text",
    "navigation_bar": "navigation_bar",
    "celebrity_card": "celebrity_card",
    "movie_title": "title",
    "series_title": "title",
    "description": "description",
    "search_bar": "search_bar",
    "rank_number": "rank_number",
    "text_input": "text_input",
    "logo": "logo",
}

NAME_TO_ID_16 = {name: i for i, name in enumerate(CLASSES_16)}


def build_id_map(classes_22):
    """Old class id -> new class id. Raises on any name we don't know."""
    unknown = [c for c in classes_22 if c not in MAPPING]
    if unknown:
        raise KeyError(
            f"classes.txt names missing from MAPPING: {unknown}. "
            "Add them explicitly — guessing would mislabel client data.")
    return {old_id: NAME_TO_ID_16[MAPPING[name]]
            for old_id, name in enumerate(classes_22)}


def load_corrections():
    """stem -> {line number: corrected 16-class id}. Absent file means none."""
    if not CORRECTIONS.exists():
        return {}
    import re
    import yaml
    text = CORRECTIONS.read_text()

    # YAML lets a repeated key win silently, which would drop every correction
    # in the earlier block for that image without a word.
    keys = re.findall(r"^([A-Za-z0-9_]+):", text, re.M)
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"{CORRECTIONS.name}: {dupes} appear more than once; "
                         "YAML keeps only the last, so merge them into one entry")

    raw = yaml.safe_load(text) or {}
    out = {}
    for stem, fixes in raw.items():
        out[stem] = {}
        for lineno, name in (fixes or {}).items():
            if name not in NAME_TO_ID_16:
                raise KeyError(f"{CORRECTIONS.name}: {stem}:{lineno} names "
                               f"'{name}', which is not one of the 16 classes")
            out[stem][int(lineno)] = NAME_TO_ID_16[name]
    return out


def main():
    classes_22 = [c.strip() for c in
                  (SRC / "classes_22.txt").read_text().splitlines() if c.strip()]
    id_map = build_id_map(classes_22)
    corrections = load_corrections()

    OUT.mkdir(parents=True, exist_ok=True)
    before, after, mapped = Counter(), Counter(), Counter()
    files = sorted((SRC / "labels").glob("*.txt"))
    applied, unused = [], dict(corrections)

    for path in files:
        fixes = corrections.get(path.stem, {})
        unused.pop(path.stem, None)
        seen_lines = set()
        lines = []
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            seen_lines.add(lineno)
            cid, rest = line.split(maxsplit=1)
            old = int(cid)
            if old not in id_map:
                raise ValueError(f"{path.name}:{lineno} class id {old} "
                                 f"outside classes_22.txt (0-{len(classes_22)-1})")
            new = id_map[old]
            mapped[CLASSES_16[new]] += 1        # pre-correction, for the merge check
            if lineno in fixes and fixes[lineno] != new:
                applied.append((path.stem, lineno, CLASSES_16[new],
                                CLASSES_16[fixes[lineno]]))
                new = fixes[lineno]
            before[classes_22[old]] += 1
            after[CLASSES_16[new]] += 1
            lines.append(f"{new} {rest}")

        # a correction aimed at a line that does not exist is a silent no-op,
        # which is exactly how a stale corrections file rots unnoticed
        missing = sorted(set(fixes) - seen_lines)
        if missing:
            raise ValueError(f"{CORRECTIONS.name}: {path.stem} has no line(s) "
                             f"{missing} - the label file has {len(seen_lines)}")
        (OUT / path.name).write_text("\n".join(lines) + "\n" if lines else "")

    if unused:
        raise ValueError(f"{CORRECTIONS.name}: no label file for "
                         f"{sorted(unused)} - stale correction entry")

    CLASSES_OUT.write_text("\n".join(CLASSES_16) + "\n")

    print("=" * 62)
    print("REMAP 22 -> 16")
    print("=" * 62)
    print(f"label files : {len(files)}")
    print(f"instances   : {sum(before.values())} -> {sum(after.values())}")
    print(f"\n{'new id':>6}  {'new class':<16}{'count':>6}   = sum of originals")
    print("-" * 62)
    for new_id, new_name in enumerate(CLASSES_16):
        sources = [c for c in classes_22 if MAPPING[c] == new_name]
        arithmetic = " + ".join(f"{c} {before[c]}" for c in sources)
        delta = after[new_name] - mapped[new_name]
        note = f"   then {delta:+d} from corrections" if delta else ""
        print(f"{new_id:>6}  {new_name:<16}{mapped[new_name]:>6}   = {arithmetic}{note}")
        assert mapped[new_name] == sum(before[c] for c in sources), new_name

    assert sum(before.values()) == sum(after.values()), "instances lost in remap"

    if applied:
        print(f"\nCorrections applied from {CORRECTIONS.name}: {len(applied)}")
        moved = Counter((was, now) for _, _, was, now in applied)
        for (was, now), n in moved.most_common():
            print(f"   {n:>3}x  {was:<16} -> {now}")
        print("   (the merge table above shows counts BEFORE these corrections)")
    print(f"\nWrote {len(files)} files to {OUT}")
    print(f"Wrote {CLASSES_OUT}")


if __name__ == "__main__":
    sys.exit(main())
