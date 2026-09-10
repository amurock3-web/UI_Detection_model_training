"""Checks the 22->16 mapping table and the remapped label files on disk."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from remap_labels import CLASSES_16, MAPPING, NAME_TO_ID_16, build_id_map

LABELS_16 = ROOT / "data" / "labels_16"


def test_taxonomy_order_is_the_contract():
    assert CLASSES_16 == (ROOT / "config" / "classes_16.txt").read_text().split()
    assert len(CLASSES_16) == 16


@pytest.mark.parametrize("old_name, new_id", [
    ("content_card", 0),
    ("nav_item", 1),
    ("carousel_filter", 1),
    ("episode_badge", 2),
    ("duration", 2),
    ("movie_title", 10),
    ("series_title", 10),
    ("logo", 15),
])
def test_known_name_mappings(old_name, new_id):
    assert NAME_TO_ID_16[MAPPING[old_name]] == new_id


def test_cast_and_celebrity_stay_separate():
    assert MAPPING["cast_card"] != MAPPING["celebrity_card"]


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        build_id_map(["content_card", "some_new_class"])


def test_every_original_class_is_mapped():
    classes_22 = (ROOT / "data" / "renamed" / "classes_22.txt").read_text().split()
    assert set(classes_22) == set(MAPPING)


def test_output_ids_in_range():
    files = sorted(LABELS_16.glob("*.txt"))
    assert files, "run scripts/remap_labels.py first"
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            cid = int(line.split()[0])
            assert 0 <= cid <= 15, f"{path.name}:{lineno} class id {cid}"
