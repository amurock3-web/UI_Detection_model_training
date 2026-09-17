"""verify_bundle.py must reject a bundle whose class order was tampered with.

Needs a built bundle (best.pt is not in git), so it skips when there is none.
"""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "models" / "ui16_v1"
VERIFY = ROOT / "scripts" / "verify_bundle.py"

pytestmark = pytest.mark.skipif(not (BUNDLE / "best.pt").exists(),
                                reason="no built bundle - run package_model.py")


def verify(bundle):
    return subprocess.run([sys.executable, str(VERIFY), str(bundle)],
                          capture_output=True, text=True)


def test_real_bundle_passes():
    r = verify(BUNDLE)
    assert r.returncode == 0, r.stdout + r.stderr


def test_swapped_classes_fail(tmp_path):
    bad = tmp_path / "ui16_v1"                 # same name, so only the swap differs
    shutil.copytree(BUNDLE, bad)
    lines = (bad / "classes_16.txt").read_text().split()
    lines[5], lines[9] = lines[9], lines[5]    # cast_card <-> celebrity_card
    (bad / "classes_16.txt").write_text("\n".join(lines) + "\n")
    r = verify(bad)
    assert r.returncode != 0
    assert "taxonomy mismatch" in r.stdout
