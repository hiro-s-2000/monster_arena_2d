"""Gen5 attack strikes. The body does not spin flat.

Same rule as gen4 (2026-11-04): frames 0-4 stay the authored roar.
Frames 5-12 may open limbs, jaws, wings, horns, tails, and pincers
the still already has. A whole-sprite in-plane spin is cleared.
Peaks live in gen5_attack_peaks.json so this file is not rewritten
per species.
"""

import json
import fcntl
from pathlib import Path

ENV = (
    0, 0, 0, 0, 0,
    0.88, 1.0, 1.0, 1.0, 0.96, 0.88,
    0.72, 0.48,
    0.18, 0.06, 0,
)

PEAKS_PATH = Path(__file__).with_name("gen5_attack_peaks.json")
PEAKS_LOCK = Path(__file__).with_name("gen5_attack_peaks.lock")


def load_peaks():
    if not PEAKS_PATH.exists():
        return {}
    return json.loads(PEAKS_PATH.read_text())


def save_peaks(stem, peaks):
    """Merge one script's strike peaks. Other species stay untouched."""

    def edit():
        data = load_peaks()
        data[stem] = {k: float(v) for k, v in peaks.items()}
        tmp = PEAKS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(PEAKS_PATH)

    PEAKS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with PEAKS_LOCK.open("a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            edit()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def finish(p, action, f, script_file):
    """Drop the in-plane spin and extend the species strike."""
    if action != "attack":
        return p
    if "spin" in p:
        p["spin"] = 0
    if "rigid" in p and 5 <= f <= 10:
        try:
            if abs(float(p["rigid"])) > 18:
                p["rigid"] = 0.0
        except (TypeError, ValueError):
            pass
    if "spin_dy" in p and 5 <= f <= 10:
        p["spin_dy"] = 0
    stem = Path(script_file).stem
    table = load_peaks().get(stem)
    if not table or not (0 <= f < len(ENV)):
        return p
    env = ENV[f]
    if env <= 0:
        return p
    for key, peak in table.items():
        if key not in p:
            continue
        cur = p[key]
        if isinstance(cur, (tuple, list, str)):
            continue
        try:
            cur_f = float(cur)
        except (TypeError, ValueError):
            continue
        target = float(peak) * env
        if abs(target) <= abs(cur_f) + 0.05:
            continue
        if isinstance(cur, int) and abs(float(peak)) <= 8:
            p[key] = int(round(target))
        else:
            p[key] = target
    return p
