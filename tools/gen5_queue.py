#!/usr/bin/env python3
"""One gen5 form per fresh context. Claim, mark, and the weekly stop."""

import json
import sys
import fcntl
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
ARENA = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d")
QUEUE = ROOT / "monsterSprite/gen5/_forge_queue.json"
LOCK = ROOT / "monsterSprite/gen5/_forge_queue.lock"
HOLD = ROOT / "monsterSprite/gen5/_forge_hold.json"
USAGE = Path.home() / ".grok/statusline-usage.json"
WEEKLY_STOP = 95
# 2026-11-04 user: stop starting forms around 21:35 JST. In-flight forms finish.
CLOCK_STOP = datetime(2026, 11, 4, 21, 35, tzinfo=ZoneInfo("Asia/Tokyo"))
KIND_ORDER = {"mega": 0, "gigantamax": 1, "regional": 2, "normal": 3}


def _flock(fn):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _load():
    return json.loads(QUEUE.read_text())


def _save(data):
    tmp = QUEUE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(QUEUE)


def usage_snapshot():
    if not USAGE.exists():
        return None
    raw = json.loads(USAGE.read_text())
    snap = raw.get("snapshot", raw)
    cfg = snap.get("config", snap)
    period = cfg.get("currentPeriod") or {}
    return {
        "percent": cfg.get("creditUsagePercent"),
        "end": period.get("end") or cfg.get("billingPeriodEnd"),
        "fetched_at": raw.get("fetched_at"),
        "products": cfg.get("productUsage") or [],
    }


def gate(write_hold=True):
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    if now >= CLOCK_STOP:
        print(f"CLOCK_HOLD local {now.strftime('%H:%M')} stop 21:35", file=sys.stderr)
        return 4
    snap = usage_snapshot()
    if not snap or snap["percent"] is None:
        print("usage unknown; not starting a form", file=sys.stderr)
        return 4
    pct = float(snap["percent"])
    print(f"weekly {pct:.0f}% stop {WEEKLY_STOP} ends {snap['end']}", file=sys.stderr)
    if pct >= WEEKLY_STOP:
        if write_hold:
            HOLD.write_text(json.dumps({
                "reason": "weekly credit usage reached the stop line",
                "percent": pct,
                "stop": WEEKLY_STOP,
                "period_end": snap["end"],
                "at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, indent=2) + "\n")
        print("WEEKLY_HOLD", file=sys.stderr)
        return 4
    return 0


def _png_size(path):
    import struct
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return None
        length = struct.unpack(">I", f.read(4))[0]
        if f.read(4) != b"IHDR":
            return None
        w, h = struct.unpack(">II", f.read(8))
        return int(w), int(h)


def _items_from_disk():
    root = ARENA / "monsterSprite/gen5"
    items = []
    for front in sorted(root.rglob("front.png")):
        rel = front.relative_to(root)
        parts = rel.parts[:-1]
        if any(p in ("shiny", "box-icons", "legends-arceus") for p in parts):
            continue
        species = parts[0]
        if "mega" in parts:
            kind = "mega"
        elif "gigantamax" in parts:
            kind = "gigantamax"
        elif "regional" in parts:
            kind = "regional"
        else:
            kind = "normal"
        variant = parts[-1]
        if variant == "normal":
            variant = species.split("_", 1)[1]
            item_id = species
        else:
            item_id = species.split("_", 1)[0] + "_" + variant
        revenge = ROOT / "monsterSprite/gen5" / Path(*parts)
        arena = ARENA / "monsterSprite/gen5" / Path(*parts)
        if not (revenge / "front.png").is_file():
            raise SystemExit(f"missing revenge still: {revenge}")
        size = _png_size(front)
        dex = int(species.split("_", 1)[0])
        items.append({
            "id": item_id,
            "dex": dex,
            "species": species,
            "variant": variant,
            "kind": kind,
            "status": "pending",
            "still": f"{size[0]}x{size[1]}" if size else "unknown",
            "revenge": str(revenge),
            "arena2d": str(arena),
            "generator": None,
            "notes": "Measure this form's own front.png. Do not copy another species' masks.",
        })
    items.sort(key=lambda it: (it["dex"], KIND_ORDER.get(it["kind"], 9), it["id"]))
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise SystemExit(f"duplicate ids: {dup}")
    return items


def cmd_init():
    def edit():
        if QUEUE.exists():
            data = _load()
            if any(it["status"] != "pending" for it in data["items"]):
                raise SystemExit("queue already has progress; not rebuilding")
        items = _items_from_disk()
        data = {
            "version": 1,
            "created": "2026-11-04",
            "policy": (
                "One gen5 form per fresh grok process. "
                "Up to 8 processes run at once. "
                "Each process exits when its one form finishes."
            ),
            "resume": (
                "Claim the first pending item. "
                "Do not take an in_progress item another process owns. "
                "Skip needs_fix and done."
            ),
            "exclude": "shiny, box-icons, legends-arceus",
            "weekly_stop_percent": WEEKLY_STOP,
            "contract": "arena_revenge_claude/md/gen5-forge.md",
            "items": items,
        }
        _save(data)
        print(f"queued {len(items)}")

    _flock(edit)


def _next_pending(items):
    for it in items:
        if it["status"] == "pending":
            return it
    return None


def cmd_claim():
    if gate() != 0:
        return 4

    def edit():
        data = _load()
        it = _next_pending(data["items"])
        if it is None:
            print("QUEUE_EMPTY")
            return 3
        it["status"] = "in_progress"
        it["attempts"] = int(it.get("attempts") or 0) + 1
        it["claimed_at"] = datetime.now(timezone.utc).isoformat()
        _save(data)
        sys.stdout.write(json.dumps(it, ensure_ascii=False) + "\n")
        return 0

    return _flock(edit)


def cmd_mark(item_id, status, note):
    if status not in ("pending", "in_progress", "done", "needs_fix"):
        raise SystemExit(f"bad status {status}")

    def edit():
        data = _load()
        hits = [it for it in data["items"] if it["id"] == item_id]
        if len(hits) != 1:
            raise SystemExit(f"{item_id} count {len(hits)}")
        it = hits[0]
        it["status"] = status
        it["marked_at"] = datetime.now(timezone.utc).isoformat()
        if note:
            it["notes"] = note
        _save(data)
        print(item_id, status)

    _flock(edit)
    return 0


def cmd_counts():
    data = _load()
    counts = {}
    for it in data["items"]:
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    print(json.dumps({"total": len(data["items"]), **counts}, ensure_ascii=False))
    return 0


def cmd_release(item_id):
    """A worker exited without marking. Retry once, then needs_fix."""

    def edit():
        data = _load()
        hits = [it for it in data["items"] if it["id"] == item_id]
        if len(hits) != 1:
            raise SystemExit(f"{item_id} count {len(hits)}")
        it = hits[0]
        if it["status"] in ("done", "needs_fix"):
            print(it["status"])
            return
        attempts = int(it.get("attempts") or 1)
        it["marked_at"] = datetime.now(timezone.utc).isoformat()
        if attempts >= 2:
            it["status"] = "needs_fix"
            it["notes"] = "process exited before done"
            print("needs_fix")
        else:
            it["status"] = "pending"
            it["attempts"] = attempts
            it["notes"] = "retry after early exit"
            print("pending")
        _save(data)

    _flock(edit)
    return 0


def cmd_status_one(item_id):
    data = _load()
    for it in data["items"]:
        if it["id"] == item_id:
            print(it["status"])
            return 0
    print("missing")
    return 1


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: gen5_queue.py init|gate|claim|mark ID STATUS [note]|release ID|counts|status-one ID")
        return 2
    cmd = argv[0]
    if cmd == "init":
        cmd_init()
        return 0
    if cmd == "gate":
        return gate()
    if cmd == "claim":
        return cmd_claim()
    if cmd == "mark":
        note = " ".join(argv[3:]) if len(argv) > 3 else ""
        return cmd_mark(argv[1], argv[2], note)
    if cmd == "release":
        return cmd_release(argv[1])
    if cmd == "counts":
        return cmd_counts()
    if cmd == "status-one":
        return cmd_status_one(argv[1])
    raise SystemExit(f"unknown {cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
