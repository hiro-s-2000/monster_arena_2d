#!/usr/bin/env python3
"""Rebuild gen2 action sheets from each front.gif, keeping the GIF's own frames.

agent-sprite-forge has no in-between that preserves a 1px outline. gait_loop.py
(select --kind idle --dedupe-cap 0) chooses a loop from those frames. This script
keeps that loop when it is almost the whole clip, and otherwise keeps every
non-duplicate frame up to 48. Walk, attack and hurt play those same poses with a
longer integer-pixel path than the old 4-hop / 8-hop / 2-shake sheets.

No resample, no palette change, no finish_frames.py.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

FORGE_SCRIPTS = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge/skills/video2dsprite/scripts")
sys.path.insert(0, str(FORGE_SCRIPTS))
import gait_loop  # noqa: E402

GAME = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
GEN2 = GAME / "monsterSprite" / "gen2"
DIR8 = GAME / "data" / "dir8.js"
CAP = 48
PAD = 6  # room for a 4px lunge and a 2px shake


def load_gif(path: Path):
    im = Image.open(path)
    frames, durs = [], []
    i = 0
    while True:
        frames.append(im.convert("RGBA"))
        durs.append(int(im.info.get("duration") or 100))
        i += 1
        try:
            im.seek(i)
        except EOFError:
            break
    return frames, durs


def drop_exact(frames, durs):
    kept, kd = [frames[0]], [durs[0]]
    prev = np.asarray(frames[0])
    for fr, dur in zip(frames[1:], durs[1:]):
        cur = np.asarray(fr)
        if cur.shape == prev.shape and np.array_equal(cur, prev):
            kd[-1] += dur
        else:
            kept.append(fr)
            kd.append(dur)
            prev = cur
    return kept, kd


def dense(frames, durs, cap=CAP):
    n = len(frames)
    if n <= cap:
        return list(range(n)), frames, durs
    total = sum(durs) or n
    acc = 0
    edges = [0]
    for dur in durs:
        acc += dur
        edges.append(acc)
    idxs = []
    for i in range(cap):
        target = (i + 0.5) * total / cap
        j = 0
        while j < n - 1 and edges[j + 1] < target:
            j += 1
        if not idxs or idxs[-1] != j:
            idxs.append(j)
    return idxs, [frames[j] for j in idxs], [durs[j] for j in idxs]


def forge_idle_span(frames, durs, work: Path):
    """Return index list from gait_loop, or None when the clip cannot be looped."""
    if len(frames) < 8:
        return None
    raw = work / "raw"
    raw.mkdir()
    for i, fr in enumerate(frames):
        fr.save(raw / f"frame_{i:04d}.png")
    median = int(np.median(durs)) or 100
    fps = Fraction_fps(median)
    try:
        analysis = gait_loop.analyse(
            raw, fps, kind="idle", state="idle", dedupe_cap=0,
            min_seconds=0.4, max_seconds=min(4.0, (sum(durs) / 1000) or 0.4),
        )
        rec = gait_loop.recommend(analysis, "auto")
    except ValueError:
        return None
    w = rec["window"]
    indices = gait_loop.kept_indices(analysis, w["start"], w["endExclusive"])
    if rec["policy"] == "pingpong" and len(indices) >= 3 and len(indices) < 12:
        indices = indices + indices[-2:0:-1]
    return indices


def Fraction_fps(median_ms: int):
    # 100ms -> 10 fps. Keep it a real fraction gait_loop can parse.
    return gait_loop.parse_fps(f"{1000}/{median_ms}")


def offsets(kind: str, i: int, n: int):
    if kind == "idle" or n < 2:
        return 0, 0
    if kind == "walk":
        table = (0, -1, -1, -2, -1, 0, 1, 2)
        return (1 if (i // 4) % 2 else -1), table[i % 8]
    if kind == "attack":
        t = i / (n - 1)
        if t < 0.22:
            u = t / 0.22
            return round(-2 * u), round(1 * u)
        u = (t - 0.22) / 0.78
        return round(math.sin(math.pi * u) * 4), round(-math.sin(math.pi * u) * 3)
    shake = (-2, 2, -2, 1, -1, 2, -1, 0)
    return shake[i % 8], 0


def opaque_box(arr):
    ys, xs = np.where(arr[:, :, 3] > 16)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def paint(frames, kind: str):
    n = len(frames)
    w, h = frames[0].size
    cw, ch = w + PAD * 2, h + PAD * 2
    cells = []
    for i, fr in enumerate(frames):
        dx, dy = offsets(kind, i, n)
        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        canvas.paste(fr, (PAD + dx, PAD + dy), fr)
        if kind == "hurt" and i % 2 == 0:
            arr = np.array(canvas)
            box = opaque_box(arr)
            if box:
                x0, y0, x1, _y1 = box
                cx = (x0 + x1) // 2
                y = max(0, y0 - 4)
                for sx in (cx - 3, cx + 2):
                    if 0 <= sx < cw - 1 and 0 <= y < ch - 1:
                        arr[y:y + 2, sx:sx + 2] = (255, 255, 255, 255)
                canvas = Image.fromarray(arr, "RGBA")
        cells.append(canvas)
    return cells, cw, ch


def sheet_from(frames):
    rows = []
    cw = ch = None
    for kind in ("idle", "walk", "attack", "hurt"):
        cells, cw, ch = paint(frames, kind)
        rows.append(cells)
    n = len(frames)
    sheet = Image.new("RGBA", (cw * n, ch * 4), (0, 0, 0, 0))
    for ri, cells in enumerate(rows):
        for k, cell in enumerate(cells):
            sheet.paste(cell, (k * cw, ri * ch))
    return sheet, cw, ch


def preview(sheet: Image.Image) -> Image.Image:
    big = sheet.resize((sheet.width * 2, sheet.height * 2), Image.Resampling.NEAREST)
    bg = Image.new("RGB", big.size, (18, 22, 36))
    bg.paste(big, mask=big.split()[-1])
    return bg


def form_dirs():
    found = []
    for species in sorted(p for p in GEN2.iterdir() if p.is_dir()):
        for gif in species.rglob("front.gif"):
            if "shiny" in gif.name:
                continue
            folder = gif.parent
            dex = species.name.split("_", 1)[0]
            sid = species.name if folder.name == "normal" else f"{dex}_{folder.name}"
            found.append((sid, folder))
    return found


def build_one(sid: str, folder: Path, work_root: Path):
    frames, durs = drop_exact(*load_gif(folder / "front.gif"))
    _idx, frames, durs = dense(frames, durs, CAP)
    chosen = "dense"
    work = work_root / sid
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        span = forge_idle_span(frames, durs, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if span and len(span) >= max(8, int(len(frames) * 0.75)):
        frames = [frames[i] for i in span]
        durs = [durs[i] for i in span]
        chosen = "gait_loop"
    if len(frames) > CAP:
        _idx, frames, durs = dense(frames, durs, CAP)
    sheet, cw, ch = sheet_from(frames)
    out = folder / "actions"
    out.mkdir(exist_ok=True)
    sheet.save(out / "sheet.png")
    preview(sheet).save(out / "preview.png", quality=90)
    body_h = frames[0].size[1]
    delays = [round(d / 1000, 3) for d in durs]
    n = len(frames)
    return {
        "layout": "actions",
        "cw": cw,
        "ch": ch,
        "h": body_h,
        "n": [n] * 8,
        "d": [delays for _ in range(8)],
        "an": [n, n, n, n],
        "raw": 1,
        "arow": 0,
        "src": "gif-fine",
        "_via": chosen,
        "_frames": n,
    }


def main():
    only = sys.argv[1:] 
    forms = form_dirs()
    if only:
        forms = [item for item in forms if item[0] in only or item[1].name in only]
    if not forms:
        raise SystemExit("no forms")
    text = DIR8.read_text()
    start, end = text.index("{"), text.rindex("}")
    data = json.loads(text[start:end + 1])
    report = []
    with tempfile.TemporaryDirectory(prefix="gen2-fine-") as tmp:
        root = Path(tmp)
        for i, (sid, folder) in enumerate(forms, 1):
            entry = build_one(sid, folder, root)
            via, n = entry.pop("_via"), entry.pop("_frames")
            if sid not in data or data[sid].get("layout") != "actions":
                raise SystemExit(f"{sid} is not an actions entry")
            data[sid] = entry
            report.append(f"{sid} {via} {n} {entry['cw']}x{entry['ch']}")
            print(f"[{i}/{len(forms)}] {report[-1]}", flush=True)
    body = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    DIR8.write_text(
        "// gen2 action sheets: tools/forge_gen2_fine_actions.py (gait_loop idle, no resample).\n"
        f"window.ARENA_DIR8 = {body};\n"
    )
    print("WROTE", DIR8, "forms", len(report))


if __name__ == "__main__":
    main()
