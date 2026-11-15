#!/usr/bin/env python3
"""Rebuild poor gen2 action sheets by splitting each sprite into pixel rows.

agent-sprite-forge cannot invent in-between pixels that keep a 1px outline.
gait_loop.py still scores the idle loop (its window is recorded, not used to
drop frames). Walk, attack and hurt then move those original GIF rows by whole
pixels, one pixel per frame, so a large mega reads as motion instead of a
2px slide of the whole body.

No resample, no palette change, no finish_frames.py, no colour-lock.
Front/back GIFs and PNGs are not touched.
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

FORGE_SCRIPTS = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge/skills/video2dsprite/scripts"
)
sys.path.insert(0, str(FORGE_SCRIPTS))
import gait_loop  # noqa: E402

GAME = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
GEN2 = GAME / "monsterSprite" / "gen2"

# Short loops and every mega. Longer gen2 sheets that already step smoothly stay.
POOR = [
    "0154_meganium-mega",
    "0160_feraligatr-mega",
    "0169_crobat",
    "0181_ampharos-mega",
    "0208_steelix",
    "0208_steelix-mega",
    "0212_scizor-mega",
    "0214_heracross-mega",
    "0224_octillery",
    "0227_skarmory-mega",
    "0229_houndoom-mega",
    "0231_phanpy",
    "0248_tyranitar-mega",
]

PAD = 14


def load_gif(path: Path):
    im = Image.open(path)
    frames, durs = [], []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(np.array(im.convert("RGBA")))
        durs.append(int(im.info.get("duration") or 100))
    return frames, durs


def drop_exact(frames, durs):
    kept, kd = [frames[0]], [durs[0]]
    prev = frames[0]
    for fr, dur in zip(frames[1:], durs[1:]):
        if fr.shape == prev.shape and np.array_equal(fr, prev):
            kd[-1] += dur
        else:
            kept.append(fr)
            kd.append(dur)
            prev = fr
    return kept, kd


def form_dirs():
    found = []
    for species in sorted(p for p in GEN2.iterdir() if p.is_dir()):
        for gif in species.rglob("front.gif"):
            if "shiny" in gif.name:
                continue
            folder = gif.parent
            sp = next(p for p in gif.parents if p.parent == GEN2)
            dex = sp.name.split("_", 1)[0]
            sid = sp.name if folder.name == "normal" else f"{dex}_{folder.name}"
            found.append((sid, folder))
    return found


def blank_canvas(h, w):
    return np.zeros((h + PAD * 2, w + PAD * 2, 4), np.uint8)


def place(arr, dxs=None, bob=0):
    """Copy each source row to an integer x, then bob the whole sprite by whole pixels."""
    h, w = arr.shape[:2]
    canvas = blank_canvas(h, w)
    for y in range(h):
        dx = 0 if dxs is None else int(dxs[y])
        if abs(dx) > PAD - 1:
            dx = max(-(PAD - 1), min(PAD - 1, dx))
        canvas[PAD + y, PAD + dx : PAD + dx + w] = arr[y]
    if bob:
        bob = max(-(PAD - 1), min(PAD - 1, int(bob)))
        moved = np.zeros_like(canvas)
        if bob > 0:
            moved[bob:] = canvas[:-bob]
        else:
            moved[:bob] = canvas[-bob:]
        canvas = moved
    return canvas


def squash_rows(arr, dropped, protect=()):
    """Drop evenly spaced rows and pack the rest onto the feet. Rows stay exact.

    `protect` rows are never removed. Other species pass nothing, so their
    dropped rows stay the same.
    """
    h, w = arr.shape[:2]
    dropped = int(max(0, min(dropped, h // 3)))
    if dropped == 0:
        return arr.copy()
    remove = set()
    protect = {int(y) for y in protect}
    for i in range(dropped):
        y = int(round((i + 0.5) * (h - 1) / dropped))
        y = min(h - 2, max(1, y))
        if y in remove or y in protect:
            hit = None
            for d in range(1, h):
                for cand in (y - d, y + d):
                    if 1 <= cand <= h - 2 and cand not in remove and cand not in protect:
                        hit = cand
                        break
                if hit is not None:
                    y = hit
                    break
            if hit is None:
                continue
        remove.add(y)
    keep = [y for y in range(h) if y not in remove]
    out = np.zeros_like(arr)
    y0 = h - len(keep)
    for dy, sy in enumerate(keep):
        out[y0 + dy] = arr[sy]
    return out


def ramp_dx(h, head, foot, foot_start=0.58, foot_end=0.86):
    """Head shifts by `head` px, feet by `foot` px. Neighbour rows differ by at most 1."""
    dxs = []
    prev = 0
    span = max(1, h - 1)
    for y in range(h):
        t = y / span
        body = head * (1 - t)
        if t <= foot_start:
            ramp = 0.0
        elif t >= foot_end:
            ramp = 1.0
        else:
            ramp = (t - foot_start) / (foot_end - foot_start)
        target = int(round(body + foot * ramp))
        if target > prev + 1:
            target = prev + 1
        elif target < prev - 1:
            target = prev - 1
        dxs.append(target)
        prev = target
    return dxs


def forge_note(frames, durs, work: Path):
    """Score the idle with gait_loop. The sheet keeps every unique frame either way."""
    raw = work / "raw"
    raw.mkdir(parents=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr, "RGBA").save(raw / f"frame_{i:04d}.png")
    median = int(np.median(durs)) or 100
    try:
        analysis = gait_loop.analyse(
            raw,
            gait_loop.parse_fps(f"{1000}/{median}"),
            kind="idle",
            state="idle",
            dedupe_cap=0,
            min_seconds=0.4,
            max_seconds=min(4.0, (sum(durs) / 1000) or 0.4),
        )
        rec = gait_loop.recommend(analysis, "auto")
        w = rec["window"]
        return {
            "policy": rec.get("policy"),
            "start": int(w["start"]),
            "endExclusive": int(w["endExclusive"]),
            "kept": len(gait_loop.kept_indices(analysis, w["start"], w["endExclusive"])),
        }
    except (ValueError, OSError, KeyError) as exc:
        return {"error": type(exc).__name__}


def _head_left(arr):
    opaque = arr[:, :, 3] > 16
    xs = [int(np.where(opaque[y])[0].min()) for y in range(5, 15) if opaque[y].any()]
    return int(np.median(xs)) if xs else 0


def _head_top(arr):
    ys = np.where((arr[:, :, 3] > 16).any(1))[0]
    return int(ys.min()) if len(ys) else 0


def _light(arr):
    red = arr[:, :, 0].astype(np.int16)
    green = arr[:, :, 1].astype(np.int16)
    return (arr[:, :, 3] > 16) & (np.abs(red - 173) < 30) & (np.abs(green - 214) < 40)


def restore_phanpy_trunk(frames):
    """Put Phanpy's own frame-0 trunk back on frames where that nose collapsed.

    The trunk is copied at an integer shift that follows the forehead, so a
    lowered head does not swallow it. Swings that already reach left of x=4
    stay as they are. No resample and no new colours.
    """
    donor = frames[0]
    height, width = donor.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    box = (yy >= 18) & (yy < 26) & (xx >= 7) & (xx < 17) & (donor[:, :, 3] > 16)
    hl0, ht0 = _head_left(donor), _head_top(donor)
    ys, xs = np.where(box)
    box_n = int(box.sum())
    restored = []
    for fr in frames:
        mask = _light(fr)
        left = int(np.where(mask)[1].min()) if mask.any() else 99
        if left <= 4:
            restored.append(fr)
            continue
        dx = int(np.clip(_head_left(fr) - hl0, -8, 6))
        dy = int(np.clip(_head_top(fr) - ht0, -3, 3))
        matched = 0
        for y, x in zip(ys, xs):
            ny, nx = int(y + dy), int(x + dx)
            if 0 <= ny < height and 0 <= nx < width and np.array_equal(fr[ny, nx], donor[y, x]):
                matched += 1
        if matched >= box_n - 8 and left >= 7:
            restored.append(fr)
            continue
        out = fr.copy()
        for y, x in zip(ys, xs):
            ny, nx = int(y + dy), int(x + dx)
            if 0 <= ny < height and 0 <= nx < width:
                out[ny, nx] = donor[y, x]
        restored.append(out)
    return restored


def _trunk_rows(arr):
    """Rows whose left side is the trunk, so attack squash cannot delete it."""
    rows = []
    for y in range(arr.shape[0]):
        xs = np.where(arr[y, :, 3] > 16)[0]
        if len(xs) and int(xs.min()) <= 12 and 16 <= y <= 26:
            rows.append(y)
    return rows


def build_rows(frames, protect_trunk=False):
    h, w = frames[0].shape[:2]
    n = len(frames)
    foot_amp = max(3, int(round(h / 16)))
    body_amp = max(2, int(round(h / 30)))
    drop_max = max(5, h // 7)
    lean_max = min(4, max(2, h // 24))

    idle = [place(fr) for fr in frames]

    walk_n = max(n, foot_amp * 8, 24)
    walk_n = min(48, walk_n) if n < 36 else n
    walk = []
    for i in range(walk_n):
        src = frames[i % n]
        phase = 2 * math.pi * (i / walk_n)
        sway = math.sin(phase)
        step = math.sin(2 * phase)
        dxs = ramp_dx(h, head=sway * body_amp, foot=step * foot_amp)
        bob = int(round(math.sin(2 * phase)))
        walk.append(place(src, dxs, bob))

    # One authored pose, shrunk a row at a time, with a 1px head lean.
    diffs = [0]
    for i in range(1, n):
        diffs.append(int(np.abs(frames[i].astype(np.int16) - frames[0].astype(np.int16)).sum()))
    pose = frames[int(np.argmax(diffs))]
    attack = []
    total = drop_max * 2
    for i in range(total + 1):
        if i <= drop_max:
            dropped = i
            # anticipate back, then lunge through the deepest squash
            if i < drop_max * 0.55:
                lean = -min(lean_max, i)
            else:
                span = max(1, int(drop_max * 0.45))
                along = (i - drop_max * 0.55) / span
                lean = int(round(-lean_max + along * (2 * lean_max)))
                lean = max(-lean_max, min(lean_max, lean))
        else:
            dropped = total - i
            back = i - drop_max
            lean = max(0, lean_max - back)
        guard = _trunk_rows(pose) if protect_trunk else ()
        img = squash_rows(pose, dropped, guard)
        dxs = ramp_dx(img.shape[0], head=lean, foot=0)
        attack.append(place(img, dxs))

    # 1px shake. Sweat is the only pixel that is not copied from the GIF.
    hurt_src = frames[min(n - 1, n // 3)]
    path = [0]
    for target in (3, 0, -3, 0, 2, 0, -2, 0):
        while path[-1] != target:
            path.append(path[-1] + (1 if target > path[-1] else -1))
    hurt = []
    for i, dx in enumerate(path):
        canvas = place(hurt_src, [dx] * h)
        rising = i + 1 < len(path) and abs(path[i + 1]) > abs(dx)
        if rising and abs(dx) >= 1:
            opaque = canvas[:, :, 3] > 16
            ys, xs = np.where(opaque)
            if len(ys):
                y = max(1, int(ys.min()) - 4)
                x = int(np.median(xs))
                for sx in (x - max(6, w // 10), x + max(4, w // 12)):
                    if 1 <= sx < canvas.shape[1] - 2 and 1 <= y < canvas.shape[0] - 2:
                        canvas[y : y + 2, sx : sx + 2] = (255, 255, 255, 255)
        hurt.append(canvas)
    return {"idle": idle, "walk": walk, "attack": attack, "hurt": hurt}


def palette_of(frames):
    colors = set()
    for fr in frames:
        px = fr[fr[:, :, 3] > 0][:, :3]
        if len(px) == 0:
            continue
        # Quantise to the bytes that are actually in the GIF.
        uniq = np.unique(px, axis=0)
        for rgb in uniq:
            colors.add((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return colors


def large_pieces(fr):
    from scipy import ndimage
    mask = fr[:, :, 3] > 16
    labeled, n = ndimage.label(mask)
    return sum(1 for k in range(1, n + 1) if (labeled == k).sum() >= 12)


def check_row(frames, source_colors, label, piece_limit):
    issues = []
    comps = []
    for i, fr in enumerate(frames):
        alpha = fr[:, :, 3] > 16
        if not alpha.any():
            issues.append(f"{label}[{i}] empty")
            continue
        ys, xs = np.where(alpha)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == fr.shape[0] - 1 or xs.max() == fr.shape[1] - 1:
            issues.append(f"{label}[{i}] touches the cell edge")
        rgb = fr[alpha][:, :3]
        uniq = np.unique(rgb, axis=0)
        for c in uniq:
            key = (int(c[0]), int(c[1]), int(c[2]))
            if key not in source_colors and key != (255, 255, 255):
                issues.append(f"{label}[{i}] new color {key}")
                break
        comps.append(large_pieces(fr))
    # Mega Steelix and a few others are already several separate pieces in the GIF.
    if comps and max(comps) > piece_limit:
        issues.append(f"{label} split into {max(comps)} large pieces")
    return issues


def sheet_from(rows):
    order = ("idle", "walk", "attack", "hurt")
    ch = max(rows[k][0].shape[0] for k in order)
    cw = max(rows[k][0].shape[1] for k in order)
    width = max(len(rows[k]) for k in order)
    sheet = np.zeros((ch * 4, cw * width, 4), np.uint8)
    for ri, key in enumerate(order):
        for i, fr in enumerate(rows[key]):
            sheet[ri * ch : ri * ch + fr.shape[0], i * cw : i * cw + fr.shape[1]] = fr
    return Image.fromarray(sheet, "RGBA"), cw, ch


def write_review(rows, path: Path):
    order = ("idle", "walk", "attack", "hurt")
    shown = []
    for key in order:
        seq = rows[key]
        if len(seq) <= 16:
            shown.append(seq)
        else:
            idx = [round(i * (len(seq) - 1) / 15) for i in range(16)]
            shown.append([seq[i] for i in idx])
    ch = max(f.shape[0] for row in shown for f in row)
    cw = max(f.shape[1] for row in shown for f in row)
    scale = 2
    canvas = Image.new("RGB", (cw * 16 * scale, ch * 4 * scale), (16, 20, 32))
    for ri, row in enumerate(shown):
        for k, fr in enumerate(row):
            im = Image.fromarray(fr, "RGBA").resize(
                (fr.shape[1] * scale, fr.shape[0] * scale), Image.Resampling.NEAREST
            )
            canvas.paste(im, (k * cw * scale, ri * ch * scale), im)
    canvas.save(path)


def build_one(sid: str, folder: Path):
    frames, durs = load_gif(folder / "front.gif")
    if sid == "0231_phanpy":
        frames = restore_phanpy_trunk(frames)
    frames, durs = drop_exact(frames, durs)
    colors = palette_of(frames)
    with tempfile.TemporaryDirectory(prefix=f"forge-{sid}-") as tmp:
        note = forge_note(frames, durs, Path(tmp))
    rows = build_rows(frames, protect_trunk=(sid == "0231_phanpy"))
    piece_limit = max(large_pieces(fr) for fr in frames)
    issues = []
    for key, seq in rows.items():
        issues.extend(check_row(seq, colors, key, piece_limit))
    if issues:
        raise SystemExit(sid + " QC failed: " + "; ".join(issues[:8]))
    sheet, cw, ch = sheet_from(rows)
    out = folder / "actions"
    out.mkdir(exist_ok=True)
    sheet.save(out / "sheet.png")
    write_review(rows, out / "review.png")
    preview = sheet.resize((sheet.width * 2, sheet.height * 2), Image.Resampling.NEAREST)
    if preview.width <= 8192:
        bg = Image.new("RGB", preview.size, (16, 20, 32))
        bg.paste(preview, mask=preview.split()[-1])
        bg.save(out / "preview.png")
    else:
        shutil.copyfile(out / "review.png", out / "preview.png")
    counts = [len(rows[k]) for k in ("idle", "walk", "attack", "hurt")]
    total_ms = sum(durs) or 1000
    delay = max(0.04, round(total_ms / max(1, counts[0]) / 1000, 3))
    delays = [delay] * counts[0]
    body_h = int(frames[0].shape[0])
    entry = {
        "layout": "actions",
        "cw": cw,
        "ch": ch,
        "h": body_h,
        "n": [counts[0]] * 8,
        "d": [delays for _ in range(8)],
        "an": counts,
        "raw": 1,
        "arow": 0,
        "src": "gif-fine",
    }
    meta = dict(entry)
    meta["_forge"] = note
    meta["_gifFrames"] = len(frames)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n")
    print(
        f"{sid} idle {counts[0]} walk {counts[1]} attack {counts[2]} hurt {counts[3]} "
        f"{cw}x{ch} forge {note}",
        flush=True,
    )
    return entry


def main():
    only = sys.argv[1:] or POOR
    forms = [item for item in form_dirs() if item[0] in only or item[1].name in only]
    missing = [sid for sid in only if sid not in {f[0] for f in forms} and sid not in {f[1].name for f in forms}]
    if missing:
        raise SystemExit("unknown forms: " + ", ".join(missing))
    if not forms:
        raise SystemExit("no forms")
    for sid, folder in forms:
        build_one(sid, folder)


if __name__ == "__main__":
    main()
