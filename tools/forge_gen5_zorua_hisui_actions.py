#!/usr/bin/env python3
"""Hisuian Zorua (0570 regional) 8-direction actions from this form's own stills.

Measured on regional/zorua-hisui/front.png (96x96, bbox x27-69 y8-70, 1341 px,
alpha 0/255, 15 colours, foot y70):
  Eyes are the only gold (255,174,26) and amber (223,120,6). The far eye is
  two pixels at x31 y47-48. The near eye is x39-42 y47-50 (mean x 40.6), with
  a white highlight. Combined eye mean x is 39.0. The face at that band, before
  the side tail, reaches x53, which is right of the eyes, so the still faces
  screen-right.
  The forked red crest on the head is x43-57 y8-28 (163 px) and touches the
  white cap at three pixels. A small red tuft sits on the left of the cap
  (x27-31 y31-38). There is no mouth pixel, so the roar does not draw one.
  The tail is the right-hand flame and white puff, x54-69 y33-55. Four toes
  split at the bottom: x37-41, x43-47, x50-53, x55-59. The lowest pixel is
  y70 on the second toe.

regional/zorua-hisui/back.png (bbox x25-69 y8-70, 1325 px, foot y70, 13 colours,
no gold and no amber): crest x36-50 y8-28, right tuft x61-69 y33-41, tail is
the left flame x25-38. Four toes: x33-37, x39-43, x47-50, x52-56.

Directions, front faces screen-right:
  S SE E front as drawn. W SW front mirrored. NE back as drawn.
  N NW back mirrored. E is screen-right, W is screen-left, N is the back.
  Attack 5-8 is the opposite still at equal scale. No width squash and no
  in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_zorua_hisui_actions.py
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen5_attack_strike import finish, save_peaks

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen5/0570_zorua/regional/zorua-hisui"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0570_zorua/regional/zorua-hisui/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0570_zorua-hisui"
VERSION = "zorua-hisui_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
SRC_FOOT = 70
FOOT_Y = OFF_Y + SRC_FOOT
# Front as drawn looks screen-right. Back as drawn is the NE rear.
VIEW_OF = {
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "W": ("front", True), "SW": ("front", True),
    "NE": ("back", False), "N": ("back", True), "NW": ("back", True),
}
MAX_LEAN = 38.0
# Strike peaks this script draws. finish() will not go past them.
PEAKS = {"tail": 26.0, "paw": 30.0}
LIMBS = ("crest", "tuft", "tail", "hind_far", "hind_near", "fore_far", "fore_near")

INK = (16, 16, 16)
CHAR = (74, 77, 83)
DEEP = (95, 0, 2)
DRED = (148, 36, 41)
RED = (213, 58, 62)
PINK = (240, 115, 118)
PALE = (247, 172, 174)
MAROON = (167, 72, 79)
WHITE = (250, 250, 250)
SILVER = (203, 207, 216)
GREY = (179, 179, 187)
TAUPE = (146, 141, 150)
PLUM = (109, 77, 98)
GOLD = (255, 174, 26)
AMBER = (223, 120, 6)
RED_COLS = (RED, DRED, PINK, DEEP, PALE, MAROON)
EYE_COLS = (GOLD, AMBER)

SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def is_color(img, color):
    return (img[..., 3] >= 128) & np.all(img[..., :3] == np.array(color, np.uint8), axis=-1)


def any_color(img, cols):
    m = np.zeros(img.shape[:2], bool)
    for c in cols:
        m |= is_color(img, c)
    return m


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1, constant_values=False)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def flood(allowed, seeds):
    out = seeds & allowed
    while True:
        nxt = dilate(out, 1) & allowed
        if np.array_equal(nxt, out):
            return out
        out = nxt


def shift(layer, dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest in-plane rotation. Positive moves the top toward screen-left. No squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    sc = 4
    up = np.repeat(np.repeat(layer, sc, axis=0), sc, axis=1)
    h, w = up.shape[:2]
    px, py = pivot[0] * sc + 1.5, pivot[1] * sc + 1.5
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    r = math.radians(ang)
    c, s = math.cos(r), math.sin(r)
    sx = np.rint(px + c * rx + s * ry).astype(np.int32)
    sy = np.rint(py - s * rx + c * ry).astype(np.int32)
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.zeros_like(up)
    out[ok] = up[sy[ok], sx[ok]]
    return out[sc // 2::sc, sc // 2::sc].copy()


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1, constant_values=False)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1, constant_values=False)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not hole.any():
            break
        up = np.zeros_like(out)
        up[1:] = out[:-1]
        out[hole] = up[hole]
    return out


def snap(img, palette):
    out = img.copy()
    op = out[..., 3] >= 128
    out[~op] = 0
    if not op.any():
        return out
    px = out[op][:, :3].astype(np.int32)
    pal = palette.astype(np.int32)
    d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    out[op, :3] = palette[d.argmin(axis=1)]
    out[op, 3] = 255
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def put(img, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and img[y, x, 3] < 128:
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps):
    """Grow this layer's own colours into allowed pixels. Hidden at rest."""
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        grown = []
        up = np.zeros_like(src)
        dn = np.zeros_like(src)
        lt = np.zeros_like(src)
        rt = np.zeros_like(src)
        up[:-1] = src[1:]
        dn[1:] = src[:-1]
        lt[:, :-1] = src[:, 1:]
        rt[:, 1:] = src[:, :-1]
        grown.extend((up, dn, lt, rt))
        for g in grown:
            take = (acc[..., 3] < 128) & allowed & (g[..., 3] >= 128)
            acc[take] = g[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def component_list(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(pts)
    return groups


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {int(acc.sum())} vs {int(op.sum())}")


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def require_piece(name, mask):
    if not mask.any():
        raise SystemExit(f"{name} is empty")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart")


def side_runs(mask):
    """Leftmost and rightmost solid runs on each row that has a gap."""
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(h):
        xs = np.nonzero(mask[y])[0]
        if len(xs) == 0:
            continue
        segs = []
        s = int(xs[0])
        prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x > prev + 1:
                segs.append((s, prev))
                s = x
            prev = x
        segs.append((s, prev))
        if len(segs) < 2:
            continue
        x0, x1 = segs[0]
        left[y, x0:x1 + 1] = True
        x0, x1 = segs[-1]
        right[y, x0:x1 + 1] = True
    return left, right


def red_mask(img, op):
    return any_color(img, RED_COLS) & op


def foot_column(op, yy, xx, x0, x1, y0):
    seed = op & (yy >= max(y0, 67)) & (xx >= x0) & (xx <= x1)
    return flood(op & (yy >= y0) & (xx >= x0) & (xx <= x1), seed)


def absorb_islands(op, parts):
    """Outline specks left beside a part join that part. The rest is the torso."""
    used = np.zeros_like(op)
    for m in parts.values():
        used |= m
    rest = op & ~used
    groups = component_list(rest)
    if not groups:
        raise SystemExit("no torso")
    groups.sort(key=len)
    main = np.zeros_like(op)
    for y, x in groups[-1]:
        main[y, x] = True
    for g in groups[:-1]:
        island = np.zeros_like(op)
        for y, x in g:
            island[y, x] = True
        host = None
        best = 0
        for name, m in parts.items():
            n = int((island & dilate(m, 1)).sum())
            if n > best:
                best = n
                host = name
        if host is None:
            raise SystemExit(f"island of {len(g)} touches nothing")
        parts[host] = parts[host] | island
    parts = dict(parts)
    parts["torso"] = main
    return parts


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    red = red_mask(img, op)
    eye = any_color(img, EYE_COLS)
    if int(eye.sum()) != 12:
        raise SystemExit("front eyes changed")
    ex = float(np.nonzero(eye)[1].mean())
    face = op & (yy >= 46) & (yy <= 52) & (xx < 54)
    tip = int(np.nonzero(face)[1].max())
    if tip <= ex:
        raise SystemExit(f"front does not face right: muzzle {tip} eye {ex:.1f}")
    crest = flood(red, red & (yy <= 16))
    tuft = flood(red & (xx <= 36) & (yy <= 42), red & (xx <= 31) & (yy >= 31) & (yy <= 36))
    # Toes measured on y66: (37-41), (43-47), (50-53), (55-59). y70 is x44-45.
    hind_far = foot_column(op, yy, xx, 37, 41, 64)
    hind_near = foot_column(op, yy, xx, 43, 47, 66)
    fore_far = foot_column(op, yy, xx, 50, 53, 66)
    fore_near = foot_column(op, yy, xx, 55, 59, 65)
    _left_runs, right_runs = side_runs(op)
    seed = right_runs & (xx >= 58) & (yy >= 36) & (yy <= 52)
    tail = flood(op & (xx >= 54) & (yy >= 33) & (yy <= 55), seed)
    parts = {
        "crest": crest, "tuft": tuft, "tail": tail,
        "hind_far": hind_far, "hind_near": hind_near,
        "fore_far": fore_far, "fore_near": fore_near,
    }
    parts = absorb_islands(op, parts)
    check_partition(op, parts)
    if (eye & ~parts["torso"]).any():
        raise SystemExit("front eyes left the head")
    if int(np.nonzero(op)[0].max()) != SRC_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(parts["hind_near"])[0].max()) != SRC_FOOT:
        raise SystemExit("front ground toe left the foot line")
    for name in LIMBS:
        require_piece(name, parts[name])
        if not (parts[name] & dilate(parts["torso"], 1)).any():
            raise SystemExit(f"front {name} does not touch the body")
    if largest_component(parts["torso"]) != int(parts["torso"].sum()):
        raise SystemExit("front torso fell apart")
    if int(parts["crest"].sum()) < 140 or int(np.nonzero(parts["crest"])[0].min()) != 8:
        raise SystemExit("front crest is not the head flame")
    if int(np.nonzero(parts["tail"])[1].min()) < 54:
        raise SystemExit("front tail crossed into the head")
    return parts


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    if any_color(img, EYE_COLS).any():
        raise SystemExit("back gained an eye")
    red = red_mask(img, op)
    crest = flood(red, red & (yy <= 16))
    tuft = flood(red & (xx >= 58) & (yy <= 44), red & (xx >= 64) & (yy >= 33) & (yy <= 40))
    # Toes measured on y66: (34-37), (39-43), (47-50), (52-56). y70 is x41-42.
    hind_far = foot_column(op, yy, xx, 33, 37, 64)
    hind_near = foot_column(op, yy, xx, 39, 43, 66)
    fore_far = foot_column(op, yy, xx, 47, 50, 66)
    fore_near = foot_column(op, yy, xx, 52, 56, 65)
    left_runs, _right_runs = side_runs(op)
    seed = left_runs & (xx <= 38) & (yy >= 34) & (yy <= 50)
    tail = flood(op & (xx <= 38) & (yy >= 33) & (yy <= 54), seed)
    tail &= ~crest & ~tuft
    parts = {
        "crest": crest, "tuft": tuft, "tail": tail,
        "hind_far": hind_far, "hind_near": hind_near,
        "fore_far": fore_far, "fore_near": fore_near,
    }
    parts = absorb_islands(op, parts)
    check_partition(op, parts)
    if int(np.nonzero(op)[0].max()) != SRC_FOOT:
        raise SystemExit("back foot line changed")
    if int(np.nonzero(parts["hind_near"])[0].max()) != SRC_FOOT:
        raise SystemExit("back ground toe left the foot line")
    for name in LIMBS:
        require_piece(name, parts[name])
        if not (parts[name] & dilate(parts["torso"], 1)).any():
            raise SystemExit(f"back {name} does not touch the body")
    if largest_component(parts["torso"]) != int(parts["torso"].sum()):
        raise SystemExit("back torso fell apart")
    if int(np.nonzero(parts["tail"])[1].max()) > 42:
        raise SystemExit("back tail is not the left flame")
    return parts


def place(arr):
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), bool)
    else:
        out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = OFF_Y
    x0 = OFF_X
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(part, torso):
    """Contact pixel of the part nearest the torso."""
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 4)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    near = d <= d.min() + 0.5
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs.astype(np.float64) - pivot[0]) ** 2 + (ys.astype(np.float64) - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing):
    """Which rotation sign lifts the tip, and which sends it toward facing."""
    if int((layer[..., 3] >= 128).sum()) < 4:
        return {"up": 1.0, "fwd": 1.0}

    def tip_after(ang):
        return tip_xy(rotate_nearest(layer, ang, pivot), pivot)

    x_pos, y_pos = tip_after(12.0)
    x_neg, y_neg = tip_after(-12.0)
    up = -1.0 if y_neg < y_pos else 1.0
    if facing > 0:
        fwd = 1.0 if x_pos >= x_neg else -1.0
    else:
        fwd = 1.0 if x_pos <= x_neg else -1.0
    return {"up": up, "fwd": fwd}


def build_view(img, masks, kind, mirror):
    canvas = place(img)
    placed = {k: place(v) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    for name, depth in (
        ("crest", 4), ("tuft", 3), ("tail", 5),
        ("hind_far", 4), ("hind_near", 4), ("fore_far", 3), ("fore_near", 4),
    ):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth
        )
    facing = 1.0 if not mirror else -1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(placed[name], torso_m)
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        crest=0.0, tuft=0.0, tail=0.0, paw=0.0,
        tail_mode="up", paw_mode="fwd",
        step_a=(0, 0), step_b=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The head flame, the side tuft, and the tail sway.
        p["crest"] = wave(f, n, 8.0, phase=0.4)
        p["tuft"] = wave(f, n, 6.0, phase=0.9)
        p["tail"] = wave(f, n, 8.0, phase=0.2)
    elif action == "walk":
        # Four drawn toes. Diagonal pairs. Only the airborne toes leave the ground.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_a"] = (int(round(2 * max(0.0, sa))), -int(round(2 * max(0.0, sa))))
        p["step_b"] = (int(round(2 * max(0.0, sb))), -int(round(2 * max(0.0, sb))))
        p["crest"] = wave(f, n, 6.0, phase=0.5)
        p["tail"] = wave(f, n, 10.0, cycles=2.0, phase=0.3)
    elif action == "attack":
        # 0 rest. 1-4 the crest and tuft rise (no mouth on this still).
        # 5-8 the other side, equal scale. 9-12 the tail and the nose-side
        # paw strike. 11-12 impact. 13-15 recover.
        crest = [0, 8, 14, 18, 18, 0, 0, 0, 0, 6, 4, 4, 2, 0, 0, 0]
        tuft = [0, 6, 10, 14, 14, 0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 0]
        tail = [0, 4, 8, 10, 12, 0, 0, 0, 0, 18, 24, 26, 26, 12, 4, 0]
        paw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 24, 30, 30, 12, 4, 0]
        bdx = [0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["crest"] = float(crest[f])
        p["tuft"] = float(tuft[f])
        p["tail"] = float(tail[f])
        p["paw"] = float(paw[f])
        p["bdx"] = int(bdx[f])
        p["tail_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["paw_mode"] = "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["crest"] = 10.0 * env
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # 5-8 is the opposite still at equal scale. Drop the wind-up so the
    # silhouette is not swung, and do not squash the width.
    if action == "attack" and 5 <= f <= 8:
        p["crest"] = 0.0
        p["tuft"] = 0.0
        p["tail"] = 0.0
        p["paw"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_a"] = (0, 0)
        p["step_b"] = (0, 0)
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
        p["tail_mode"] = "up"
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    if mode == "up":
        return sign["up"] * amp
    return sign["fwd"] * amp


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def limb_motion(rig, p, name):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    if name in ("hind_far", "fore_near"):
        sx, sy = p["step_a"]
    elif name in ("hind_near", "fore_far"):
        sx, sy = p["step_b"]
    else:
        sx, sy = 0, 0
    if name == "fore_near":
        ang = part_angle(rig, name, p["paw_mode"], p["paw"])
    elif name == "tail":
        ang = part_angle(rig, name, p["tail_mode"], p["tail"])
    elif name == "crest":
        ang = part_angle(rig, name, "up", p["crest"])
    elif name == "tuft":
        ang = part_angle(rig, name, "up", p["tuft"])
    else:
        ang = 0.0
    dx = bdx + int(round(fc * sx))
    dy = bdy + int(sy)
    return ang, dx, dy


def compose(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    out = np.zeros_like(rig["canvas"])
    bdx = int(round(rig["facing"] * p["bdx"]))
    bdy = int(round(p["bdy"]))
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("hind_far", "hind_near", "tail", "fore_far", "fore_near", "tuft", "crest"):
        ang, dx, dy = limb_motion(rig, p, name)
        paste(out, moved(L[name], ang, piv[name], dx, dy))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, (INK, CHAR, DEEP))
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def fore_tip(rig, p):
    """Toe of the nose-side paw after the strike pose."""
    ang, dx, dy = limb_motion(rig, p, "fore_near")
    layer = moved(rig["layers"]["fore_near"], ang, rig["pivots"]["fore_near"], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    fc = rig["facing"]
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, PALE, RED)
    r = 2 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r), (1, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name in LIMBS:
        ang, dx, dy = limb_motion(rig, p, name)
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    return rig


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    for key in ("crest", "tuft", "tail", "paw"):
        if abs(p[key]) >= 0.45:
            return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    body = compose(rig, p)
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        # Positive rotation sends the top screen-left. Facing right recoils left.
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, fc * p["lean"], piv)
    if rotated(p):
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = fore_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def assemble(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png").convert("RGBA"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = t
        sheets[act] = sh
    return sheets


def write_outputs(sheets):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for act, sh in sheets.items():
        im = Image.fromarray(sh, "RGBA")
        im.save(OUT_DIR / f"{act}.png")
        im.save(OUT_DIR / f"{act}_8dir.png")
        Image.fromarray(sh[:CELL_H], "RGBA").save(OUT_DIR / f"{act}_S_strip.png")
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    return master


def tile_grid(tiles, cols, bg=(18, 22, 34, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def crop_box(canvas, pad=8):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 18)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 16)
    x0 = max(0, int(xs.min()) - pad - 28)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 28)
    return y0, y1, x0, x1


def preview_image(sheets):
    pick = {"idle": 4, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def motion_strip(sheets, di, box):
    y0, y1, x0, x1 = box
    tiles = []
    for act in ACTIONS:
        for f in range(16):
            if f < FRAMES[act]:
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                tiles.append(t[y0:y1, x0:x1])
            else:
                tiles.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
    return tile_grid(tiles, 16)


def locked_palette(front_path, back_path, work):
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    img = np.array(Image.open(front_path).convert("RGBA"))
    own = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
    if not (lock_dir / "palette.json").exists():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
             "--colors", str(len(hexes)), "--reserve", *hexes,
             "--output-dir", str(pal_dir)],
            capture_output=True, text=True,
        )
        if r.returncode:
            raise SystemExit("palette build failed: " + r.stdout + r.stderr)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "lock", "--palette", str(pal_dir / "palette.json"),
             "--source", str(front_path), "--strict", "--output-dir", str(lock_dir)],
            capture_output=True, text=True,
        )
        if r.returncode:
            raise SystemExit("palette lock failed: " + r.stdout + r.stderr)
    data = json.loads((lock_dir / "palette.json").read_text())
    cols = np.array([[int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)] for c in data["colors"]], np.uint8)
    got = {tuple(int(v) for v in c) for c in cols}
    want = {tuple(int(v) for v in c) for c in own}
    if got != want:
        raise SystemExit(f"locked palette differs from the still: {sorted(got ^ want)}")
    back = np.array(Image.open(back_path).convert("RGBA"))
    extra = []
    for c in np.unique(back[back[..., 3] >= 128][:, :3], axis=0):
        t = tuple(int(v) for v in c)
        if t not in got:
            extra.append(c)
    if extra:
        cols = np.vstack([cols, np.array(extra, np.uint8)])
    return cols


def params_hash():
    src = Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def build_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 stills")
    for im in (front, back):
        im[im[..., 3] < 128] = 0
        im[im[..., 3] >= 128, 3] = 255
    if int(np.nonzero(front[..., 3])[0].max()) != SRC_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != SRC_FOOT:
        raise SystemExit("back foot line changed")
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "crest": (220, 40, 40, 255),
        "tuft": (255, 150, 40, 255),
        "tail": (240, 220, 40, 255),
        "hind_far": (40, 180, 255, 255),
        "hind_near": (40, 90, 220, 255),
        "fore_far": (180, 70, 220, 255),
        "fore_near": (80, 220, 120, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        a = img[..., 3] >= 128
        ys, xs = np.nonzero(a)
        y0, y1 = int(ys.min()) - 1, int(ys.max()) + 2
        x0, x1 = int(xs.min()) - 1, int(xs.max()) + 2
        tiles.append(np.repeat(np.repeat(part[y0:y1, x0:x1], 4, 0), 4, 1))
    h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[0] == h:
            padded.append(t)
            continue
        z = np.zeros((h, t.shape[1], 4), np.uint8)
        z[:t.shape[0]] = t
        padded.append(z)
    gap = np.zeros((h, 8, 4), np.uint8)
    Image.fromarray(np.concatenate([padded[0], gap, padded[1]], axis=1), "RGBA").save(work / "parts.png")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 570, "form": "regional", "variant": "zorua-hisui", "gen": "gen5",
        "dir": "monsterSprite/gen5/0570_zorua/regional/zorua-hisui",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def measure_frames(frames, palette):
    """frames is a list of (tag, rgba). Returns the four gate counts."""
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    edge = []
    for tag, t in frames:
        a = t[..., 3]
        partial += int(((a != 0) & (a != 255)).sum())
        op = a == 255
        if op.any():
            used = {tuple(int(v) for v in c) for c in np.unique(t[op][:, :3], axis=0)}
            off |= used - pal
        if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
            edge.append(tag)
    return len(off), partial, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)
    print("parts", {k: int(v.sum()) for k, v in fm.items()}, {k: int(v.sum()) for k, v in bm.items()}, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
        r3 = synth(rigs, d, "attack", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r3 != rigs[d]["canvas"], axis=2).sum())
    rest_max = max(rest_diff.values())
    print("rest diff", rest_diff, flush=True)
    if rest_max:
        raise SystemExit("frame 0 does not match the still")

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        for d in ("S", "W", "N", "E"):
            rows = []
            for act in ACTIONS:
                frames = []
                for f in range(16):
                    if f < FRAMES[act]:
                        frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
                    else:
                        frames.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
                rows.append(tile_grid(frames, 16))
            sheet = np.concatenate(rows, axis=0)
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        lost = []
        bank = []
        bodies = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
                    img = synth(rigs, d, act, f, palette)
                    bank.append((f"{act}:{d}:{f}", img))
                    if act in ("attack", "hurt", "walk") and f != 0:
                        bodies.append(largest_fraction(img))
        off_n, partial, edge = measure_frames(bank, palette)
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["crest"], 1), round(p["tuft"], 1), round(p["tail"], 1),
                        round(p["paw"], 1), p["tail_mode"], p["turned"], p["spin"], p["bdx"], p["spark"]))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        print(
            "metrics",
            f"rest_diff_max={rest_max}",
            f"off_palette={off_n}",
            f"partial_alpha={partial}",
            f"edge_n={len(edge)}",
            f"edge={edge[:8]}",
            f"body_main_min={round(min(bodies), 3) if bodies else 1}",
            flush=True,
        )
        print("samples", work, flush=True)
        return

    frames_dir = work / "frames"
    state_p = work / "state.json"
    digest = params_hash()
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if args.force or state.get("hash") != digest:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": digest, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(DIRS) * len(ACTIONS)
    for act in ACTIONS:
        for d in DIRS:
            key = f"{act}/{d}"
            if key in state["done"] and not args.force:
                continue
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if fp.exists() and fp.stat().st_size > 0 and not args.force:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion = np.repeat(np.repeat(motion, 2, 0), 2, 1)
    Image.fromarray(motion, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

    pal = {tuple(int(v) for v in c) for c in palette}
    op = master[..., 3] == 255
    used = {tuple(int(v) for v in c) for c in np.unique(master[op][:, :3], axis=0)}
    off = sorted(used - pal)
    partial = int(((master[..., 3] != 0) & (master[..., 3] != 255)).sum())
    clip = []
    bodies = []
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = t[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    import gen5_publish
    bounds = gen5_publish.measure_idle_bounds(OUT_DIR / "sheet.png", CELL_W, CELL_H, rows=8)
    sprite_h = max(bounds[i + 3] for i in range(0, 32, 4))
    idle0 = {}
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        idle0[d] = int(np.any(t != rigs[d]["canvas"], axis=2).sum())
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "off_palette_colors": len(off), "partial_alpha": partial, "edge_touching": clip[:12],
        "edge_n": len(clip), "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (
        not any(rest_diff.values()) and not any(idle0.values()) and not off
        and partial == 0 and not clip
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    print(
        "metrics",
        f"rest_diff_max={max(rest_diff.values()) if rest_diff else 0}",
        f"off_palette={len(off)}",
        f"partial_alpha={partial}",
        f"edge_n={len(clip)}",
        flush=True,
    )
    meta = {
        "generator": "forge_gen5_zorua_hisui_actions", "version": VERSION, "key": KEY,
        "dex": 570, "variant": "zorua-hisui", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "quadruped fox (head flame, side tuft, curled tail, four toes, no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-right (face tip x53 > eye mean x39); back tail is screen-left",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
