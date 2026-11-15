#!/usr/bin/env python3
"""Galarian Darmanitan standard (0555) 8-direction actions from this form's stills.

Measured on regional/darmanitan-galar-standard/front.png (96x96, bbox x15-72
y6-77, 2423 px, 13 colours, alpha 0/255) and back.png (bbox x20-72 y7-72,
2094 px, 11 colours). Back adds outline black (0,0,0). Both alpha are 0/255.
Front ground is y77. Back ground is y72, so the back is shifted down 5px and
both pegs share one foot line.

Facing: the 16 white pixels in the chin (y34-42) sit at mean x 44.4. The blue
chin's leftmost pixel is x35, and the skull centre (y<=34) is x50.3. The face
tip is left of the eyes, so the still faces screen-left. The heavy ice fist
is the separated mass at x15-32 y52-70. The back keeps its visible arm on the
screen-left; N uses that drawing, and the turn is the N row at equal scale.

No mouth cavity is drawn. The whites in the chin are ice highlights, so the
roar does not invent a wedge or rings. It lifts the ice brows, the ice beard,
and the fists that are already in the still.

Front parts, screen axes as drawn:
  Heavy fist x15-32 y52-70. Small fist x62-72 y61-68. Ice peg x42-59 y69-77.
  Ice brows y17-25. Ice beard x36-46 y55-62. Head and belly stay on the torso.
Back:
  One arm x20-30 y62-67. Centre foot x35-49 y65-72. Right foot x57-72 y61-71.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  Attack 5-8 swaps to that direction's back at equal scale. No width squash,
  no in-plane body spin.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_darmanitan-galar-standard_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0555_darmanitan-standard/regional/darmanitan-galar-standard"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/"
    "0555_darmanitan-standard/regional/darmanitan-galar-standard/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0555_darmanitan-galar-standard"
VERSION = "darmanitan-galar-standard_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 208, 208
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 20
FRONT_FOOT = 77
BACK_FOOT = 72
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. N is the back drawing, unmirrored.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Turn-frame ceiling. The ice punch itself is authored at 34, above these peaks.
PEAKS = {"arm": 12.0, "arm_rear": 8.0}
ROTATE = ("arm_fore", "arm_rear", "brow", "beard")
ORDER = {
    "front": ("foot", "beard", "brow", "arm_rear", "arm_fore"),
    "back": ("foot_l", "foot_r", "arm_fore"),
}

INK = (16, 16, 16)
BLACK = (0, 0, 0)
WHITE = (246, 246, 246)
ICE = (106, 234, 208)
BLUE = (72, 153, 201)
TEAL = (38, 162, 140)
TEAL_COLS = (TEAL, (41, 197, 169), (43, 107, 103), ICE)
BLUE_COLS = (BLUE, (40, 99, 173), (28, 49, 100))

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
    """4x nearest in-plane rotation. Positive angle follows the shared forge formula. No squash."""
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


def component_list(mask, diag=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        nbrs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
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
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(pts)
    groups.sort(key=len, reverse=True)
    return groups


def reconnect(img):
    """Close a 1px gap after a turn. Sparks are drawn later."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(3):
        groups = component_list(out[..., 3] >= 128)
        if len(groups) <= 1:
            return out
        groups.sort(key=len, reverse=True)
        main = np.zeros((h, w), bool)
        for y, x in groups[0]:
            main[y, x] = True
        changed = False
        for g in groups[1:]:
            if len(g) < 4:
                continue
            best = None
            for y, x in g:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and main[ny, nx]:
                            dist = max(abs(dy), abs(dx))
                            if best is None or dist < best[0]:
                                best = (dist, y, x, ny, nx)
            if best is None or best[0] > 2:
                continue
            _, y, x, ny, nx = best
            if abs(ny - y) == 1 and abs(nx - x) == 1:
                candidates = ((y, nx), (ny, x))
            else:
                sy = 0 if ny == y else (1 if ny > y else -1)
                sx = 0 if nx == x else (1 if nx > x else -1)
                candidates = ((y + sy, x + sx),)
            for fy, fx in candidates:
                if 0 <= fy < h and 0 <= fx < w and out[fy, fx, 3] < 128:
                    out[fy, fx, :3] = out[y, x, :3]
                    out[fy, fx, 3] = 255
                    changed = True
                    break
        if not changed:
            return out
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
        grown_up = np.zeros_like(src)
        grown_dn = np.zeros_like(src)
        grown_lt = np.zeros_like(src)
        grown_rt = np.zeros_like(src)
        grown_up[:-1] = src[1:]
        grown_dn[1:] = src[:-1]
        grown_lt[:, :-1] = src[:, 1:]
        grown_rt[:, 1:] = src[:, :-1]
        for grown in (grown_up, grown_dn, grown_lt, grown_rt):
            take = (acc[..., 3] < 128) & allowed & (grown[..., 3] >= 128)
            acc[take] = grown[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


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


def row_runs(op, y):
    xs = np.nonzero(op[y])[0]
    if len(xs) == 0:
        return []
    segs = []
    s = p = int(xs[0])
    for x in map(int, xs[1:]):
        if x > p + 1:
            segs.append((s, p))
            s = x
        p = x
    segs.append((s, p))
    return segs


def feature_mask(img, y0, y1, x0, x1, extras):
    """Ice crystals plus the outline pixels that touch them, inside the box."""
    core = any_color(img, TEAL_COLS)
    region = np.zeros(core.shape, bool)
    region[y0:y1 + 1, x0:x1 + 1] = True
    core = core & region
    acc = core.copy()
    near = dilate(core, 1) & region
    for c in extras:
        acc |= is_color(img, c) & near
    return acc


def solid(mask, torso, name, allow_pieces=1):
    if int(mask.sum()) < 8:
        raise SystemExit(f"{name} is missing from the still")
    if not (mask & dilate(torso, 1)).any():
        raise SystemExit(f"{name} does not touch the body")
    groups = component_list(mask, diag=True)
    if len(groups) > allow_pieces:
        bits = []
        for g in groups[:6]:
            ys = [p[0] for p in g]
            xs = [p[1] for p in g]
            bits.append(f"n{len(g)}@{min(xs)},{min(ys)}")
        raise SystemExit(f"{name} fell apart ({len(groups)} pieces: {', '.join(bits)})")
    for g in groups:
        piece = np.zeros_like(mask)
        for y, x in g:
            piece[y, x] = True
        if not (piece & dilate(torso, 1)).any():
            raise SystemExit(f"{name} has a piece that does not touch the body")


def segment_front(img):
    op = img[..., 3] >= 128
    arm_fore = np.zeros_like(op)
    for y in range(52, 71):
        for a, b in row_runs(op, y):
            if b <= 33:
                arm_fore[y, a:b + 1] = True
    arm_rear = np.zeros_like(op)
    for y in range(61, 69):
        for a, b in row_runs(op, y):
            if a >= 62:
                arm_rear[y, a:b + 1] = True
    foot = np.zeros_like(op)
    for y in range(69, 78):
        for a, b in row_runs(op, y):
            if a >= 40 and b <= 62:
                foot[y, a:b + 1] = True
    brow = feature_mask(img, 16, 26, 28, 58, (INK, WHITE))
    beard = feature_mask(img, 54, 62, 32, 52, (INK,))
    masks = {
        "arm_fore": arm_fore, "arm_rear": arm_rear, "foot": foot,
        "brow": brow, "beard": beard,
    }
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        acc |= m
    masks["torso"] = op & ~acc
    check_partition(op, masks)
    torso = masks["torso"]
    solid(arm_fore, torso, "arm_fore")
    solid(arm_rear, torso, "arm_rear")
    solid(foot, torso, "foot")
    solid(beard, torso, "beard")
    solid(brow, torso, "brow", allow_pieces=3)
    eye = is_color(img, WHITE) & (np.arange(96)[:, None] >= 34) & (np.arange(96)[:, None] <= 42)
    if int(eye.sum()) != 16 or int((eye & torso).sum()) != 16:
        raise SystemExit("front eye highlights left the head")
    chin = any_color(img, BLUE_COLS) & (np.arange(96)[:, None] >= 35) & (np.arange(96)[:, None] <= 42)
    chin &= np.arange(96)[None, :] <= 56
    if int((chin & torso).sum()) != int(chin.sum()) or chin.sum() < 40:
        raise SystemExit("blue chin left the head")
    eye_x = float(np.nonzero(eye)[1].mean())
    tip_x = int(np.nonzero(chin)[1].min())
    if tip_x >= eye_x - 4:
        raise SystemExit("face tip is not left of the eyes")
    if int(np.nonzero(foot)[0].max()) != FRONT_FOOT:
        raise SystemExit("front peg is not the ground line")
    if float(np.nonzero(arm_fore)[1].mean()) >= eye_x:
        raise SystemExit("heavy fist is not left of the eyes")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    arm_fore = np.zeros_like(op)
    for y in range(62, 68):
        for a, b in row_runs(op, y):
            if b <= 32:
                arm_fore[y, a:b + 1] = True
    foot_l = np.zeros_like(op)
    for y in range(65, 73):
        for a, b in row_runs(op, y):
            if 34 <= a and b <= 52:
                foot_l[y, a:b + 1] = True
    foot_r = np.zeros_like(op)
    for y in range(61, 72):
        for a, b in row_runs(op, y):
            if a >= 57:
                foot_r[y, a:b + 1] = True
    masks = {"arm_fore": arm_fore, "foot_l": foot_l, "foot_r": foot_r}
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        acc |= m
    masks["torso"] = op & ~acc
    check_partition(op, masks)
    torso = masks["torso"]
    solid(arm_fore, torso, "back arm")
    solid(foot_l, torso, "back foot_l")
    solid(foot_r, torso, "back foot_r")
    if int(np.nonzero(foot_l)[0].max()) != BACK_FOOT:
        raise SystemExit("back centre foot is not the ground line")
    if float(np.nonzero(arm_fore)[1].mean()) > 40:
        raise SystemExit("back arm is not on the screen-left")
    return masks


def place(arr, y_shift):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = OFF_Y + y_shift
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


def pivot_of(layer, torso_layer):
    part = layer[..., 3] >= 128
    torso = torso_layer[..., 3] >= 128
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 6)
    if not touch.any():
        ys, xs = np.nonzero(part)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    near = d <= d.min() + 6.0
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
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


def facing_of(kind, mirror):
    if kind == "front":
        return 1.0 if mirror else -1.0
    return -1.0 if mirror else 1.0


def build_view(img, masks, kind, mirror):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    depths = {"arm_fore": 5, "arm_rear": 4, "foot": 4, "foot_l": 5, "foot_r": 5, "brow": 3, "beard": 4}
    for name in ORDER[kind]:
        if int(placed[name].sum()) < 4:
            continue
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depths[name]), depths[name]
        )
    facing = facing_of(kind, mirror)
    pivots = {}
    signs = {}
    for name in ORDER[kind]:
        if name not in ROTATE:
            continue
        if int((layers[name][..., 3] >= 128).sum()) < 4:
            continue
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "order": ORDER[kind],
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        arm=0.0, arm_rear=0.0, brow=0.0, beard=0.0,
        arm_mode="up", arm_rear_mode="up", brow_mode="up", beard_mode="fwd",
        step_l=(0, 0), step_r=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Peg stays planted. Fists, brows, and the beard breathe. The head does not slide.
        p["arm"] = wave(f, n, 4.0, phase=0.4)
        p["arm_rear"] = wave(f, n, 3.0, phase=0.4 + math.pi)
        p["brow"] = wave(f, n, 3.0, phase=0.9)
        p["beard"] = wave(f, n, 6.0, phase=0.2)
    elif action == "walk":
        # Back has two feet, so they alternate. The front peg stays with the body.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        p["arm"] = wave(f, n, 14.0)
        p["arm_rear"] = wave(f, n, 10.0, phase=math.pi)
        p["arm_mode"] = "fwd"
        p["arm_rear_mode"] = "fwd"
        p["beard"] = wave(f, n, 6.0, phase=1.1)
        p["beard_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 roar: brows, beard, and fists rise. No mouth, so no wedge.
        # 5-8 the back, equal scale. 9-12 the heavy ice fist punches, max 34.
        # 13-15 recover. 15 is the still. The head does not slide.
        arm = [0, 6, 10, 12, 12, 8, 10, 12, 10, 18, 26, 34, 34, 14, 6, 0]
        rear = [0, 4, 8, 10, 10, 6, 8, 8, 6, 8, 10, 12, 10, 6, 2, 0]
        brow = [0, 4, 8, 10, 10, 0, 0, 0, 0, 2, 4, 4, 2, 0, 0, 0]
        beard = [0, 6, 10, 14, 14, 0, 0, 0, 0, 2, 4, 2, 0, 0, 0, 0]
        p["arm"] = float(arm[f])
        p["arm_rear"] = float(rear[f])
        p["brow"] = float(brow[f])
        p["beard"] = float(beard[f])
        p["arm_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["arm_rear_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["brow_mode"] = "up"
        p["beard_mode"] = "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.84, 0.62, 0.40, 0.22, 0.10, 0.03, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["arm"] = 8.0 * env
        p["brow"] = 4.0 * env
        p["beard"] = 6.0 * env
        p["flash"] = f == 1
    return finish(p, action, f, __file__)


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
    if name == "foot":
        return 0.0, bdx, bdy
    if name == "foot_l":
        sx, sy = p["step_l"]
        return 0.0, bdx + int(round(fc * sx)), bdy + int(sy)
    if name == "foot_r":
        sx, sy = p["step_r"]
        return 0.0, bdx + int(round(fc * sx)), bdy + int(sy)
    if name == "arm_fore":
        ang = part_angle(rig, name, p["arm_mode"], p["arm"])
    elif name == "arm_rear":
        ang = part_angle(rig, name, p["arm_rear_mode"], p["arm_rear"])
    elif name == "brow":
        ang = part_angle(rig, name, p["brow_mode"], p["brow"])
    elif name == "beard":
        ang = part_angle(rig, name, p["beard_mode"], p["beard"])
    else:
        ang = 0.0
    return ang, bdx, bdy


def compose(rig, p):
    L = rig["layers"]
    bdx = int(round(rig["facing"] * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in rig["order"]:
        ang, dx, dy = limb_motion(rig, p, name)
        if name not in rig["pivots"] or abs(ang) < 0.45:
            paste(out, shift(L[name], dx, dy))
        else:
            paste(out, moved(L[name], ang, rig["pivots"][name], dx, dy))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK) | is_color(out, BLACK)
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
    fc = rig["facing"]
    if "arm_fore" not in rig["pivots"]:
        return rig["pivots"]["center"]
    ang, dx, dy = limb_motion(rig, p, "arm_fore")
    layer = moved(rig["layers"]["arm_fore"], ang, rig["pivots"]["arm_fore"], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, ICE, BLUE, TEAL)
    r = 4 + age * 3
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r), (1, r))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    bdx = int(round(rig["facing"] * p["bdx"]))
    bdy = int(round(p["bdy"]))
    torso = shift(rig["layers"]["torso"], bdx, bdy)
    for name in rig["order"]:
        ang, dx, dy = limb_motion(rig, p, name)
        if name in rig["pivots"] and abs(ang) >= 0.45:
            layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
        else:
            layer = shift(rig["layers"][name], dx, dy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        other = other | (torso[..., 3] >= 128)
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    return rig


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    moved_part = (
        abs(p["arm"]) >= 0.45 or abs(p["arm_rear"]) >= 0.45
        or abs(p["brow"]) >= 0.45 or abs(p["beard"]) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0)
        or p["bdx"] or p["bdy"] or p["knock"] or abs(p["lean"]) > 0.4
    )
    if not moved_part and not p["turned"]:
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        fc = rig["facing"]
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            piv = rear_pivot(body, fc)
            body = rotate_nearest(body, fc * p["lean"], piv)
        body = reconnect(fill_pinholes(despeckle(body)))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = fore_tip(rig, p)
        impact(img, (tx + rig["facing"] * 2, ty), rig["facing"], p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    groups = component_list(op)
    return len(groups[0]) / float(op.sum())


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
    if int(np.nonzero(front[..., 3])[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    fa = rigs[("front", False)]["canvas"]
    ba = rigs[("back", False)]["canvas"]
    if int(np.nonzero(fa[..., 3] >= 128)[0].max()) != int(np.nonzero(ba[..., 3] >= 128)[0].max()):
        raise SystemExit("front and back feet do not share a line")
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if rigs["E"]["facing"] != 1.0 or rigs["W"]["facing"] != -1.0:
        raise SystemExit("E/W do not face screen-right/left")
    if rig_for(rigs, "S", {"turned": True}) is not rigs["N"]:
        raise SystemExit("south turn is not the north back")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "arm_fore": (40, 180, 255, 255),
        "arm_rear": (180, 70, 220, 255),
        "foot": (40, 210, 80, 255),
        "foot_l": (80, 255, 180, 255),
        "foot_r": (255, 180, 40, 255),
        "brow": (255, 80, 80, 255),
        "beard": (80, 220, 255, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 4, 0), 4, 1))
    gap = np.zeros((tiles[0].shape[0], 8, 4), np.uint8)
    Image.fromarray(np.concatenate([tiles[0], gap, tiles[1]], axis=1), "RGBA").save(work / "parts.png")


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
        "dex": 555, "form": "regional", "variant": "darmanitan-galar-standard", "gen": "gen5",
        "dir": "monsterSprite/gen5/0555_darmanitan-standard/regional/darmanitan-galar-standard",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_faults(img, palette_set):
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    op = img[..., 3] == 255
    off = set()
    if op.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
        off = used - palette_set
    edge = bool(op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any())
    return off, partial, edge


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
    print(
        "parts front", {k: int(v.sum()) for k, v in fm.items()},
        "back", {k: int(v.sum()) for k, v in bm.items()},
        "palette", len(palette), "foot", FOOT_Y, flush=True,
    )
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, flush=True)

    ident = 0
    zero = base_pose()
    for key in (("front", False), ("front", True), ("back", False), ("back", True)):
        body = snap(compose(rigs[key], zero), palette)
        ident += int(np.any(body != rigs[key]["canvas"], axis=2).sum())
    print("compose_identity", ident, flush=True)
    if ident:
        raise SystemExit("rest compose does not match the still")

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        pal = {tuple(int(v) for v in c) for c in palette}
        off = set()
        partial = 0
        clip = []
        bodies = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    bad, part, edge = frame_faults(img, pal)
                    off |= bad
                    partial += part
                    if edge:
                        clip.append(f"{act}:{d}:{f}")
                    if act in ("attack", "hurt", "walk"):
                        bodies.append(largest_fraction(img))
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
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["arm"], 1), round(p["arm_rear"], 1), p["arm_mode"],
                        p["turned"], p["spin"], round(p["brow"], 1), round(p["beard"], 1)))
        print("attack poses", atk, flush=True)
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        print("qa_numbers", {
            "rest_diff": int(sum(rest_diff.values())),
            "off_palette_colors": len(off),
            "partial_alpha": partial,
            "edge_n": len(clip),
            "edge_touching": clip[:12],
            "body_main_min": round(min(bodies), 3) if bodies else 1,
        }, flush=True)
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
            if key in state["done"]:
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
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
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
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_darmanitan-galar-standard_actions", "version": VERSION, "key": KEY,
        "dex": 555, "variant": "darmanitan-galar-standard", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "snowman biped: heavy ice fist, small ice fist, one front peg, two back feet, ice brows and beard, no mouth",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (chin tip x35 < eye-white mean x44.4); N is the back drawing",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
