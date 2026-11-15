#!/usr/bin/env python3
"""Tranquill normal (0520) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x28-70 y16-71, 1046 px, alpha 0/255,
15 colours):
  Beak gold+olive of the bill itself, mean x 30.9, tip x28 y35.
  Pink eye plus the white catchlight, mean x 35.2. The bill tip is left of
  the eye, so the still faces screen-left.
  Heart crest is the top knot y16-22 x37-45 (the y21 cleft is the heart).
  No open mouth is drawn inside the bill, so the roar does not cut a wedge.
  It lifts the bill, the crest, and the tail, and puts rings in front of the bill.
  Tail fan is the rear feathers on the screen-right (the gap-separated upper
  feather plus the lower fan that joins the rump). Folded wings stay on the
  torso; they are not a separate piece.
  Two pink feet from y59. Left foot x<=50, right foot x>=51. Soles end y71.
normal/back.png (bbox x19-64 y18-73, 1071 px, lowest y73) has the bill on the
  screen-right (gold+olive mean x 60.4, pink glimpse mean x 55.7), so the
  back still looks away to screen-right. Tail is the left feather x19-34.
  Feet from y63. The back is shifted up 2px so both soles share y71.

Directions, because the front faces left and the back bill is on the right:
  S SE E front mirrored (bill to screen-right). W SW front as drawn.
  N NE back as drawn. NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 shows that
  direction's back at equal scale (S/SE/E become the N still). No width
  squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_tranquill_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0520_tranquill/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0520_tranquill/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0520_tranquill"
VERSION = "tranquill_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
OFF_Y = 72
FRONT_FOOT = 71
BACK_FOOT = 73
FOOT_Y = OFF_Y + FRONT_FOOT
# kind, mirror. Front as drawn faces screen-left. Back as drawn has the bill
# on the screen-right, so away-right uses the back unmirrored.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up peaks finish() may raise on frames 5-12. The peck and the tail
# whip are drawn at 34 deg, above these peaks, so finish does not pass them.
PEAKS = {"beak": 18.0, "crest": 12.0, "tail": 16.0}
LIMBS = ("crest", "beak", "tail", "foot_l", "foot_r")

INK = (0, 0, 0)
DK = (49, 49, 49)
GRAY = (74, 74, 74)
MID = (131, 131, 131)
SILVER = (164, 164, 164)
PALE = (189, 189, 189)
WHITE_G = (222, 222, 222)
WHITE = (255, 255, 255)
GOLD = (255, 205, 57)
OLIVE = (156, 123, 8)
PINK = (255, 65, 123)
ROSE = (189, 98, 98)
LPINK = (255, 156, 156)
DRED = (139, 32, 49)
FOOT_COLS = (ROSE, LPINK, DRED)
RIM_COLS = (INK, DK, GRAY)

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
        if (nxt == out).all():
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
    """4x nearest in-plane rotation (positive = clockwise on screen). No squash."""
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
        for name in ("up", "down", "left", "right"):
            grown = np.zeros_like(src)
            if name == "up":
                grown[:-1] = src[1:]
            elif name == "down":
                grown[1:] = src[:-1]
            elif name == "left":
                grown[:, :-1] = src[:, 1:]
            else:
                grown[:, 1:] = src[:, :-1]
            take = (acc[..., 3] < 128) & allowed & (grown[..., 3] >= 128)
            acc[take] = grown[take]
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
        raise SystemExit(f"{name} is missing")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart ({largest_component(mask)} vs {int(mask.sum())})")


def absorb_islands(torso, limbs):
    """A torso speck that touches a limb rides with that limb."""
    groups = component_list(torso)
    if len(groups) <= 1:
        return torso
    groups.sort(key=len, reverse=True)
    keep = np.zeros_like(torso)
    for y, x in groups[0]:
        keep[y, x] = True
    h, w = torso.shape
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    order = ("crest", "beak", "tail", "foot_l", "foot_r")
    for pts in groups[1:]:
        target = None
        for name in order:
            mask = limbs[name]
            hit = False
            for y, x in pts:
                for dy, dx in neigh:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                target = name
                break
        if target is None:
            for y, x in pts:
                keep[y, x] = True
        else:
            for y, x in pts:
                limbs[target][y, x] = True
    return keep


def far_side(op, y0, y1, side):
    """Pixels past a gap of two or more empty cells. The feather behind the body."""
    out = np.zeros_like(op)
    for y in range(y0, y1):
        xs = np.nonzero(op[y])[0]
        if len(xs) == 0:
            continue
        if side == "right":
            start = None
            for i in range(1, len(xs)):
                if xs[i] >= xs[i - 1] + 3:
                    start = int(xs[i])
            if start is not None:
                out[y, start:] = op[y, start:]
        else:
            end = None
            for i in range(1, len(xs)):
                if xs[i] >= xs[i - 1] + 3:
                    end = int(xs[i - 1])
                    break
            if end is not None:
                out[y, : end + 1] = op[y, : end + 1]
    return out


def bill_core(img, facing_left):
    core = is_color(img, GOLD) | is_color(img, OLIVE)
    picked = np.zeros(core.shape, bool)
    for pts in component_list(core):
        xs = [p[1] for p in pts]
        ys = [p[0] for p in pts]
        if facing_left:
            keep = min(xs) <= 35 and max(ys) >= 32
        else:
            keep = max(xs) >= 57
        if keep:
            for y, x in pts:
                picked[y, x] = True
    return picked


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    crest = flood(op & (yy <= 22), op & (yy <= 18))
    beak_core = bill_core(img, True)
    rim = any_color(img, (INK, DK, DRED)) & dilate(beak_core, 1)
    rim &= (yy >= 29) & (yy <= 39) & (xx <= 40)
    rim &= ~is_color(img, PINK) & ~is_color(img, WHITE)
    beak = flood(beak_core | rim, beak_core)
    tail = far_side(op, 29, 47, "right")
    white = is_color(img, WHITE_G)
    wing_seed = op & (xx >= 62) & (yy >= 34) & (yy <= 50) & ~tail & ~white
    wing_allow = op & (xx >= 56) & (yy >= 32) & (yy <= 52) & ~tail & ~white & ~beak & ~crest
    # The lower fan is the same tail, not a second wing. Folded wings stay on the torso.
    tail = tail | flood(wing_allow, wing_seed)
    footc = any_color(img, FOOT_COLS) & (yy >= 59)
    fl = footc & (xx <= 50)
    fr = footc & (xx >= 51)
    rimf = any_color(img, RIM_COLS) & (yy >= 59)
    foot_l = flood(fl | (dilate(fl, 1) & rimf & (xx <= 50)), fl)
    foot_r = flood(fr | (dilate(fr, 1) & rimf & (xx >= 51)), fr)
    limbs = {"crest": crest, "beak": beak, "tail": tail, "foot_l": foot_l, "foot_r": foot_r}
    torso = op & ~crest & ~beak & ~tail & ~foot_l & ~foot_r
    torso = absorb_islands(torso, limbs)
    masks = dict(limbs)
    masks["torso"] = torso
    check_partition(op, masks)
    for name in LIMBS:
        require_piece(name, masks[name])
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"front {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    eye = is_color(img, PINK) | is_color(img, WHITE)
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) != 25:
        raise SystemExit("eyes left the head")
    # The pale feather edge at x61-64 rides with the tail. The chest stays.
    belly = white & (xx < 58)
    if int((belly & torso).sum()) != int(belly.sum()):
        raise SystemExit("belly was cut into the tail")
    bill = is_color(img, GOLD) | is_color(img, OLIVE)
    if int((bill & beak).sum()) < 25:
        raise SystemExit("bill lost its gold")
    ey, ex = np.nonzero(eye)
    by, bx = np.nonzero(beak)
    if int(bx.min()) >= float(ex.mean()):
        raise SystemExit("front bill is not left of the eye")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    expect = {"crest": 32, "beak": 37, "tail": 199, "foot_l": 74, "foot_r": 49, "torso": 655}
    if counts != expect:
        raise SystemExit(f"front parts changed {counts}")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    crest = flood(op & (yy <= 24), op & (yy <= 20))
    beak_core = bill_core(img, False)
    rim = any_color(img, (INK, DK, DRED, GRAY)) & dilate(beak_core, 1)
    rim &= (xx >= 55) & (yy >= 28) & (yy <= 35)
    rim &= ~is_color(img, PINK) & ~is_color(img, WHITE)
    beak = flood(beak_core | rim, beak_core)
    tail = far_side(op, 40, 53, "left")
    footc = any_color(img, FOOT_COLS) & (yy >= 63)
    fl = footc & (xx <= 42)
    fr = footc & (xx >= 43)
    rimf = any_color(img, RIM_COLS) & (yy >= 63)
    foot_l = flood(fl | (dilate(fl, 1) & rimf & (xx <= 42)), fl)
    foot_r = flood(fr | (dilate(fr, 1) & rimf & (xx >= 43)), fr)
    limbs = {"crest": crest, "beak": beak, "tail": tail, "foot_l": foot_l, "foot_r": foot_r}
    torso = op & ~crest & ~beak & ~tail & ~foot_l & ~foot_r
    torso = absorb_islands(torso, limbs)
    masks = dict(limbs)
    masks["torso"] = torso
    check_partition(op, masks)
    for name in LIMBS:
        require_piece(name, masks[name])
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"back {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
    pink = is_color(img, PINK)
    bill = is_color(img, GOLD) | is_color(img, OLIVE)
    if int((bill & beak).sum()) != int(bill.sum()):
        raise SystemExit("back bill was split")
    if float(np.nonzero(beak)[1].mean()) <= float(np.nonzero(pink)[1].mean()):
        raise SystemExit("back bill is no longer right of the eye glimpse")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back lowest pixel changed")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    expect = {"crest": 43, "beak": 32, "tail": 57, "foot_l": 41, "foot_r": 73, "torso": 825}
    if counts != expect:
        raise SystemExit(f"back parts changed {counts}")
    return masks


def place(arr, y_shift):
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), bool)
    else:
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
    depth = {"crest": 3, "beak": 4, "tail": 6, "foot_l": 4, "foot_r": 4}
    for name in LIMBS:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name] + 2
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in ("crest", "beak", "tail"):
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    for name in ("foot_l", "foot_r"):
        pivots[name] = pivot_of(layers[name], layers["torso"])
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    foot_cx = {n: float(np.nonzero(placed[n])[1].mean()) for n in ("foot_l", "foot_r")}
    if facing > 0:
        near = max(("foot_l", "foot_r"), key=lambda n: foot_cx[n])
    else:
        near = min(("foot_l", "foot_r"), key=lambda n: foot_cx[n])
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "near_foot": near,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, beak=0.0, crest=0.0, tail=0.0,
        beak_mode="up", crest_mode="up", tail_mode="up",
        step_l=(0, 0), step_r=(0, 0), kick=(0, 0),
        lean=0.0, knock=0, flash=False, ring=-1, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. Crest and tail breathe. The bill stays shut.
        p["bdy"] = -1 if wave(f, n, 1.0) > 0.55 else 0
        p["crest"] = wave(f, n, 6.0, phase=0.4)
        p["tail"] = wave(f, n, 8.0, phase=0.9)
        p["crest_mode"] = "up"
        p["tail_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step_l"] = (int(round(2 * max(0.0, sa))), -int(round(3 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * max(0.0, -sa))), -int(round(3 * max(0.0, -sa))))
        p["bdy"] = -1 if abs(math.sin(u)) > 0.65 else 0
        p["crest"] = wave(f, n, 5.0, phase=0.3)
        p["tail"] = wave(f, n, 10.0, phase=math.pi / 2, cycles=2.0)
        p["crest_mode"] = "up"
        p["tail_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 raise the bill, crest, and tail. No mouth wedge.
        # 5-8 the back, at rest, equal scale. 9-12 peck and tail whip. 13-15 recover.
        beak = [0, 8, 14, 18, 18, 0, 0, 0, 0, 22, 30, 34, 34, 14, 6, 0]
        crest = [0, 4, 8, 12, 12, 0, 0, 0, 0, 8, 10, 12, 10, 4, 2, 0]
        tail = [0, 6, 12, 16, 16, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        bdx = [0, 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0]
        p["beak"] = float(beak[f])
        p["crest"] = float(crest[f])
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        p["beak_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["crest_mode"] = "up"
        p["tail_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if 2 <= f <= 4:
            p["ring"] = f - 2
        if f in (11, 12):
            p["kick"] = (2, -2)
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["beak"] = 6.0 * env
        p["crest"] = 10.0 * env
        p["tail"] = 12.0 * env
        p["beak_mode"] = "up"
        p["crest_mode"] = "up"
        p["tail_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # Frames 5-8 are the back still at equal scale. finish() would swing the
    # wind-up on these frames; drop it so the turned silhouette stays whole.
    if action == "attack" and 5 <= f <= 8:
        p["beak"] = 0.0
        p["crest"] = 0.0
        p["tail"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["kick"] = (0, 0)
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["spin"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
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


def limb_layers(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    parts = {}
    for name in LIMBS:
        if name == "foot_l":
            sx, sy = p["step_l"]
            ang = 0.0
        elif name == "foot_r":
            sx, sy = p["step_r"]
            ang = 0.0
        else:
            sx, sy = 0, 0
            ang = part_angle(rig, name, p[name + "_mode"], p[name])
        if name == rig["near_foot"]:
            kx, ky = p["kick"]
            sx += kx
            sy += ky
        dx = bdx + int(round(fc * sx))
        dy = bdy + int(sy)
        parts[name] = moved(L[name], ang, piv[name], dx, dy)
    return parts


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    parts = limb_layers(rig, p)
    # Torso first, so the joint cushion stays hidden under each part at rest.
    paste(out, shift(L["torso"], bdx, bdy))
    paste(out, parts["tail"])
    paste(out, parts["foot_l"])
    paste(out, parts["foot_r"])
    paste(out, parts["crest"])
    paste(out, parts["beak"])
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def beak_tip(rig, p):
    layer = limb_layers(rig, p)["beak"]
    fc = rig["facing"]
    if not np.any(layer[..., 3] >= 128):
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (GOLD, WHITE, PINK)
    rad = 8 + age * 5
    for deg in range(-50, 51, 8):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[abs(deg) // 16 % 3])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (GOLD, WHITE, PINK)
    r = 4 + age * 3
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -3), (2, 3))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name, layer in limb_layers(rig, p).items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    """Attack 5-8 faces away. Front rows become that side's back, unscaled."""
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", not mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    rotated = (
        abs(p["beak"]) >= 0.45 or abs(p["crest"]) >= 0.45 or abs(p["tail"]) >= 0.45
        or abs(p["lean"]) > 0.4
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        # Lean backward, away from facing, around the rear foot. Not a flat spin.
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if rotated:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["ring"] >= 0 and action == "attack" and not p["turned"]:
        tx, ty = beak_tip(rig, p)
        roar_rings(img, (tx + fc * 3, ty), fc, p["ring"])
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = beak_tip(rig, p)
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
    return snap(img, palette)


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


def crop_box(canvas, pad_x=28, pad_y=22):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(CELL_H, int(ys.max()) + 1 + 16)
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad_x)
    return y0, y1, x0, x1


def preview_image(sheets):
    pick = {"idle": 0, "walk": 4, "attack": 11, "hurt": 3}
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


def still_colors(*imgs):
    cols = []
    seen = set()
    for img in imgs:
        for c in np.unique(img[img[..., 3] >= 128][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in seen:
                seen.add(t)
                cols.append(t)
    return cols


def locked_palette(front_path, back_path, work):
    front = np.array(Image.open(front_path).convert("RGBA"))
    back = np.array(Image.open(back_path).convert("RGBA"))
    own = still_colors(front, back)
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    if not (lock_dir / "palette.json").exists():
        py = str(FORGE / ".venv/bin/python")
        hexes = ["#%02x%02x%02x" % c for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(front_path),
             "--colors", str(len(hexes)), "--reserve", *hexes, "--output-dir", str(pal_dir)],
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
    want = set(own)
    if got != want:
        raise SystemExit(f"locked palette differs from the stills: {sorted(got ^ want)}")
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
        "crest": (255, 40, 40, 255),
        "beak": (255, 210, 40, 255),
        "tail": (40, 220, 80, 255),
        "foot_l": (255, 80, 200, 255),
        "foot_r": (80, 255, 220, 255),
        "torso": (90, 90, 100, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        pair = np.concatenate([img, part], axis=1)
        tiles.append(np.repeat(np.repeat(pair, 4, 0), 4, 1))
    Image.fromarray(np.concatenate(tiles, axis=0), "RGBA").save(work / "parts.png")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    actions = OUT_DIR / "actions" / "preview.png"
    if actions.is_file():
        dest = SYNC_OUT / "actions"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actions, dest / "preview.png")
        n += 1
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 520, "form": "normal", "variant": "tranquill", "gen": "gen5",
        "dir": "monsterSprite/gen5/0520_tranquill/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def head_x(img):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    top = ys <= int(ys.min()) + 8
    return float(xs[top].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    facing = {d: rigs[d]["facing"] for d in DIRS}
    print("facing", facing, "near", {d: rigs[d]["near_foot"] for d in ("S", "W", "N", "E")}, flush=True)
    print("signs S", rigs["S"]["signs"], "W", rigs["W"]["signs"], flush=True)

    rest_diff = {}
    for d in DIRS:
        r = synth(rigs, d, "idle", 0, palette, effects=False)
        rest_diff[d] = int(np.any(r != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    for d, back_d in (("S", "N"), ("E", "N"), ("SE", "N"), ("W", "NW"), ("SW", "NW")):
        got = synth(rigs, d, "attack", 6, palette, effects=False)
        if not np.array_equal(got, rigs[back_d]["canvas"]):
            raise SystemExit(f"attack turn {d} is not the {back_d} still")
    for d in ("N", "NE", "NW"):
        got = synth(rigs, d, "attack", 7, palette, effects=False)
        if not np.array_equal(got, rigs[d]["canvas"]):
            raise SystemExit(f"attack turn {d} changed the back still")
    for d in DIRS:
        a0 = synth(rigs, d, "attack", 0, palette, effects=False)
        a15 = synth(rigs, d, "attack", 15, palette, effects=False)
        if not np.array_equal(a0, rigs[d]["canvas"]) or not np.array_equal(a15, rigs[d]["canvas"]):
            raise SystemExit(f"attack ends do not match the still for {d}")

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        for d in ("S", "W", "N", "E"):
            frames = []
            for act, fs in (
                ("idle", (0, 4, 8, 12)),
                ("walk", (0, 2, 4, 6, 8, 10, 12, 14)),
                ("attack", (0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15)),
                ("hurt", (0, 1, 3, 6, 11)),
            ):
                for f in fs:
                    frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
            sheet = tile_grid(frames, len(frames))
            z = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_{d}.png")
        zoom_frames = []
        for act, f in (
            ("idle", 0), ("walk", 4), ("walk", 12),
            ("attack", 0), ("attack", 4), ("attack", 5), ("attack", 8),
            ("attack", 9), ("attack", 11), ("attack", 12), ("attack", 15),
            ("hurt", 3),
        ):
            zoom_frames.append(synth(rigs, "S", act, f, palette)[y0:y1, x0:x1])
        for act, f in (("attack", 11), ("hurt", 3), ("walk", 4), ("idle", 0)):
            zoom_frames.append(synth(rigs, "N", act, f, palette)[y0:y1, x0:x1])
        z = tile_grid(zoom_frames, 8)
        Image.fromarray(np.repeat(np.repeat(z, 3, 0), 3, 1), "RGBA").save(work / "samples_zoom.png")
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
            atk.append((f, round(p["beak"], 1), p["beak_mode"], round(p["tail"], 1), p["turned"], p["spin"], p["lean"]))
        print("attack poses", atk, flush=True)
        hurt = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", hurt, "max", max(hurt), flush=True)
        idle_s = synth(rigs, "S", "idle", 0, palette, effects=False)
        hurt_s = synth(rigs, "S", "hurt", 3, palette, effects=False)
        idle_w = synth(rigs, "W", "idle", 0, palette, effects=False)
        hurt_w = synth(rigs, "W", "hurt", 3, palette, effects=False)
        print("head x S idle", round(head_x(idle_s), 1), "hurt", round(head_x(hurt_s), 1),
              "W idle", round(head_x(idle_w), 1), "hurt", round(head_x(hurt_w), 1), flush=True)
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
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion_z = np.repeat(np.repeat(motion, 2, 0), 2, 1)
    Image.fromarray(motion_z, "RGBA").save(OUT_DIR / "preview_motion.png")
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
          and partial == 0 and not clip and min(bodies) >= 0.9)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_tranquill_actions", "version": VERSION, "key": KEY,
        "dex": 520, "variant": "tranquill", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "avian pigeon (bill, heart crest, rear tail fan, two feet; wings folded into the torso)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front bill tip x28 is left of the eye mean x35.2; back bill mean x60.4 is right of the pink glimpse x55.7",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.publish:
        print("synced", sync_outputs(), flush=True)
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
