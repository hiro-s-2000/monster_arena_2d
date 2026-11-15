#!/usr/bin/env python3
"""Krokorok normal (0552) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x23-78 y19-78, 1506 px, alpha 0/255,
13 colours):
  Two eye glints of (180, 180, 189): (y26, x39) and (y25, x47). Midpoint x 43.0.
  The muzzle lip is the bright pink (238, 148, 164) cluster at x38-42 y39-40
  (mean x 40.1). The muzzle outline reaches x34 at y40 and x32 at y42, left of
  the eyes, so the still faces screen-left. The navy band y25-30 x37-49 is the
  sunglasses and stays on the head.
  The large pink panel x33-45 y53-62 is the belly, bordered by cream. It is not
  a mouth. The roar uses only the muzzle lip.
  No separate arm is drawn. Do not add one.
  Tail: the right curl, one piece, x60-78 y45-65 (163 px). Tip rises at x70-78.
  Feet split under the gap at y69. Left foot x23-37 y68-75 (76 px) with white
  and grey claws. Right foot x51-62 y68-78 (93 px) with claws at the toe.
normal/back.png (bbox x7-68 y18-80, 1485 px, 12 colours, no eye glint):
  The tail is the left sweep x7-40 y62-76 (300 px). Clawed foot x47-62 y74-80
  (77 px), grey claws x59-62 y77-79. Only one foot is drawn. Cream on the head
  reaches x66, so the away-side of this still is screen-right. The back is
  shifted up 2 px so its sole shares the front's lowest toe (y78).

Directions, because the front faces screen-left and the back faces away-right:
  S SE E front mirrored (forward is screen-right). W SW front as drawn.
  NE N back as drawn (away-right). NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 shows that
  direction's back at equal scale. No width squash and no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_krokorok_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0552_krokorok/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0552_krokorok/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0552_krokorok"
VERSION = "krokorok_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
OFF_Y = (CELL_H - 96) // 2
FRONT_FOOT = 78
BACK_FOOT = 80
FOOT_Y = OFF_Y + FRONT_FOOT
# kind, mirror. Front as drawn faces screen-left. Back as drawn faces away-right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up peak finish() may raise frames 5-12 toward. The tail slap is drawn
# at 34, above this, so finish does not carry the hit past the authored angle.
PEAKS = {"tail": 22.0}

TAN = (197, 156, 90)
NAVY = (41, 41, 49)
INK = (16, 16, 16)
BROWN = (139, 115, 49)
SHADOW = (24, 24, 32)
PINK = (238, 148, 164)
RIDGE = (82, 57, 16)
CREAM = (222, 189, 131)
LIP = (205, 115, 123)
WHITE = (255, 255, 255)
GRAY = (90, 90, 106)
SILVER = (197, 197, 197)
GLINT = (180, 180, 189)

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
    """4x nearest in-plane rotation. Positive is clockwise on screen. No squash."""
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
    d = ((px[:, None, :] - palette.astype(np.int32)[None, :, :]) ** 2).sum(axis=2)
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


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


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


def require_piece(name, mask, torso):
    if not mask.any():
        raise SystemExit(f"{name} is missing")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart ({largest_component(mask)} vs {int(mask.sum())})")
    if not (mask & dilate(torso, 1)).any():
        raise SystemExit(f"{name} does not touch the body")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    foot_l = op & (yy >= 68) & (xx <= 38)
    foot_r = op & (yy >= 68) & (xx >= 51)
    tail = op & (xx >= 60) & (yy <= 67)
    torso = op & ~foot_l & ~foot_r & ~tail
    masks = {"tail": tail, "foot_l": foot_l, "foot_r": foot_r, "torso": torso}
    check_partition(op, masks)
    for name in ("tail", "foot_l", "foot_r"):
        require_piece(name, masks[name], torso)
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    glint = is_color(img, GLINT)
    if int(glint.sum()) != 2 or int((glint & torso).sum()) != 2:
        raise SystemExit("eye glints left the head")
    lip = is_color(img, PINK) & (yy <= 42)
    belly = is_color(img, PINK) & (yy >= 50)
    if int(lip.sum()) != 7 or int((lip & torso).sum()) != 7:
        raise SystemExit("muzzle lip left the head")
    if int(belly.sum()) < 90 or int((belly & torso).sum()) != int(belly.sum()):
        raise SystemExit("belly left the torso")
    ex = float(np.nonzero(glint)[1].mean())
    lx = float(np.nonzero(lip)[1].mean())
    if not (lx < ex - 2.0):
        raise SystemExit(f"front no longer faces left: lip {lx:.1f} eyes {ex:.1f}")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    expect = {"tail": 163, "foot_l": 76, "foot_r": 93, "torso": 1174}
    if counts != expect:
        raise SystemExit(f"front parts changed {counts}")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    foot = op & (yy >= 74) & (xx >= 46)
    tail = op & (xx <= 40) & (yy >= 62) & ~foot
    torso = op & ~foot & ~tail
    masks = {"tail": tail, "foot": foot, "torso": torso}
    check_partition(op, masks)
    for name in ("tail", "foot"):
        require_piece(name, masks[name], torso)
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
    if is_color(img, GLINT).any():
        raise SystemExit("back gained an eye glint")
    if float(np.nonzero(tail)[1].mean()) >= float(np.nonzero(torso)[1].mean()):
        raise SystemExit("back tail is no longer on the left")
    claws = any_color(img, (GRAY, WHITE, SILVER)) & (yy >= 76)
    if int(claws.sum()) < 4 or int((claws & foot).sum()) != int(claws.sum()):
        raise SystemExit("back claws left the foot")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back lowest pixel changed")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    expect = {"tail": 300, "foot": 77, "torso": 1108}
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
    feet = ("foot_l", "foot_r") if kind == "front" else ("foot",)
    depth = {"tail": 8, "foot_l": 5, "foot_r": 5, "foot": 5}
    for name in ("tail",) + feet:
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth[name]), depth[name]
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {"tail": pivot_of(layers["tail"], layers["torso"])}
    signs = {"tail": signs_for(layers["tail"], pivots["tail"], facing)}
    for name in feet:
        pivots[name] = pivot_of(layers[name], layers["torso"])
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    foot_cx = {n: float(np.nonzero(placed[n])[1].mean()) for n in feet}
    if facing > 0:
        near = max(feet, key=lambda n: foot_cx[n])
    else:
        near = min(feet, key=lambda n: foot_cx[n])
    mouth = None
    if kind == "front":
        lip = is_color(img, PINK)
        yy = np.mgrid[0:96, 0:96][0]
        lip = lip & (yy <= 42)
        ly, lx = np.nonzero(lip)
        cx = OFF_X + float(lx.mean())
        cy = OFF_Y + y_shift + float(ly.mean())
        if mirror:
            cx = (CELL_W - 1) - cx
        mouth = (cx, cy)
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "feet": feet,
        "near_foot": near, "mouth": mouth,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, tail=0.0, tail_mode="up",
        step_a=(0, 0), step_b=(0, 0), kick=(0, 0),
        lean=0.0, knock=0, flash=False, wedge=0, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The tail curls. The head does not bob.
        p["tail"] = wave(f, n, 5.0, phase=0.4)
        p["tail_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step_a"] = (int(round(2 * max(0.0, sa))), -int(round(2 * max(0.0, sa))))
        p["step_b"] = (int(round(2 * max(0.0, -sa))), -int(round(2 * max(0.0, -sa))))
        p["tail"] = wave(f, n, 8.0, phase=math.pi / 2)
        p["tail_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 raise the tail and open the muzzle. 5-8 the back, at
        # rest, equal scale. 9-12 tail slap and a kick. 13-15 recover.
        tail = [0, 8, 14, 18, 22, 0, 0, 0, 0, 22, 28, 34, 34, 14, 6, 0]
        bdx = [0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0]
        p["tail"] = float(tail[f])
        p["bdx"] = int(bdx[f])
        p["tail_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["turned"] = 5 <= f <= 8
        p["wedge"] = f if 1 <= f <= 4 else 0
        if f in (11, 12):
            p["kick"] = (2, -2)
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.82, 0.60, 0.40, 0.22, 0.10, 0.03, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["tail"] = 6.0 * env
        p["tail_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # Frames 5-8 are the other side at equal scale. finish() would swing the
    # tail on these frames; drop it so the turned silhouette stays whole.
    if action == "attack" and 5 <= f <= 8:
        p["tail"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_a"] = (0, 0)
        p["step_b"] = (0, 0)
        p["kick"] = (0, 0)
        p["wedge"] = 0
        p["spark"] = 0
        p["spin"] = 0
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
    ang = part_angle(rig, "tail", p["tail_mode"], p["tail"])
    parts["tail"] = moved(L["tail"], ang, piv["tail"], bdx, bdy)
    for name in rig["feet"]:
        if name == "foot_r":
            sx, sy = p["step_b"]
        else:
            sx, sy = p["step_a"]
        if name == rig["near_foot"]:
            kx, ky = p["kick"]
            sx += kx
            sy += ky
        dx = bdx + int(round(fc * sx))
        dy = int(sy)
        parts[name] = shift(L[name], dx, dy)
    return parts


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    parts = limb_layers(rig, p)
    for name in rig["feet"]:
        paste(out, parts[name])
    paste(out, parts["tail"])
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, (INK, SHADOW))
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def tail_tip(rig, p):
    layer = limb_layers(rig, p)["tail"]
    fc = rig["facing"]
    if not np.any(layer[..., 3] >= 128):
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def roar(img, origin, fc, age):
    """A small wedge and a ring in front of the muzzle lip. Palette colours only."""
    ox, oy = origin
    ox = ox + fc * 8
    cols = (PINK, LIP, INK)
    span = age + 1
    for i in range(span):
        put(img, ox + fc * (2 + i), oy, cols[i % 3])
        put(img, ox + fc * (2 + i), oy - 1 - i // 2, PINK)
        put(img, ox + fc * (2 + i), oy + 1 + i // 2, LIP)
    rad = 6 + age * 2
    for deg in range(-48, 49, 8):
        a = math.radians(deg)
        x = ox + fc * (6 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, WHITE if (deg // 8) % 2 == 0 else CREAM)


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, CREAM, PINK)
    r = 4 + age * 3
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r), (2, r))):
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
    rotated = abs(p["tail"]) >= 0.45 or abs(p["lean"]) > 0.4
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if rotated:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["wedge"] and rig["kind"] == "front" and rig["mouth"] is not None and not p["turned"]:
        mx, my = rig["mouth"]
        bdx = int(round(fc * p["bdx"]))
        roar(img, (mx + bdx, my + int(p["bdy"])), fc, p["wedge"])
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = tail_tip(rig, p)
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


def crop_box(canvas, pad_x=36, pad_y=28):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(CELL_H, int(ys.max()) + 1 + 14)
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
        "tail": (220, 80, 40, 255),
        "foot_l": (250, 220, 40, 255),
        "foot_r": (80, 140, 255, 255),
        "foot": (80, 200, 120, 255),
        "torso": (170, 176, 186, 255),
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
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 552, "form": "normal", "variant": "krokorok", "gen": "gen5",
        "dir": "monsterSprite/gen5/0552_krokorok/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_faults(img, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    op = img[..., 3] == 255
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)} if op.any() else set()
    off = len(used - pal)
    a = img[..., 3] == 255
    edge = int(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())
    return off, partial, edge


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
    print("signs S", rigs["S"]["signs"], "W", rigs["W"]["signs"], "N", rigs["N"]["signs"], flush=True)
    print("mouth S", rigs["S"]["mouth"], "W", rigs["W"]["mouth"], "N", rigs["N"]["mouth"], flush=True)

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

    if args.samples:
        off = partial = edge = 0
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    o, p, e = frame_faults(img, palette)
                    off += o
                    partial += p
                    edge += e
            print("scanned", d, flush=True)
        still = int(sum(rest_diff.values()))
        print(f"still_diff {still} off_palette {off} partial_alpha {partial} edge_touch {edge}", flush=True)
        box = crop_box(rigs["S"]["canvas"])
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
            atk.append((f, round(p["tail"], 1), p["tail_mode"], p["turned"], p["spin"], round(p["lean"], 1)))
        print("attack poses", atk, flush=True)
        hurt = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", hurt, "max", max(hurt), flush=True)
        print("samples", work, flush=True)
        return

    box = crop_box(rigs["S"]["canvas"])
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
    still = int(sum(rest_diff.values()))
    report = {
        "version": VERSION, "hash": digest, "rest_diff": rest_diff, "idle0": idle0,
        "still_diff": still,
        "off_palette_colors": len(off), "partial_alpha": partial,
        "edge_touching": clip[:12], "edge_n": len(clip),
        "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (still == 0 and not any(idle0.values()) and not off
          and partial == 0 and not clip)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(
        f"still_diff {still} off_palette {len(off)} partial_alpha {partial} edge_touch {len(clip)}",
        flush=True,
    )
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_krokorok_actions", "version": VERSION, "key": KEY,
        "dex": 552, "variant": "krokorok", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped desert croc (two feet, right-side tail, muzzle lip, no separate arm)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front lip mean x 40.1 is left of eye midpoint 43.0, muzzle outline x34; back tail trails left so away is screen-right",
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
