#!/usr/bin/env python3
"""Excadrill normal (0530) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x8-85 y14-76, 2760 px, alpha 0/255,
14 colours, sole y76):
  White eye pixels mean x 35.15. The pink nose (ee837b, ac5252, cd736a,
  13 px, x29-32 y40-43) mean x 30.23, left of the eyes, so the still faces
  screen-left. The largest eye is x30-39 y36-41. There is no mouth cavity
  under the nose (the pixels there are outline and brown). Maroon at x25-27
  y44 is a chest mark, not an opening, so the roar does not invent a wedge
  or rings. It raises the head drill and the two drill-claws.
  Head drill: steel, x16-45 y14-37, tip toward the upper left.
  Screen-left claw x8-25 y39-61. Screen-right claw x62-85 y45-73.
  Short left foot x27-40 y66-71. Right foot, the sole, x55-70 y67-76.
  Red bands stay on the torso. No tail is drawn.
normal/back.png (bbox x4-81 y12-78, 2820 px, sole y78) has the white cheek
  mean x 64.21 and the nose mean x 66.5, so the back faces screen-right.
  Drill on the upper right, left claw x4-21, right claw x58-81, short left
  foot and a right sole at y78. The back is shifted up 2px so both soles
  share y76.

The front faces left, so the mirror rule is the reverse of a right-facing
still: S SE E NE are mirrored, N NW W SW are not.
  S SE E: front mirrored (screen-right). W SW: front as drawn (screen-left).
  N NW: back as drawn (nose on the right). NE: back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 shows that
  side's back at equal scale. No width squash and no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_excadrill_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0530_excadrill/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0530_excadrill/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0530_excadrill"
VERSION = "excadrill_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 176
OFF_X = (CELL_W - 96) // 2
OFF_Y = 36
FRONT_FOOT = 76
BACK_FOOT = 78
FOOT_Y = OFF_Y + FRONT_FOOT
# kind, mirror. Front faces screen-left. Back faces screen-right.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", True),
    "N": ("back", False), "NW": ("back", False),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Roar ceiling. The claw strike is drawn at 34 and the drill at 28, both
# above these, so finish() does not push the hit past the drawn pose.
# Frames 5-8 are cleared after finish().
PEAKS = {"drill": 22.0, "claw_l": 22.0, "claw_r": 22.0}
LIMBS = ("drill", "claw_l", "claw_r", "foot_l", "foot_r")
WEAPONS = ("drill", "claw_l", "claw_r")

INK = (16, 16, 16)
BROWN = (90, 74, 65)
DKBROWN = (65, 49, 41)
VDK = (41, 32, 24)
RED = (189, 65, 65)
DRED = (131, 65, 65)
MAROON = (131, 41, 49)
GRAY = (164, 164, 164)
LGRAY = (213, 213, 213)
MGRAY = (98, 98, 98)
WHITE = (255, 255, 255)
PINK = (238, 131, 123)
ROSE = (172, 82, 82)
BLUSH = (205, 115, 106)
METAL_COLS = (GRAY, LGRAY, MGRAY)
BODY_COLS = (BROWN, DKBROWN, VDK, RED, DRED, MAROON, PINK, ROSE, BLUSH, WHITE)
FACE_COLS = (WHITE, PINK, ROSE, BLUSH)
NOSE_COLS = (PINK, ROSE, BLUSH)

FRONT_COUNTS = {
    "drill": 370, "claw_l": 197, "claw_r": 383,
    "foot_l": 50, "foot_r": 76, "torso": 1684,
}
BACK_COUNTS = {
    "drill": 250, "claw_l": 202, "claw_r": 385,
    "foot_l": 28, "foot_r": 82, "torso": 1873,
}
FOOT_BOX = {
    "front": {"foot_l": (67, 71, 26, 42), "foot_r": (68, 76, 54, 71)},
    "back": {"foot_l": (72, 75, 24, 38), "foot_r": (72, 78, 46, 66)},
}

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


def components(mask):
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
    groups.sort(key=len, reverse=True)
    return groups


def mask_from(pts, shape):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


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
    """4x nearest in-plane rotation. Positive is clockwise. No squash."""
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
    groups = components(mask)
    if not groups:
        return 0
    return len(groups[0])


def require_piece(name, mask):
    if not mask.any():
        raise SystemExit(f"{name} is missing")
    if largest_component(mask) != int(mask.sum()):
        raise SystemExit(f"{name} fell apart ({largest_component(mask)} vs {int(mask.sum())})")


def segment(img, kind):
    """Steel drill, two drill-claws, two short feet. Face stays on the torso."""
    op = img[..., 3] >= 128
    metal = any_color(img, METAL_COLS)
    ink = is_color(img, INK)
    body = any_color(img, BODY_COLS)
    face = any_color(img, FACE_COLS)
    parts = {k: np.zeros_like(op) for k in ("drill", "claw_l", "claw_r")}
    for pts in components(metal):
        if len(pts) < 80:
            continue
        core = mask_from(pts, op.shape)
        ys, xs = np.nonzero(core)
        if float(ys.mean()) < 40.0:
            key = "drill"
        elif float(xs.mean()) < 40.0:
            key = "claw_l"
        else:
            key = "claw_r"
        allowed = ((metal & dilate(core, 4)) | (ink & dilate(core, 1))) & ~face
        parts[key] |= flood(allowed, core)
    for keep, drop in (("drill", "claw_l"), ("drill", "claw_r"), ("claw_l", "claw_r")):
        parts[drop] &= ~parts[keep]
    yy, xx = np.mgrid[0:96, 0:96]
    boxes = FOOT_BOX[kind]
    metal_now = parts["drill"] | parts["claw_l"] | parts["claw_r"]

    def foot(box):
        y0, y1, x0, x1 = box
        core = body & (yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1) & ~face
        zone = (yy >= y0) & (yy <= y1 + 1) & (xx >= x0 - 1) & (xx <= x1 + 1)
        rim = ink & zone & dilate(core, 1) & ~metal_now
        return core | rim

    parts["foot_l"] = foot(boxes["foot_l"])
    parts["foot_r"] = foot(boxes["foot_r"]) & ~parts["foot_l"]
    limb = np.zeros_like(op)
    for name in LIMBS:
        limb |= parts[name]
    torso = op & ~limb
    for pts in components(torso)[1:]:
        cols = {tuple(int(v) for v in img[y, x, :3]) for y, x in pts}
        target = None
        if cols == {INK}:
            for name in ("foot_l", "foot_r", "claw_l", "claw_r", "drill"):
                if any(dilate(parts[name], 1)[y, x] for y, x in pts):
                    target = name
                    break
        if target is None:
            raise SystemExit(f"{kind} torso island {len(pts)} at {pts[0]} cols {cols}")
        for y, x in pts:
            parts[target][y, x] = True
            torso[y, x] = False
    parts["torso"] = torso
    check_partition(op, parts)
    for name in LIMBS:
        require_piece(name, parts[name])
        if not (parts[name] & dilate(parts["torso"], 1)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")
    if largest_component(parts["torso"]) != int(parts["torso"].sum()):
        raise SystemExit(f"{kind} torso fell apart")
    white = is_color(img, WHITE)
    nose = any_color(img, NOSE_COLS)
    if int((white & parts["torso"]).sum()) != int(white.sum()) or white.sum() < 8:
        raise SystemExit(f"{kind} eyes left the head")
    if int((nose & parts["torso"]).sum()) != int(nose.sum()) or not nose.any():
        raise SystemExit(f"{kind} nose left the head")
    ex = float(np.nonzero(white)[1].mean())
    nx = float(np.nonzero(nose)[1].mean())
    if kind == "front" and not (nx < ex - 2.0):
        raise SystemExit(f"front no longer faces left: nose {nx:.2f} eyes {ex:.2f}")
    if kind == "back" and not (nx > ex + 1.0):
        raise SystemExit(f"back no longer faces right: nose {nx:.2f} eyes {ex:.2f}")
    expect = FRONT_COUNTS if kind == "front" else BACK_COUNTS
    counts = {k: int(v.sum()) for k, v in parts.items()}
    if counts != expect:
        raise SystemExit(f"{kind} parts changed {counts}")
    sole = int(np.nonzero(op)[0].max())
    want = FRONT_FOOT if kind == "front" else BACK_FOOT
    if sole != want:
        raise SystemExit(f"{kind} sole changed {sole}")
    if kind == "front" and int(np.nonzero(parts["foot_r"])[0].max()) != FRONT_FOOT:
        raise SystemExit("front sole is not the right foot")
    if kind == "back" and int(np.nonzero(parts["foot_r"])[0].max()) != BACK_FOOT:
        raise SystemExit("back sole is not the right foot")
    return parts


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
    depth = {"drill": 10, "claw_l": 8, "claw_r": 8, "foot_l": 6, "foot_r": 6}
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
    for name in WEAPONS:
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
        bdx=0, bdy=0, drill=0.0, claw_l=0.0, claw_r=0.0,
        drill_mode="up", claw_mode="up", thrust=0,
        step_l=(0, 0), step_r=(0, 0), kick=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0, turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The drill and the claws breathe.
        p["drill"] = wave(f, n, 4.0, phase=0.4)
        p["claw_l"] = wave(f, n, 3.0, phase=0.2)
        p["claw_r"] = wave(f, n, 3.0, phase=1.1)
        p["drill_mode"] = "up"
        p["claw_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step_l"] = (int(round(2 * max(0.0, sa))), -int(round(2 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * max(0.0, -sa))), -int(round(2 * max(0.0, -sa))))
        p["claw_l"] = wave(f, n, 8.0, phase=0.2)
        p["claw_r"] = wave(f, n, 8.0, phase=0.2 + math.pi)
        p["drill"] = wave(f, n, 3.0, phase=0.6)
        p["drill_mode"] = "up"
        p["claw_mode"] = "fwd"
    elif action == "attack":
        # 0 rest. 1-4 raise the drill and both claws. No mouth, so no rings.
        # 5-8 the back, at rest, equal scale. 9-12 claw strike and drill thrust.
        # 13-15 recover.
        drill = [0, 8, 14, 18, 22, 0, 0, 0, 0, 22, 26, 28, 28, 12, 4, 0]
        claw = [0, 8, 14, 18, 22, 0, 0, 0, 0, 22, 28, 34, 34, 14, 5, 0]
        bdx = [0, 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0]
        thrust = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0]
        p["drill"] = float(drill[f])
        p["claw_l"] = float(claw[f])
        p["claw_r"] = float(claw[f])
        p["bdx"] = int(bdx[f])
        p["thrust"] = int(thrust[f])
        p["drill_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["claw_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["kick"] = (1, -1)
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["drill"] = 10.0 * env
        p["claw_l"] = 8.0 * env
        p["claw_r"] = 8.0 * env
        p["drill_mode"] = "up"
        p["claw_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    # Frames 5-8 are the back still at equal scale. finish() would swing the
    # wind-up on these frames; drop it so the turned silhouette stays whole.
    if action == "attack" and 5 <= f <= 8:
        p["drill"] = 0.0
        p["claw_l"] = 0.0
        p["claw_r"] = 0.0
        p["thrust"] = 0
        p["bdx"] = 0
        p["bdy"] = 0
        p["kick"] = (0, 0)
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["spin"] = 0
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
    thrust = int(p["thrust"])
    parts = {}
    for name in LIMBS:
        if name == "foot_l":
            sx, sy = p["step_l"]
            ang = 0.0
            extra = 0
        elif name == "foot_r":
            sx, sy = p["step_r"]
            ang = 0.0
            extra = 0
        elif name == "drill":
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["drill_mode"], p["drill"])
            extra = thrust
        else:
            sx, sy = 0, 0
            ang = part_angle(rig, name, p["claw_mode"], p[name])
            extra = thrust
        if name == rig["near_foot"]:
            kx, ky = p["kick"]
            sx += kx
            sy += ky
        dx = bdx + int(round(fc * (sx + extra)))
        dy = bdy + int(sy)
        parts[name] = moved(L[name], ang, piv[name], dx, dy)
    return parts


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    parts = limb_layers(rig, p)
    for name in ("foot_l", "foot_r", "claw_l", "claw_r", "drill"):
        paste(out, parts[name])
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


def weapon_tip(rig, p):
    parts = limb_layers(rig, p)
    fc = rig["facing"]
    best = None
    best_x = None
    for name in ("claw_l", "claw_r", "drill"):
        layer = parts[name]
        if not np.any(layer[..., 3] >= 128):
            continue
        ys, xs = np.nonzero(layer[..., 3] >= 128)
        i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
        x = float(xs[i])
        if best_x is None or (fc > 0 and x > best_x) or (fc < 0 and x < best_x):
            best_x = x
            best = (x, float(ys[i]))
    if best is None:
        return rig["pivots"]["center"]
    return best


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, LGRAY, RED)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
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
        abs(p["drill"]) >= 0.45 or abs(p["claw_l"]) >= 0.45 or abs(p["claw_r"]) >= 0.45
        or abs(p["lean"]) > 0.4
    )
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
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
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


def crop_box(canvas, pad_x=28, pad_y=20):
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
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "drill": (80, 200, 255, 255),
        "claw_l": (255, 180, 40, 255),
        "claw_r": (255, 60, 60, 255),
        "foot_l": (80, 220, 80, 255),
        "foot_r": (180, 80, 255, 255),
        "torso": (170, 150, 130, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        pair = np.concatenate([img, part], axis=1)
        tiles.append(np.repeat(np.repeat(pair, 3, 0), 3, 1))
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
        "dex": 530, "form": "normal", "variant": "excadrill", "gen": "gen5",
        "dir": "monsterSprite/gen5/0530_excadrill/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_stats(img, pal):
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    a = img[..., 3] == 255
    edge = bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())
    off = set()
    if a.any():
        for c in np.unique(img[a][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in pal:
                off.add(t)
    return partial, edge, off


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
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    # Attack 5-8 is that side's back still, same pixels, equal scale.
    turn_of = {"S": "N", "SE": "N", "E": "N", "W": "NE", "SW": "NE"}
    for d, back_d in turn_of.items():
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

    pal = {tuple(int(v) for v in c) for c in palette}
    if args.samples:
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
        partial = 0
        edge = []
        off = set()
        bodies = []
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    part_n, hit, bad = frame_stats(img, pal)
                    partial += part_n
                    off |= bad
                    if hit:
                        edge.append(f"{act}:{d}:{f}")
                    if act in ("attack", "hurt", "walk"):
                        bodies.append(largest_fraction(img))
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad_limbs = limb_lost(rig, p)
                    if bad_limbs:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad_limbs)}")
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["claw_l"], 1), round(p["drill"], 1), p["claw_mode"], p["turned"], p["spin"]))
        hurt = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("attack poses", atk, flush=True)
        print("hurt lean", hurt, "max", max(hurt), flush=True)
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        print(
            "samples_qa",
            "rest_diff", rest_diff,
            "off_palette_colors", len(off),
            "partial_alpha", partial,
            "edge_touching", edge[:12],
            "edge_n", len(edge),
            "body_main_min", round(min(bodies), 3) if bodies else 1,
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
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    box = crop_box(rigs["S"]["canvas"])
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
    motion_z = np.repeat(np.repeat(motion, 2, 0), 2, 1)
    Image.fromarray(motion_z, "RGBA").save(OUT_DIR / "preview_motion.png")
    preview_image(sheets).save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    shutil.copy2(OUT_DIR / "preview.png", OUT_DIR / "actions" / "preview.png")

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
        "generator": "forge_gen5_excadrill_actions", "version": VERSION, "key": KEY,
        "dex": 530, "variant": "excadrill", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped mole (head drill, two drill-claws, two short feet, no mouth, no tail)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front nose mean x 30.23 is left of the eye whites 35.15, so the still faces screen-left; back nose 66.5 is right of the cheek whites 64.21",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
