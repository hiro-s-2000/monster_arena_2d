#!/usr/bin/env python3
"""Zoroark normal (0571) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x15-82 y18-81, 2053 px, alpha 0/255,
13 colours, foot y81):
  The face eye is the cyan/teal/silver cluster at x27-28 y37-39 (mean x 27.5).
  The grey muzzle's leftmost pixel in y28-44 is x20, left of the eye, so the
  still faces screen-left. The red mane trails to the right (x82).
  No mouth is drawn. The single pink pixel (205, 74, 90) at (29, 74) is the
  near claw, so the roar does not invent a wedge or rings. It lifts the two
  mane tips that are already on the head, and cocks that claw.
  Near mane tip x25-27 y21-24. Far mane tip x43-48 y19-22.
  Near claw arm x26-42 y62-75 (does not reach the ground).
  Planted foot x49-66 y63-81, three red toes at y80-81.
  Trailing mane x66-82 y50-64.
normal/back.png (bbox x7-80 y14-78, 2250 px, 11 colours, foot y78) faces
screen-right: head eye x66-68 y33-35, head tip x73, mane trails to x7.
  Crest spike x49-55 y14-18. Mane tail x7-32 y51-68. Rear claw x28-37 y66-74.
  Planted foot x53-70 y66-78. The back is shifted down 3px so both ground
  lines match. Both stills share the front's 13-colour palette.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (head screen-right). NW back mirrored (head screen-left).
  E is screen-right, W is screen-left, N is the back. Attack 5-8 shows that
  direction's back at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_zoroark_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0571_zoroark/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0571_zoroark/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0571_zoroark"
VERSION = "zoroark_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 224, 224
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 32
FRONT_FOOT = 81
BACK_FOOT = 78
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Back as drawn looks screen-right.
# S (front mirrored) turns into N (back as drawn): same facing, equal scale.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Wind-up peaks finish() may raise on frames 5-12. The claw and mane strike
# is drawn higher than these, so the impact frames stay as authored.
PEAKS = {"crest": 12.0, "mane": 14.0, "claw": 12.0}

INK = (16, 16, 16)
NAVY = (32, 32, 49)
DKBLUE = (41, 49, 65)
GREY = (74, 74, 82)
FACE = (98, 98, 106)
LGRAY = (123, 123, 131)
MAROON = (123, 41, 65)
RED = (172, 16, 65)
DKRED = (74, 16, 41)
CYAN = (24, 180, 180)
TEAL = (49, 131, 131)
SILVER = (205, 205, 205)
PINK = (205, 74, 90)
EYE_COLS = (CYAN, TEAL, SILVER)

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


def reconnect(img):
    """Close a 1px gap with the island's own colour. Loose specks stay loose."""
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
    delta = px[:, None, :] - pal[None, :, :]
    dist = (delta * delta).sum(axis=2, dtype=np.int32)
    out[op, :3] = palette[dist.argmin(axis=1)]
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


def largest_component(mask):
    groups = component_list(mask)
    if not groups:
        return 0
    return max(len(g) for g in groups)


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


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


def require_part(masks, torso, name):
    part = masks[name]
    if int(part.sum()) < 8:
        raise SystemExit(f"{name} is missing")
    if largest_component(part) != int(part.sum()):
        raise SystemExit(f"{name} fell apart")
    if not touches(part, torso):
        raise SystemExit(f"{name} does not touch the body")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    # Two mane tips only. The face, the eye, and the hood stay on the torso.
    crest_a = flood(op & (yy <= 24) & (xx <= 30), op & (yy <= 22) & (xx <= 28))
    crest_b = flood(op & (yy <= 22) & (xx >= 42) & (xx <= 52), op & (yy <= 20) & (xx >= 44))
    # Near claw. It ends above the ground. The pink pixel is this claw, not a mouth.
    arm = flood(op & (yy >= 62) & (xx <= 43), op & (yy >= 74) & (xx <= 36))
    leg = flood(op & (yy >= 63) & (xx >= 45) & (xx <= 68), op & (yy >= 80))
    # Trailing mane on the far side. The chest glint at x63-64 stays on the torso.
    mane = flood(
        op & (yy >= 48) & (yy <= 68) & (xx >= 66) & ~leg & ~arm,
        op & (yy >= 53) & (yy <= 57) & (xx >= 78),
    )
    used = crest_a | crest_b | arm | leg | mane
    torso = op & ~used
    masks = {
        "crest_a": crest_a, "crest_b": crest_b, "arm": arm,
        "leg": leg, "mane": mane, "torso": torso,
    }
    check_partition(op, masks)
    eye = any_color(img, EYE_COLS) & (yy >= 36) & (yy <= 41) & (xx <= 40)
    if int(eye.sum()) != 4 or int((eye & torso).sum()) != 4:
        raise SystemExit("front eye left the face")
    ex = float(np.nonzero(eye)[1].mean())
    face = is_color(img, FACE) & (yy >= 28) & (yy <= 44)
    tip = int(np.nonzero(face)[1].min())
    if not tip < ex - 4:
        raise SystemExit(f"front does not face left: muzzle {tip} eye {ex:.1f}")
    glint = any_color(img, EYE_COLS) & (xx >= 60)
    if (glint & mane).any():
        raise SystemExit("chest glint was cut into the mane")
    pink = is_color(img, PINK)
    if int(pink.sum()) != 1 or int((pink & arm).sum()) != 1:
        raise SystemExit("pink claw pixel left the arm")
    if int(np.nonzero(leg)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot moved")
    if int(np.nonzero(arm)[0].max()) >= FRONT_FOOT:
        raise SystemExit("the claw arm was treated as a foot")
    for name in ("crest_a", "crest_b", "arm", "leg", "mane"):
        require_part(masks, torso, name)
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    crest = flood(op & (yy <= 18) & (xx >= 46) & (xx <= 58), op & (yy <= 16))
    hair = flood(op & (yy >= 51) & (xx <= 32), op & (yy >= 56) & (xx <= 16))
    arm = flood(op & (yy >= 66) & (xx >= 26) & (xx <= 42), op & (yy >= 71) & (xx <= 38) & (xx >= 28))
    foot = flood(op & (yy >= 66) & (xx >= 50), op & (yy >= 77))
    used = crest | hair | arm | foot
    torso = op & ~used
    masks = {"crest": crest, "hair": hair, "arm": arm, "foot": foot, "torso": torso}
    check_partition(op, masks)
    eye = any_color(img, EYE_COLS) & (yy <= 40)
    if int(eye.sum()) < 4 or int((eye & torso).sum()) != int(eye.sum()):
        raise SystemExit("back eye left the head")
    ex = float(np.nonzero(eye)[1].mean())
    hx = float(np.nonzero(hair)[1].mean())
    head = op & (yy <= 36) & (xx >= 40)
    tip = int(np.nonzero(head)[1].max())
    if not (hx < ex - 20 and tip > ex):
        raise SystemExit(f"back does not face right: mane {hx:.1f} eye {ex:.1f} head {tip}")
    if int(np.nonzero(foot)[0].max()) != BACK_FOOT:
        raise SystemExit("back foot moved")
    if int(np.nonzero(arm)[0].max()) >= BACK_FOOT:
        raise SystemExit("back claw reached the ground line")
    for name in ("crest", "hair", "arm", "foot"):
        require_part(masks, torso, name)
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("back torso fell apart")
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
    d = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
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
    if kind == "front":
        depths = {"crest_a": 3, "crest_b": 3, "mane": 5, "arm": 4, "leg": 4}
        order = ("torso", "leg", "mane", "crest_b", "arm", "crest_a")
        roles = {
            "crest_a": "crest", "crest_b": "crest", "mane": "mane",
            "arm": "claw", "leg": "foot",
        }
        facing = 1.0 if mirror else -1.0
    else:
        depths = {"crest": 3, "hair": 5, "arm": 4, "foot": 4}
        order = ("torso", "hair", "foot", "arm", "crest")
        roles = {"crest": "crest", "hair": "mane", "arm": "claw", "foot": "foot"}
        facing = -1.0 if mirror else 1.0
    for name, depth in depths.items():
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth)
    pivots = {}
    signs = {}
    for name in roles:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas,
        "order": order, "roles": roles,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        crest=0.0, mane=0.0, claw=0.0,
        claw_mode="up", mane_mode="up",
        step_a=(0, 0), step_b=(0, 0),
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The mane tips and the trailing hair breathe.
        p["crest"] = wave(f, n, 6.0, phase=0.4)
        p["mane"] = wave(f, n, 8.0, phase=0.9)
        p["mane_mode"] = "up"
    elif action == "walk":
        # The drawn claw and the one planted foot, opposite phase.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_a"] = (int(round(2.2 * sa)), -int(round(2.4 * max(0.0, sa))))
        p["step_b"] = (int(round(2.4 * sb)), -int(round(3.0 * max(0.0, sb))))
        p["crest"] = wave(f, n, 7.0)
        p["mane"] = wave(f, n, 10.0, phase=0.7)
        p["mane_mode"] = "up"
        p["claw_mode"] = "up"
    elif action == "attack":
        # 0 still. 1-4 mane tips and claw (no mouth). 5-8 the matching back.
        # 9-12 the claw and the trailing mane strike. 13-15 recover. 15 is the still.
        crest = [0, 8, 14, 18, 20, 12, 10, 8, 6, 8, 10, 8, 6, 3, 0, 0]
        mane = [0, 6, 12, 16, 18, 10, 8, 8, 8, 16, 26, 32, 28, 12, 4, 0]
        claw = [0, 4, 8, 12, 14, 6, 4, 4, 8, 18, 28, 34, 30, 12, 4, 0]
        bdx = [0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["crest"] = float(crest[f])
        p["mane"] = float(mane[f])
        p["claw"] = float(claw[f])
        p["bdx"] = int(bdx[f])
        p["mane_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["claw_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
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


def limb_delta(rig, p, name):
    role = rig["roles"][name]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    if role == "foot":
        sx, sy = p["step_b"]
        return 0.0, bdx + int(round(fc * sx)), bdy + int(sy)
    if role == "claw":
        sx, sy = p["step_a"]
        ang = part_angle(rig, name, p["claw_mode"], p["claw"])
        return ang, bdx + int(round(fc * sx)), bdy + int(sy)
    if role == "mane":
        ang = part_angle(rig, name, p["mane_mode"], p["mane"])
        return ang, bdx, bdy
    ang = part_angle(rig, name, "up", p["crest"])
    return ang, bdx, bdy


def compose(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    for name in rig["order"]:
        if name == "torso":
            paste(out, shift(L["torso"], bdx, bdy))
            continue
        ang, dx, dy = limb_delta(rig, p, name)
        paste(out, moved(L[name], ang, piv[name], dx, dy))
    return out


def flash_silver(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(SILVER, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def strike_tip(rig, p):
    fc = rig["facing"]
    name = None
    for n, role in rig["roles"].items():
        if role == "claw":
            name = n
            break
    if name is None:
        return rig["pivots"]["center"]
    ang, dx, dy = limb_delta(rig, p, name)
    layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (RED, SILVER, INK)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -r // 2), (2, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def limb_layers(rig, p):
    parts = {}
    for name in rig["roles"]:
        ang, dx, dy = limb_delta(rig, p, name)
        parts[name] = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
    return parts


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    for name, layer in limb_layers(rig, p).items():
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    """Attack 5-8 shows the back at the same screen facing. Equal scale, no squash."""
    kind, mirror = VIEW_OF[d]
    if p["turned"] and kind == "front":
        return rigs[("back", not mirror)]
    return rigs[d]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["crest"]) >= 0.45 or abs(p["mane"]) >= 0.45 or abs(p["claw"]) >= 0.45
        or p["bdx"] or p["bdy"]
        or p["step_a"] != (0, 0) or p["step_b"] != (0, 0)
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if moved_part or abs(p["lean"]) > 0.4:
        body = reconnect(fill_pinholes(despeckle(body)))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_silver(img)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = strike_tip(rig, p)
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
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "crest_a": (220, 40, 40, 255), "crest_b": (255, 150, 40, 255),
        "crest": (220, 40, 40, 255), "mane": (240, 220, 40, 255),
        "hair": (240, 220, 40, 255), "arm": (40, 180, 255, 255),
        "leg": (180, 70, 220, 255), "foot": (180, 70, 220, 255),
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
        tiles.append(np.repeat(np.repeat(part[y0:y1, x0:x1], 3, 0), 3, 1))
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
        "dex": 571, "form": "normal", "variant": "zoroark", "gen": "gen5",
        "dir": "monsterSprite/gen5/0571_zoroark/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def audit_synth(rigs, palette):
    """Count the four gates over every frame. Does not read preview images."""
    pal = {tuple(int(v) for v in c) for c in palette}
    still = 0
    off = 0
    partial = 0
    edge = []
    opaque = {}
    for d in DIRS:
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                img = synth(rigs, d, act, f, palette, effects=True)
                if f == 0:
                    still += int(np.any(img != rigs[d]["canvas"], axis=2).sum())
                    bare = synth(rigs, d, act, f, palette, effects=False)
                    still += int(np.any(bare != rigs[d]["canvas"], axis=2).sum())
                a = img[..., 3]
                partial += int(((a != 0) & (a != 255)).sum())
                op = a == 255
                if op.any():
                    used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
                    off += len(used - pal)
                if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
                    edge.append(f"{act}:{d}:{f}")
                if d == "S" and act == "attack" and f in (0, 6, 11):
                    opaque[f] = int(op.sum())
    return still, off, partial, edge, opaque


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
    print("facing", facing, flush=True)
    turns = {}
    p6 = pose("attack", 6)
    for d in DIRS:
        rig = rig_for(rigs, d, p6)
        turns[d] = (rig["kind"], bool(rig["mirror"]), rig["facing"])
    print("turn6", turns, "spin", p6["spin"], flush=True)

    if args.samples:
        still, off, partial, edge, opaque = audit_synth(rigs, palette)
        print("still_diff", still, flush=True)
        print("off_palette", off, flush=True)
        print("partial_alpha", partial, flush=True)
        print("edge_touch", len(edge), edge[:12], flush=True)
        print("opaque S attack", opaque, flush=True)
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    if abs(p["lean"]) > 0.4:
                        continue
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        print("disconnected", lost[:24], "count", len(lost), flush=True)
        atk = []
        leans = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["crest"], 1), round(p["mane"], 1), round(p["claw"], 1),
                        p["turned"], p["spin"], p["bdx"], p["mane_mode"], p["claw_mode"]))
        for f in range(12):
            leans.append(round(pose("hurt", f)["lean"], 2))
        print("attack poses", atk, flush=True)
        print("hurt lean", leans, "max", max(leans), flush=True)
        box = crop_box(rigs["S"]["canvas"])
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
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            fresh = key not in state["done"] or args.force
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if not fresh and fp.exists() and fp.stat().st_size > 0:
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
    box = crop_box(rigs["S"]["canvas"])
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
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                a = t[..., 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
    import gen5_publish
    bounds = gen5_publish.measure_idle_bounds(OUT_DIR / "sheet.png", CELL_W, CELL_H, rows=8)
    sprite_h = max(bounds[i + 3] for i in range(0, 32, 4))
    idle0 = {}
    still_diff = 0
    for di, d in enumerate(DIRS):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        idle0[d] = int(np.any(t != rigs[d]["canvas"], axis=2).sum())
        still_diff += idle0[d]
        for act in ACTIONS:
            t0 = sheets[act][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
            still_diff += int(np.any(t0 != rigs[d]["canvas"], axis=2).sum())
    report = {
        "version": VERSION, "hash": digest, "idle0": idle0,
        "still_diff": still_diff,
        "off_palette": len(off), "partial_alpha": partial,
        "edge_touch": len(clip), "edge": clip[:12],
        "sprite_h": sprite_h, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = still_diff == 0 and not off and partial == 0 and not clip
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("still_diff", still_diff, flush=True)
    print("off_palette", len(off), flush=True)
    print("partial_alpha", partial, flush=True)
    print("edge_touch", len(clip), clip[:12], flush=True)
    meta = {
        "generator": "forge_gen5_zoroark_actions", "version": VERSION, "key": KEY,
        "dex": 571, "variant": "zoroark", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped fox (two mane tips, trailing mane, one claw arm, one planted foot; no mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (muzzle x20 < eye x27.5); back mane trails left of the head eye",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
