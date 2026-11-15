#!/usr/bin/env python3
"""Crustle normal (0558) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x13-82 y16-78, 3252 px, 13 colours,
alpha 0/255, sole y78):
  Two equal eyes under the rock, each 13 px. Left eye mean x 33.69 (x32-35),
  right eye mean x 42.38 (x41-44). Combined eye mean x 38.04.
  The gray mouth mark below the eyes (y>=62) has mean x 31.91, left of the
  eyes, so the still faces screen-left. There is no separate open-mouth
  colour, so the roar does not invent a wedge or a ring. It opens the claws.
  The light shell face (172, 156, 131) sits left of the brown mass. The back
  still's light rim is on the right, which matches a left-facing front.
  Near claw is the orange pincer on the screen-left (x16-32). Far claw is the
  longer pincer on the screen-right, including the rust fringe that only
  touches that pincer. Center legs reach the sole at x50 y78. The near-side
  leg column is x13-28. No wings, horns, or tail are drawn.
normal/back.png (bbox x11-85 y18-77, 3132 px, 9 colours, all in the front
  palette). Sole is one pixel higher, so the cell drops it 1 px onto y78.
  No eyes and no mouth. One pincer on the right (mean x 76). Leg tips split
  below y72. The back as drawn faces away to the right.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  Attack 5-8 is that direction's back at equal scale. W turns into N.
  S turns into N flipped. No |cos| squash and no in-plane spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_crustle_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0558_crustle/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0558_crustle/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0558_crustle"
VERSION = "crustle_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 224, 224
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 78
BACK_FOOT = 78
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Its back, at equal scale, is N.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# finish() may lift a claw this far. The snap is drawn at 34, above this.
PEAKS = {"claw_l": 16.0, "claw_r": 16.0}
LIMBS = ("claw_l", "claw_r", "leg_l", "leg_c", "leg_r")

INK = (16, 16, 16)
DARK = (57, 49, 32)
GRAY = (57, 57, 57)
DBROWN = (90, 65, 49)
DRED = (106, 41, 24)
BROWN = (139, 106, 74)
BEIGE = (172, 156, 131)
RUST = (189, 82, 49)
GOLD = (197, 139, 24)
ORANGE = (238, 131, 74)
PALE = (255, 189, 123)
YELLOW = (246, 189, 57)
WHITE = (255, 255, 255)
ROCK_COLS = (BROWN, BEIGE, DBROWN, YELLOW, GOLD)
CLAW_COLS = (ORANGE, PALE, RUST, DRED)

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


def flood(mask, seeds):
    out = seeds & mask
    while True:
        nxt = dilate(out, 1) & mask
        if (nxt == out).all():
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


def largest_component(mask):
    groups = components(mask)
    if not groups:
        return 0
    return len(groups[0])


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
    """4x nearest in-plane rotation. Positive is clockwise on screen."""
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
    dist = ((px[:, None, :] - palette.astype(np.int32)[None, :, :]) ** 2).sum(axis=2)
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
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            grown = np.zeros_like(src)
            y0, y1 = max(0, dy), src.shape[0] + min(0, dy)
            x0, x1 = max(0, dx), src.shape[1] + min(0, dx)
            grown[y0:y1, x0:x1] = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            take = (acc[..., 3] < 128) & allowed & (grown[..., 3] >= 128)
            acc[take] = grown[take]
    return acc


def wave(f, n, amp, phase=0.0, cycles=1.0):
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


def mask_from_pts(shape, pts):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


def absorb_outline(img, core, rock, claw_col):
    acc = core.copy()
    extra = is_color(img, INK) | is_color(img, DARK) | is_color(img, WHITE) | is_color(img, PALE)
    for _ in range(5):
        take = extra & dilate(acc, 1) & ~dilate(rock | (claw_col & ~acc), 1)
        if not take.any():
            break
        acc |= take
    return acc


def split_legs(leg):
    out = {name: np.zeros_like(leg) for name in ("leg_l", "leg_c", "leg_r")}
    for pts in components(leg):
        xs = np.array([p[1] for p in pts])
        mx = float(xs.mean())
        name = "leg_l" if mx < 36 else ("leg_r" if mx >= 60 else "leg_c")
        out[name] |= mask_from_pts(leg.shape, pts)
    return out


def segment(img, kind):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    claw_col = any_color(img, CLAW_COLS)
    rock = any_color(img, ROCK_COLS)
    if kind == "front":
        left = claw_col & (xx <= 32) & (yy >= 56)
        right = claw_col & (xx >= 46) & (yy >= 56)
        y_band, y_seed = 69, 72
    else:
        left = np.zeros_like(op)
        right = claw_col & (xx >= 68) & (yy >= 54)
        y_band, y_seed = 72, 73
    claw_l = absorb_outline(img, left, rock, claw_col)
    claw_r = absorb_outline(img, right, rock, claw_col)
    band = op & (yy >= y_band) & ~claw_l & ~claw_r
    leg = flood(band, band & (yy >= y_seed))
    leg |= band & dilate(leg, 1)
    legs = split_legs(leg)
    torso = op & ~(claw_l | claw_r | legs["leg_l"] | legs["leg_c"] | legs["leg_r"])
    # Rust fringe that never meets the shell rides on the far pincer.
    if legs["leg_r"].any() and not (legs["leg_r"] & dilate(torso, 2)).any():
        if (dilate(legs["leg_r"], 2) & claw_r).any():
            claw_r = claw_r | legs["leg_r"]
            legs["leg_r"] = np.zeros_like(claw_r)
            torso = op & ~(claw_l | claw_r | legs["leg_l"] | legs["leg_c"] | legs["leg_r"])
    masks = {
        "claw_l": claw_l, "claw_r": claw_r,
        "leg_l": legs["leg_l"], "leg_c": legs["leg_c"], "leg_r": legs["leg_r"],
        "torso": torso,
    }
    check_partition(op, masks)
    if kind == "front":
        eye = (is_color(img, YELLOW) | is_color(img, WHITE) | is_color(img, GRAY))
        eye = eye & (yy >= 56) & (yy <= 62) & (xx >= 28) & (xx <= 48)
        mouth = is_color(img, GRAY) & (yy >= 62)
        if int(eye.sum()) < 20 or int((eye & torso).sum()) != int(eye.sum()):
            raise SystemExit("front eyes left the head")
        ex = float(np.nonzero(eye)[1].mean())
        mx = float(np.nonzero(mouth)[1].mean())
        if not (mx < ex - 4):
            raise SystemExit(f"front facing changed: mouth {mx:.2f} eye {ex:.2f}")
        left_eye = eye & (xx <= 38)
        right_eye = eye & (xx >= 39)
        if abs(int(left_eye.sum()) - int(right_eye.sum())) > 2:
            raise SystemExit("front eyes are no longer a pair")
        chin = is_color(img, ORANGE) & (xx >= 34) & (xx <= 45) & (yy >= 63)
        if int((chin & torso).sum()) != int(chin.sum()) or int(chin.sum()) < 10:
            raise SystemExit("front jaw left the head")
        if not masks["leg_c"][FRONT_FOOT].any():
            raise SystemExit("front sole left the center leg")
        if int(claw_l.sum()) < 70 or int(claw_r.sum()) < 140:
            raise SystemExit(f"front claws look short: {int(claw_l.sum())} {int(claw_r.sum())}")
    else:
        face = any_color(img, (YELLOW, WHITE, GRAY, PALE))
        if face.any():
            raise SystemExit("back gained a face colour")
        if int(claw_r.sum()) < 100:
            raise SystemExit(f"back pincer looks short: {int(claw_r.sum())}")
        if not masks["leg_c"][BACK_FOOT].any():
            raise SystemExit("back sole left the center leg")
        beige = is_color(img, BEIGE)
        if not beige.any() or float(np.nonzero(beige)[1].mean()) < 60:
            raise SystemExit("back light rim is not on the right")
    for name in ("claw_l", "claw_r", "leg_l", "leg_c", "leg_r"):
        m = masks[name]
        if not m.any():
            continue
        if not (m & dilate(torso, 2)).any():
            raise SystemExit(f"{kind} {name} does not touch the shell")
    if largest_component(torso) < int(torso.sum()) * 0.9:
        raise SystemExit(f"{kind} shell fell apart")
    return masks


def place(arr):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[OFF_Y:OFF_Y + arr.shape[0], OFF_X:OFF_X + arr.shape[1]] = arr
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
        if len(xs) == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ty, tx = np.nonzero(torso)
    cx, cy = float(tx.mean()), float(ty.mean())
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    near = d <= d.min() + 8.0
    return float(xs[near].mean()), float(ys[near].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing):
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
    for name, depth in (("claw_l", 6), ("claw_r", 6), ("leg_l", 4), ("leg_c", 4), ("leg_r", 4)):
        if int(placed[name].sum()) == 0:
            continue
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth)
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        if int((layers[name][..., 3] >= 128).sum()) == 0:
            continue
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        claw_l=0.0, claw_r=0.0, mode="up",
        leg_l=(0, 0), leg_c=(0, 0), leg_r=(0, 0),
        lean=0.0, knock=0, flash=False,
        turned=False, spin=0, spark=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        sway = wave(f, n, 4.0, phase=0.5)
        p["claw_l"] = sway
        p["claw_r"] = sway * 0.75
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.2) > 0.55 else 0
        p["mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["leg_l"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["leg_r"] = (int(round(2 * sa)), -int(round(3 * max(0.0, sa))))
        p["leg_c"] = (int(round(2 * sb)), -int(round(3 * max(0.0, sb))))
        sway = wave(f, n, 6.0, phase=0.4)
        p["claw_l"] = sway
        p["claw_r"] = sway * 0.75
        p["bdy"] = -int(round(0.9 * abs(math.sin(u))))
        p["mode"] = "up"
    elif action == "attack":
        # 0 still. 1-4 the claws open. 5-8 the back, cleared after finish().
        # 9 wind-up. 10-12 the pincers snap. 13-15 return. 15 is the still.
        near = [0, 8, 14, 18, 22, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        far = [0, 8, 12, 16, 18, 0, 0, 0, 0, 14, 22, 28, 28, 10, 4, 0]
        p["claw_l"] = float(near[f])
        p["claw_r"] = float(far[f])
        p["mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        # The back still, at equal scale. finish() must not swing it.
        p["claw_l"] = 0.0
        p["claw_r"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["leg_l"] = (0, 0)
        p["leg_c"] = (0, 0)
        p["leg_r"] = (0, 0)
        p["spark"] = 0
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["mode"] = "up"
        p["turned"] = True
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


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("leg_l", "leg_c", "leg_r"):
        if int((L[name][..., 3] >= 128).sum()) == 0:
            continue
        sx, sy = p[name]
        # Feet stay on the sole. They do not rise with the shell's breath.
        paste(out, shift(L[name], int(round(fc * sx)), int(sy)))
    for name in ("claw_l", "claw_r"):
        if int((L[name][..., 3] >= 128).sum()) == 0:
            continue
        ang = part_angle(rig, name, p["mode"], p[name])
        paste(out, moved(L[name], ang, piv[name], bdx, bdy))
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


def claw_tip(rig, p, name):
    L = rig["layers"]
    fc = rig["facing"]
    if name not in rig["pivots"] or int((L[name][..., 3] >= 128).sum()) == 0:
        return rig["pivots"]["center"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    ang = part_angle(rig, name, p["mode"], p[name])
    layer = moved(L[name], ang, rig["pivots"][name], bdx, bdy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (YELLOW, WHITE, GOLD, ORANGE)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (r // 2, -3), (r // 2, 3))):
        put(img, ox + fc * dx, oy + dy, cols[i % 4])


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    _kind, mirror = VIEW_OF[d]
    return rigs[("back", mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["claw_l"]) >= 0.45 or abs(p["claw_r"]) >= 0.45
        or p["leg_l"] != (0, 0) or p["leg_c"] != (0, 0) or p["leg_r"] != (0, 0)
        or p["bdx"] or p["bdy"]
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
    if moved_part or abs(p["lean"]) > 0.4 or p["knock"]:
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    else:
        img = rig["canvas"].copy()
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = claw_tip(rig, p, "claw_l")
        impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def limb_lost(rig, p):
    body = compose(rig, p)
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    lost = []
    for name in LIMBS:
        if int((L[name][..., 3] >= 128).sum()) == 0:
            continue
        if name in ("leg_l", "leg_c", "leg_r"):
            sx, sy = p[name]
            layer = shift(L[name], int(round(fc * sx)), int(sy))
        else:
            ang = part_angle(rig, name, p["mode"], p[name])
            layer = moved(L[name], ang, rig["pivots"][name], bdx, bdy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


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


def tile_grid(tiles, cols, bg=(15, 20, 31, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def content_box(canvases, pad=6):
    acc = np.zeros(canvases[0].shape[:2], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(acc.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(acc.shape[1], int(xs.max()) + 1 + pad)
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


def locked_palette(front_path, back_path, work):
    pal_dir = work / "palette"
    lock_dir = work / "palette_lock"
    imgs = [np.array(Image.open(p).convert("RGBA")) for p in (front_path, back_path)]
    own = []
    for img in imgs:
        cols = np.unique(img[img[..., 3] >= 128][:, :3], axis=0)
        own.extend(tuple(int(v) for v in c) for c in cols)
    own = sorted(set(own))
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


def align_back(back):
    """The back sole is y77. Drop it 1 px onto the front sole. The png is not written."""
    if int(np.nonzero(back[..., 3])[0].max()) != 77:
        raise SystemExit("back sole is not y77")
    if back[-1].any():
        raise SystemExit("back has no empty row to shift into")
    out = np.zeros_like(back)
    out[1:] = back[:-1]
    return out


def build_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = align_back(np.array(Image.open(SRC_DIR / "back.png").convert("RGBA")))
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 stills")
    for im in (front, back):
        im[im[..., 3] < 128] = 0
        im[im[..., 3] >= 128, 3] = 255
    if int(np.nonzero(front[..., 3])[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "claw_l": (220, 80, 40, 255),
        "claw_r": (255, 180, 40, 255),
        "leg_l": (40, 200, 90, 255),
        "leg_c": (40, 160, 220, 255),
        "leg_r": (180, 80, 220, 255),
        "torso": (170, 170, 170, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 3, 0), 3, 1))
    Image.fromarray(np.concatenate(tiles, axis=1), "RGBA").save(work / "parts.png")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    preview = OUT_DIR / "actions" / "preview.png"
    if preview.is_file():
        dest = SYNC_OUT / "actions"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preview, dest / "preview.png")
        n += 1
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 558, "form": "normal", "variant": "crustle", "gen": "gen5",
        "dir": "monsterSprite/gen5/0558_crustle/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_metrics(rigs, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    clip = []
    bodies = []
    for d in DIRS:
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                t = synth(rigs, d, act, f, palette)
                a = t[..., 3]
                partial += int(((a != 0) & (a != 255)).sum())
                op = a == 255
                if op.any():
                    used = {tuple(int(v) for v in c) for c in np.unique(t[op][:, :3], axis=0)}
                    off |= used - pal
                if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
                if act in ("attack", "hurt", "walk"):
                    bodies.append(largest_fraction(t))
    return {
        "off_palette": len(off),
        "partial_alpha": partial,
        "edge_touch": len(clip),
        "edge_touching": clip[:8],
        "body_main_min": round(min(bodies), 3) if bodies else 1,
    }


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
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)

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

    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    turn_n = synth(rigs, "N", "attack", 6, palette, effects=False)
    turn_e = synth(rigs, "E", "attack", 6, palette, effects=False)
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(turn_s != rigs["NW"]["canvas"]):
        raise SystemExit("S turn is not N flipped at equal scale")
    if np.any(turn_n != rigs["N"]["canvas"]):
        raise SystemExit("N turn left the back still")
    if np.any(turn_e != rigs["NW"]["canvas"]):
        raise SystemExit("E turn is not the mirrored back")

    if args.samples:
        lost = {}
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
        print("limb lost", lost or "none", flush=True)
        metrics = frame_metrics(rigs, palette)
        print("qa_samples", "rest_diff", int(sum(rest_diff.values())),
              "off_palette", metrics["off_palette"],
              "partial_alpha", metrics["partial_alpha"],
              "edge_touch", metrics["edge_touch"],
              "body_main_min", metrics["body_main_min"],
              "clip", metrics["edge_touching"], flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(float(p["claw_l"]), 1), round(float(p["claw_r"]), 1),
                        p["mode"], p["turned"], p["spin"], p["spark"]))
        print("attack poses", atk, flush=True)
        leans = [round(float(pose("hurt", f)["lean"]), 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 11),
                ("hurt", 3),
            ):
                tiles.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
        sheet = np.repeat(np.repeat(tile_grid(tiles, 9), 2, 0), 2, 1)
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", flush=True)
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
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    box = content_box([rigs[d]["canvas"] for d in DIRS])
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.repeat(np.repeat(np.concatenate(strips, axis=0), 2, 0), 2, 1)
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
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("qa_full", "rest_diff", int(sum(rest_diff.values())),
          "off_palette", len(off),
          "partial_alpha", partial,
          "edge_touch", len(clip),
          "body_main_min", report["body_main_min"],
          "idle0", idle0, flush=True)
    meta = {
        "generator": "forge_gen5_crustle_actions", "version": VERSION, "key": KEY,
        "dex": 558, "variant": "crustle", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "stone-shell crab (paired eyes, two claws, near and center legs, no open mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth-mark mean x 31.91 < eye mean x 38.04)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
