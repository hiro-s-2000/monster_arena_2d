#!/usr/bin/env python3
"""Dwebble normal (0557) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x30-67 y31-63, 887 px, 12 colours,
alpha 0/255):
  Faces screen-left. Two eye whites at (36,38), (36,39), (44,39), (44,40),
  mean x 40.0. The near claw tip reaches x30, left of the eyes, so the face
  points left. No mouth colour is drawn (the face is solid orange between the
  eye stalks), so the roar does not invent a wedge or a ring. It raises the
  two claws the still already has.
  Near claw (peach x33-35 y48-50) is the left pincer, x30-39 y47-61.
  Far claw (peach x47-49 y49-52) is the right pincer, x47-55 y44-56.
  Three walking feet under the body: leg_a x37-40, leg_b x43-48 (the toe at
  y63), leg_c x49-53. The rock shell and both eye stalks stay on the torso.
normal/back.png (bbox x28-66 y33-62, 872 px, 11 colours, no white) shows the
  shell. The peach pincer is x62-63 y50-51, right of the shell, so the back
  as drawn faces screen-right. The outer pincer tip is the claw. Two feet
  hang under the right side. Lowest back pixels sit one row higher than the
  front toe, so the back is shifted down 1 px and both soles share y63.

Directions:
  S SE E front mirrored (forward is screen-right). W SW front as drawn.
  N NE back as drawn. NW back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 is that back at equal scale: W and SW show N, S SE E show N
  flipped. Back rows stay the back. No |cos| squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_dwebble_actions.py
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen5_attack_strike import finish, save_peaks

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen5/0557_dwebble/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0557_dwebble/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0557_dwebble"
VERSION = "dwebble_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 63
BACK_FOOT = 62
OFF_Y = CELL_H - 96 - (FRONT_FOOT - BACK_FOOT)
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. A 180 yaw keeps the mirror flag:
# W (front, False) shows N (back, False); S (front, True) shows NW.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
STRIKE_MAX = 34.0
# Wind-up peaks finish() may raise on frames 5-12. The claw strike is drawn
# at 34 and 28, above these, so finish does not pass the drawn strike.
PEAKS = {"claw_l": 20.0, "claw_r": 16.0}

INK = (16, 16, 16)
PUPIL = (65, 65, 65)
BROWN = (98, 65, 41)
MIDBROWN = (148, 98, 74)
ORANGE_D = (197, 98, 32)
ORANGE = (238, 156, 74)
PEACH = (255, 189, 123)
WHITE = (255, 255, 255)
CRAB_COLS = (ORANGE, ORANGE_D, BROWN, PEACH, MIDBROWN)

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
        if np.array_equal(nxt, out):
            return out
        out = nxt


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
    groups.sort(key=len, reverse=True)
    return groups


def largest_component(mask):
    groups = component_list(mask)
    return len(groups[0]) if groups else 0


def claim_ink(op, ink, parts):
    """Give every outline pixel to the nearest already-labelled part."""
    names = list(parts)
    lab = np.zeros(op.shape, np.int16)
    for i, name in enumerate(names, 1):
        lab[parts[name]] = i
    dist = np.full(op.shape, 999, np.int16)
    q = deque()
    ys, xs = np.nonzero(lab > 0)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((int(y), int(x)))
    h, w = op.shape
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and ink[ny, nx] and dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                lab[ny, nx] = lab[y, x]
                q.append((ny, nx))
    out = {name: lab == i for i, name in enumerate(names, 1)}
    left = op & (lab == 0)
    if left.any():
        raise SystemExit(f"ink left over: {int(left.sum())}")
    return out


def absorb_torso_islands(parts):
    """A torso speck that only touches limbs rides with the limb it touches."""
    groups = component_list(parts["torso"])
    if len(groups) <= 1:
        return parts
    h, w = parts["torso"].shape
    limbs = [name for name in parts if name != "torso"]
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for pts in groups[1:]:
        votes = {name: 0 for name in limbs}
        for y, x in pts:
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    for name in limbs:
                        if parts[name][ny, nx]:
                            votes[name] += 1
        if sum(votes.values()) == 0:
            continue
        best = max(limbs, key=lambda name: votes[name])
        for y, x in pts:
            parts["torso"][y, x] = False
            parts[best][y, x] = True
    return parts


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, mask in masks.items():
        if (mask & ~op).any():
            raise SystemExit(f"{name} left the still")
        if (mask & acc).any():
            raise SystemExit(f"overlap in {name}")
        acc |= mask
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition the still: {int(acc.sum())} vs {int(op.sum())}")


def assert_parts_touch(parts):
    op = np.zeros_like(parts["torso"])
    for mask in parts.values():
        op |= mask
    for name, mask in parts.items():
        if name == "torso":
            continue
        if not mask.any():
            raise SystemExit(f"{name} is missing")
        for pts in component_list(mask):
            comp = np.zeros_like(mask)
            for y, x in pts:
                comp[y, x] = True
            other = op & ~comp
            if not (comp & dilate(other, 1)).any():
                raise SystemExit(f"{name} has a floating component ({len(pts)} px)")


def segment_front(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    crab = any_color(img, CRAB_COLS)
    ink = is_color(img, INK)
    claw_l = flood(crab, crab & (xx <= 36) & (yy >= 47))
    claw_r = crab & (xx >= 47) & (yy >= 44) & (yy <= 56) & ~claw_l
    leg_a = crab & (yy >= 56) & (xx >= 37) & (xx <= 42) & ~claw_l & ~claw_r
    leg_b = crab & (yy >= 56) & (xx >= 43) & (xx <= 48) & ~claw_l & ~claw_r
    leg_c = crab & (yy >= 56) & (xx >= 49) & ~claw_l & ~claw_r
    used = claw_l | claw_r | leg_a | leg_b | leg_c
    torso = op & ~ink & ~used
    parts = claim_ink(op, ink, {
        "claw_l": claw_l, "claw_r": claw_r,
        "leg_a": leg_a, "leg_b": leg_b, "leg_c": leg_c, "torso": torso,
    })
    parts = absorb_torso_islands(parts)
    check_partition(op, parts)
    assert_parts_touch(parts)
    white = is_color(img, WHITE)
    if int(white.sum()) != 4 or int((white & parts["torso"]).sum()) != 4:
        raise SystemExit("front eyes left the head")
    eye_x = float(np.nonzero(white)[1].mean())
    tip_x = int(np.nonzero(parts["claw_l"])[1].min())
    if tip_x >= eye_x:
        raise SystemExit(f"front does not face left: claw {tip_x} eye {eye_x:.1f}")
    peach = is_color(img, PEACH)
    if not (peach & parts["claw_l"]).any() or not (peach & parts["claw_r"]).any():
        raise SystemExit("a front claw lost its peach highlight")
    bottom = int(np.nonzero(op)[0].max())
    if bottom != FRONT_FOOT or int(parts["leg_b"][bottom].sum()) < 1:
        raise SystemExit("front toe is not on leg_b")
    return parts


def segment_back(img):
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    crab = any_color(img, CRAB_COLS)
    ink = is_color(img, INK)
    if is_color(img, WHITE).any():
        raise SystemExit("back gained eyes")
    # Outer pincer tip only. The arm beside the shell stays on the torso so
    # the feet keep a body to stand on.
    claw = crab & (xx >= 62) & (yy >= 46) & (yy <= 56)
    leg_a = crab & (yy >= 57) & (xx <= 61) & ~claw
    leg_b = crab & (yy >= 57) & (xx >= 62) & ~claw
    used = claw | leg_a | leg_b
    torso = op & ~ink & ~used
    parts = claim_ink(op, ink, {
        "claw": claw, "leg_a": leg_a, "leg_b": leg_b, "torso": torso,
    })
    parts = absorb_torso_islands(parts)
    check_partition(op, parts)
    assert_parts_touch(parts)
    peach = is_color(img, PEACH)
    if int(peach.sum()) != 4 or int((peach & parts["claw"]).sum()) != 4:
        raise SystemExit("back pincer lost its peach")
    cx = float(np.nonzero(parts["claw"])[1].mean())
    tx = float(np.nonzero(parts["torso"])[1].mean())
    if cx <= tx:
        raise SystemExit(f"back pincer is not on the right: {cx:.1f} vs shell {tx:.1f}")
    bottom = int(np.nonzero(op)[0].max())
    if bottom != BACK_FOOT or int(parts["leg_a"][bottom].sum()) < 1:
        raise SystemExit("back sole is not on leg_a")
    return parts


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
    """4x nearest in-plane rotation (positive = clockwise). No width squash."""
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


def paste_empty(dst, layer):
    m = (layer[..., 3] >= 128) & (dst[..., 3] < 128)
    dst[m] = layer[m]


def put(img, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and img[y, x, 3] < 128:
        img[y, x, :3] = col
        img[y, x, 3] = 255


def paint_cushion(layer, allowed, steps, dirs=("up", "down", "left", "right")):
    acc = layer.copy()
    for _ in range(steps):
        src = acc.copy()
        for d in dirs:
            mv = np.zeros_like(src)
            if d == "up":
                mv[:-1] = src[1:]
            elif d == "down":
                mv[1:] = src[:-1]
            elif d == "left":
                mv[:, :-1] = src[:, 1:]
            else:
                mv[:, 1:] = src[:, :-1]
            take = (acc[..., 3] < 128) & allowed & (mv[..., 3] >= 128)
            acc[take] = mv[take]
    return acc


def extra_growth(layer, allowed, steps):
    grown = paint_cushion(layer, allowed, steps, ("up",))
    extra = grown.copy()
    extra[layer[..., 3] >= 128] = 0
    return extra


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def joint_band(part, torso, depth):
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_toward_center(part, torso, center_x):
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part
    ys, xs = np.nonzero(touch)
    if float(xs.mean()) < center_x:
        band = xs >= xs.max() - 1
    else:
        band = xs <= xs.min() + 1
    return float(xs[band].mean()), float(ys[band].mean())


def pivot_top(part, torso):
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(part, 0)
        # fall back to the part's own top if it hangs off another limb
        if not (part & dilate(torso, 2)).any():
            ys, xs = np.nonzero(part)
            band = ys <= ys.min() + 1
            return float(xs[band].mean()), float(ys[band].mean())
    ys, xs = np.nonzero(touch if touch.any() else part)
    band = ys <= ys.min() + 1
    return float(xs[band].mean()), float(ys[band].mean())


def tip_xy(layer, pivot):
    m = layer[..., 3] >= 128
    ys, xs = np.nonzero(m)
    d = (xs - pivot[0]) ** 2 + (ys - pivot[1]) ** 2
    i = int(d.argmax())
    return float(xs[i]), float(ys[i])


def signs_for(layer, pivot, facing):
    if int((layer[..., 3] >= 128).sum()) < 4:
        return {"up": -1.0, "fwd": 1.0}

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


def build_view(img, masks, kind, mirror):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items()}
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
    layers = {}
    for name, mask in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[mask] = canvas[mask]
        layers[name] = lyr
    torso_m = placed["torso"]
    limb_names = [name for name in masks if name != "torso"]
    for name in limb_names:
        depth = 4 if "claw" in name else 3
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth + 2
        )
    growth = {}
    for name in limb_names:
        if name.startswith("leg"):
            growth[name] = extra_growth(layers[name], torso_m, 4)
        else:
            growth[name] = np.zeros_like(canvas)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    center = (float(xs.mean()), float(ys.mean()))
    if kind == "front":
        facing = 1.0 if mirror else -1.0
        claws = ("claw_l", "claw_r")
        lift_legs = ("leg_a", "leg_c")
        plant_leg = "leg_b"
    else:
        facing = -1.0 if mirror else 1.0
        claws = ("claw",)
        lift_legs = ("leg_b",)
        plant_leg = "leg_a"
    rides = {}
    for name in limb_names:
        touches = bool((placed[name] & dilate(torso_m, 1)).any())
        rides[name] = ("claw" in name) or (name.startswith("leg") and not touches)
    pivots = {"center": center}
    signs = {}
    for name in claws:
        pivots[name] = pivot_toward_center(placed[name], torso_m, center[0])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    for name in limb_names:
        if name not in pivots:
            pivots[name] = pivot_top(placed[name], torso_m)
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "layers": layers, "growth": growth, "pivots": pivots, "signs": signs,
        "canvas": canvas, "claws": claws, "lift_legs": lift_legs,
        "plant_leg": plant_leg, "rides": rides, "limbs": tuple(limb_names),
        "has_mouth": False,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0, claw_l=0.0, claw_r=0.0, claw_mode="up",
        step_lift=(0, 0), step_plant=(0, 0), thrust=0,
        lean=0.0, knock=0, flash=False, spark=0,
        turned=False, spin=0, head_dx=0, head_dy=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["claw_l"] = wave(f, n, 4.0, phase=0.4)
        p["claw_r"] = wave(f, n, 3.0, phase=1.1)
        p["claw_mode"] = "up"
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.2) > 0.55 else 0
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        p["step_lift"] = (int(round(2 * sa)), -int(round(2 * max(0.0, sa))))
        p["step_plant"] = (int(round(-1.5 * sa)), 0)
        p["claw_l"] = wave(f, n, 6.0, phase=0.3)
        p["claw_r"] = wave(f, n, 5.0, phase=0.3 + math.pi)
        p["claw_mode"] = "up"
        p["bdy"] = -1 if abs(math.sin(u)) > 0.65 else 0
    elif action == "attack":
        # 0 rest. 1-4 raise the drawn claws (no mouth). 5-8 the back still.
        # 9-12 the claws strike, at most 34 deg. 13-15 recover.
        lead = [0, 8, 14, 18, 20, 0, 0, 0, 0, 22, 30, 34, 34, 16, 6, 0]
        off = [0, 6, 10, 14, 16, 0, 0, 0, 0, 14, 22, 28, 28, 12, 4, 0]
        p["claw_l"] = float(lead[f])
        p["claw_r"] = float(off[f])
        p["claw_mode"] = "fwd" if 9 <= f <= 12 else "up"
        p["bdy"] = [0, -1, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["bdx"] = [0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0][f]
        p["thrust"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0][f]
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["claw_l"] = 12.0 * env
        p["claw_r"] = 8.0 * env
        p["claw_mode"] = "up"
        p["flash"] = f == 1
        p["spark"] = 1 if f in (2, 3) else 0
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p = base_pose()
        p["turned"] = True
        p["spin"] = 0
    if action == "attack" and (abs(p["claw_l"]) > STRIKE_MAX + 0.01 or abs(p["claw_r"]) > STRIKE_MAX + 0.01):
        raise SystemExit(f"strike passed 34 on frame {f}: {p['claw_l']} {p['claw_r']}")
    if action == "hurt" and p["lean"] > MAX_LEAN + 0.01:
        raise SystemExit("hurt lean passed 38")
    return p


def part_angle(rig, name, mode, amp):
    if abs(amp) < 0.45 or name not in rig["signs"]:
        return 0.0
    sign = rig["signs"][name]
    if mode == "up":
        return sign["up"] * amp
    return sign["fwd"] * amp


def claw_amp(rig, name, p):
    if rig["kind"] == "back":
        return p["claw_l"]
    if name == "claw_l":
        return p["claw_l"]
    return p["claw_r"]


def pose_moved(p):
    return (
        abs(p["claw_l"]) >= 0.45 or abs(p["claw_r"]) >= 0.45
        or p["step_lift"] != (0, 0) or p["step_plant"] != (0, 0)
        or p["thrust"] or p["bdx"] or p["bdy"]
        or p["head_dx"] or p["head_dy"]
    )


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    body_dx = int(round(fc * p["bdx"]))
    body_dy = int(p["bdy"])
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], body_dx, body_dy))
    moved_legs = {}
    for name in rig["limbs"]:
        if "claw" in name:
            continue
        if name == rig["plant_leg"]:
            sx, sy = p["step_plant"]
        elif name in rig["lift_legs"]:
            sx, sy = p["step_lift"]
        else:
            sx, sy = (0, 0)
        dx = body_dx + int(round(fc * sx))
        dy = int(sy) + (body_dy if rig["rides"].get(name) else 0)
        moved_legs[name] = shift(L[name], dx, dy)
        paste(out, moved_legs[name])
    for name in rig["claws"]:
        amp = claw_amp(rig, name, p)
        ang = part_angle(rig, name, p["claw_mode"], amp)
        px, py = rig["pivots"][name]
        layer = shift(L[name], body_dx, body_dy)
        if abs(ang) >= 0.45:
            layer = despeckle(rotate_nearest(layer, ang, (px + body_dx, py + body_dy)))
        if p["thrust"]:
            layer = shift(layer, int(round(fc * p["thrust"])), 0)
        paste(out, layer)
    for name, layer in moved_legs.items():
        if name == rig["plant_leg"]:
            sx, sy = p["step_plant"]
        elif name in rig["lift_legs"]:
            sx, sy = p["step_lift"]
        else:
            sx, sy = (0, 0)
        dx = body_dx + int(round(fc * sx))
        dy = int(sy) + (body_dy if rig["rides"].get(name) else 0)
        paste_empty(out, shift(rig["growth"][name], dx, dy))
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 8
    xb = int(xs[band].max()) if facing < 0 else int(xs[band].min())
    col = np.nonzero(a[:, xb])[0]
    return float(xb), float(col.max())


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK)
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, PEACH, ORANGE)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r // 2), (1, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def weapon_tip(rig, p):
    """Tip of the lead claw after the strike pose, in cell pixels."""
    fc = rig["facing"]
    name = rig["claws"][0]
    body_dx = int(round(fc * p["bdx"]))
    body_dy = int(p["bdy"])
    ang = part_angle(rig, name, p["claw_mode"], claw_amp(rig, name, p))
    px, py = rig["pivots"][name]
    layer = shift(rig["layers"][name], body_dx, body_dy)
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, (px + body_dx, py + body_dy))
    if p["thrust"]:
        layer = shift(layer, int(round(fc * p["thrust"])), 0)
    if not np.any(layer[..., 3] >= 128):
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(layer[..., 3] >= 128)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def rig_for(rigs, d, p):
    """Attack 5-8 shows this side's back at equal scale. Back rows stay."""
    if not p.get("turned"):
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if not pose_moved(p) and abs(p["lean"]) <= 0.4 and not p["knock"]:
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            piv = rear_pivot(body, fc)
            body = rotate_nearest(body, -fc * p["lean"], piv)
            body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if p["spark"] and action == "attack" and not p["turned"]:
        wx, wy = weapon_tip(rig, p)
        impact(img, (wx + fc * 2, wy), fc, p["spark"])
    elif p["spark"] and action == "hurt":
        cx, cy = rig["pivots"]["center"]
        impact(img, (cx - fc * 4, cy - 6), -fc, p["spark"])
    if p["flash"] or p["spark"]:
        return snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def limb_lost(rig, p):
    if p.get("turned") or (not pose_moved(p) and abs(p["lean"]) <= 0.4 and not p["knock"]):
        return []
    # A whole-sprite lean keeps the parts together. Per-limb masks are not
    # rotated on their own, so only the composed silhouette is checked.
    if abs(p["lean"]) > 0.4 or p["knock"]:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-rig["facing"] * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            body = rotate_nearest(body, -rig["facing"] * p["lean"], rear_pivot(body, rig["facing"]))
        if largest_fraction(body) < 0.90:
            return ["body"]
        return []
    body = compose(rig, p)
    lost = []
    op = body[..., 3] >= 128
    if not op.any():
        return list(rig["limbs"])
    # A limb is lost when its pixels have no neighbour in the composed body
    # outside themselves. Re-draw each limb the same way compose did and test.
    fc = rig["facing"]
    body_dx = int(round(fc * p["bdx"]))
    body_dy = int(p["bdy"])
    for name in rig["limbs"]:
        if "claw" in name:
            ang = part_angle(rig, name, p["claw_mode"], claw_amp(rig, name, p))
            px, py = rig["pivots"][name]
            layer = shift(rig["layers"][name], body_dx, body_dy)
            if abs(ang) >= 0.45:
                layer = rotate_nearest(layer, ang, (px + body_dx, py + body_dy))
            if p["thrust"]:
                layer = shift(layer, int(round(fc * p["thrust"])), 0)
        else:
            if name == rig["plant_leg"]:
                sx, sy = p["step_plant"]
            elif name in rig["lift_legs"]:
                sx, sy = p["step_lift"]
            else:
                sx, sy = (0, 0)
            dx = body_dx + int(round(fc * sx))
            dy = int(sy) + (body_dy if rig["rides"].get(name) else 0)
            layer = shift(rig["layers"][name], dx, dy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = op & ~limb
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


def tile_grid(tiles, cols, bg=(18, 22, 34, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def content_box(canvases, pad=8):
    acc = np.zeros((CELL_H, CELL_W), bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad)
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
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("dwebble has no drawn mouth")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "claw_l": (220, 80, 40, 255), "claw_r": (240, 180, 40, 255), "claw": (220, 80, 40, 255),
        "leg_a": (40, 180, 80, 255), "leg_b": (40, 140, 220, 255), "leg_c": (180, 80, 220, 255),
        "torso": (150, 150, 150, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(np.concatenate([img, part], axis=1), 3, 0), 3, 1))
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
        "dex": 557, "form": "normal", "variant": "dwebble", "gen": "gen5",
        "dir": "monsterSprite/gen5/0557_dwebble/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def foot_delta(img, canvas):
    ys = np.nonzero(img[..., 3] >= 128)[0]
    cy = np.nonzero(canvas[..., 3] >= 128)[0]
    if len(ys) == 0 or len(cy) == 0:
        return 99
    return int(ys.max()) - int(cy.max())


def frame_metrics(rigs, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    clip = []
    bodies = []
    feet = []
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
                    bodies.append(largest_fraction(synth(rigs, d, act, f, palette, effects=False)))
                if act in ("idle", "walk", "attack"):
                    feet.append(abs(foot_delta(t, rigs[d]["canvas"] if not pose(act, f)["turned"] else rig_for(rigs, d, pose(act, f))["canvas"])))
    return {
        "off_palette": len(off),
        "partial_alpha": partial,
        "edge_touch": len(clip),
        "edge_touching": clip[:8],
        "body_main_min": round(min(bodies), 3) if bodies else 1,
        "foot_delta_max": max(feet) if feet else 0,
    }


def check_turns(rigs, palette):
    pairs = (("W", "N"), ("SW", "N"), ("S", "NW"), ("SE", "NW"), ("E", "NW"))
    for d, back_d in pairs:
        for f in (5, 6, 7, 8):
            got = synth(rigs, d, "attack", f, palette, effects=False)
            if not np.array_equal(got, rigs[back_d]["canvas"]):
                raise SystemExit(f"attack turn {d} frame {f} is not the {back_d} still")
    for d in ("N", "NE", "NW"):
        got = synth(rigs, d, "attack", 6, palette, effects=False)
        if not np.array_equal(got, rigs[d]["canvas"]):
            raise SystemExit(f"attack turn {d} changed the back still")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    save_peaks(Path(__file__).stem, PEAKS)
    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    palette = locked_palette(SRC_DIR / "front.png", SRC_DIR / "back.png", work)
    front, back, fm, bm, rigs = build_rigs()
    save_parts(work, front, fm, back, bm)
    print("parts front", {k: int(v.sum()) for k, v in fm.items()},
          "back", {k: int(v.sum()) for k, v in bm.items()},
          "palette", len(palette), "foot", FOOT_Y, "mouth", 0, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("signs front", rigs[("front", False)]["signs"],
          "back", rigs[("back", False)]["signs"], flush=True)

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
    check_turns(rigs, palette)
    print("turn_equal 1", flush=True)

    if args.samples:
        lost = {}
        for d in ("S", "W", "N", "E"):
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
              "foot_delta_max", metrics["foot_delta_max"],
              "clip", metrics["edge_touching"], flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(float(p["claw_l"]), 1), round(float(p["claw_r"]), 1),
                        p["claw_mode"], p["turned"], p["spin"], p["bdx"], p["bdy"], p["thrust"]))
        print("attack poses", atk, flush=True)
        leans = [round(float(pose("hurt", f)["lean"]), 2) for f in range(12)]
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
                    bare = synth(rigs, d, act, f, palette, effects=False)
                    bodies.append(largest_fraction(bare))
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
        and partial == 0 and not clip and min(bodies) >= 0.90
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
          "idle0", int(sum(idle0.values())),
          flush=True)
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_dwebble_actions", "version": VERSION, "key": KEY,
        "dex": 557, "variant": "dwebble", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "hermit crab (rock shell, two claws, three front feet, no drawn mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (near claw tip x30 < eye mean x 40.0); back faces screen-right (pincer mean x right of the shell)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    if not args.no_publish:
        publish(bounds, sprite_h)
        print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
