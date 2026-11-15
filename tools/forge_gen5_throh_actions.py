#!/usr/bin/env python3
"""Throh normal (0538) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x12-80 y21-71, 1874 px, 10 colours,
alpha 0/255):
  Faces screen-left. Both eyes are drawn: white #f6f6f6 at x38-39 y41-42 and
  x45-48 y40-42 (mean x 44.2). The closed mouth is the ink slit #101010 at
  x42-45 y46 (mean x 43.5), left of the eyes. The lead fist is the red arm
  on the screen-left (x12-32 y49-65). The raised fist is the red arm on the
  screen-right (x60-80 y21-44). Two feet split at y66. The lead foot ends
  y70. The ground foot x61-67 ends y71. No wing, horn, tail, or fang.

normal/back.png (bbox x18-77 y22-72, 1734 px, 9 colours, all in the front
  palette). No face and no white. The raised fist is the left nub x21-34
  y22-29. The lead arm is the right mass x59-77 y50-65. Feet split at y66.
  Left foot ends y71. Ground foot ends y72. The back is shifted up 1px so
  both ground feet share a line.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn (raised fist left). NW back mirrored.
  E is screen-right, W is screen-left, N is the back. Attack 5-8 swaps to
  the matching back at equal scale. No width squash, no in-plane body spin.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_throh_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0538_throh/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0538_throh/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0538_throh"
VERSION = "throh_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = (CELL_H - 96) // 2
FRONT_FOOT = 71
BACK_FOOT = 72
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left (mouth x < eye x, lead fist on the left).
# S (front mirrored) turns into N (back as drawn) by flipping the mirror flag.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Open amounts finish() may raise frames 5-12 toward. The punches themselves
# are drawn to 34 deg, above these, so finish() does not lower them.
PEAKS = {"arm_hi": 16.0, "arm_lo": 12.0}
LIMBS = ("arm_hi", "arm_lo", "leg_l", "leg_r")

INK = (16, 16, 16)
BELT = (41, 41, 32)
GRAY = (57, 57, 57)
WHITE = (246, 246, 246)
SKIN = (222, 90, 90)
SKIN_MID = (172, 49, 49)
SKIN_DARK = (98, 24, 24)
GI = (238, 230, 213)
GI_MID = (197, 189, 172)
GI_SHADE = (148, 139, 131)
LIP = (SKIN, SKIN_MID, SKIN_DARK, INK, BELT, GRAY)

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
    """Close a 1-2px gap after a limb moves. Sparks are drawn later."""
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


def row_segments(mask_row):
    xs = np.nonzero(mask_row)[0]
    if len(xs) == 0:
        return []
    segs = []
    start = int(xs[0])
    prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x > prev + 1:
            segs.append((start, prev))
            start = x
        prev = x
    segs.append((start, prev))
    return segs


def paint_seg(mask, y, seg):
    mask[y, seg[0]:seg[1] + 1] = True


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


def touches(part, body):
    return bool((part & dilate(body, 1)).any())


def sole_y(mask):
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        raise SystemExit("empty sole")
    return int(ys.max())


def require_solid(masks, torso):
    for name in LIMBS:
        if not touches(masks[name], torso):
            raise SystemExit(f"{name} does not touch the body")
        if largest_component(masks[name]) != int(masks[name].sum()):
            raise SystemExit(f"{name} fell apart ({int(masks[name].sum())} px)")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("torso fell apart")


def flood_from(seeds, region):
    seen = np.zeros_like(region, bool)
    stack = [(int(y), int(x)) for y, x in seeds]
    for y, x in stack:
        seen[y, x] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < region.shape[0] and 0 <= nx < region.shape[1] and region[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return seen


def feet_from(op, y0):
    """Split the two feet on the rows where the silhouette actually gaps."""
    leg_l = np.zeros_like(op)
    leg_r = np.zeros_like(op)
    for y in range(y0, op.shape[0]):
        segs = row_segments(op[y])
        if len(segs) >= 2:
            paint_seg(leg_l, y, segs[0])
            paint_seg(leg_r, y, segs[-1])
        elif len(segs) == 1:
            paint_seg(leg_r, y, segs[0])
    return leg_l, leg_r


def segment_front(img):
    op = img[..., 3] >= 128
    xx = np.arange(96)
    arm_hi = np.zeros_like(op)
    for y in range(0, 34):
        for seg in row_segments(op[y]):
            if seg[0] >= 60:
                paint_seg(arm_hi, y, seg)
    region = np.zeros_like(op)
    region[34:45, 60:] = op[34:45, 60:]
    ys, xs = np.nonzero(dilate(arm_hi, 1) & region)
    arm_hi = arm_hi | (flood_from(zip(ys, xs), region) & region)
    arm_lo = np.zeros_like(op)
    for y in range(49, 52):
        arm_lo[y] = op[y] & (xx <= 32)
    for y in range(52, 66):
        segs = row_segments(op[y] & ~arm_hi[y] & ~arm_lo[y])
        if len(segs) < 2:
            continue
        for seg in segs[:-1]:
            if seg[0] <= 26 and seg[1] <= 36:
                paint_seg(arm_lo, y, seg)
    leg_l, leg_r = feet_from(op & ~arm_hi & ~arm_lo, 66)
    torso = op & ~(arm_hi | arm_lo | leg_l | leg_r)
    masks = {"arm_hi": arm_hi, "arm_lo": arm_lo, "leg_l": leg_l, "leg_r": leg_r, "torso": torso}
    check_partition(op, masks)
    require_solid(masks, torso)
    eyes = is_color(img, WHITE)
    mouth = np.zeros_like(op)
    mouth[46, 42:46] = True
    if int(eyes.sum()) != 6 or int((eyes & torso).sum()) != 6:
        raise SystemExit("front eyes left the head")
    if int((mouth & ink_row(img)).sum()) != 4 or int((mouth & torso).sum()) != 4:
        raise SystemExit("front mouth left the head")
    eye_x = float(np.nonzero(eyes)[1].mean())
    if 43.5 > eye_x:
        raise SystemExit("front mouth is not left of the eyes")
    if float(np.nonzero(arm_lo)[1].mean()) >= float(np.nonzero(arm_hi)[1].mean()):
        raise SystemExit("front lead fist is not left of the raised fist")
    if sole_y(leg_r) != FRONT_FOOT or sole_y(leg_l) != FRONT_FOOT - 1:
        raise SystemExit("front feet moved")
    gap = sole_y(leg_r) - sole_y(leg_l)
    return masks, "leg_r", gap


def ink_row(img):
    return is_color(img, INK)


def segment_back(img):
    op = img[..., 3] >= 128
    xx = np.arange(96)
    arm_hi = np.zeros_like(op)
    for y in range(0, 30):
        for seg in row_segments(op[y]):
            if seg[1] <= 36:
                paint_seg(arm_hi, y, seg)
    arm_lo = np.zeros_like(op)
    for y in range(50, 54):
        row = op[y] & ~arm_hi[y]
        xs = np.nonzero(row)[0]
        if len(xs) and int(xs.max()) >= 70:
            arm_lo[y] |= row & (xx >= 59)
    for y in range(54, 66):
        segs = row_segments(op[y] & ~arm_hi[y] & ~arm_lo[y])
        if len(segs) < 2:
            continue
        for seg in segs:
            if seg[0] >= 59:
                paint_seg(arm_lo, y, seg)
    leg_l, leg_r = feet_from(op & ~arm_hi & ~arm_lo, 66)
    torso = op & ~(arm_hi | arm_lo | leg_l | leg_r)
    masks = {"arm_hi": arm_hi, "arm_lo": arm_lo, "leg_l": leg_l, "leg_r": leg_r, "torso": torso}
    check_partition(op, masks)
    require_solid(masks, torso)
    if is_color(img, WHITE).any():
        raise SystemExit("back gained an eye")
    if float(np.nonzero(arm_hi)[1].mean()) >= float(np.nonzero(torso)[1].mean()):
        raise SystemExit("back raised fist is not on the left")
    if float(np.nonzero(arm_lo)[1].mean()) <= float(np.nonzero(torso)[1].mean()):
        raise SystemExit("back lead arm is not on the right")
    if sole_y(leg_r) != BACK_FOOT or sole_y(leg_l) != BACK_FOOT - 1:
        raise SystemExit("back feet moved")
    gap = sole_y(leg_r) - sole_y(leg_l)
    return masks, "leg_r", gap


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


def contact_pivot(layer, torso_layer, mode):
    """Joint on the contact with the torso. low = the bottom edge, high = the top edge."""
    part = layer[..., 3] >= 128
    torso = torso_layer[..., 3] >= 128
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part
    ys, xs = np.nonzero(touch)
    if mode == "low":
        band = ys >= int(ys.max()) - 1
    else:
        band = ys <= int(ys.min()) + 1
    return float(xs[band].mean()), float(ys[band].mean())


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


def build_view(img, masks, kind, mirror, ground_leg, sole_gap):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items()}
    mouth = np.zeros(canvas.shape[:2], bool)
    if kind == "front":
        src = np.zeros((96, 96), bool)
        src[46, 42:46] = True
        mouth = place(src, y_shift)
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
        mouth = np.ascontiguousarray(mouth[:, ::-1])
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    torso_m = placed["torso"]
    for name, depth in (("arm_hi", 5), ("arm_lo", 5), ("leg_l", 4), ("leg_r", 4)):
        layers["torso"] = paint_cushion(layers["torso"], joint_band(placed[name], torso_m, depth), depth)
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    pivots = {
        "arm_hi": contact_pivot(layers["arm_hi"], layers["torso"], "low"),
        "arm_lo": contact_pivot(layers["arm_lo"], layers["torso"], "high"),
        "leg_l": contact_pivot(layers["leg_l"], layers["torso"], "high"),
        "leg_r": contact_pivot(layers["leg_r"], layers["torso"], "high"),
    }
    signs = {name: signs_for(layers[name], pivots[name], facing) for name in ("arm_hi", "arm_lo")}
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "mouth": mouth,
        "ground_leg": ground_leg, "sole_gap": int(sole_gap),
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        arm_hi=0.0, arm_lo=0.0,
        cock_hi=0.0, cock_lo=0.0,
        step_l=(0, 0), step_r=(0, 0),
        feet_follow=False,
        lean=0.0, knock=0, flash=False, spark=0, roar=0,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The chest and both fists lift 1px. The fists sway.
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.5) > 0.55 else 0
        p["arm_hi"] = wave(f, n, 5.0, phase=0.4)
        p["arm_lo"] = wave(f, n, 4.0, phase=0.4 + math.pi)
    elif action == "walk":
        # Two drawn feet, opposite steps. A foot only swings while it is up.
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)

        def step(s):
            up = max(0.0, s)
            lift = -2 if up > 0.45 else 0
            fwd = 2 if lift else 0
            return (fwd, lift)

        p["step_l"] = step(sa)
        p["step_r"] = step(sb)
        p["arm_hi"] = wave(f, n, 8.0, phase=math.pi / 2)
        p["arm_lo"] = wave(f, n, 7.0, phase=math.pi / 2 + math.pi)
        p["feet_follow"] = True
    elif action == "attack":
        # 0 rest. 1-4 roar: the drawn mouth opens and the fists cock back
        # (windup 8-22). 5-8 the back, equal scale, fists starting to open.
        # 9-12 both fists strike, at most 34 deg. 13-15 recover.
        cock_hi = [0, 8, 14, 18, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cock_lo = [0, 6, 12, 16, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        arm_hi = [0, 0, 0, 0, 0, 16, 16, 16, 16, 24, 30, 34, 34, 12, 4, 0]
        arm_lo = [0, 0, 0, 0, 0, 12, 12, 12, 12, 22, 28, 34, 30, 10, 3, 0]
        bdx = [0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        bdy = [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p["cock_hi"] = float(cock_hi[f])
        p["cock_lo"] = float(cock_lo[f])
        p["arm_hi"] = float(arm_hi[f])
        p["arm_lo"] = float(arm_lo[f])
        p["bdx"] = int(bdx[f])
        p["bdy"] = int(bdy[f])
        p["roar"] = f if 1 <= f <= 4 else 0
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.85, 0.62, 0.42, 0.24, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["cock_hi"] = 10.0 * env
        p["cock_lo"] = 8.0 * env
        p["flash"] = f == 1
    return finish(p, action, f, __file__)


def arm_angle(rig, name, p):
    cock_key = "cock_hi" if name == "arm_hi" else "cock_lo"
    amp = float(p[name]) - float(p[cock_key])
    if abs(amp) < 0.45:
        return 0.0
    return rig["signs"][name]["fwd"] * amp


def plant_drop(rig, p):
    """When the long foot is up, drop so the short foot reaches the ground line."""
    gap = int(rig["sole_gap"])
    if gap <= 0:
        return 0
    g = rig["ground_leg"]
    sy = p["step_r"][1] if g == "leg_r" else p["step_l"][1]
    if sy < 0:
        return min(gap, -int(sy))
    return 0


def moved(layer, ang, pivot, dx, dy):
    if abs(ang) >= 0.45:
        layer = rotate_nearest(layer, ang, pivot)
    return shift(layer, dx, dy)


def limb_layers(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    leg_dy = bdy if p["feet_follow"] else 0
    parts = {}
    for name in ("arm_hi", "arm_lo"):
        parts[name] = moved(L[name], arm_angle(rig, name, p), piv[name], bdx, bdy)
    for name, step in (("leg_l", p["step_l"]), ("leg_r", p["step_r"])):
        sx, sy = step
        parts[name] = moved(L[name], 0.0, piv[name], bdx + int(round(fc * sx)), leg_dy + int(sy))
    return parts


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    parts = limb_layers(rig, p)
    for name in ("leg_l", "leg_r", "arm_lo", "arm_hi"):
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
    band = ys >= yb - 1
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def open_mouth(img, mouth, fc, age):
    """Widen the drawn ink slit. Only lip colours already on the face."""
    if age <= 0 or not mouth.any():
        return
    ys, xs = np.nonzero(mouth)
    cx, cy = float(xs.mean()), float(ys.mean())
    h, w = img.shape[:2]
    pts = []
    for i in range(1, int(age) + 1):
        pts.append((cx + fc * i, cy))
        pts.append((cx + fc * (i - 1), cy + 1))
        if i >= 2:
            pts.append((cx + fc * i, cy + 1))
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= yi < h and 0 <= xi < w) or img[yi, xi, 3] < 128:
            continue
        col = tuple(int(v) for v in img[yi, xi, :3])
        if col in LIP:
            img[yi, xi, :3] = INK


def roar_ring(img, mouth, fc, age):
    """A small ring in the air, in the facing direction, outside the body."""
    if age <= 0 or not mouth.any():
        return
    cy = int(round(float(np.nonzero(mouth)[0].mean())))
    if not (0 <= cy < img.shape[0]):
        return
    xs = np.nonzero(img[cy, :, 3] >= 128)[0]
    if len(xs) == 0:
        return
    edge = int(xs.min() if fc < 0 else xs.max())
    ox = edge + fc * (5 + int(age))
    oy = cy - 6
    cols = (SKIN, WHITE, INK)
    dots = ((0, 0), (3, -3), (3, 3), (6, -5), (6, 0), (6, 5), (4, -1), (4, 2))
    for i, (dx, dy) in enumerate(dots[: 4 + int(age)]):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def fist_tip(layer, pivot):
    m = layer[..., 3] >= 128
    if not m.any():
        return pivot
    return tip_xy(layer, pivot)


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, SKIN, INK)
    r = 3 + age * 2
    dots = ((r, 0), (r - 1, -3), (r - 1, 3), (2, -2), (3, 2))
    for i, (dx, dy) in enumerate(dots):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def body_moved(p):
    return (
        abs(p["arm_hi"]) >= 0.45 or abs(p["arm_lo"]) >= 0.45
        or abs(p["cock_hi"]) >= 0.45 or abs(p["cock_lo"]) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0)
        or p["bdx"] or p["bdy"] or p["knock"] or abs(p["lean"]) > 0.4
    )


def apply_plant(rig, p):
    p = dict(p)
    if p["feet_follow"]:
        p["bdy"] = int(p["bdy"]) + plant_drop(rig, p)
    return p


def rig_for(rigs, d, p):
    rig = rigs[d]
    if p["turned"]:
        kind, mirror = VIEW_OF[d]
        rig = rigs[("back" if kind == "front" else "front", not mirror)]
    return rig


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    p = apply_plant(rig, p)
    if body_moved(p):
        body = compose(rig, p)
        fc = rig["facing"]
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            piv = rear_pivot(body, fc)
            # Head goes backward, away from facing.
            body = rotate_nearest(body, -fc * p["lean"], piv)
        body = reconnect(fill_pinholes(despeckle(body)))
        img = snap(body, palette)
    else:
        img = rig["canvas"].copy()
        fc = rig["facing"]
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["roar"] and rig["kind"] == "front":
        open_mouth(img, rig["mouth"], fc, p["roar"])
        roar_ring(img, rig["mouth"], fc, p["roar"])
        img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        parts = limb_layers(rig, p)
        for name in ("arm_lo", "arm_hi"):
            tx, ty = fist_tip(parts[name], rig["pivots"][name])
            impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def head_x(img):
    """Front: the white eyes. Back: the upper back, under the raised fist."""
    eyes = is_color(img, WHITE)
    if int(eyes.sum()) >= 4:
        return float(np.nonzero(eyes)[1].mean())
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0 = int(ys.min())
    band = (ys >= y0 + 10) & (ys <= y0 + 22)
    if not band.any():
        band = ys <= y0 + 16
    return float(xs[band].mean())


def facing_extreme(layer, fc):
    xs = np.nonzero(layer[..., 3] >= 128)[1]
    if len(xs) == 0:
        return 0.0
    return float(xs.min() if fc < 0 else xs.max())


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


def crop_box(canvas, pad=14):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 18)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 16)
    x0 = max(0, int(xs.min()) - pad - 28)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 36)
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
    fm, fg, fgap = segment_front(front)
    bm, bg, bgap = segment_back(back)
    rigs = {}
    for kind, img, masks, gleg, gap in (
        ("front", front, fm, fg, fgap),
        ("back", back, bm, bg, bgap),
    ):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror, gleg, gap)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    fa = rigs[("front", False)]["canvas"]
    ba = rigs[("back", False)]["canvas"]
    if int(np.nonzero(fa[..., 3] >= 128)[0].max()) != int(np.nonzero(ba[..., 3] >= 128)[0].max()):
        raise SystemExit("front and back feet do not share a line")
    if rigs[("front", False)]["facing"] != -1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "arm_hi": (240, 80, 40, 255),
        "arm_lo": (240, 200, 40, 255),
        "leg_l": (40, 210, 80, 255),
        "leg_r": (80, 80, 255, 255),
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
        "dex": 538, "form": "normal", "variant": "throh", "gen": "gen5",
        "dir": "monsterSprite/gen5/0538_throh/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_metrics(img, palette):
    """Return (off_palette, partial_alpha, edge_touch) for one cell."""
    pal = {tuple(int(v) for v in c) for c in palette}
    a = img[..., 3]
    partial = int(((a != 0) & (a != 255)).sum())
    op = a == 255
    off = 0
    if op.any():
        used = {tuple(int(v) for v in c) for c in np.unique(img[op][:, :3], axis=0)}
        off = len(used - pal)
    edge = 1 if (op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any()) else 0
    return off, partial, edge


def assert_motion(rigs, palette):
    for d, expect_back in (("S", True), ("W", True), ("N", False), ("E", True)):
        p = pose("attack", 6)
        rig = rig_for(rigs, d, p)
        if expect_back and rig["kind"] != "back":
            raise SystemExit(f"{d} frame 6 did not turn to the back")
        if (not expect_back) and rig["kind"] != "front":
            raise SystemExit(f"{d} frame 6 left the front")
        if p["spin"] != 0:
            raise SystemExit("attack spin was not cleared")
    for f in list(range(0, 5)) + list(range(9, 16)):
        if pose("attack", f)["turned"]:
            raise SystemExit(f"attack {f} should face the front")
    leans = [pose("hurt", f)["lean"] for f in range(12)]
    if max(leans) > MAX_LEAN + 0.01 or leans[0] != 0 or leans[-1] != 0:
        raise SystemExit(f"hurt lean out of range {leans}")
    if max(pose("attack", f)["arm_hi"] for f in range(16)) > 34.01:
        raise SystemExit("strike exceeds 34 deg")
    if max(pose("attack", f)["cock_hi"] for f in range(5)) > 22.01:
        raise SystemExit("windup exceeds 22 deg")
    for d in ("W", "E", "N", "S"):
        rest = synth(rigs, d, "hurt", 0, palette, effects=False)
        hit = synth(rigs, d, "hurt", 3, palette, effects=False)
        fc = rigs[d]["facing"]
        dx = head_x(hit) - head_x(rest)
        if fc > 0 and dx >= -1:
            raise SystemExit(f"{d} hurt leaned forward ({dx:.1f})")
        if fc < 0 and dx <= 1:
            raise SystemExit(f"{d} hurt leaned forward ({dx:.1f})")
    for d in ("W", "E"):
        rig0 = rigs[d]
        p0 = apply_plant(rig0, pose("attack", 0))
        p11 = apply_plant(rig0, pose("attack", 11))
        fc = rig0["facing"]
        for name in ("arm_lo", "arm_hi"):
            x0 = facing_extreme(limb_layers(rig0, p0)[name], fc)
            x1 = facing_extreme(limb_layers(rig0, p11)[name], fc)
            # Remove the shared body step so this is the fist itself.
            delta = (x1 - fc * p11["bdx"]) - (x0 - fc * p0["bdx"])
            forward = delta * fc
            if forward < 2.0:
                raise SystemExit(f"{d} {name} did not strike forward ({forward:.1f})")


def rest_diffs(rigs, palette):
    rest = {}
    for d in DIRS:
        rest[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    return rest


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
    print("soles",
          "front", rigs[("front", False)]["ground_leg"], rigs[("front", False)]["sole_gap"],
          "back", rigs[("back", False)]["ground_leg"], rigs[("back", False)]["sole_gap"],
          flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    assert_motion(rigs, palette)

    rest_diff = rest_diffs(rigs, palette)
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    box = crop_box(rigs["S"]["canvas"])
    if args.samples:
        y0, y1, x0, x1 = box
        off_n = partial_n = edge_n = 0
        soles = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette)
                    off, partial, edge = frame_metrics(img, palette)
                    off_n += off
                    partial_n += partial
                    edge_n += edge
                    if act in ("idle", "walk") or (act == "attack" and f not in range(5, 9)):
                        ys = np.nonzero(img[..., 3] >= 128)[0]
                        if len(ys) and int(ys.max()) != FOOT_Y:
                            soles.append(f"{act}:{d}:{f}:{int(ys.max())}")
        print("off_palette", off_n, flush=True)
        print("partial_alpha", partial_n, flush=True)
        print("edge_touching", edge_n, flush=True)
        print("sole off", soles[:12], "count", len(soles), flush=True)
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
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("hurt lean", leans, "max", max(leans), flush=True)
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
        "sprite_h": sprite_h, "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    ok = (not any(rest_diff.values()) and not any(idle0.values()) and not off
          and partial == 0 and not clip and min(bodies) >= 0.9)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_throh_actions", "version": VERSION, "key": KEY,
        "dex": 538, "variant": "throh", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped judoka (raised fist, lead fist, two feet, closed mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth x 43.5 < eye x 44.2); back raised fist is on the left",
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
