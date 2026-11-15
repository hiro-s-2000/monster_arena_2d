#!/usr/bin/env python3
"""Whimsicott normal (0547) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x19-73 y21-72, 2170 px, 15 colours,
alpha 0/255, foot y72):
  Two eyes. Left orange/white x40-41 y47-50 (mean x 40.5). Right x50-51
  y46-49 (mean x 50.5). Eye mean x 45.5.
  Mouth is the pink stroke x43-49 y52-53, 10 px, mean x 46.6. The mouth is
  the face tip and sits right of the eyes, so the still faces screen-right.
  Two green leaves flank the face and do not hold the eyes or the mouth:
  left x30-39 y41-56 (93 px), right x53-62 y38-51 (106 px). Those are the
  hands already in the still.
  The brown body is one mass. It forks only at y70-72 into two hem points.
  Those points are not feet. This form floats. No wing, horn, fang, tail,
  or pincer is drawn.

normal/back.png (bbox x19-74 y23-72, 2073 px, foot y72). No eye, no mouth,
  no pink, no orange, no white, and none of the front brown (148, 106, 65).
  One green leaf x35-47 y39-53, mean x 41.2, left of the body mean x 46.4,
  so the back still looks away to the left. The hem points are y70-72.

Directions, front faces right, back faces away-left:
  S SE E front as drawn (screen-right). W SW front mirrored (screen-left).
  N NW back as drawn. NE back mirrored.
  E is screen-right, W is screen-left, N is the back.
  Attack 5-8 is that back at equal scale. S SE E show N. W SW show N
  flipped. Back rows stay the back. No |cos| squash and no in-plane spin.

  idle 16: float, the two leaves breathe. Frame 0 is the still.
  walk 16: a glide. The leaves sway out of phase. No feet are grown.
  attack: 0 still, 1-4 leaves rise and the mouth wedge and ring (front
          only), 5-8 the back still, 9-12 both leaves strike, 11-12
          impact, 13-15 recover.
  hurt 12: white flash, knock-back lean (<=38 deg) and recover.

Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_whimsicott_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0547_whimsicott/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0547_whimsicott/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0547_whimsicott"
VERSION = "whimsicott_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 192, 192
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 24
FRONT_FOOT = 72
BACK_FOOT = 72
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-right. Back as drawn looks away to the left.
VIEW_OF = {
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "NE": ("back", True),
    "N": ("back", False), "NW": ("back", False),
    "W": ("front", True), "SW": ("front", True),
}
MAX_LEAN = 38.0
# The leaf slap is drawn at 34 deg. finish() must not pass that.
# Frames 5-8 are cleared after finish() so the back still stays whole.
PEAKS = {"arm_l": 34.0, "arm_r": 34.0}

INK = (16, 16, 16)
CREAM = (230, 222, 205)
BEIGE = (197, 180, 139)
TAN = (172, 148, 90)
OLIVE = (106, 90, 65)
BROWN = (148, 106, 65)
BROWN_D = (115, 82, 65)
BROWN_K = (82, 57, 41)
LEAF = (82, 164, 74)
MID_G = (65, 123, 65)
DARK_G = (24, 74, 24)
PINK = (172, 74, 98)
ORANGE = (230, 131, 0)
ORANGE_D = (172, 106, 16)
WHITE = (255, 255, 255)
GREEN = (LEAF, MID_G, DARK_G)
EYE_COLS = (ORANGE, ORANGE_D, WHITE)
FRONT_ONLY = (PINK, ORANGE, ORANGE_D, WHITE, BROWN)

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
    """4x nearest in-plane rotation. Positive follows the shared forge formula. No squash."""
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
        up = np.zeros_like(src)
        dn = np.zeros_like(src)
        lt = np.zeros_like(src)
        rt = np.zeros_like(src)
        up[:-1] = src[1:]
        dn[1:] = src[:-1]
        lt[:, :-1] = src[:, 1:]
        rt[:, 1:] = src[:, :-1]
        for g in (up, dn, lt, rt):
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


def mask_from(shape, pts):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m


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


def claim_islands(host, torso):
    """Ink specks that touch only this leaf belong to the leaf."""
    changed = True
    while changed:
        changed = False
        for pts in component_list(torso):
            if len(pts) > 8:
                continue
            blob = np.zeros_like(torso)
            for y, x in pts:
                blob[y, x] = True
            if (blob & dilate(host, 1)).any() and not (blob & dilate(torso & ~blob, 1)).any():
                host |= blob
                torso &= ~blob
                changed = True
                break
    return host, torso


def segment_front(img):
    op = img[..., 3] >= 128
    green = any_color(img, GREEN)
    groups = component_list(green)
    if len(groups) != 2:
        raise SystemExit(f"front leaves {len(groups)}")
    groups.sort(key=lambda pts: sum(p[1] for p in pts) / len(pts))
    arm_l = mask_from(op.shape, groups[0])
    arm_r = mask_from(op.shape, groups[1])
    if int(arm_l.sum()) != 93 or int(arm_r.sum()) != 106:
        raise SystemExit(f"front leaf sizes {int(arm_l.sum())} {int(arm_r.sum())}")
    torso = op & ~arm_l & ~arm_r
    # Three ink pixels at x33 y49-51 sit inside the left leaf.
    arm_l, torso = claim_islands(arm_l, torso)
    arm_r, torso = claim_islands(arm_r, torso)
    masks = {"arm_l": arm_l, "arm_r": arm_r, "torso": torso}
    check_partition(op, masks)
    eye = any_color(img, EYE_COLS)
    mouth = is_color(img, PINK)
    if int(eye.sum()) != 12 or (eye & ~torso).any():
        raise SystemExit("front eyes left the face")
    if int(mouth.sum()) != 10 or (mouth & ~torso).any():
        raise SystemExit("front mouth left the face")
    eye_x = float(np.nonzero(eye)[1].mean())
    mouth_x = float(np.nonzero(mouth)[1].mean())
    if not (mouth_x > eye_x):
        raise SystemExit(f"front does not face right: mouth {mouth_x:.1f} eye {eye_x:.1f}")
    if float(np.nonzero(arm_l)[1].mean()) >= eye_x:
        raise SystemExit("left leaf is not left of the eyes")
    if float(np.nonzero(arm_r)[1].mean()) <= eye_x:
        raise SystemExit("right leaf is not right of the eyes")
    for name in ("arm_l", "arm_r"):
        require_piece(name, masks[name])
        if not (masks[name] & dilate(torso, 1)).any():
            raise SystemExit(f"front {name} does not touch the body")
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit("front torso fell apart")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front foot line changed")
    return masks


def segment_back(img):
    op = img[..., 3] >= 128
    for c in FRONT_ONLY:
        if is_color(img, c).any():
            raise SystemExit(f"back gained a front-only colour {c}")
    green = any_color(img, GREEN)
    groups = component_list(green)
    if len(groups) != 1 or len(groups[0]) != 122:
        raise SystemExit(f"back leaf {len(groups)} {[len(g) for g in groups]}")
    arm = mask_from(op.shape, groups[0])
    torso = op & ~arm
    masks = {"arm": arm, "torso": torso}
    check_partition(op, masks)
    require_piece("arm", arm)
    require_piece("torso", torso)
    if not (arm & dilate(torso, 1)).any():
        raise SystemExit("back leaf does not touch the body")
    body_x = float(np.nonzero(op)[1].mean())
    leaf_x = float(np.nonzero(arm)[1].mean())
    if leaf_x >= body_x:
        raise SystemExit(f"back leaf is not left of the body: {leaf_x:.1f} {body_x:.1f}")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back foot line changed")
    return masks


def place(arr):
    extra = arr.shape[2:]
    out = np.zeros((CELL_H, CELL_W) + extra, arr.dtype)
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
    """Contact pixel of the part nearest the torso centre."""
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
    if kind == "front":
        facing = 1.0 if not mirror else -1.0
    else:
        facing = -1.0 if not mirror else 1.0
    pivots = {}
    signs = {}
    for name in list(layers):
        if name == "torso":
            continue
        depth = 18
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth
        )
        pivots[name] = pivot_of(placed[name], torso_m)
        signs[name] = signs_for(layers[name], pivots[name], facing)
        covered = (layers["torso"][..., 3] >= 128) & (layers[name][..., 3] >= 128)
        need = int((layers[name][..., 3] >= 128).sum())
        if int(covered.sum()) != need:
            raise SystemExit(f"{kind} {name} cushion {int(covered.sum())} of {need}")
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    mouth = is_color(canvas, PINK)
    has_mouth = bool(mouth.any())
    if has_mouth:
        my, mx = np.nonzero(mouth)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
    else:
        pivots["mouth"] = pivots["center"]
    if kind == "front" and not has_mouth:
        raise SystemExit("front lost its mouth")
    if kind == "back" and has_mouth:
        raise SystemExit("back grew a mouth")
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "layers": layers,
        "pivots": pivots, "signs": signs, "canvas": canvas, "has_mouth": has_mouth,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        arm_l=0.0, arm_r=0.0, arm_mode="up",
        lean=0.0, knock=0, flash=False, wedge=0, ring=-1, spark=0,
        turned=False, spin=0, head_dx=0, head_dy=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        p["bdy"] = int(round(wave(f, n, 1.0)))
        p["arm_l"] = wave(f, n, 7.0, phase=0.4)
        p["arm_r"] = wave(f, n, 7.0, phase=0.9)
        p["arm_mode"] = "up"
    elif action == "walk":
        p["bdy"] = int(round(wave(f, n, 2.0)))
        p["bdx"] = int(round(wave(f, n, 1.0, cycles=1.0)))
        p["arm_l"] = wave(f, n, 10.0, phase=0.15)
        p["arm_r"] = wave(f, n, 10.0, phase=0.15 + math.pi)
        p["arm_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 leaves rise, and the front mouth roars.
        # 5-8 the back still, cleared again after finish().
        # 9-12 both leaves strike at 34 deg. 11-12 impact. 13-15 recover.
        arm = [0, 8, 14, 20, 18, 0, 0, 0, 0, 34, 34, 34, 34, 16, 6, 0]
        bdx = [0, -1, -1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        wedge = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ring = [-1, 0, 1, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        p["arm_l"] = float(arm[f])
        p["arm_r"] = float(arm[f])
        p["bdx"] = int(bdx[f])
        p["wedge"] = int(wedge[f])
        p["ring"] = int(ring[f])
        p["arm_mode"] = "fwd" if 9 <= f <= 14 else "up"
        p["turned"] = 5 <= f <= 8
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["arm_l"] = 0.0
        p["arm_r"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
        p["head_dx"] = 0
        p["head_dy"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
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


def angle_of(rig, p, name):
    if name == "arm_l":
        return part_angle(rig, name, p["arm_mode"], p["arm_l"])
    if name == "arm_r":
        return part_angle(rig, name, p["arm_mode"], p["arm_r"])
    if name == "arm":
        amp = p["arm_l"] if abs(p["arm_l"]) >= abs(p["arm_r"]) else p["arm_r"]
        return part_angle(rig, name, p["arm_mode"], amp)
    return 0.0


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])

    def one(name):
        ang = angle_of(rig, p, name)
        return moved(L[name], ang, piv[name], bdx, bdy)

    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("arm", "arm_l", "arm_r"):
        if name in L:
            paste(out, one(name))
    return out


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = any_color(out, (INK, DARK_G, BROWN_K))
    out[op & ~keep, :3] = np.array(WHITE, np.uint8)
    return out


def rear_pivot(img, facing):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    yb = int(ys.max())
    band = ys >= yb - 2
    xb = int(xs[band].min()) if facing > 0 else int(xs[band].max())
    return float(xb), float(yb)


def arm_tip(rig, p):
    fc = rig["facing"]
    names = [n for n in ("arm_r", "arm_l", "arm") if n in rig["layers"]]
    best = None
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    for name in names:
        layer = moved(
            rig["layers"][name], angle_of(rig, p, name), rig["pivots"][name], bdx, bdy,
        )
        m = layer[..., 3] >= 128
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
        pt = (float(xs[i]), float(ys[i]))
        if best is None:
            best = pt
        elif fc > 0 and pt[0] > best[0]:
            best = pt
        elif fc < 0 and pt[0] < best[0]:
            best = pt
    if best is None:
        return rig["pivots"]["center"]
    return best


def mouth_exit(img, origin, fc):
    """First empty pixel in front of the mouth, outside the cotton."""
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    step = 1 if fc >= 0 else -1
    h, w = img.shape[:2]
    for _ in range(48):
        x += step
        if not (1 <= x < w - 1 and 1 <= y < h - 1):
            break
        if img[y, x, 3] < 128:
            return float(x), float(y)
    return float(x), float(y)


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the existing mouth. Empty pixels only."""
    ox, oy = origin
    cols = (INK, PINK, WHITE)
    step = 1 if fc >= 0 else -1
    for i in range(max(0, int(age))):
        x = ox + step * i
        span = 0 if i == 0 else 1
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, CREAM, LEAF)
    rad = 5 + int(age) * 3
    step = 1.0 if fc >= 0 else -1.0
    for deg in range(-40, 41, 10):
        a = math.radians(deg)
        x = ox + step * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.62
        put(img, x, y, cols[int(age) % 3])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, CREAM, LEAF)
    r = 2 + int(age) * 2
    step = 1 if fc >= 0 else -1
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r))):
        put(img, ox + step * dx, oy + dy, cols[i % 3])


def limb_names(rig):
    return [n for n in rig["layers"] if n != "torso"]


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    for name in limb_names(rig):
        layer = moved(
            rig["layers"][name], angle_of(rig, p, name), rig["pivots"][name], bdx, bdy,
        )
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    """Attack 5-8 shows this side's back at equal scale. Back rows stay."""
    if not p.get("turned"):
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    if kind == "back":
        return rigs[d]
    return rigs[("back", mirror)]


def moving(p):
    if p["bdx"] or p["bdy"] or p["knock"]:
        return True
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["arm_l"]) >= 0.45 or abs(p["arm_r"]) >= 0.45:
        return True
    return False


def rotated(p):
    if abs(p["lean"]) > 0.4:
        return True
    if abs(p["arm_l"]) >= 0.45 or abs(p["arm_r"]) >= 0.45:
        return True
    return False


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if not moving(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
        if rotated(p):
            body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if rig["has_mouth"] and not p["turned"] and (p["wedge"] or p["ring"] >= 0):
        mx, my = rig["pivots"]["mouth"]
        origin = mouth_exit(img, (mx + int(round(fc * p["bdx"])), my + int(round(p["bdy"]))), fc)
        if p["wedge"]:
            mouth_wedge(img, origin, fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, origin, fc, p["ring"])
        img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = arm_tip(rig, p)
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
    y0 = max(0, int(ys.min()) - pad - 10)
    y1 = min(CELL_H, int(ys.max()) + 1 + pad + 18)
    x0 = max(0, int(xs.min()) - pad - 16)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad + 16)
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
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs["E"]["facing"] != 1.0 or rigs["W"]["facing"] != -1.0:
        raise SystemExit("front facing is not right-as-drawn")
    if rigs["N"]["facing"] != -1.0 or rigs["NE"]["facing"] != 1.0:
        raise SystemExit("back facing is not away-left as drawn")
    if not rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("mouth side is wrong")
    return front, back, fm, bm, rigs


def save_parts(work, fm, bm):
    cols = {
        "arm_l": (80, 200, 80, 255),
        "arm_r": (40, 120, 255, 255),
        "arm": (255, 180, 40, 255),
        "torso": (230, 220, 200, 255),
    }
    tiles = []
    for masks in (fm, bm):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 2, 0), 2, 1))
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
        "dex": 547, "form": "normal", "variant": "whimsicott", "gen": "gen5",
        "dir": "monsterSprite/gen5/0547_whimsicott/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_metrics(img, pal):
    partial = int(((img[..., 3] != 0) & (img[..., 3] != 255)).sum())
    a = img[..., 3] == 255
    edge = bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())
    off = set()
    if a.any():
        for c in np.unique(img[a][:, :3], axis=0):
            t = tuple(int(v) for v in c)
            if t not in pal:
                off.add(t)
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
    save_parts(work, fm, bm)
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
    print("rest diff", rest_diff, flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    if args.samples:
        pal = {tuple(int(v) for v in c) for c in palette}
        off = set()
        partial = 0
        clip = []
        bodies = []
        lost = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    img = synth(rigs, d, act, f, palette, effects=True)
                    o, part, edge = frame_metrics(img, pal)
                    off |= o
                    partial += part
                    if edge:
                        clip.append(f"{act}:{d}:{f}")
                    if act in ("attack", "hurt", "walk"):
                        bodies.append(largest_fraction(img))
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost.append(f"{act}:{d}:{f}:{','.join(bad)}")
        box = crop_box(rigs["S"]["canvas"])
        y0, y1, x0, x1 = box
        rows = []
        for d in ("S", "W", "N", "E"):
            for act in ACTIONS:
                frames = []
                for f in range(16):
                    if f < FRAMES[act]:
                        frames.append(synth(rigs, d, act, f, palette)[y0:y1, x0:x1])
                    else:
                        frames.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
                rows.append(tile_grid(frames, 16))
        Image.fromarray(np.concatenate(rows, axis=0), "RGBA").save(work / "samples.png")
        hole = synth(rigs, "S", "attack", 4, palette, effects=False)
        rest = rigs["S"]["canvas"]
        uncovered = int(((rest[..., 3] >= 128) & (hole[..., 3] < 128)).sum())
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(p["arm_l"], 1), round(p["arm_r"], 1), p["arm_mode"],
                        p["turned"], p["spin"], p["bdx"], p["wedge"], p["ring"], p["spark"]))
        leans = [round(pose("hurt", f)["lean"], 1) for f in range(12)]
        print("attack poses", atk, flush=True)
        print("hurt lean", leans, "max", max(leans), flush=True)
        print("disconnected", lost[:12], "count", len(lost), flush=True)
        print("uncovered", uncovered, flush=True)
        print("body_main_min", round(min(bodies), 3) if bodies else 1, flush=True)
        rest_sum = int(sum(rest_diff.values()))
        print(
            f"qa_samples 静止差分 {rest_sum} パレット外 {len(off)} "
            f"中間アルファ {partial} 枠接触 {len(clip)}",
            flush=True,
        )
        if clip:
            print("edge", clip[:8], flush=True)
        print("samples", work / "samples.png", flush=True)
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
            if key not in state["done"]:
                state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    y0, y1, x0, x1 = box
    strips = [motion_strip(sheets, DIRS.index(d), box) for d in ("S", "N", "E", "W")]
    motion = np.concatenate(strips, axis=0)
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
          and partial == 0 and not clip and min(bodies) >= 0.9)
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    rest_sum = int(sum(rest_diff.values()) + sum(idle0.values()))
    print(
        f"qa_full 静止差分 {rest_sum} パレット外 {len(off)} "
        f"中間アルファ {partial} 枠接触 {len(clip)}",
        flush=True,
    )
    meta = {
        "generator": "forge_gen5_whimsicott_actions", "version": VERSION, "key": KEY,
        "dex": 547, "variant": "whimsicott", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "floating cotton (two side leaves, pink mouth, forked hem, no feet)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front mouth mean x 46.6 is right of eye mean x 45.5; back leaf is left of the body",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
