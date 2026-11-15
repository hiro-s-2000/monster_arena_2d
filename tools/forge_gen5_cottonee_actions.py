#!/usr/bin/env python3
"""Cottonee normal (0546) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x26-69 y34-60, 707 px, 12 colours,
alpha 0/255, cotton tip y60). No feet. The body is one cotton puff.
  Irises are brown (173, 107, 16) plus two orange (231, 132, 0) pixels,
  mean x 43.67. The mouth is the ink bar between those irises, x42-46 y46,
  mean x 44.0. The mouth sits to the right of the irises, so the still faces
  screen-right. The back has neither brown nor orange, so it has no mouth.
  Two leaves already stick out of the puff. Front, 8-connected green:
  screen-left x26-35 (45 px) and screen-right x54-68 (64 px).
  The back's side leaves are x26-34 and x56-69. The green coat across the
  back of the puff (x35-50) stays on the torso. It is not a third limb.
normal/back.png (bbox x26-69 y35-60, 682 px, 10 colours, all in the front
  palette, tip y60). A small white tuft sits on the right (x58-61), so the
  back as drawn also points screen-right.

Directions, front faces right:
  S SE E front as drawn. W SW front mirrored.
  N NE back as drawn. NW back mirrored.
  Attack 5-8 shows that direction's back at equal scale. No |cos| squash
  and no in-plane spin. The cotton floats. The leaves are the only strike.

  idle 16, walk 16, attack 16, hurt 12.
  Python: agent-sprite-forge/.venv/bin/python -I tools/forge_gen5_cottonee_actions.py
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
SRC_DIR = ROOT / "monsterSprite/gen5/0546_cottonee/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0546_cottonee/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0546_cottonee"
VERSION = "cottonee_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 160
OFF_X = (CELL_W - 96) // 2
OFF_Y = CELL_H - 96 - 20
TIP_Y = 60
FOOT_Y = OFF_Y + TIP_Y
# Front as drawn looks screen-right. S keeps that picture. Its back is N.
VIEW_OF = {
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", True), "SW": ("front", True),
}
MAX_LEAN = 38.0
# finish() may lift a leaf this far. The slap is drawn at 34, above this.
PEAKS = {"leaf_l": 16.0, "leaf_r": 16.0}
LIMBS = ("leaf_l", "leaf_r")

INK = (16, 16, 16)
WHITE = (255, 255, 255)
PALE = (222, 231, 198)
SILVER = (206, 206, 214)
LEAF_HI = (82, 165, 74)
BROWN = (173, 107, 16)
ORANGE = (231, 132, 0)
LEAF_COLS = ((82, 165, 74), (66, 123, 66), (24, 74, 24))
IRIS_COLS = (BROWN, ORANGE)

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


def component_list(mask, diag=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    groups = []
    neigh = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        neigh += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
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
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(pts)
    return groups


def mask_from_pts(shape, pts):
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


def largest_fraction(img):
    op = img[..., 3] >= 128
    if not op.any():
        return 1.0
    return largest_component(op) / float(op.sum())


def side_leaves(img):
    """The two leaves that stick out of the puff. Body-green stays put."""
    leaf = any_color(img, LEAF_COLS)
    left = np.zeros(leaf.shape, bool)
    right = np.zeros(leaf.shape, bool)
    for pts in component_list(leaf, diag=True):
        xs = [x for _y, x in pts]
        mean = sum(xs) / float(len(xs))
        m = mask_from_pts(leaf.shape, pts)
        if mean <= 40.0 and max(xs) <= 42:
            left |= m
        elif mean >= 52.0 and min(xs) >= 50:
            right |= m
    ink = is_color(img, INK)
    op = img[..., 3] >= 128

    def rim(part):
        if not part.any():
            return part
        other = op & ~part & ~ink
        return ink & dilate(part, 1) & ~dilate(other, 1)

    left = left | rim(left)
    right = right | rim(right)
    right &= ~left
    return left, right


def mouth_between(img):
    """Ink strictly between the two irises. Empty when this view has no face."""
    iris = any_color(img, IRIS_COLS)
    if not iris.any():
        return iris, iris
    groups = [g for g in component_list(iris, diag=True) if len(g) >= 2]
    if len(groups) < 2:
        raise SystemExit("irises did not split into two eyes")
    groups.sort(key=lambda g: sum(x for _y, x in g) / float(len(g)))
    left, right = groups[0], groups[-1]
    x0 = max(x for _y, x in left)
    x1 = min(x for _y, x in right)
    ys = [y for y, _x in left + right]
    y0, y1 = min(ys), max(ys)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    ink = is_color(img, INK)
    mouth = ink & (xx > x0) & (xx < x1) & (yy >= y0) & (yy <= y1)
    return iris, mouth


def segment(img, kind):
    op = img[..., 3] >= 128
    leaf_l, leaf_r = side_leaves(img)
    iris, mouth = mouth_between(img)
    used = leaf_l | leaf_r
    torso = op & ~used
    if (iris & ~torso).any():
        raise SystemExit(f"{kind} iris was cut into a leaf")
    if (mouth & ~torso).any():
        raise SystemExit(f"{kind} mouth was cut into a leaf")
    masks = {"leaf_l": leaf_l, "leaf_r": leaf_r, "torso": torso}
    check_partition(op, masks)
    if kind == "front":
        if int(iris.sum()) != 12 or int(mouth.sum()) < 4:
            raise SystemExit(f"front face changed: iris {int(iris.sum())} mouth {int(mouth.sum())}")
        ix = float(np.nonzero(iris)[1].mean())
        mx = float(np.nonzero(mouth)[1].mean())
        if not (mx > ix):
            raise SystemExit(f"mouth {mx:.2f} is not right of irises {ix:.2f}")
        if not (40 <= int(leaf_l.sum()) <= 80 and 55 <= int(leaf_r.sum()) <= 110):
            raise SystemExit(f"front leaves look wrong: {int(leaf_l.sum())} {int(leaf_r.sum())}")
    else:
        if iris.any() or mouth.any():
            raise SystemExit("back grew a face")
        if not (35 <= int(leaf_l.sum()) <= 90 and 40 <= int(leaf_r.sum()) <= 100):
            raise SystemExit(f"back leaves look wrong: {int(leaf_l.sum())} {int(leaf_r.sum())}")
    return masks


def place(arr):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    out[OFF_Y:OFF_Y + 96, OFF_X:OFF_X + 96] = arr
    return out


def root_band(part, torso, depth=4):
    """A few leaf pixels at the root, grown a short way into the leaf."""
    touch = part & dilate(torso, 1)
    if not touch.any():
        touch = part & dilate(torso, 4)
    if not touch.any():
        return touch
    ys, xs = np.nonzero(torso)
    cx, cy = float(xs.mean()), float(ys.mean())
    ty, tx = np.nonzero(touch)
    d = (tx - cx) ** 2 + (ty - cy) ** 2
    k = min(8, len(d) - 1)
    limit = np.partition(d, k)[k]
    band = np.zeros_like(part)
    keep = d <= limit
    band[ty[keep], tx[keep]] = True
    for _ in range(depth):
        band = dilate(band, 1) & part
    return band


def pivot_of(layer, torso_layer):
    part = layer[..., 3] >= 128
    torso = torso_layer[..., 3] >= 128
    touch = part & dilate(torso, 2)
    if not touch.any():
        touch = part & dilate(torso, 5)
    ys, xs = np.nonzero(touch if touch.any() else part)
    if len(xs) == 0:
        return 0.0, 0.0
    ty, tx = np.nonzero(torso)
    if len(tx) == 0:
        return float(xs.mean()), float(ys.mean())
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
    for name in LIMBS:
        if int(placed[name].sum()) == 0:
            continue
        layers["torso"] = paint_cushion(layers["torso"], root_band(placed[name], torso_m, 4), 4)
    facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    _iris, mouth = mouth_between(canvas)
    has_mouth = bool(mouth.any())
    if has_mouth:
        my, mx = np.nonzero(mouth)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
        j = int(mx.argmax()) if facing > 0 else int(mx.argmin())
        pivots["muzzle"] = (float(mx[j]), float(my[j]))
    else:
        pivots["mouth"] = pivots["center"]
        pivots["muzzle"] = pivots["center"]
    return {
        "kind": kind, "mirror": mirror, "facing": facing, "has_mouth": has_mouth,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        leaf_l=0.0, leaf_r=0.0, leaf_mode="up",
        wedge=0, ring=-1, spark=0,
        lean=0.0, knock=0, flash=False,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # The puff floats. The two leaves sway. Frame 0 is the still.
        p["bdy"] = -1 if wave(f, n, 1.0, phase=0.3) > 0.55 else 0
        p["leaf_l"] = wave(f, n, 5.0, phase=0.4)
        p["leaf_r"] = wave(f, n, 5.0, phase=1.5)
        p["leaf_mode"] = "up"
    elif action == "walk":
        # No feet. A glide, with the leaves fluttering out of phase.
        glide = wave(f, n, 1.4)
        p["bdx"] = int(round(glide))
        p["bdy"] = -int(round(abs(wave(f, n, 1.0, phase=1.0))))
        p["leaf_l"] = wave(f, n, 8.0, phase=0.2)
        p["leaf_r"] = wave(f, n, 8.0, phase=0.2 + math.pi)
        p["leaf_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 roar and lift the leaves. 5-8 the back, equal scale.
        # 9 wind-up. 10-12 the leaves slap forward. 13-15 return.
        leaf = [0, 8, 14, 18, 20, 0, 0, 0, 0, 18, 28, 34, 30, 12, 4, 0]
        bdx = [0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        bdy = [0, -1, -1, -2, -2, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0]
        p["leaf_l"] = float(leaf[f])
        p["leaf_r"] = float(leaf[f])
        p["bdx"] = int(bdx[f])
        p["bdy"] = int(bdy[f])
        p["leaf_mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["wedge"] = [0, 1, 2, 3, 3][f]
        if 2 <= f <= 4:
            p["ring"] = f - 2
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.45, 0.78, 1.0, 0.86, 0.64, 0.44, 0.26, 0.12, 0.04, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["leaf_l"] = 10.0 * env
        p["leaf_r"] = 10.0 * env
        p["leaf_mode"] = "up"
        p["flash"] = f == 1
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        # The back is the still, at equal scale. finish() must not swing it.
        p["leaf_l"] = 0.0
        p["leaf_r"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
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


def compose(rig, p):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    out = np.zeros_like(rig["canvas"])
    paste(out, shift(L["torso"], bdx, bdy))
    for name in LIMBS:
        ang = part_angle(rig, name, p["leaf_mode"], p[name])
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


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the existing mouth. Empty pixels only."""
    ox, oy = origin
    cols = (INK, BROWN, ORANGE)
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, PALE, SILVER)
    rad = 7 + age * 4
    for deg in range(-50, 51, 8):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, cols[age % 3])


def leaf_tip(rig, p, name):
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    ang = part_angle(rig, name, p["leaf_mode"], p[name])
    layer = moved(L[name], ang, piv[name], bdx, bdy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def impact(img, origin, fc, age):
    ox, oy = origin
    cols = (LEAF_HI, WHITE, PALE)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (2, -3), (2, 3))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    other = "back" if kind == "front" else "front"
    return rigs[(other, mirror)]


def moved_any(p):
    return (
        abs(p["leaf_l"]) >= 0.45 or abs(p["leaf_r"]) >= 0.45
        or p["bdx"] or p["bdy"] or p["knock"] or abs(p["lean"]) > 0.4
    )


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if not moved_any(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            body = rotate_nearest(body, -fc * p["lean"], rear_pivot(body, fc))
        body = fill_pinholes(despeckle(body))
        img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if rig["has_mouth"] and not p["turned"]:
        if p["wedge"]:
            mouth_wedge(img, rig["pivots"]["muzzle"], fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, rig["pivots"]["muzzle"], fc, p["ring"])
        if p["wedge"] or p["ring"] >= 0:
            img = snap(img, palette)
    if p["spark"] and action == "attack" and not p["turned"]:
        for name in LIMBS:
            tx, ty = leaf_tip(rig, p, name)
            impact(img, (tx + fc * 2, ty), fc, p["spark"])
        img = snap(img, palette)
    return img


def limb_lost(rig, p):
    body = compose(rig, p)
    L = rig["layers"]
    fc = rig["facing"]
    piv = rig["pivots"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    lost = []
    other = body[..., 3] >= 128
    for name in LIMBS:
        ang = part_angle(rig, name, p["leaf_mode"], p[name])
        layer = moved(L[name], ang, piv[name], bdx, bdy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        rest = other & ~limb
        if not (limb & dilate(rest, 1)).any():
            lost.append(name)
    return lost


def opaque_width(img):
    xs = np.nonzero(img[..., 3] >= 128)[1]
    if len(xs) == 0:
        return 0
    return int(xs.max() - xs.min() + 1)


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


def motion_contact(sheets, box):
    y0, y1, x0, x1 = box
    rows = []
    for di in range(8):
        row = []
        for act in ACTIONS:
            for f in range(16):
                if f < FRAMES[act]:
                    t = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                    row.append(t[y0:y1, x0:x1])
                else:
                    row.append(np.zeros((y1 - y0, x1 - x0, 4), np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


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


def build_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 stills")
    for im in (front, back):
        im[im[..., 3] < 128] = 0
        im[im[..., 3] >= 128, 3] = 255
    if int(np.nonzero(front[..., 3])[0].max()) != TIP_Y:
        raise SystemExit("front tip line changed")
    if int(np.nonzero(back[..., 3])[0].max()) != TIP_Y:
        raise SystemExit("back tip line changed")
    fm = segment(front, "front")
    bm = segment(back, "back")
    rigs = {}
    for kind, img, masks in (("front", front, fm), ("back", back, bm)):
        for mirror in (False, True):
            rigs[(kind, mirror)] = build_view(img, masks, kind, mirror)
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    if rigs[("front", False)]["facing"] != 1.0 or rigs[("back", False)]["facing"] != 1.0:
        raise SystemExit("facing locked to the wrong side")
    if not rigs[("front", False)]["has_mouth"] or rigs[("back", False)]["has_mouth"]:
        raise SystemExit("mouth is on the wrong view")
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "leaf_l": (40, 200, 90, 255),
        "leaf_r": (40, 160, 220, 255),
        "torso": (230, 210, 160, 255),
    }
    tiles = []
    for _img, masks in ((front, fm), (back, bm)):
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
    return n


def publish(bounds, sprite_h):
    import gen5_publish

    entry = {
        "dex": 546, "form": "normal", "variant": "cottonee", "gen": "gen5",
        "dir": "monsterSprite/gen5/0546_cottonee/normal", "sheetDir": "agent-sprite-forge",
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
    iris = any_color(front, IRIS_COLS)
    _i, mouth = mouth_between(front)
    print("face iris_x", round(float(np.nonzero(iris)[1].mean()), 2),
          "mouth_x", round(float(np.nonzero(mouth)[1].mean()), 2),
          "faces", "screen-right", flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    rest_sum = int(sum(rest_diff.values()))
    print("rest diff", rest_diff, flush=True)
    if rest_sum:
        raise SystemExit("frame 0 does not match the still")

    back_w = opaque_width(rigs[("back", False)]["canvas"])
    turned_w = []
    for d in DIRS:
        img = synth(rigs, d, "attack", 6, palette, effects=False)
        turned_w.append(opaque_width(img))
    print("back width", back_w, "attack6 widths", dict(zip(DIRS, turned_w)), flush=True)

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
        print("qa_samples", "rest_diff", rest_sum,
              "off_palette", metrics["off_palette"],
              "partial_alpha", metrics["partial_alpha"],
              "edge_touch", metrics["edge_touch"],
              "body_main_min", metrics["body_main_min"],
              "clip", metrics["edge_touching"], flush=True)
        atk = []
        for f in range(16):
            p = pose("attack", f)
            atk.append((f, round(float(p["leaf_l"]), 1), p["leaf_mode"], p["turned"],
                        p["spin"], p["wedge"], p["bdx"], p["bdy"]))
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
    box = content_box([rigs[d]["canvas"] for d in DIRS])
    motion = motion_contact(sheets, box)
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
        "version": VERSION, "hash": digest, "rest_diff": rest_sum, "idle0": idle0,
        "off_palette_colors": len(off), "partial_alpha": partial,
        "edge_touch": len(clip), "edge_touching": clip[:12],
        "body_main_min": round(min(bodies), 3) if bodies else 1,
        "sprite_h": sprite_h, "bounds": bounds, "cell": [CELL_W, CELL_H, FOOT_Y],
        "attack6_widths": dict(zip(DIRS, turned_w)), "back_width": back_w,
    }
    ok = (
        rest_sum == 0 and not any(idle0.values()) and not off
        and partial == 0 and not clip
        and min(turned_w) == back_w and max(turned_w) == back_w
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("qa_full", "rest_diff", rest_sum,
          "off_palette", len(off),
          "partial_alpha", partial,
          "edge_touch", len(clip),
          "idle0", idle0,
          "body_main_min", report["body_main_min"],
          "back_width", back_w,
          "attack6", dict(zip(DIRS, turned_w)),
          "ok", ok, flush=True)
    meta = {
        "generator": "forge_gen5_cottonee_actions", "version": VERSION, "key": KEY,
        "dex": 546, "variant": "cottonee", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "floating cotton puff (no feet; two side leaves; mouth on the front only)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-right (iris x43.67, mouth x44.0); back tuft is on the right",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
