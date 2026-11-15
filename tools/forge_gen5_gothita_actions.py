#!/usr/bin/env python3
"""Gothita normal (0574) 8-direction actions from this form's own stills.

Measured on normal/front.png (96x96, bbox x34-61 y27-65, 640 px, 13 colours,
alpha 0/255, sole y65):
  Eyes (8,74,139) and (57,164,255), 18 px, mean x 50.50.
  Mouth (255,106,74) and (123,74,65), 10 px, mean x 53.20, right of the eyes.
  The still faces screen-right. The mouth is drawn, so the roar is a wedge
  and rings. Nothing else is given a mouth.
  The white bow is the hair (x34-61 y27-43). The pink face, blue eyes and
  the white dress stay on the torso. No arms, wings, tail, horns or pincers
  are drawn. Two black feet: screen-left x42-45 y62-65 (the sole), and
  screen-right x48-51 y62-64.

normal/back.png (bbox x35-63 y28-67, 656 px, 9 colours, all in the front
palette, sole y67): no eyes and no mouth. The bow is the white/grey hair
above the pink back of the head. Two feet, split under y63. The back sole
is 2px lower, so the back is shifted up 2px and both forms share one ground.

Directions, front faces screen-right:
  S SE E front as drawn. W SW front mirrored.
  NE back as drawn. N NW back mirrored.
  E is screen-right, W is screen-left, N is the back row.
  Attack 5-8 shows that direction's back at equal scale (NE, or N which is
  NE flipped). No |cos| squash and no in-plane body spin.

  idle 16: the bow breathes. Feet stay down. Frame 0 is the still.
  walk 16: the two feet that are drawn take turns. The bow counters.
  attack: 0 still, 1-4 the mouth roars and the bow lifts,
          5-8 the back, 9-10 the leading foot cocks (14-22 deg),
          11-12 it kicks (34 deg), 13-15 recover.
  hurt 12: white flash, knock-back lean (<=38 deg) and recover.
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
SRC_DIR = ROOT / "monsterSprite/gen5/0574_gothita/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0574_gothita/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0574_gothita"
VERSION = "gothita_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 65
BACK_FOOT = 67
# Back sole is 2px lower. Shift it up. The 96px blit stays inside the cell.
OFF_Y = CELL_H - 96 - max(0, FRONT_FOOT - BACK_FOOT)
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn faces screen-right. Its 180 turn is the back as drawn.
VIEW_OF = {
    "S": ("front", False), "SE": ("front", False), "E": ("front", False),
    "NE": ("back", False),
    "N": ("back", True), "NW": ("back", True),
    "W": ("front", True), "SW": ("front", True),
}
MAX_LEAN = 38.0
# Roar ceilings. The kick is drawn at 34 and the bow lift at 14, both above
# these, so finish() does not push a frame past the angle this script draws.
# Frames 5-8 are cleared after finish().
PEAKS = {"bow": 6.0, "foot": 14.0}
LIMBS = ("bow", "foot_l", "foot_r")
STRIKE_FOOT = "foot_r"

INK = (24, 24, 24)
DARK = (49, 49, 49)
GREY = (65, 65, 65)
MGREY = (115, 115, 115)
LGREY = (180, 180, 180)
WHITE = (255, 255, 255)
PINK = (205, 139, 189)
DPINK = (90, 49, 74)
MAUVE = (148, 106, 131)
NAVY = (8, 74, 139)
BLUE = (57, 164, 255)
BROWN = (123, 74, 65)
ORANGE = (255, 106, 74)
EYE_COLS = (NAVY, BLUE)
MOUTH_COLS = (ORANGE, BROWN)
PINK_COLS = (PINK, DPINK)

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
    """4x nearest in-plane rotation. Positive follows the forge formula. No squash."""
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


def keep_main(zone):
    groups = component_list(zone)
    if not groups:
        raise SystemExit("empty part")
    groups.sort(key=len, reverse=True)
    part = mask_from(zone.shape, groups[0])
    for g in groups[1:]:
        sm = mask_from(zone.shape, g)
        if (sm & dilate(part, 1)).any():
            part |= sm
    return part


def check_partition(op, masks):
    acc = np.zeros_like(op)
    for name, m in masks.items():
        if (m & acc).any():
            raise SystemExit(f"overlap in {name}")
        if (m & ~op).any():
            raise SystemExit(f"{name} left the still")
        acc |= m
    if int(acc.sum()) != int(op.sum()):
        raise SystemExit(f"parts do not partition: {int(acc.sum())} vs {int(op.sum())}")


def require_limbs(masks, kind):
    torso = masks["torso"]
    if len(component_list(torso)) != 1:
        raise SystemExit(f"{kind} torso fell apart")
    for name in LIMBS:
        part = masks[name]
        if int(part.sum()) < 8:
            raise SystemExit(f"{kind} {name} is too small")
        if len(component_list(part)) != 1:
            raise SystemExit(f"{kind} {name} fell apart")
        if not (part & dilate(torso, 1)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")


def segment_front(img):
    """Bow, two feet. Face, eyes, mouth and the dress stay on the torso."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    pink = any_color(img, PINK_COLS)
    eye = any_color(img, EYE_COLS)
    mouth = any_color(img, MOUTH_COLS)
    zone = op & ~pink & ~eye & ~mouth & (
        (yy <= 37) | ((yy <= 43) & ((xx <= 44) | (xx >= 55)))
    )
    bow = keep_main(zone)
    foot_l = op & (yy >= 62) & (xx <= 46) & ~bow
    foot_r = op & (yy >= 62) & (xx >= 47) & ~bow
    torso = op & ~bow & ~foot_l & ~foot_r
    masks = {"bow": bow, "foot_l": foot_l, "foot_r": foot_r, "torso": torso}
    check_partition(op, masks)
    require_limbs(masks, "front")
    if int((eye & torso).sum()) != int(eye.sum()) or int(eye.sum()) != 18:
        raise SystemExit("front eyes left the head")
    if int((mouth & torso).sum()) != int(mouth.sum()) or int(mouth.sum()) != 10:
        raise SystemExit("front mouth left the face")
    my, mx = np.nonzero(mouth)
    ey, ex = np.nonzero(eye)
    if float(mx.mean()) <= float(ex.mean()):
        raise SystemExit(
            f"front does not face right: mouth {float(mx.mean()):.2f} eye {float(ex.mean()):.2f}"
        )
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front sole moved")
    if int(np.nonzero(bow)[0].max()) > 43:
        raise SystemExit("front bow ate the dress")
    return masks


def segment_back(img):
    """Bow and two feet. No face is drawn. The pink back of the head stays put."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    pink = any_color(img, PINK_COLS)
    zone = op & ~pink & (
        (yy <= 40) | ((yy <= 44) & ((xx <= 40) | (xx >= 55)))
    )
    bow = keep_main(zone)
    foot_l = op & (yy >= 63) & (xx <= 42) & ~bow
    foot_r = op & (yy >= 63) & (xx >= 43) & ~bow
    torso = op & ~bow & ~foot_l & ~foot_r
    masks = {"bow": bow, "foot_l": foot_l, "foot_r": foot_r, "torso": torso}
    check_partition(op, masks)
    require_limbs(masks, "back")
    if any_color(img, EYE_COLS).any() or any_color(img, MOUTH_COLS).any():
        raise SystemExit("back grew a face")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back sole moved")
    if int(np.nonzero(bow)[0].max()) > 44:
        raise SystemExit("back bow ate the dress")
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
    """Which rotation sign lifts the tip, and which sends it toward facing.

    A 9px foot does not move at 12 degrees under nearest-neighbour, and the
    34 degree kick then points the other way. Probe small parts at 28.
    """
    n = int((layer[..., 3] >= 128).sum())
    if n < 4:
        return {"up": 1.0, "fwd": 1.0}
    probe = 28.0 if n < 24 else 12.0

    def tip_after(ang):
        return tip_xy(rotate_nearest(layer, ang, pivot), pivot)

    x_pos, y_pos = tip_after(probe)
    x_neg, y_neg = tip_after(-probe)
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
    # Unmirrored art faces screen-right on the front. The back is the same
    # turn, so an unmirrored back also aims screen-right.
    facing = -1.0 if mirror else 1.0
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    torso_m = placed["torso"]
    for name, depth in (("bow", 5), ("foot_l", 4), ("foot_r", 4)):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth + 1,
        )
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    mouth = any_color(canvas, MOUTH_COLS)
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
        bow=0.0, bow_mode="up",
        foot=0.0, foot_mode="fwd",
        step_l=(0, 0), step_r=(0, 0),
        wedge=0, ring=-1,
        lean=0.0, knock=0, flash=False,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay on the ground. The bow breathes. The head does not slide.
        p["bow"] = wave(f, n, 5.0, phase=0.4)
        p["bow_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(2 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(2 * max(0.0, sb))))
        p["bow"] = wave(f, n, 6.0, cycles=2.0, phase=0.6)
        p["bow_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 the drawn mouth roars and the bow lifts. 5-8 the back,
        # equal scale. 9-10 the leading foot cocks. 11-12 it kicks. 13-15 return.
        bow = [0, 6, 10, 14, 12, 0, 0, 0, 0, 6, 6, 5, 3, 2, 0, 0]
        foot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 22, 34, 34, 12, 4, 0]
        # The drawn foot is 9px. The 34 deg kick also steps 2px toward facing.
        reach = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 0, 0]
        lift = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0]
        p["bow"] = float(bow[f])
        p["foot"] = float(foot[f])
        p["step_r"] = (reach[f], lift[f])
        p["bow_mode"] = "up"
        p["foot_mode"] = "fwd"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["wedge"] = [0, 1, 2, 3, 3][f]
        if 2 <= f <= 4:
            p["ring"] = f - 2
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(2 * env))
        p["flash"] = f == 1
        p["bow"] = 8.0 * env
        p["bow_mode"] = "up"
        p["foot"] = 6.0 * env
        p["foot_mode"] = "up"
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        # The back is the still, at equal scale. finish() must not swing it.
        p["bow"] = 0.0
        p["foot"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["wedge"] = 0
        p["ring"] = -1
        p["spin"] = 0
        p["lean"] = 0.0
        p["knock"] = 0
        p["turned"] = True
    if action == "attack" and abs(float(p["foot"])) > 34.05:
        p["foot"] = 34.0 if p["foot"] > 0 else -34.0
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


def limb_motion(rig, p):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    ang = {
        "bow": part_angle(rig, "bow", p["bow_mode"], p["bow"]),
        "foot_l": 0.0,
        "foot_r": part_angle(rig, "foot_r", p["foot_mode"], p["foot"]),
    }
    step = {"foot_l": p["step_l"], "foot_r": p["step_r"]}
    delta = {"bow": (bdx, bdy)}
    for name in ("foot_l", "foot_r"):
        sx, sy = step[name]
        delta[name] = (bdx + int(round(fc * sx)), bdy + int(sy))
    return ang, delta


def compose(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    ang, delta = limb_motion(rig, p)
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = delta["bow"]
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("foot_l", "foot_r", "bow"):
        dx, dy = delta[name]
        paste(out, moved(L[name], ang[name], piv[name], dx, dy))
    return out


def bridge_cracks(img):
    """Close a 1px resampling crack so a moved part stays on the body."""
    out = img.copy()
    op = out[..., 3] >= 128
    groups = component_list(op)
    if len(groups) <= 1:
        return out
    groups.sort(key=len, reverse=True)
    main = mask_from(op.shape, groups[0])
    h, w = op.shape
    for g in groups[1:]:
        if len(g) < 4:
            continue
        part = mask_from(op.shape, g)
        crack = dilate(main, 1) & dilate(part, 1) & (out[..., 3] < 128)
        if not crack.any():
            continue
        ys, xs = np.nonzero(crack)
        for y, x in zip(ys.tolist(), xs.tolist()):
            src = None
            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx, 3] >= 128:
                    src = out[ny, nx].copy()
                    break
            if src is None:
                continue
            out[y, x] = src
            out[y, x, 3] = 255
        main = main | part | crack
    return out


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the existing mouth. Empty pixels only."""
    ox, oy = origin
    cols = (INK, ORANGE, PINK)
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, LGREY, BLUE)
    rad = 5 + age * 3
    for deg in range(-50, 51, 10):
        a = math.radians(deg)
        x = ox + fc * (3 + rad * math.cos(a))
        y = oy + rad * math.sin(a) * 0.5
        put(img, x, y, cols[age % 3])


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


def limb_lost(rig, p):
    if motionless(p):
        return []
    body = compose(rig, p)
    ang, delta = limb_motion(rig, p)
    lost = []
    for name in LIMBS:
        dx, dy = delta[name]
        layer = moved(rig["layers"][name], ang[name], rig["pivots"][name], dx, dy)
        limb = layer[..., 3] >= 128
        if not limb.any():
            lost.append(name)
            continue
        other = (body[..., 3] >= 128) & ~limb
        if not (limb & dilate(other, 1)).any():
            lost.append(name)
    return lost


def rig_for(rigs, d, p):
    if not p["turned"]:
        return rigs[d]
    kind, mirror = VIEW_OF[d]
    other = "back" if kind == "front" else "front"
    return rigs[(other, mirror)]


def motionless(p):
    return (
        abs(p["bow"]) < 0.45 and abs(p["foot"]) < 0.45
        and p["step_l"] == (0, 0) and p["step_r"] == (0, 0)
        and not p["bdx"] and not p["bdy"]
        and abs(p["lean"]) < 0.4 and not p["knock"]
    )


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    fc = rig["facing"]
    if motionless(p):
        img = rig["canvas"].copy()
    else:
        body = compose(rig, p)
        if p["knock"]:
            body = shift(body, int(round(-fc * p["knock"])), 0)
        if abs(p["lean"]) > 0.4:
            px, py = rear_pivot(body, fc)
            body = rotate_nearest(body, -fc * p["lean"], (px, py))
        if abs(p["lean"]) > 0.4 or abs(p["bow"]) >= 0.45 or abs(p["foot"]) >= 0.45:
            body = bridge_cracks(body)
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
    return img


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


def tile_grid(tiles, cols, bg=(18, 22, 32, 255)):
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    out = np.full((rows * th, cols * tw, 4), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        m = t[..., 3] >= 128
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
    return out


def preview_image(sheets):
    pick = {"idle": 4, "walk": 4, "attack": 11, "hurt": 3}
    tiles = []
    for act in ACTIONS:
        for di in range(8):
            f = pick[act]
            tiles.append(sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W])
    return Image.fromarray(tile_grid(tiles, 8), "RGBA")


def content_box(canvases, pad=8):
    acc = np.zeros(canvases[0].shape[:2], bool)
    for c in canvases:
        acc |= c[..., 3] >= 128
    ys, xs = np.nonzero(acc)
    y0 = max(0, int(ys.min()) - pad - 18)
    y1 = min(CELL_H, int(ys.max()) + pad + 16)
    x0 = max(0, int(xs.min()) - pad - 28)
    x1 = min(CELL_W, int(xs.max()) + pad + 28)
    return y0, y1, x0, x1


def motion_strip(sheets, di, box):
    y0, y1, x0, x1 = box
    rows = []
    for act in ACTIONS:
        row = []
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
    fm = segment_front(front)
    bm = segment_back(back)
    rigs = {
        ("front", False): build_view(front, fm, "front", False),
        ("front", True): build_view(front, fm, "front", True),
        ("back", False): build_view(back, bm, "back", False),
        ("back", True): build_view(back, bm, "back", True),
    }
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "torso": (70, 130, 210, 255),
        "bow": (255, 220, 80, 255),
        "foot_l": (250, 250, 250, 255),
        "foot_r": (255, 80, 40, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 4, 0), 4, 1))
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
        "dex": 574, "form": "normal", "variant": "gothita", "gen": "gen5",
        "dir": "monsterSprite/gen5/0574_gothita/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def metrics_of(frames, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = 0
    partial = 0
    edge = 0
    seen = set()
    for t in frames:
        a = t[..., 3]
        partial += int(((a != 0) & (a != 255)).sum())
        op = a == 255
        if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
            edge += 1
        if op.any():
            cols = {tuple(int(v) for v in c) for c in np.unique(t[op][:, :3], axis=0)}
            seen |= cols
    off = len(seen - pal)
    return off, partial, edge


def tip_shift(rigs, d, action, f, name):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    ang, delta = limb_motion(rig, p)
    dx, dy = delta[name]
    layer = moved(rig["layers"][name], ang[name], rig["pivots"][name], dx, dy)
    rest = rig["layers"][name]
    def center(img):
        m = img[..., 3] >= 128
        ys, xs = np.nonzero(m)
        return float(xs.mean()), float(ys.mean())
    x1, y1 = center(layer)
    x0, y0 = center(rest)
    return x1 - x0, y1 - y0


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
    print(
        "parts front", {k: int(v.sum()) for k, v in fm.items()},
        "back", {k: int(v.sum()) for k, v in bm.items()},
        "palette", len(palette), "foot", FOOT_Y, flush=True,
    )
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("mouth", {d: rigs[d]["has_mouth"] for d in DIRS}, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
        r2 = synth(rigs, d, "idle", 0, palette, effects=True)
        rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
        composed = snap(compose(rigs[d], base_pose()), palette)
        rest_diff[d] += int(np.any(composed != rigs[d]["canvas"], axis=2).sum())
    print("rest diff", rest_diff, "sum", sum(rest_diff.values()), flush=True)
    if any(rest_diff.values()):
        raise SystemExit("frame 0 does not match the still")

    turn_e = synth(rigs, "E", "attack", 6, palette, effects=False)
    turn_w = synth(rigs, "W", "attack", 6, palette, effects=False)
    turn_s = synth(rigs, "S", "attack", 6, palette, effects=False)
    if np.any(turn_e != rigs["NE"]["canvas"]) or np.any(turn_s != rigs["NE"]["canvas"]):
        raise SystemExit("E/S turn is not the unmirrored back at equal scale")
    if np.any(turn_w != rigs["N"]["canvas"]):
        raise SystemExit("W turn is not the N back at equal scale")
    if np.any(rigs["N"]["canvas"] != rigs["NE"]["canvas"][:, ::-1]):
        raise SystemExit("N is not NE flipped at equal scale")

    bow_dx, bow_dy = tip_shift(rigs, "S", "attack", 3, "bow")
    kick_dx, kick_dy = tip_shift(rigs, "S", "attack", 11, STRIKE_FOOT)
    print(
        "bow_tip", round(bow_dx, 2), round(bow_dy, 2),
        "kick_tip", round(kick_dx, 2), round(kick_dy, 2),
        "facing_S", rigs["S"]["facing"], flush=True,
    )

    if args.samples:
        lost = {}
        frames = []
        for d in DIRS:
            for act in ACTIONS:
                for f in range(FRAMES[act]):
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
                    frames.append(synth(rigs, d, act, f, palette))
        off, partial, edge = metrics_of(frames, palette)
        print("limb lost", lost or "none", flush=True)
        print(
            "off_palette", off, "partial_alpha", partial, "edge_touching", edge,
            flush=True,
        )
        box = content_box([rigs[d]["canvas"] for d in DIRS])
        y0, y1, x0, x1 = box
        tiles = []
        for d in ("S", "W", "N", "E"):
            for act, f in (
                ("idle", 0), ("idle", 8),
                ("walk", 4), ("walk", 12),
                ("attack", 0), ("attack", 3), ("attack", 6), ("attack", 8),
                ("attack", 11), ("attack", 14),
                ("hurt", 1), ("hurt", 3),
            ):
                t = synth(rigs, d, act, f, palette)
                tiles.append(t[y0:y1, x0:x1])
        sheet = np.repeat(np.repeat(tile_grid(tiles, 12), 2, 0), 2, 1)
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
        and partial == 0 and not clip and min(bodies) >= 0.90
    )
    report["ok"] = ok
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_gothita_actions", "version": VERSION, "key": KEY,
        "dex": 574, "variant": "gothita", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "small biped (white hair bow, drawn mouth, two feet, no arms)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-right (mouth mean x 53.20 > eye mean x 50.50)",
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
