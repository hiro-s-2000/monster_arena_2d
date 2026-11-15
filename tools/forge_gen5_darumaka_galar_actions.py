#!/usr/bin/env python3
"""Galarian Darumaka (0554 regional) 8-direction actions from this form's stills.

Measured on regional/darumaka-galar/front.png (96x96, bbox x29-66 y31-65,
788 px, alpha 0/255, sole y65):
  Eyes (white plus the grey pupils they sit in) mean x 47.11.
  The teal mouth is x40-45 y47-53, mean x 42.58, min x 40, left of the eyes,
  so the still faces screen-left. A short ink bar closes the mouth at y49-50.
  Forward ice horn x39-45 y31-36. Rear ice crystal x51-60 y33-39.
  Forward arm (screen-left) x29-37 y51-55. Far arm x62-66 y42-46.
  Two blue feet under y62, split by the ink gap at x50. Brow and eyes stay
  on the torso.
regional/darumaka-galar/back.png (bbox x28-64 y29-64, 775 px, sole y64,
outline is pure black) has no eyes and no mouth. Top ice x44-51 y29-35,
side ice x31-38 y35-40, left arm x28-33 y44-51, right arm x57-62 y36-40,
feet under y62 split at x45. The back sole is 1px higher, so the back is
shifted down 1px and both forms share one ground line.

Directions, front faces left:
  S SE E front mirrored (screen-right). W SW front as drawn (screen-left).
  N NE back as drawn. NW back mirrored.
  Attack 5-8 shows that direction's other side at equal scale: W turns into
  N, S turns into N flipped. No |cos| squash, no in-plane body spin.
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
SRC_DIR = ROOT / "monsterSprite/gen5/0554_darumaka/regional/darumaka-galar"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0554_darumaka/regional/darumaka-galar/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0554_darumaka-galar"
VERSION = "darumaka-galar_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 144, 144
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 65
BACK_FOOT = 64
OFF_Y = 20
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. Its 180 turn is the back as drawn.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 38.0
# Drawn strike is horn 28 and arm 34, both above these, so finish() does not
# grow the hit past the pose this script already paints.
PEAKS = {"horn": 16.0, "crystal": 8.0, "arm": 18.0, "arm_low": 4.0}
LIMBS = ("horn", "crystal", "arm_l", "arm_r", "foot_l", "foot_r")

INK_F = (16, 16, 16)
INK_B = (0, 0, 0)
CREAM = (235, 238, 230)
GREY = (189, 191, 193)
MGREY = (114, 115, 120)
DGREY = (164, 164, 172)
LGRAY = (215, 215, 217)
WHITE = (255, 255, 255)
TEAL = (38, 162, 140)
DTEAL = (43, 107, 103)
MINT = (41, 197, 169)
BMINT = (106, 234, 208)
BLUE = (40, 99, 173)
LBLUE = (72, 153, 201)
NAVY = (28, 49, 100)
ICE_COLS = (TEAL, DTEAL, MINT, BMINT)
BLUE_COLS = (BLUE, LBLUE, NAVY)
EYE_COLS = (WHITE, DGREY, LGRAY)
WEDGE_COLS = (INK_F, TEAL, BMINT)
RING_COLS = (WHITE, BMINT, LBLUE)
SPARK_COLS = (WHITE, BMINT, CREAM)

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
    if not out.any():
        return out
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
    """4x nearest in-plane rotation. No horizontal squash."""
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
            raise SystemExit(f"{kind} {name} is too small ({int(part.sum())})")
        if len(component_list(part)) != 1:
            raise SystemExit(f"{kind} {name} fell apart")
        if not (part & dilate(torso, 1)).any():
            raise SystemExit(f"{kind} {name} does not touch the body")


def segment_front(img):
    """Ice horn, rear crystal, two arms, two feet. Face and mouth stay on the torso."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ice = any_color(img, ICE_COLS)
    blue = any_color(img, BLUE_COLS)
    ink = is_color(img, INK_F)
    horn = flood((ice | ink) & (yy <= 40) & (xx <= 48), ice & (xx <= 48) & (yy <= 37))
    crystal = flood((ice | ink) & (yy <= 41) & (xx >= 49), ice & (xx >= 50) & (yy <= 40))
    arm_l = flood((blue | ink) & (xx <= 40) & (yy >= 48) & (yy <= 58), blue & (xx <= 38) & (yy >= 50))
    arm_r = flood((blue | ink) & (xx >= 59) & (yy >= 40) & (yy <= 49), blue & (xx >= 61) & (yy <= 47))
    feet = op & (yy >= 62)
    foot_l = feet & (xx <= 49)
    foot_r = feet & (xx >= 50)
    used = horn | crystal | arm_l | arm_r | foot_l | foot_r
    masks = {
        "horn": horn, "crystal": crystal, "arm_l": arm_l, "arm_r": arm_r,
        "foot_l": foot_l, "foot_r": foot_r, "torso": op & ~used,
    }
    check_partition(op, masks)
    require_limbs(masks, "front")
    eyes = any_color(img, EYE_COLS)
    mouth = ice & (yy >= 46)
    torso = masks["torso"]
    if int((eyes & torso).sum()) != int(eyes.sum()) or int(eyes.sum()) < 30:
        raise SystemExit("front eyes left the head")
    if int((mouth & torso).sum()) != int(mouth.sum()) or int(mouth.sum()) < 8:
        raise SystemExit("front mouth left the face")
    my, mx = np.nonzero(mouth)
    ey, ex = np.nonzero(eyes)
    if float(mx.mean()) >= float(ex.mean()) - 1.0:
        raise SystemExit(f"front does not face left: mouth {float(mx.mean()):.1f} eye {float(ex.mean()):.1f}")
    if int(np.nonzero(op)[0].max()) != FRONT_FOOT:
        raise SystemExit("front sole moved")
    return masks, mouth


def segment_back(img):
    """Top ice, side ice, two arms, two feet. No face is drawn."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    ice = any_color(img, ICE_COLS)
    blue = any_color(img, BLUE_COLS)
    ink = is_color(img, INK_B)
    horn = flood((ice | ink) & (yy <= 37) & (xx >= 42), ice & (xx >= 43) & (yy <= 36))
    crystal = flood((ice | ink) & (yy <= 42) & (xx <= 42), ice & (xx <= 40) & (yy <= 41))
    arm_l = flood((blue | ink) & (xx <= 33) & (yy >= 43) & (yy <= 54), blue & (xx <= 32) & (yy >= 45))
    arm_r = flood((blue | ink) & (xx >= 55) & (yy <= 42), blue & (xx >= 56) & (yy <= 41))
    feet = op & (yy >= 62)
    foot_l = feet & (xx <= 44)
    foot_r = feet & (xx >= 45)
    used = horn | crystal | arm_l | arm_r | foot_l | foot_r
    masks = {
        "horn": horn, "crystal": crystal, "arm_l": arm_l, "arm_r": arm_r,
        "foot_l": foot_l, "foot_r": foot_r, "torso": op & ~used,
    }
    check_partition(op, masks)
    require_limbs(masks, "back")
    if any_color(img, EYE_COLS).any():
        raise SystemExit("back grew eyes")
    plate = blue & (yy >= 43) & (yy <= 50) & (xx >= 35) & (xx <= 50)
    if plate.any() and int((plate & masks["torso"]).sum()) != int(plate.sum()):
        raise SystemExit("back plate left the torso")
    if int(np.nonzero(op)[0].max()) != BACK_FOOT:
        raise SystemExit("back sole moved")
    mouth = np.zeros(op.shape, bool)
    return masks, mouth


def place(arr, y_shift):
    if arr.ndim == 2:
        out = np.zeros((CELL_H, CELL_W), arr.dtype)
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


def build_view(img, masks, mouth_src, kind, mirror):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    canvas = place(img, y_shift)
    placed = {k: place(v, y_shift) for k, v in masks.items()}
    mouth = place(mouth_src, y_shift)
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
    for name, depth in (
        ("horn", 4), ("crystal", 4), ("arm_l", 4), ("arm_r", 4),
        ("foot_l", 4), ("foot_r", 4),
    ):
        layers["torso"] = paint_cushion(
            layers["torso"], joint_band(placed[name], torso_m, depth), depth + 1,
        )
    if kind == "front":
        facing = 1.0 if mirror else -1.0
        strike, off = "arm_l", "arm_r"
    else:
        facing = -1.0 if mirror else 1.0
        strike, off = "arm_r", "arm_l"
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], layers["torso"])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    has_mouth = bool(mouth.any())
    if has_mouth:
        my, mx = np.nonzero(mouth)
        j = int(mx.argmax()) if facing > 0 else int(mx.argmin())
        pivots["muzzle"] = (float(mx[j]), float(my[j]))
    else:
        pivots["muzzle"] = pivots["center"]
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "strike": strike, "off": off, "has_mouth": has_mouth,
        "layers": layers, "pivots": pivots, "signs": signs, "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        horn=0.0, horn_mode="up",
        crystal=0.0, crystal_mode="up",
        arm=0.0, arm_mode="fwd",
        arm_low=0.0, arm_low_mode="up",
        step_l=(0, 0), step_r=(0, 0),
        wedge=0, ring=-1, spark=0,
        lean=0.0, knock=0, flash=False,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay down. The ice and the arms fidget.
        p["horn"] = wave(f, n, 6.0, phase=0.4)
        p["crystal"] = wave(f, n, 4.0, phase=1.1)
        p["arm"] = wave(f, n, 5.0, phase=0.2)
        p["arm_low"] = wave(f, n, 4.0, phase=0.2 + math.pi)
        p["horn_mode"] = "up"
        p["crystal_mode"] = "up"
        p["arm_mode"] = "up"
        p["arm_low_mode"] = "up"
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        p["step_l"] = (int(round(2 * sa)), -int(round(2 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * sb)), -int(round(2 * max(0.0, sb))))
        p["arm"] = 12.0 * sa
        p["arm_low"] = 10.0 * sb
        p["horn"] = wave(f, n, 5.0, phase=0.5)
        p["crystal"] = wave(f, n, 4.0, phase=0.5 + math.pi)
        p["arm_mode"] = "fwd"
        p["arm_low_mode"] = "fwd"
        p["horn_mode"] = "up"
        p["crystal_mode"] = "up"
    elif action == "attack":
        # 0 rest. 1-4 roar from the mouth and raise the ice. 5-8 the other
        # side, equal scale. 9-12 the forward arm punches and the horn thrusts.
        # 13-15 return.
        horn = [0, 6, 12, 16, 16, 0, 0, 0, 0, 18, 24, 28, 28, 10, 4, 0]
        crystal = [0, 4, 8, 10, 10, 0, 0, 0, 0, 8, 12, 14, 14, 6, 2, 0]
        arm = [0, -6, -10, -12, -10, 0, 0, 0, 0, 18, 28, 34, 34, 12, 4, 0]
        arm_low = [0, 4, 6, 8, 6, 0, 0, 0, 0, -4, -6, -8, -8, -3, 0, 0]
        bdx = [0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0]
        p["horn"] = float(horn[f])
        p["crystal"] = float(crystal[f])
        p["arm"] = float(arm[f])
        p["arm_low"] = float(arm_low[f])
        p["bdx"] = int(bdx[f])
        p["horn_mode"] = "up" if f <= 4 else "fwd"
        p["crystal_mode"] = "up" if f <= 4 else "fwd"
        p["arm_mode"] = "fwd"
        p["arm_low_mode"] = "up" if f <= 4 else "fwd"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["wedge"] = [0, 1, 2, 3, 3][f]
        if 2 <= f <= 4:
            p["ring"] = f - 2
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.88, 0.68, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["horn"] = 8.0 * env
        p["horn_mode"] = "up"
        p["flash"] = f == 1
        p["spark"] = 1 if f in (1, 2) else 0
    p = finish(p, action, f, __file__)
    if action == "attack" and 5 <= f <= 8:
        p["horn"] = 0.0
        p["crystal"] = 0.0
        p["arm"] = 0.0
        p["arm_low"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
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


def limb_motion(rig, p):
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(p["bdy"])
    strike, off = rig["strike"], rig["off"]
    ang = {
        "horn": part_angle(rig, "horn", p["horn_mode"], p["horn"]),
        "crystal": part_angle(rig, "crystal", p["crystal_mode"], p["crystal"]),
        strike: part_angle(rig, strike, p["arm_mode"], p["arm"]),
        off: part_angle(rig, off, p["arm_low_mode"], p["arm_low"]),
        "foot_l": 0.0,
        "foot_r": 0.0,
    }
    step = {"foot_l": p["step_l"], "foot_r": p["step_r"]}
    delta = {}
    for name in LIMBS:
        if name in step:
            sx, sy = step[name]
            delta[name] = (bdx + int(round(fc * sx)), bdy + int(sy))
        else:
            delta[name] = (bdx, bdy)
    delta["torso"] = (bdx, bdy)
    return ang, delta


def compose(rig, p):
    L = rig["layers"]
    piv = rig["pivots"]
    ang, delta = limb_motion(rig, p)
    out = np.zeros_like(rig["canvas"])
    bdx, bdy = delta["torso"]
    paste(out, shift(L["torso"], bdx, bdy))
    for name in ("crystal", rig["off"], "horn", rig["strike"], "foot_l", "foot_r"):
        dx, dy = delta[name]
        paste(out, moved(L[name], ang[name], piv[name], dx, dy))
    return out


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the existing mouth. Empty pixels only."""
    ox, oy = origin
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 1 if i == 0 else 2
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, WEDGE_COLS[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    rad = 6 + age * 3
    for deg in range(-50, 51, 8):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.55
        put(img, x, y, RING_COLS[age % 3])


def spark(img, origin, fc, age):
    ox, oy = origin
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r // 2), (1, r // 2))):
        put(img, ox + fc * dx, oy + dy, SPARK_COLS[i % 3])


def flash_white(img):
    out = img.copy()
    op = out[..., 3] >= 128
    keep = is_color(out, INK_F) | is_color(out, INK_B)
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
    name = rig["strike"]
    fc = rig["facing"]
    ang, delta = limb_motion(rig, p)
    dx, dy = delta[name]
    layer = moved(rig["layers"][name], ang[name], rig["pivots"][name], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def horn_tip(rig, p):
    fc = rig["facing"]
    ang, delta = limb_motion(rig, p)
    dx, dy = delta["horn"]
    layer = moved(rig["layers"]["horn"], ang["horn"], rig["pivots"]["horn"], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def limb_lost(rig, p):
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


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["horn"]) >= 0.45 or abs(p["crystal"]) >= 0.45
        or abs(p["arm"]) >= 0.45 or abs(p["arm_low"]) >= 0.45
        or p["step_l"] != (0, 0) or p["step_r"] != (0, 0) or p["bdx"] or p["bdy"]
    )
    fc = rig["facing"]
    if p["knock"]:
        body = shift(body, int(round(-fc * p["knock"])), 0)
    if abs(p["lean"]) > 0.4:
        piv = rear_pivot(body, fc)
        body = rotate_nearest(body, -fc * p["lean"], piv)
    if moved_part or abs(p["lean"]) > 0.4:
        body = fill_pinholes(despeckle(body))
    img = snap(body, palette)
    if not effects:
        return img
    if p["flash"]:
        img = flash_white(img)
    if rig["has_mouth"] and not p["turned"]:
        mx, my = rig["pivots"]["muzzle"]
        mx += fc * p["bdx"]
        my += p["bdy"]
        if p["wedge"]:
            mouth_wedge(img, (mx, my), fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, (mx + fc * 2, my), fc, p["ring"])
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = horn_tip(rig, p)
        spark(img, (tx + fc * 2, ty), fc, p["spark"])
    elif p["spark"] and action == "hurt":
        cx, cy = rig["pivots"]["center"]
        spark(img, (cx - fc * 4, cy - 8), -fc, p["spark"])
    return snap(img, palette)


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
    x0 = max(0, int(xs.min()) - pad - 24)
    x1 = min(CELL_W, int(xs.max()) + pad + 24)
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
        combo = np.zeros((96, 192, 4), np.uint8)
        combo[:, :96] = imgs[0]
        combo[:, 96:] = imgs[1]
        combo_path = work / "palette_src.png"
        Image.fromarray(combo, "RGBA").save(combo_path)
        hexes = ["#%02x%02x%02x" % c for c in own]
        for d in (pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "build", "--input", str(combo_path),
             "--colors", str(len(hexes)), "--reserve", *hexes, "--output-dir", str(pal_dir)],
            capture_output=True, text=True,
        )
        if r.returncode:
            raise SystemExit("palette build failed: " + r.stdout + r.stderr)
        r = subprocess.run(
            [py, "-I", str(PALETTE_TOOL), "lock", "--palette", str(pal_dir / "palette.json"),
             "--source", str(combo_path), "--strict", "--output-dir", str(lock_dir)],
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
    fm, fmouth = segment_front(front)
    bm, bmouth = segment_back(back)
    rigs = {
        ("front", False): build_view(front, fm, fmouth, "front", False),
        ("front", True): build_view(front, fm, fmouth, "front", True),
        ("back", False): build_view(back, bm, bmouth, "back", False),
        ("back", True): build_view(back, bm, bmouth, "back", True),
    }
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "torso": (180, 180, 180, 255),
        "horn": (40, 210, 180, 255),
        "crystal": (40, 120, 220, 255),
        "arm_l": (240, 80, 80, 255),
        "arm_r": (240, 200, 40, 255),
        "foot_l": (250, 250, 250, 255),
        "foot_r": (180, 80, 220, 255),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m] = cols[name]
        tiles.append(np.repeat(np.repeat(part, 4, 0), 4, 1))
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
        "dex": 554, "form": "regional", "variant": "darumaka-galar", "gen": "gen5",
        "dir": "monsterSprite/gen5/0554_darumaka/regional/darumaka-galar",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


def frame_metrics(frames, palette):
    pal = {tuple(int(v) for v in c) for c in palette}
    off = set()
    partial = 0
    edge = 0
    for t in frames:
        a = t[..., 3]
        partial += int(((a != 0) & (a != 255)).sum())
        op = a == 255
        if op.any():
            cols = {tuple(int(v) for v in c) for c in np.unique(t[op][:, :3], axis=0)}
            off |= (cols - pal)
        if op[0].any() or op[-1].any() or op[:, 0].any() or op[:, -1].any():
            edge += 1
    return len(off), partial, edge


def top_x(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    y0 = int(ys.min())
    band = ys <= y0 + 2
    return float(xs[band].mean())


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
          "palette", len(palette), "foot", FOOT_Y, flush=True)
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("mouth", {d: rigs[d]["has_mouth"] for d in DIRS}, flush=True)
    print("strike", {d: rigs[d]["strike"] for d in ("S", "W", "N", "E")}, flush=True)

    rest_diff = {}
    for d in DIRS:
        rest_diff[d] = 0
        for act in ACTIONS:
            r = synth(rigs, d, act, 0, palette, effects=False)
            rest_diff[d] += int(np.any(r != rigs[d]["canvas"], axis=2).sum())
            r2 = synth(rigs, d, act, 0, palette, effects=True)
            rest_diff[d] += int(np.any(r2 != rigs[d]["canvas"], axis=2).sum())
    rest_sum = int(sum(rest_diff.values()))
    print("rest diff", rest_diff, "sum", rest_sum, flush=True)
    if rest_sum:
        raise SystemExit("frame 0 does not match the still")

    for f in range(5, 9):
        turn_w = synth(rigs, "W", "attack", f, palette, effects=False)
        turn_s = synth(rigs, "S", "attack", f, palette, effects=False)
        if np.any(turn_w != rigs["N"]["canvas"]):
            raise SystemExit(f"W turn f{f} is not the N back at equal scale")
        if np.any(turn_s != rigs["NW"]["canvas"]):
            raise SystemExit(f"S turn f{f} is not N flipped at equal scale")

    rest_w = rigs["W"]["canvas"]
    hurt_w = synth(rigs, "W", "hurt", 3, palette, effects=False)
    rest_s = rigs["S"]["canvas"]
    hurt_s = synth(rigs, "S", "hurt", 3, palette, effects=False)
    print("hurt_top_dx",
          "W", round(top_x(hurt_w) - top_x(rest_w), 2),
          "S", round(top_x(hurt_s) - top_x(rest_s), 2), flush=True)

    if args.samples:
        lost = {}
        sample_frames = []
        clip = []
        for d in ("S", "W", "N", "E"):
            for act, fs in (
                ("idle", (0, 8)),
                ("walk", (4, 12)),
                ("attack", (2, 4, 6, 8, 11)),
                ("hurt", (3,)),
            ):
                for f in fs:
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
                    t = synth(rigs, d, act, f, palette)
                    sample_frames.append(t)
                    a = t[..., 3] == 255
                    if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                        clip.append(f"{act}:{d}:{f}")
        off, partial, edge = frame_metrics(sample_frames, palette)
        print("limb lost", lost or "none", flush=True)
        print("sample clip", clip or "none", flush=True)
        print(
            f"MEASURE rest_diff_sum={rest_sum} off_palette={off} "
            f"partial_alpha={partial} edge_touch={edge}",
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
    print(
        f"MEASURE rest_diff_sum={int(sum(rest_diff.values()))} off_palette={len(off)} "
        f"partial_alpha={partial} edge_touch={len(clip)}",
        flush=True,
    )
    print(json.dumps({k: report[k] for k in report if k != "bounds"}), flush=True)
    meta = {
        "generator": "forge_gen5_darumaka_galar_actions", "version": VERSION, "key": KEY,
        "dex": 554, "variant": "darumaka-galar", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "round daruma (ice horn, rear crystal, two stub arms, two stub feet, mouth)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (mouth x40 < eye mean x47.1)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if args.no_publish:
        print("qa ok, publish skipped", flush=True)
        return
    print("synced", sync_outputs(), flush=True)
    publish(bounds, sprite_h)
    print("published", KEY, VERSION, flush=True)


if __name__ == "__main__":
    main()
