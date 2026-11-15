#!/usr/bin/env python3
"""Panpour normal (0515) 8-direction action sheet from this form's own stills.

Measured on normal/front.png (96x96, bbox x22-65 y27-70, 12 colours, alpha 0/255):
  Eye whites x33-46 y43-49, mean x 39.5 (left eye x33-36, right eye x43-46).
  Grey pupils are the four pixels (33,46), (36,46), (43,47), (46,47).
  The chin, the lowest tan of the muzzle, is x37-39 y55 (mean x 38), left of
  the eyes, so the still faces screen-left. The smile is the black pixels
  (39,51), (41,52), (42,52). Inner ears are the pale cyan, not a second face.
  The water tuft is every opaque pixel at y<=36. Two tan arms hang beside the
  belly (left x25-35 y57-64, right x45-58 y57-68, the right hand by the tail).
  Two feet split under y65 (left sole y69, right sole y70). The tail is the
  blue loop x57-65 y58-65, touching the right hand.
normal/back.png (bbox x27-68 y27-69, 9 colours, no eyes and no smile):
  The same tuft at y<=36, the tail loop on the left x27-35 y60-67, two arms,
  and two feet. The lowest opaque row is y69, one pixel above the front sole.

Directions: the front still looks screen-left, so W SW keep it and S SE E are
mirrored. N NE use the back shifted so the sole sits on the same ground as the
mirrored front. NW mirrors that back. E is screen-right, W is screen-left,
N is the back. Attack 5-8 swaps to that opposite view at equal scale. The body
is never laid on its side and never squashed by |cos|.

  idle   : feet planted. The tuft, hands and tail breathe. Frame 0 is the still.
  walk   : the two feet step. Hands swing. The tail counters twice a cycle.
  attack : 0 still, 1-4 the tuft and hands rise and the smile opens,
           5-8 the other side at equal scale with no extra pose,
           9-12 both hands and the tail strike (at most 32 deg), 13-15 recover.
  hurt   : 12 frames, lean at most 36 deg, knock-back, one white flash.
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
SRC_DIR = ROOT / "monsterSprite/gen5/0515_panpour/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen5/0515_panpour/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"

KEY = "0515_panpour"
VERSION = "panpour_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H = 176, 176
OFF_X = (CELL_W - 96) // 2
FRONT_FOOT = 70
BACK_FOOT = 69
# Front sole-band mean x is 44.94. Back sole-band mean x is 42.93.
# The back is placed 7px right so a mirrored front and the unmirrored back
# share the same ground point (cell centre is the mirror axis).
BACK_DX = 7
OFF_Y = CELL_H - 96 - (FRONT_FOOT - BACK_FOOT)
FOOT_Y = OFF_Y + FRONT_FOOT
# Front as drawn looks screen-left. S (front mirrored, looks right) turns
# into N (back, sole-aligned) by flipping the mirror.
VIEW_OF = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False),
    "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
MAX_LEAN = 36.0
# Wind-up peaks finish() may raise on frames 5-12. The hand and tail strike
# are drawn above these, so finish does not shrink the hit.
PEAKS = {"arm": 12.0, "tail": 12.0, "kick": 8.0}
LIMBS = ("crest", "tail", "arm_l", "arm_r", "foot_l", "foot_r")

INK = (0, 0, 0)
NAVY = (0, 57, 115)
TEAL = (24, 139, 172)
MID = (32, 115, 156)
CYAN = (41, 180, 222)
GREY = (82, 82, 82)
BROWN = (115, 82, 32)
CREAM = (156, 230, 238)
TAN_D = (164, 123, 57)
TAN_M = (205, 164, 90)
TAN = (255, 205, 123)
WHITE = (255, 255, 255)
TAN_COLS = (BROWN, TAN_D, TAN_M, TAN)
BLUE_COLS = (NAVY, TEAL, MID, CYAN)
EYE_COLS = (WHITE, GREY)

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
        g = []
        while stack:
            y, x = stack.pop()
            g.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(g)
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


def paint_cushion(layer, allowed, steps, dirs=("up", "down", "left", "right")):
    """Grow this layer's own colours into allowed pixels. Hidden at rest."""
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


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """0 on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def mask_from(group):
    m = np.zeros((96, 96), bool)
    for y, x in group:
        m[y, x] = True
    return m


def touches(part, body):
    if not part.any() or not body.any():
        return False
    return bool((part & dilate(body, 1)).any())


def seal(op, masks):
    """Keep each limb one piece. Stray pixels return to whatever they touch."""
    limbs = list(LIMBS)
    taken = np.zeros_like(op)
    for name in limbs:
        masks[name] = masks[name] & op & ~taken
        taken |= masks[name]
    for name in limbs:
        groups = component_list(masks[name])
        if len(groups) <= 1:
            continue
        groups.sort(key=len, reverse=True)
        keep = mask_from(groups[0])
        masks[name] = keep
    torso = op.copy()
    for name in limbs:
        torso &= ~masks[name]
    groups = component_list(torso)
    if len(groups) > 1:
        groups.sort(key=len, reverse=True)
        for g in groups[1:]:
            sm = mask_from(g)
            hits = [name for name in limbs if touches(sm, masks[name])]
            if not hits:
                continue
            best = max(hits, key=lambda n: int((sm & dilate(masks[n], 1)).sum()))
            masks[best] |= sm
            torso &= ~sm
    # A limb that gained a scrap stays one piece because the scrap touches it.
    masks["torso"] = torso
    return masks


def check_parts(img, masks, kind):
    op = img[..., 3] >= 128
    total = sum(int(m.sum()) for m in masks.values())
    if total != int(op.sum()):
        raise SystemExit(f"{kind} parts do not partition: {total} vs {int(op.sum())}")
    overlap = np.zeros(op.shape, np.int16)
    for m in masks.values():
        overlap += m.astype(np.int16)
    if int((overlap > 1).sum()):
        raise SystemExit(f"{kind} parts overlap")
    torso = masks["torso"]
    if largest_component(torso) != int(torso.sum()):
        raise SystemExit(f"{kind} torso fell apart")
    for name in LIMBS:
        part = masks[name]
        if int(part.sum()) < 8:
            raise SystemExit(f"{kind} {name} is too small: {int(part.sum())}")
        if not touches(part, op & ~part):
            raise SystemExit(f"{kind} {name} does not touch the body")
        groups = component_list(part)
        scrap = sum(len(g) for g in groups[1:]) if len(groups) > 1 else 0
        if scrap > 6:
            raise SystemExit(f"{kind} {name} fell apart ({len(groups)} pieces, scrap {scrap})")


def crest_of(op):
    yy = np.arange(96)[:, None]
    region = op & (yy <= 36)
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        raise SystemExit("no water tuft")
    top = int(ys.min())
    seeds = np.zeros_like(op)
    seeds[top, xs[ys == top]] = True
    return flood(region, seeds)


def segment_front(img):
    """Tuft, two arms, two feet, the tail loop. Ears, face and belly stay torso."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    tan = any_color(img, TAN_COLS)
    blue = any_color(img, BLUE_COLS)
    black = is_color(img, INK)
    crest = crest_of(op)

    region = blue & (xx >= 56) & (yy >= 57)
    ys, xs = np.nonzero(region)
    i = int(xs.argmax())
    seeds = np.zeros_like(op)
    seeds[int(ys[i]), int(xs[i])] = True
    tail = flood(region, seeds)
    tail = tail | (black & dilate(tail, 1) & (xx >= 55) & (yy >= 56))

    sole_y = int(np.nonzero(tan)[0].max())
    if sole_y != FRONT_FOOT:
        raise SystemExit(f"front sole moved to {sole_y}")
    legs = flood(tan, tan & (yy == sole_y))
    arm_l = tan & ~legs & (xx <= 40) & (yy >= 56)
    arm_r = tan & ~legs & (xx >= 44) & (yy >= 56)
    for _ in range(3):
        extra = black & dilate(arm_r, 1) & (xx >= 48) & (yy >= 57) & (yy <= 67)
        extra |= black & dilate(arm_r, 1) & (xx >= 45) & (yy >= 57) & (yy <= 68) & ~dilate(legs, 1)
        arm_r = arm_r | extra
    feet = legs & (yy >= 65)
    foot_l = feet & (xx <= 40)
    foot_r = feet & (xx >= 41)
    arm_l = arm_l | (black & dilate(arm_l, 1) & (xx <= 42) & (yy >= 55) & (yy <= 66))
    arm_r = arm_r | (black & dilate(arm_r, 1) & (xx >= 44) & (yy >= 56) & (yy <= 68) & ~feet)
    foot_l = foot_l | (black & dilate(foot_l, 1) & (yy >= 64) & (xx <= 42))
    foot_r = foot_r | (black & dilate(foot_r, 1) & (yy >= 64) & (xx >= 39) & (xx <= 52) & ~arm_r)
    tail = tail & ~arm_r & ~arm_l & ~crest

    masks = {
        "crest": crest, "tail": tail, "arm_l": arm_l, "arm_r": arm_r,
        "foot_l": foot_l, "foot_r": foot_r,
    }
    masks = seal(op, masks)
    check_parts(img, masks, "front")

    eyes = any_color(img, EYE_COLS)
    if int((eyes & masks["torso"]).sum()) != int(eyes.sum()) or int(eyes.sum()) < 30:
        raise SystemExit("front eyes left the head")
    ey, ex = np.nonzero(eyes & is_color(img, WHITE))
    eye_x = float(ex.mean())
    chin = tan & (yy == 55) & (xx >= 33) & (xx <= 47)
    cy, cx = np.nonzero(chin)
    chin_x = float(cx.mean())
    if chin_x >= eye_x:
        raise SystemExit(f"front chin is not left of the eyes: chin {chin_x:.1f} eye {eye_x:.1f}")
    mouth = black & (yy >= 50) & (yy <= 53) & (xx >= 36) & (xx <= 44) & dilate(tan, 1)
    if int(mouth.sum()) < 3 or int((mouth & masks["torso"]).sum()) != int(mouth.sum()):
        raise SystemExit("front smile left the face")
    return masks, mouth


def segment_back(img):
    """Tuft, tail on the left, two arms, two feet. No face is drawn."""
    op = img[..., 3] >= 128
    yy, xx = np.mgrid[0:96, 0:96]
    tan = any_color(img, TAN_COLS)
    blue = any_color(img, BLUE_COLS)
    black = is_color(img, INK)
    crest = crest_of(op)

    region = blue & (xx <= 36) & (yy >= 58)
    ys, xs = np.nonzero(region)
    i = int(xs.argmin())
    seeds = np.zeros_like(op)
    seeds[int(ys[i]), int(xs[i])] = True
    tail = flood(region, seeds)
    tail = tail | (black & dilate(tail, 1) & (xx <= 38) & (yy >= 58))

    sole_y = int(np.nonzero(op)[0].max())
    if sole_y != BACK_FOOT:
        raise SystemExit(f"back sole moved to {sole_y}")
    legs = flood(tan, tan & (yy == int(np.nonzero(tan)[0].max())))
    arm_l = tan & ~legs & (xx <= 50) & (yy >= 56)
    arm_r = tan & ~legs & (xx >= 54) & (yy >= 56)
    for _ in range(2):
        arm_l = arm_l | (black & dilate(arm_l, 1) & (xx <= 50) & (yy >= 54) & (yy <= 63) & ~dilate(legs, 1) & ~tail)
        arm_r = arm_r | (black & dilate(arm_r, 1) & (xx >= 54) & (yy >= 54) & (yy <= 65) & ~dilate(legs, 1))

    box_l = (xx <= 43) & (yy >= 65) & (yy <= 69)
    box_r = (xx >= 44) & (xx <= 58) & (yy >= 65) & (yy <= 69)
    seed_l = np.zeros_like(op)
    ys, xs = np.nonzero(tan & box_l)
    i = int(xs.argmin())
    seed_l[int(ys[i]), int(xs[i])] = True
    foot_l = flood((tan | black) & box_l, seed_l)
    foot_r = flood((tan | black) & box_r, tan & (yy == int(np.nonzero(tan)[0].max())) & box_r)
    arm_l = arm_l & ~foot_l & ~tail
    arm_r = arm_r & ~foot_r & ~tail
    tail = tail & ~foot_l & ~arm_l & ~crest

    masks = {
        "crest": crest, "tail": tail, "arm_l": arm_l, "arm_r": arm_r,
        "foot_l": foot_l, "foot_r": foot_r,
    }
    masks = seal(op, masks)
    check_parts(img, masks, "back")
    if any_color(img, EYE_COLS).any() or is_color(img, CREAM).any():
        raise SystemExit("back grew a face")
    return masks


def place(arr, y_shift, x_shift):
    out = np.zeros((CELL_H, CELL_W) + arr.shape[2:], arr.dtype)
    y0 = OFF_Y + y_shift
    x0 = OFF_X + x_shift
    h, w = arr.shape[:2]
    if y0 < 0 or x0 < 0 or y0 + h > CELL_H or x0 + w > CELL_W:
        raise SystemExit(f"place overflow y {y0} x {x0}")
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def joint_band(part, neighbor, depth):
    """Pixels of `part` that sit against `neighbor`, grown a few steps into the part."""
    touch = part & dilate(neighbor, 1)
    if not touch.any():
        touch = part & dilate(neighbor, 3)
    band = touch.copy()
    for _ in range(max(0, depth - 1)):
        band = dilate(band, 1) & part
    return band


def pivot_of(layer, anchor):
    part = layer[..., 3] >= 128
    base = anchor[..., 3] >= 128
    touch = part & dilate(base, 2)
    if not touch.any():
        touch = part & dilate(base, 6)
    if not touch.any():
        ys, xs = np.nonzero(part)
        if len(xs) == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.nonzero(touch)
    ay, ax = np.nonzero(base)
    cx, cy = float(ax.mean()), float(ay.mean())
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


def pick_forward(layers, pivots, names, facing):
    best = names[0]
    best_s = -1e18
    for name in names:
        x, _y = tip_xy(layers[name], pivots[name])
        score = facing * x
        if score > best_s:
            best, best_s = name, score
    return best


def build_view(img, masks, mouth, kind, mirror):
    y_shift = 0 if kind == "front" else (FRONT_FOOT - BACK_FOOT)
    x_shift = 0 if kind == "front" else BACK_DX
    canvas = place(img, y_shift, x_shift)
    placed = {k: place(v, y_shift, x_shift) for k, v in masks.items()}
    mouth_p = place(mouth, y_shift, x_shift) if mouth is not None else np.zeros((CELL_H, CELL_W), bool)
    if mirror:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
        placed = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in placed.items()}
        mouth_p = np.ascontiguousarray(mouth_p[:, ::-1])
    layers = {}
    for k, m in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    # Cushions live on the layer that is pasted underneath, inside the part
    # that covers them. At rest the cover hides them, so frame 0 matches.
    torso_m = placed["torso"]
    for name, depth in (("foot_l", 3), ("foot_r", 3), ("arm_l", 3), ("arm_r", 3)):
        band = joint_band(torso_m, placed[name], depth)
        layers[name] = paint_cushion(layers[name], band, depth)
    # The tail's neighbour is whichever part it actually touches.
    tail_m = placed["tail"]
    neigh = "torso"
    best = int((tail_m & dilate(torso_m, 1)).sum())
    for name in ("arm_l", "arm_r", "foot_l", "foot_r"):
        n = int((tail_m & dilate(placed[name], 1)).sum())
        if n > best:
            best = n
            neigh = name
    layers[neigh] = paint_cushion(layers[neigh], joint_band(tail_m, placed[neigh], 3), 3)
    layers["torso"] = paint_cushion(layers["torso"], joint_band(placed["crest"], torso_m, 4), 4)
    if kind == "front":
        facing = 1.0 if mirror else -1.0
    else:
        facing = -1.0 if mirror else 1.0
    # Anchor pivots on the torso, except the tail, which hinges on its neighbour.
    anchors = {name: layers["torso"] for name in LIMBS}
    anchors["tail"] = layers[neigh]
    pivots = {}
    signs = {}
    for name in LIMBS:
        pivots[name] = pivot_of(layers[name], anchors[name])
        signs[name] = signs_for(layers[name], pivots[name], facing)
    op = canvas[..., 3] >= 128
    ys, xs = np.nonzero(op)
    pivots["center"] = (float(xs.mean()), float(ys.mean()))
    has_mouth = bool(mouth_p.any())
    if has_mouth:
        my, mx = np.nonzero(mouth_p)
        pivots["mouth"] = (float(mx.mean()), float(my.mean()))
    else:
        pivots["mouth"] = pivots["center"]
    lead_arm = pick_forward(layers, pivots, ("arm_l", "arm_r"), facing)
    kick_name = pick_forward(layers, pivots, ("foot_l", "foot_r"), facing)
    return {
        "kind": kind, "mirror": mirror, "facing": facing,
        "lead_arm": lead_arm, "kick_name": kick_name, "tail_neighbor": neigh,
        "has_mouth": has_mouth, "layers": layers, "pivots": pivots, "signs": signs,
        "canvas": canvas,
    }


def base_pose():
    return dict(
        bdx=0, bdy=0,
        crest=0.0, crest_mode="up",
        arm=0.0, arm_mode="up",
        tail=0.0, tail_mode="up",
        kick=0.0, kick_mode="fwd",
        step_l=(0, 0), step_r=(0, 0),
        wedge=0, ring=-1, spark=0,
        lean=0.0, knock=0, flash=False,
        turned=False, spin=0,
    )


def pose(action, f):
    n = FRAMES[action]
    p = base_pose()
    if action == "idle":
        # Feet stay put. The tuft, the hands and the tail breathe.
        p["crest"] = wave(f, n, 5.0, phase=0.4)
        p["arm"] = wave(f, n, 4.0, phase=1.1)
        p["tail"] = wave(f, n, 7.0, phase=0.9)
    elif action == "walk":
        u = 2 * math.pi * f / n
        sa = math.sin(u)
        sb = math.sin(u + math.pi)
        # Only the airborne foot moves, so the planted sole stays shut.
        p["step_l"] = (int(round(2 * max(0.0, sa))), -int(round(2 * max(0.0, sa))))
        p["step_r"] = (int(round(2 * max(0.0, sb))), -int(round(2 * max(0.0, sb))))
        p["arm"] = wave(f, n, 10.0, phase=0.0)
        p["arm_mode"] = "fwd"
        p["tail"] = wave(f, n, 8.0, cycles=2.0, phase=0.4)
        p["crest"] = wave(f, n, 3.0, phase=0.2)
    elif action == "attack":
        # 0 rest. 1-4 roar from the smile and the tuft. 5-8 the other side.
        # 9-12 hands and tail. 13-15 recover. 15 is the still.
        crest = [0, 6, 12, 14, 12, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
        arm = [0, 6, 10, 12, 12, 0, 0, 0, 0, 16, 26, 32, 30, 12, 4, 0]
        # The curl only meets the hand at two pixels. Past 16 deg it leaves.
        tail = [0, 4, 8, 12, 12, 0, 0, 0, 0, 12, 14, 16, 14, 8, 3, 0]
        kick = [0, 0, 0, 2, 4, 0, 0, 0, 0, 8, 12, 16, 12, 6, 2, 0]
        p["crest"] = float(crest[f])
        p["arm"] = float(arm[f])
        p["tail"] = float(tail[f])
        p["kick"] = float(kick[f])
        p["arm_mode"] = "up" if f <= 4 else "fwd"
        p["tail_mode"] = "up" if f <= 4 else "fwd"
        p["crest_mode"] = "up"
        p["kick_mode"] = "fwd"
        p["turned"] = 5 <= f <= 8
        if 1 <= f <= 4:
            p["wedge"] = [0, 1, 2, 3, 2][f]
        if 2 <= f <= 4:
            p["ring"] = f - 2
        if f in (11, 12):
            p["spark"] = f - 10
    else:
        env = [0.0, 0.55, 0.85, 1.0, 0.90, 0.70, 0.48, 0.30, 0.16, 0.06, 0.0, 0.0][f]
        p["lean"] = MAX_LEAN * env
        p["knock"] = int(round(3 * env))
        p["flash"] = f == 1
        p["spark"] = 1 if f in (1, 2) else 0
    p = finish(p, action, f, __file__)
    # Frames 5-8 are the other side at equal scale. finish() would swing the
    # wind-up on that still. Drop it so the silhouette stays the still.
    if action == "attack" and 5 <= f <= 8:
        p["crest"] = 0.0
        p["arm"] = 0.0
        p["tail"] = 0.0
        p["kick"] = 0.0
        p["bdx"] = 0
        p["bdy"] = 0
        p["step_l"] = (0, 0)
        p["step_r"] = (0, 0)
        p["wedge"] = 0
        p["ring"] = -1
        p["spark"] = 0
        p["spin"] = 0
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

    def foot(name, step):
        sx, sy = step
        ang = part_angle(rig, name, p["kick_mode"], p["kick"]) if name == rig["kick_name"] else 0.0
        dx = bdx + int(round(fc * sx))
        dy = int(sy)
        return moved(L[name], ang, piv[name], dx, dy)

    def arm(name):
        amp = p["arm"]
        # The hand beside the tail stays near the curl. The other hand strikes.
        if p["arm_mode"] != "up" and name != rig["lead_arm"]:
            amp = amp * 0.15
        ang = part_angle(rig, name, p["arm_mode"], amp)
        return moved(L[name], ang, piv[name], bdx, bdy)

    # Feet and arms go under the torso so their shoulder cushions stay hidden.
    paste(out, foot("foot_l", p["step_l"]))
    paste(out, foot("foot_r", p["step_r"]))
    paste(out, arm("arm_l"))
    paste(out, arm("arm_r"))
    paste(out, shift(L["torso"], bdx, bdy))
    tail_ang = part_angle(rig, "tail", p["tail_mode"], p["tail"])
    paste(out, moved(L["tail"], tail_ang, piv["tail"], bdx, bdy))
    crest_ang = part_angle(rig, "crest", p["crest_mode"], p["crest"])
    paste(out, moved(L["crest"], crest_ang, piv["crest"], bdx, bdy))
    return out


def mouth_wedge(img, origin, fc, age):
    """A few pixels in front of the smile that is already drawn. Empty pixels only."""
    ox, oy = origin
    cols = (INK, GREY, BROWN)
    for i in range(max(0, int(age))):
        x = ox + fc * (2 + i)
        span = 0 if i == 0 else 1
        for dy in range(-span, span + 1):
            put(img, x, oy + dy, cols[min(i, 2)])


def roar_rings(img, origin, fc, age):
    ox, oy = origin
    cols = (CYAN, WHITE, NAVY)
    rad = 6 + age * 3
    for deg in range(-40, 41, 10):
        a = math.radians(deg)
        x = ox + fc * rad * math.cos(a)
        y = oy + rad * math.sin(a) * 0.45
        put(img, x, y, cols[age % 3])


def spark(img, origin, fc, age):
    ox, oy = origin
    cols = (WHITE, CYAN, TAN)
    r = 3 + age * 2
    for i, (dx, dy) in enumerate(((r, 0), (r - 1, -2), (r - 1, 2), (1, -r // 2), (1, r // 2))):
        put(img, ox + fc * dx, oy + dy, cols[i % 3])


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
    """Hand of the leading arm, after the strike pose."""
    name = rig["lead_arm"]
    fc = rig["facing"]
    amp = p["arm"]
    ang = part_angle(rig, name, p["arm_mode"], amp)
    dx = int(round(fc * p["bdx"]))
    dy = int(round(p["bdy"]))
    layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
    m = layer[..., 3] >= 128
    if not m.any():
        return rig["pivots"]["center"]
    ys, xs = np.nonzero(m)
    i = int(xs.argmax()) if fc > 0 else int(xs.argmin())
    return float(xs[i]), float(ys[i])


def limb_lost(rig, p):
    body = compose(rig, p)
    lost = []
    fc = rig["facing"]
    bdx = int(round(fc * p["bdx"]))
    bdy = int(round(p["bdy"]))
    for name in LIMBS:
        if name in ("foot_l", "foot_r"):
            sx, sy = p["step_l"] if name == "foot_l" else p["step_r"]
            ang = part_angle(rig, name, p["kick_mode"], p["kick"]) if name == rig["kick_name"] else 0.0
            dx = bdx + int(round(fc * sx))
            dy = int(sy)
        elif name in ("arm_l", "arm_r"):
            amp = p["arm"]
            if p["arm_mode"] != "up" and name != rig["lead_arm"]:
                amp = amp * 0.15
            ang = part_angle(rig, name, p["arm_mode"], amp)
            dx, dy = bdx, bdy
        elif name == "tail":
            ang = part_angle(rig, name, p["tail_mode"], p["tail"])
            dx, dy = bdx, bdy
        else:
            ang = part_angle(rig, name, p["crest_mode"], p["crest"])
            dx, dy = bdx, bdy
        layer = moved(rig["layers"][name], ang, rig["pivots"][name], dx, dy)
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
    return rigs[("back" if kind == "front" else "front", not mirror)]


def synth(rigs, d, action, f, palette, effects=True):
    p = pose(action, f)
    rig = rig_for(rigs, d, p)
    body = compose(rig, p)
    moved_part = (
        abs(p["crest"]) >= 0.45 or abs(p["arm"]) >= 0.45 or abs(p["tail"]) >= 0.45
        or abs(p["kick"]) >= 0.45
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
        mx, my = rig["pivots"]["mouth"]
        mx += fc * p["bdx"]
        my += p["bdy"]
        if p["wedge"]:
            mouth_wedge(img, (mx, my), fc, p["wedge"])
        if p["ring"] >= 0:
            roar_rings(img, (mx + fc * 2, my), fc, p["ring"])
    if p["spark"] and action == "attack" and not p["turned"]:
        tx, ty = weapon_tip(rig, p)
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


def content_box(canvas, pad=8):
    ys, xs = np.nonzero(canvas[..., 3] >= 128)
    y0 = max(0, int(ys.min()) - pad - 28)
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
    fm, mouth = segment_front(front)
    bm = segment_back(back)
    empty = np.zeros((96, 96), bool)
    rigs = {
        ("front", False): build_view(front, fm, mouth, "front", False),
        ("front", True): build_view(front, fm, mouth, "front", True),
        ("back", False): build_view(back, bm, empty, "back", False),
        ("back", True): build_view(back, bm, empty, "back", True),
    }
    for d in DIRS:
        rigs[d] = rigs[VIEW_OF[d]]
    return front, back, fm, bm, rigs


def save_parts(work, front, fm, back, bm):
    cols = {
        "torso": (80, 170, 230),
        "crest": (255, 220, 40),
        "tail": (220, 40, 40),
        "arm_l": (40, 200, 80),
        "arm_r": (255, 120, 20),
        "foot_l": (170, 40, 220),
        "foot_r": (255, 80, 170),
    }
    tiles = []
    for img, masks in ((front, fm), (back, bm)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, m in masks.items():
            part[m, :3] = cols[name]
            part[m, 3] = 255
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
        "dex": 515, "form": "normal", "variant": "panpour", "gen": "gen5",
        "dir": "monsterSprite/gen5/0515_panpour/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge",
        "v": VERSION,
    }
    gen5_publish.publish_dir8(KEY, entry)
    gen5_publish.publish_bounds(KEY, bounds)


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
    print("facing", {d: rigs[d]["facing"] for d in DIRS}, flush=True)
    print("mouth", {d: rigs[d]["has_mouth"] for d in DIRS}, flush=True)
    print("lead", {d: (rigs[d]["lead_arm"], rigs[d]["kick_name"], rigs[d]["tail_neighbor"]) for d in DIRS}, flush=True)

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

    if args.samples:
        lost = {}
        for d in DIRS:
            for act, fs in (
                ("idle", (0, 8)),
                ("walk", (4, 12)),
                ("attack", (2, 4, 6, 8, 11, 12)),
                ("hurt", (3,)),
            ):
                for f in fs:
                    p = pose(act, f)
                    rig = rig_for(rigs, d, p)
                    bad = limb_lost(rig, p)
                    if bad:
                        lost[f"{act}:{d}:{f}"] = bad
        print("limb lost", lost or "none", flush=True)
        y0, y1, x0, x1 = content_box(rigs["S"]["canvas"])
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
        sheet = tile_grid(tiles, 12)
        sheet = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
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
    box = content_box(rigs["S"]["canvas"])
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
        "generator": "forge_gen5_panpour_actions", "version": VERSION, "key": KEY,
        "dex": 515, "variant": "panpour", "cell_w": CELL_W, "cell_h": CELL_H,
        "foot_y": FOOT_Y, "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS,
        "actions": ACTIONS, "views": {d: list(VIEW_OF[d]) for d in DIRS},
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "archetype": "biped water monkey (tuft, two arms, two feet, curled tail, smile)",
        "source": "front.png / back.png 96x96", "gifs_untouched": True,
        "facing": "front still faces screen-left (chin mean x38 < eye mean x39.5)",
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
