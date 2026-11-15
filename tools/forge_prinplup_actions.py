#!/usr/bin/env python3
"""Prinplup (0394) 8-direction action sheet from this form's 96x96 front.png.

Measured on front.png (opaque 1445, bbox x21-66 y18-74, 46x57, 14 colours,
alpha already 0 or 255):

  Two crown tufts, not wings. Left tuft gold is x35-46 y18-30. Right tuft gold
  is x47-53 y19-27. They sway. They are not extra flippers.
  One yellow beak between the eyes. Upper beak y31-33, lower beak (jaw) y34-36,
  tip of the lower beak at y36 x41-42. Left eye whites x37-40 y31-33, right eye
  whites x44-47 y31-33. The tip is between the eyes, so the still faces the
  camera (south), not left and not right. Unflipped rows are S SE E NE.
  N NW W SW are the horizontal mirror, so W is the screen-left counterpart.
  Two flippers only: the left lobe x21-34 y44-64, the right lobe x55-66 y40-67.
  Two webbed feet, split at the empty gap around x46, from y68 down, plus the
  one brown ankle pixel at (67, 41). No third foot, no wings, no tail.

  idle   : planted feet. Crowns and the two flippers sway. A short beak chirp.
           Frame 0 is the still.
  walk   : penguin waddle. Body rocks 1px, the two feet lift in turn, flippers
           swing. Not a glide and not a flap of invented wings.
  attack : 0-4 the existing beak opens and rings leave the beak; 5-10 one-piece
           in-place 360 with no squash; 11-12 impact; 13-15 rest.
  hurt   : one-piece recoil, 38 degrees max, about the feet. No belly flip.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_prinplup_actions.py
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _gen4_strike_sys
import pathlib as _gen4_strike_path
_gen4_strike_sys.path.insert(0, str(_gen4_strike_path.Path(__file__).resolve().parent))
import gen4_attack_strike

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0394_prinplup/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0394_prinplup/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Still faces the camera. South keeps the drawing. West is the mirror.
UNFLIPPED = {"S", "SE", "E", "NE"}
# Screen-space roar / knock. South is toward the viewer (down the cell).
DIR_VEC = {
    "S": (0.0, 1.0), "SE": (1.0, 1.0), "E": (1.0, 0.0), "NE": (1.0, -1.0),
    "N": (0.0, -1.0), "NW": (-1.0, -1.0), "W": (-1.0, 0.0), "SW": (-1.0, 1.0),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "prinplup_v2"

# Colours measured on this front.png. Effects use only these.
BLACK = (16, 16, 16)
NAVY = (32, 65, 106)
BLUE = (32, 90, 148)
MID = (57, 123, 180)
PALE = (98, 156, 246)
FACE = (156, 205, 255)
ICE = (131, 180, 255)
MIST = (189, 205, 222)
WASH = (222, 238, 255)
WHITE = (255, 255, 255)
GOLD = (238, 205, 98)
YELLOW = (255, 230, 131)
TAN = (189, 139, 106)
BROWN = (123, 82, 65)
GOLD_SET = {GOLD, YELLOW, TAN, BROWN}

DRAW = ["torso", "head", "beak_upper", "crest_l", "crest_r", "flip_l", "flip_r", "foot_l", "foot_r", "jaw"]


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
    q = deque()
    ys, xs = np.nonzero(seeds & allow)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not seen[y, x]:
            seen[y, x] = True
            q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def match_cols(img, opaque, cols):
    rgb = img[..., :3]
    m = np.zeros_like(opaque)
    for c in cols:
        m |= (rgb[..., 0] == c[0]) & (rgb[..., 1] == c[1]) & (rgb[..., 2] == c[2])
    return opaque & m


def segment(img):
    """Partition this still. Cuts are the measured tufts, beak, two flippers, two feet."""
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    gold = match_cols(img, a, GOLD_SET)
    black = match_cols(img, a, [BLACK])

    # Feet are the only gold below the belly. (67, 41) is the left ankle, brown.
    foot = (a & (yy >= 68)) | (gold & (yy == 67) & (xx <= 45))
    foot_l = foot & (xx <= 45)
    foot_r = foot & (xx >= 46)

    crest_l = gold & (yy <= 30) & (xx <= 46)
    crest_r = gold & (yy <= 27) & (xx >= 47)
    # Black caps on the tufts only. The cheek outline below the eyes stays on the head.
    crest_l = crest_l | (black & dilate(crest_l, 1) & (yy <= 31) & (xx >= 33) & (xx <= 47))
    crest_r = crest_r | (black & dilate(crest_r, 1) & (yy <= 28) & (xx >= 46))

    beak = gold & (yy >= 31) & (yy <= 40) & (xx >= 38) & (xx <= 45)
    crest_l = crest_l & ~beak
    crest_r = crest_r & ~beak
    jaw = beak & (yy >= 34)
    beak_upper = beak & ~jaw

    used = foot_l | foot_r | crest_l | crest_r | jaw | beak_upper
    # Outer lobes only. The white belly stripes stay on the torso so the belly cannot split.
    flip_l = a & (yy >= 44) & (yy <= 64) & (xx <= 34) & ~used
    flip_r = a & (yy >= 40) & (yy <= 67) & (xx >= 55) & ~used
    used = used | flip_l | flip_r
    head = a & (yy <= 42) & ~used
    torso = a & ~used & ~head

    parts = {
        "torso": torso, "head": head, "beak_upper": beak_upper,
        "crest_l": crest_l, "crest_r": crest_r,
        "flip_l": flip_l, "flip_r": flip_r,
        "foot_l": foot_l, "foot_r": foot_r, "jaw": jaw,
    }
    cover = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(cover & m):
            raise SystemExit(f"prinplup masks overlap at {name}")
        cover |= m
    if int((cover != a).sum()):
        raise SystemExit(f"prinplup masks missed {int((cover != a).sum())} pixels")
    return parts


def measure_facing(img):
    """Beak tip versus the two eye catchlights. Between them means a front view."""
    a = img[..., 3] >= 128
    rgb = img[..., :3]
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    white = a & (rgb[..., 0] == 255) & (rgb[..., 1] == 255) & (rgb[..., 2] == 255)
    eyes = white & (yy <= 36)
    left = eyes & (xx <= 41)
    right = eyes & (xx >= 44)
    if int(left.sum()) < 3 or int(right.sum()) < 3:
        raise SystemExit("expected two eyes on the face")
    left_x = float(np.nonzero(left)[1].mean())
    right_x = float(np.nonzero(right)[1].mean())
    gold = match_cols(img, a, GOLD_SET)
    beak = gold & (yy >= 31) & (yy <= 40) & (xx >= 38) & (xx <= 45)
    by, bx = np.nonzero(beak)
    tip_y = int(by.max())
    tip_xs = bx[by == tip_y]
    tip_x = float(np.median(tip_xs))
    if tip_x < left_x:
        facing = "left"
    elif tip_x > right_x:
        facing = "right"
    else:
        facing = "front"
    return facing, tip_x, tip_y, left_x, right_x


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def shift(layer, dx, dy):
    dx_i, dy_i = int(round(dx)), int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0, sx1 = max(0, -dx_i), min(w, w - dx_i)
    sy0, sy1 = max(0, -dy_i), min(h, h - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest rotation about pivot. Exact colours. No scale and no mirror."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(ang)
    c, s = math.cos(rad), math.sin(rad)
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + c * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[1::4, 1::4].copy()


def fill_pinholes(img):
    """A flipper shift can enclose one empty pixel. Fill only n==4 holes."""
    a = img[..., 3] >= 128
    if not np.any(a):
        return img
    pad = np.pad(a, 1)
    n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    holes = (~a) & (n == 4)
    if not np.any(holes):
        return img
    fill = nearest_color(img, holes, a)
    out = img.copy()
    out[holes] = fill[holes]
    return out


def despeckle(img):
    """Drop only pixels whose four neighbours are empty. Keeps eyes and outlines."""
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        n = 0
        while q:
            cy, cx = q.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - ys[None, :].astype(np.int32)) ** 2
        d += (tx[sl, None].astype(np.int32) - xs[None, :].astype(np.int32)) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    return out


def put(out, x, y, col):
    """Draw only on empty pixels so a ring cannot recolour the body."""
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def contact_pivot(part, body):
    """Contact pixel of a part closest to the body it hangs from."""
    ys, xs = np.nonzero(part)
    if len(ys) == 0:
        return 0.0, 0.0
    touch = part & dilate(body, 1)
    if np.any(touch):
        ys, xs = np.nonzero(touch)
    by, bx = np.nonzero(body)
    bcy, bcx = float(by.mean()), float(bx.mean())
    d = (ys.astype(np.float32) - bcy) ** 2 + (xs.astype(np.float32) - bcx) ** 2
    i = int(d.argmin())
    return float(xs[i]), float(ys[i])


def add_socket(host, host_mask, part_mask, canvas, reach=2):
    """Body colour under the part. Stays hidden until the part moves."""
    zone = part_mask & dilate(host_mask, reach)
    if not np.any(zone):
        return
    fill = nearest_color(canvas, zone, host_mask)
    write = zone & (host[..., 3] < 128)
    host[write] = fill[write]


def build_rig(img, masks, flip):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]

    placed = {}
    for name, m in masks.items():
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dst_y:dst_y + sh, dst_x:dst_x + sw] = m[y0:y1, x0:x1]
        placed[name] = pm

    layers = {}
    for name, pm in placed.items():
        lyr = np.zeros_like(canvas)
        lyr[pm] = canvas[pm]
        layers[name] = lyr

    body = placed["torso"] | placed["head"]
    add_socket(layers["head"], placed["head"], placed["crest_l"], canvas)
    add_socket(layers["head"], placed["head"], placed["crest_r"], canvas)
    add_socket(layers["torso"], placed["torso"], placed["flip_l"], canvas)
    add_socket(layers["torso"], placed["torso"], placed["flip_r"], canvas)
    add_socket(layers["torso"], placed["torso"], placed["foot_l"], canvas, reach=2)
    add_socket(layers["torso"], placed["torso"], placed["foot_r"], canvas, reach=2)

    pivots = {
        "crest_l": contact_pivot(placed["crest_l"], placed["head"]),
        "crest_r": contact_pivot(placed["crest_r"], placed["head"]),
        "flip_l": contact_pivot(placed["flip_l"], body),
        "flip_r": contact_pivot(placed["flip_r"], body),
        "foot_l": contact_pivot(placed["foot_l"], placed["torso"]),
        "foot_r": contact_pivot(placed["foot_r"], placed["torso"]),
    }
    fys, fxs = np.nonzero(placed["foot_l"] | placed["foot_r"])
    sole = (float(np.median(fxs)), float(fys.max()))
    by, bx = np.nonzero(placed["beak_upper"] | placed["jaw"])
    beak_pt = (float(np.median(bx)), float(by.max()))
    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    center = (float(opx.mean()), float(opy.mean()))

    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        pivots = {k: (CELL_W - 1 - x, y) for k, (x, y) in pivots.items()}
        sole = (CELL_W - 1 - sole[0], sole[1])
        beak_pt = (CELL_W - 1 - beak_pt[0], beak_pt[1])
        center = (CELL_W - 1 - center[0], center[1])
    return layers, pivots, sole, center, beak_pt


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def pose(action, f):
    n = FRAMES[action]
    p = dict(
        dx=0.0, dy=0.0,
        crest_l=0.0, crest_r=0.0,
        flip_l_px=0.0, flip_r_px=0.0, flip_up=0.0,
        lift_l=0.0, lift_r=0.0, jaw=0, spin=0.0, recoil=0.0,
        rings=0, ring=0, burst=0, flash=0, sparks=0, lunge=0.0,
    )
    if action == "idle":
        p["crest_l"] = wave(f, n, 7.0, phase=0.4)
        p["crest_r"] = wave(f, n, -7.0, phase=0.4)
        # Integer shifts only. Rotating these wide flippers tears the 1px outline.
        p["flip_l_px"] = wave(f, n, 1.0)
        p["flip_r_px"] = wave(f, n, -1.0, phase=0.6)
        if f in (8, 9, 10):
            p["jaw"] = 1
    elif action == "walk":
        step = wave(f, n, 1.0, cycles=2.0)
        p["dx"] = step
        p["lift_l"] = max(0.0, step) * 2.0
        p["lift_r"] = max(0.0, -step) * 2.0
        p["flip_l_px"] = -2.0 * step
        p["flip_r_px"] = 2.0 * step
        p["flip_up"] = -abs(step)
        p["crest_l"] = wave(f, n, 5.0, cycles=2.0)
        p["crest_r"] = wave(f, n, -5.0, cycles=2.0)
    elif action == "attack":
        # 0-4 roar, 5-10 in-place spin, 11-12 impact, 13-15 rest.
        if f <= 4:
            p["jaw"] = (2, 2, 2, 2, 1)[f]
            p["crest_l"] = (-10, -14, -16, -14, -8)[f]
            p["crest_r"] = (10, 14, 16, 14, 8)[f]
            spread = (1, 2, 2, 2, 1)[f]
            p["flip_l_px"] = -spread
            p["flip_r_px"] = spread
            p["flip_up"] = -1 if spread == 2 else 0
            p["rings"] = 3
            p["ring"] = f
        elif f <= 10:
            p["spin"] = 60.0 * (f - 4)
        elif f in (11, 12):
            p["burst"] = 1
            p["lunge"] = 1.0
        else:
            settle = (4.0, 2.0, 0.0)[f - 13]
            p["crest_l"] = -settle
            p["crest_r"] = settle
    else:
        env = math.sin(math.pi * f / 11.0)
        p["recoil"] = 38.0 * env
        p["lunge"] = 3.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
    return p


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def open_beak(img, jaw_src, bdx, bdy, drop):
    """Drop the existing lower beak. The gap is a black mouth, not a new beak."""
    if drop <= 0:
        paste(img, shift(jaw_src, bdx, bdy))
        return img
    home = shift(jaw_src, bdx, bdy)
    dropped = shift(jaw_src, bdx, bdy + drop)
    hole = (home[..., 3] >= 128) & (dropped[..., 3] < 128) & (img[..., 3] < 128)
    img[hole, :3] = BLACK
    img[hole, 3] = 255
    paste(img, dropped)
    return img


def roar_rings(out, origin, fx, fy, age, strength):
    """Rings around the beak. South's forward axis runs into the belly, so the
    circle has to clear the head on the sides and above the crown."""
    ox, oy = origin
    cols = (WHITE, YELLOW, PALE)
    for i in range(strength):
        rad = 13 + age + i * 5
        col = cols[i]
        for s in range(18):
            ang = 2 * math.pi * s / 18.0
            put(out, ox + fx * 2 + math.cos(ang) * rad, oy + fy * 2 + math.sin(ang) * rad, col)
            put(out, ox + fx * 2 + math.cos(ang) * (rad + 1), oy + fy * 2 + math.sin(ang) * (rad + 1), GOLD)
    return out


def impact_burst(out, origin, fx, fy):
    """Starburst above the crown. A burst drawn down the belly would be hidden."""
    ox, oy = origin
    cx = ox + fx * 3
    cy = oy - 8
    for k in range(12):
        ang = k * math.pi / 6.0
        for r, col in ((9, WHITE), (12, YELLOW), (15, PALE)):
            put(out, cx + r * math.cos(ang), cy + r * math.sin(ang), col)
    return out


def sparks(out, frame_idx, fx, fy):
    bb = body_box(out)
    if not bb:
        return out
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2, y0 + 4
    mag = math.hypot(fx, fy) or 1.0
    ux, uy = -fx / mag, -fy / mag
    d = frame_idx
    pts = ((4 + d, 0), (2 + d, -3), (6, 2))
    for dist, side in pts:
        put(out, cx + ux * dist + uy * side, cy + uy * dist - ux * side, WHITE)
        put(out, cx + ux * dist + uy * side + 1, cy + uy * dist - ux * side + 1, YELLOW)
    return out


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    dark = a & (out[..., :3].sum(axis=2) < 80)
    out[a & ~dark, :3] = WHITE
    return out


def moved_parts(layers, pivots, p, mirror):
    """Head uses the same integer shift as the torso. Relative head offset stays 0."""
    bdx = int(round(p["dx"] * mirror))
    bdy = int(round(p["dy"]))
    sgn = mirror
    out = {}
    out["torso"] = shift(layers["torso"], bdx, bdy)
    out["head"] = shift(layers["head"], bdx, bdy)
    out["beak_upper"] = shift(layers["beak_upper"], bdx, bdy)
    out["crest_l"] = shift(rotate_nearest(layers["crest_l"], p["crest_l"] * sgn, pivots["crest_l"]), bdx, bdy)
    out["crest_r"] = shift(rotate_nearest(layers["crest_r"], p["crest_r"] * sgn, pivots["crest_r"]), bdx, bdy)
    # Flippers slide by whole pixels. A rotation of this wide lobe breaks the outline.
    up = int(round(p["flip_up"]))
    out["flip_l"] = shift(layers["flip_l"], bdx + int(round(p["flip_l_px"] * sgn)), bdy + up)
    out["flip_r"] = shift(layers["flip_r"], bdx + int(round(p["flip_r_px"] * sgn)), bdy + up)
    out["foot_l"] = shift(layers["foot_l"], bdx, bdy - int(round(p["lift_l"])))
    out["foot_r"] = shift(layers["foot_r"], bdx, bdy - int(round(p["lift_r"])))
    out["jaw"] = shift(layers["jaw"], bdx, bdy)
    return out, bdx, bdy


def assemble(moved):
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW:
        paste(img, moved[name])
    return img


def synth(rig, action, f, palette, direction):
    layers, pivots, sole, center, beak_pt = rig
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    mirror = 1.0 if direction in UNFLIPPED else -1.0
    fx, fy = DIR_VEC[direction]
    moved, bdx, bdy = moved_parts(layers, pivots, p, mirror)

    spinning = p["spin"] > 0.5 and (p["spin"] % 360.0) > 0.5
    recoiling = p["recoil"] > 0.45
    if spinning or recoiling:
        base = assemble({k: layers[k] for k in DRAW})
        if spinning:
            img = rotate_nearest(base, p["spin"], center)
        else:
            # Positive angle swings the head to screen-right. Lean backward.
            if fx != 0:
                ang = -math.copysign(1.0, fx) * p["recoil"]
            else:
                ang = -p["recoil"] if fy >= 0 else p["recoil"]
            mag = math.hypot(fx, fy) or 1.0
            kdx = int(round(-fx / mag * p["lunge"]))
            kdy = int(round(-fy / mag * p["lunge"]))
            img = shift(rotate_nearest(base, ang, sole), kdx, kdy)
        img = despeckle(img)
    else:
        img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        for name in DRAW:
            if name == "jaw":
                continue
            paste(img, moved[name])
        img = open_beak(img, layers["jaw"], bdx, bdy, int(p["jaw"]))
        # Frame 0 of idle and walk is the still, which already has its own pixels.
        identity = (
            int(p["jaw"]) == 0 and int(round(p["dx"])) == 0 and int(round(p["dy"])) == 0
            and int(round(p["flip_l_px"] * mirror)) == 0 and int(round(p["flip_r_px"] * mirror)) == 0
            and int(round(p["flip_up"])) == 0 and int(round(p["lift_l"])) == 0
            and int(round(p["lift_r"])) == 0 and abs(p["crest_l"]) < 0.45
            and abs(p["crest_r"]) < 0.45 and not p["lunge"]
        )
        if not identity:
            img = fill_pinholes(img)
        if p["lunge"]:
            mag = math.hypot(fx, fy) or 1.0
            img = shift(img, int(round(fx / mag * p["lunge"])), int(round(fy / mag * p["lunge"])))

    img = snap(img, palette)
    if p["flash"]:
        img = flash_white(img)
    if p["rings"]:
        tip = (beak_pt[0] + bdx, beak_pt[1] + bdy + int(p["jaw"]))
        img = roar_rings(img, tip, fx, fy, p["ring"], p["rings"])
    if p["burst"]:
        tip = (beak_pt[0], beak_pt[1])
        img = impact_burst(img, tip, fx, fy)
    if p["sparks"]:
        img = sparks(img, f, fx, fy)
    return snap(img, palette)


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (SRC_DIR / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def stack_sheets(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png"))
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = tile
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
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    return master


def preview_image(sheets):
    """Every south-row frame, cropped and scaled, so the beak and the spin are readable."""
    x0, x1 = 48, 136
    y0, y1 = 70, 162
    scale = 3
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    cols = 8
    rows = sum(math.ceil(FRAMES[a] / cols) for a in ACTIONS)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for act in ACTIONS:
        band = sheets[act][:CELL_H]
        n = FRAMES[act]
        for f in range(n):
            tile = band[y0:y1, f * CELL_W + x0:f * CELL_W + x1]
            up = np.repeat(np.repeat(tile, scale, 0), scale, 1)
            rr, cc = divmod(f, cols)
            y = (r + rr) * ch
            x = cc * cw
            m = up[..., 3] >= 128
            bg[y:y + ch, x:x + cw][m] = up[m]
        r += math.ceil(n / cols)
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 8
    crops = []
    for act in ACTIONS:
        row = []
        band = sheets[act][:CELL_H]
        for f in range(FRAMES[act]):
            tile = band[:, f * CELL_W:(f + 1) * CELL_W]
            a = tile[..., 3] >= 128
            if not np.any(a):
                continue
            ys, xs = np.nonzero(a)
            y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + 1 + pad)
            x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + 1 + pad)
            row.append(tile[y0:y1, x0:x1])
        crops.append(row)
    cell_h = max(c.shape[0] for row in crops for c in row)
    cell_w = max(c.shape[1] for row in crops for c in row)
    cols = max(len(row) for row in crops)
    bg = np.full((len(crops) * cell_h, cols * cell_w, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(crops):
        for ci, c in enumerate(row):
            y = ri * cell_h + (cell_h - c.shape[0]) // 2
            x = ci * cell_w + (cell_w - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
    return Image.fromarray(bg, "RGBA")


def south_attack_ratios(rigs, palette):
    frames = [synth(rigs["S"], "attack", f, palette, "S") for f in range(16)]
    counts = [int((fr[..., 3] >= 128).sum()) for fr in frames]
    base = counts[0]
    ratios = [c / base for c in counts]
    print("SOUTH attack opaque ratios:", flush=True)
    for f, (c, r) in enumerate(zip(counts, ratios)):
        print(f"  {f:02d} {c} {r:.4f}", flush=True)
    print(f"SOUTH attack opaque min {min(ratios):.4f} max {max(ratios):.4f}", flush=True)
    return frames, ratios


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    facing, tip_x, tip_y, left_x, right_x = measure_facing(front)
    print(f"facing {facing} beak_tip ({tip_x:.1f},{tip_y}) eyes ({left_x:.1f},{right_x:.1f})", flush=True)
    if facing != "front":
        raise SystemExit(f"expected a front-facing still, measured {facing}")

    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)
    for name, need in (("crest_l", 40), ("crest_r", 20), ("beak_upper", 6), ("jaw", 6),
                       ("flip_l", 40), ("flip_r", 40), ("foot_l", 20), ("foot_r", 20),
                       ("head", 80), ("torso", 200)):
        if counts[name] < need:
            raise SystemExit(f"part {name} too small: {counts[name]}")

    rigs = {d: build_rig(front, masks, flip=d not in UNFLIPPED) for d in DIRS}

    rest = synth(rigs["S"], "idle", 0, palette, "S")
    a = front[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    placed = np.zeros_like(rest)
    dst_x = (CELL_W - (x1 - x0)) // 2
    dst_y = FOOT_Y - (y1 - y0)
    placed[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = front[y0:y1, x0:x1]
    differ = np.any(rest != placed, axis=2) & ((rest[..., 3] >= 128) | (placed[..., 3] >= 128))
    n_diff = int(differ.sum())
    print("rest-pose pixels changed", n_diff, flush=True)
    if n_diff:
        raise SystemExit("idle frame 0 does not match the still")

    west = synth(rigs["W"], "idle", 0, palette, "W")
    flipped = placed[:, ::-1].copy()
    ed = np.any(west != flipped, axis=2) & ((west[..., 3] >= 128) | (flipped[..., 3] >= 128))
    print("west mirror pixels changed", int(ed.sum()), flush=True)
    if int(ed.sum()):
        raise SystemExit("west idle 0 is not the mirror of the still")

    walk0 = synth(rigs["S"], "walk", 0, palette, "S")
    wd = np.any(walk0 != placed, axis=2) & ((walk0[..., 3] >= 128) | (placed[..., 3] >= 128))
    if int(wd.sum()):
        raise SystemExit("walk frame 0 does not match the still")

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (40, 90, 160), "head": (240, 220, 80), "beak_upper": (255, 140, 40),
        "crest_l": (220, 60, 180), "crest_r": (255, 80, 80),
        "flip_l": (40, 200, 90), "flip_r": (40, 180, 220),
        "foot_l": (255, 230, 80), "foot_r": (180, 120, 40), "jaw": (20, 220, 200),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 6, 0), 6, 1), "RGBA").save(work / "parts.png")

    atk, ratios = south_attack_ratios(rigs, palette)
    bad = [i for i, r in enumerate(ratios) if r < 0.90 or r > 1.08]
    if bad:
        raise SystemExit(f"south attack opaque ratio out of range at {bad}")

    broken = []
    for f in range(5, 11):
        fr = atk[f]
        sizes = components(fr[..., 3] >= 128)
        body = [s for s in sizes if s > 20]
        if len(body) != 1:
            broken.append(("attack", f, sizes[:6]))
    for f in (0, 3, 6, 9, 11):
        fr = synth(rigs["S"], "hurt", f, palette, "S")
        sizes = components(fr[..., 3] >= 128)
        body = [s for s in sizes if s > 20]
        if len(body) != 1:
            broken.append(("hurt", f, sizes[:6]))
    print("piece check", broken if broken else "ok", flush=True)
    if broken:
        raise SystemExit("spin or recoil broke the body")

    if args.samples:
        want = [("idle", f) for f in (0, 4, 8, 10)]
        want += [("walk", f) for f in (0, 2, 4, 8, 12)]
        want += [("attack", f) for f in range(16)]
        want += [("hurt", f) for f in (0, 1, 3, 6, 9, 11)]
        tiles = [synth(rigs["S"], act, f, palette, "S") for act, f in want]
        cols = 8
        x0, x1, y0, y1 = 48, 136, 70, 162
        scale = 3
        cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            up = np.repeat(np.repeat(tile[y0:y1, x0:x1], scale, 0), scale, 1)
            m = up[..., 3] >= 128
            sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw][m] = up[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        zoom_keys = [("idle", 0), ("idle", 8), ("walk", 4), ("walk", 12),
                     ("attack", 0), ("attack", 2), ("attack", 7), ("attack", 11), ("hurt", 6)]
        ztiles = []
        for act, f in zoom_keys:
            tile = synth(rigs["S"], act, f, palette, "S")
            bb = body_box(tile)
            x0b, y0b, x1b, y1b = bb
            x0b, y0b = max(0, x0b - 18), max(0, y0b - 18)
            x1b, y1b = min(CELL_W, x1b + 19), min(CELL_H, y1b + 19)
            crop = tile[y0b:y1b, x0b:x1b]
            ztiles.append(np.repeat(np.repeat(crop, 6, 0), 6, 1))
        zh = max(t.shape[0] for t in ztiles)
        zw = max(t.shape[1] for t in ztiles)
        board = np.full((zh, len(ztiles) * zw, 4), (15, 20, 31, 255), np.uint8)
        for i, t in enumerate(ztiles):
            m = t[..., 3] >= 128
            board[:t.shape[0], i * zw:i * zw + t.shape[1]][m] = t[m]
        Image.fromarray(board, "RGBA").save(work / "zoom.png")
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
                Image.fromarray(synth(rigs[d], act, f, palette, d), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = stack_sheets(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    contact = motion_contact(sheets)
    contact.save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    na = neutral[..., 3] >= 128
    nys = np.nonzero(na)[0]
    sprite_h = int(nys.max() - nys.min() + 1) if len(nys) else 0

    # Re-measure the written south attack row, which is what the sheet stores.
    sheet_counts = []
    for f in range(16):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        sheet_counts.append(int((tile[..., 3] >= 128).sum()))
    sheet_base = sheet_counts[0]
    sheet_ratios = [c / sheet_base for c in sheet_counts]
    print("SOUTH sheet attack opaque ratios:", flush=True)
    for f, (c, r) in enumerate(zip(sheet_counts, sheet_ratios)):
        print(f"  {f:02d} {c} {r:.4f}", flush=True)
    print(f"SOUTH sheet attack opaque min {min(sheet_ratios):.4f} max {max(sheet_ratios):.4f}", flush=True)

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted({int(v) for v in np.unique(master[..., 3])})
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "alphas": alphas,
        "edge_touching": {k: len(v) for k, v in clip.items()},
        "south_attack_ratios": [round(r, 4) for r in sheet_ratios],
        "facing": facing, "beak_tip_x": tip_x, "eye_x": [left_x, right_x],
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act, kind in (("idle", "hover"), ("walk", "hover")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", kind, "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False)[:1500], flush=True)

    bad_sheet = [i for i, r in enumerate(sheet_ratios) if r < 0.90 or r > 1.08]
    if off or clip or alphas != [0, 255] or sprite_h <= 0 or bad_sheet:
        raise SystemExit(f"qa failed off={len(off)} clip={clip} alphas={alphas} bad={bad_sheet}")

    meta = {
        "generator": "forge_prinplup_actions", "version": VERSION, "key": "0394_prinplup",
        "dex": 394, "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
        "facing": "front", "beak_tip_x": tip_x, "eye_x": [left_x, right_x],
        "south_attack_ratio_min": min(sheet_ratios), "south_attack_ratio_max": max(sheet_ratios),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
