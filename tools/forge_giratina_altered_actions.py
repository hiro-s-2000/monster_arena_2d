#!/usr/bin/env python3
"""Giratina Altered Forme (0487) normal 8-direction action sheet from this
form's own front.png / back.png (no pokeAnim, no image generation).

Measured on normal/front.png (96x96, alpha 0/255, 14 colours):
  bbox x6-89 y6-86. Heavy six-legged ghost dragon on the ground.
  The gold-masked head is near frontal (red eye x44-46 y20-21) but the body
  faces screen-left: the fore claw is bottom-left, the spiked tail curls up
  on the right and the hind leg is on the right.
  jaw     dark muzzle + gold chin under the visor, x39-46 y24-27 (real mouth)
  head    gold crown/visor and eye, stays on the torso (integer moves only)
  wing_l  shadow wing with red spikes, upper left x6-31 y7-46
  wing_r  shadow wing with red spikes, upper right x60-89 y6-46
  tail    grey spiked tail rising on the right, x71-81 y45-60
  leg_a   near fore leg + gold claw, bottom left x19-32 y66-83
  leg_d   far fore claw tucked behind, x35-47 y74-79
  leg_b   near middle leg, x48-65 y69-86 (lowest toe)
  leg_c   hind leg, x66-80 y67-80

back.png (96x96, 14 colours) faces up-right: crown top centre, both wings,
the tail wedge running left (x3-26 y48-70), three legs:
  leg_a hind x9-30 y76-85, leg_b middle x34-49 y76-87, leg_c fore claws x50-65 y65-86.

Directions: front still S SW W unflipped, SE E mirrored. back still N NE
unflipped, NW mirrored. E looks screen-right, W screen-left, N is the back.

  idle   : slow menacing breath (torso 1px), the shadow wings rise and fall,
           tail sways. Frame 0 and the loop end are the still.
  walk   : six-legged trot (near fore + hind against middle + far fore),
           lifted swing feet, 1px stomp bob, wings beat, tail swings.
  attack : 0-4 roar: the real jaw drops, wings flare wide, tail rears, red
           and grey roar rings leave the mouth (back rows: from the crown).
           5-6 the fore claw rears up, wings cock back (no body rotation),
           7-8 claw slam with shadow-claw arcs and a wing down-beat,
           9-12 ground shock + a shadow burst (dark orb with red rays),
           13-15 recover.
  hurt   : 12 frames, white flash, recoil of the one-piece body (<= 18 deg)
           around the rear foot, wings fold, sparks, recover.

Palette-locked to each still. front/back png and gif are not written.
Resumable via agent-sprite-forge/_work. --force clears only this form's frames.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_giratina_altered_actions.py [--samples]
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
import gen4_attack_strike  # noqa: E402

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0487_giratina-altered/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0487_giratina-altered/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# (still, mirrored)
VIEW = {
    "S": ("front", False), "SW": ("front", False), "W": ("front", False),
    "SE": ("front", True), "E": ("front", True),
    "N": ("back", False), "NE": ("back", False), "NW": ("back", True),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "giratina-altered_v1"
KEY = "0487_giratina-altered"

INK = (0, 0, 0)
SHADOW = (57, 57, 74)
SHADOW2 = (74, 74, 90)
RED = (255, 65, 82)
DRED = (164, 32, 57)
PALE = (197, 205, 213)
GREY = (148, 148, 164)
CREAM = (255, 238, 197)
GOLD = (255, 213, 0)
EYE = (255, 65, 82)
WHITE = CREAM

WING_COLS = ((57, 57, 74), (74, 74, 90), (255, 65, 82), (164, 32, 57), (90, 49, 16), (106, 106, 123))
PARTS = ("wing_l", "wing_r", "tail", "leg_a", "leg_b", "leg_c", "leg_d", "jaw")
LEGS = ("leg_a", "leg_b", "leg_c", "leg_d")

# ---------------------------------------------------------------- helpers
def flood(seed_yx, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
    y, x = seed_yx
    if not allow[y, x]:
        raise SystemExit(f"seed {seed_yx} is not inside its allow mask")
    q = deque([(y, x)])
    seen[y, x] = True
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


def is_col(img, rgb):
    a = img[..., 3] >= 128
    c = img[..., :3]
    return a & (c[..., 0] == rgb[0]) & (c[..., 1] == rgb[1]) & (c[..., 2] == rgb[2])


def rotate_nearest(layer, angle_deg, pivot):
    """4x nearest. Rigid, exact colours. Positive angle turns a point left of
    the pivot upward (counter-clockwise on screen)."""
    ang = float(angle_deg) % 360.0
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


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    for i in range(0, len(ty), 400):
        sl = slice(i, i + 400)
        d = (ty[sl, None].astype(np.int32) - ys[None, :].astype(np.int32)) ** 2
        d += (tx[sl, None].astype(np.int32) - xs[None, :].astype(np.int32)) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def fill_pinholes(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        p = np.pad(a, 1)
        n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not np.any(hole):
            break
        fill = nearest_color(out, hole, a)
        out[hole] = fill[hole]
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


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[..., 3] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def put(out, x, y, col, only_empty=False):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2:
        if only_empty and out[y, x, 3] >= 128:
            return
        out[y, x, :3] = col
        out[y, x, 3] = 255


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def contact_pivot(part, body):
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


def top_pivot(part):
    """Hip/shoulder joint of a leg: middle of its top two rows."""
    ys, xs = np.nonzero(part)
    top = ys <= int(ys.min()) + 1
    return float(np.median(xs[top])), float(ys.min())


def add_socket(host, host_mask, part_mask, canvas, n=4):
    zone = part_mask & dilate(host_mask, n)
    if not np.any(zone):
        return
    # keep only cushion pixels 4-connected to the host, so no island shows
    # when the part swings away
    lab = label4(zone)
    touch = np.unique(lab[zone & dilate(host_mask, 1)])
    zone = np.isin(lab, touch[touch > 0])
    if not np.any(zone):
        return
    ink = is_col(canvas, INK)
    src = host_mask & ~ink if np.any(host_mask & ~ink) else host_mask
    fill = nearest_color(canvas, zone, src)
    write = zone & (host[..., 3] < 128)
    host[write] = fill[write]


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def cols_mask(img, lst):
    a = img[..., 3] >= 128
    m = np.zeros(a.shape, bool)
    for c in lst:
        m |= a & np.all(img[..., :3] == np.array(c, np.uint8), axis=2)
    return m


def label4(mask):
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    for y, x in zip(*np.nonzero(mask)):
        if lab[y, x]:
            continue
        n += 1
        q = deque([(y, x)])
        lab[y, x] = n
        while q:
            cy, cx = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = n
                    q.append((ny, nx))
    return lab


def segment(img, kind):
    h, w = img.shape[:2]
    a = img[..., 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]
    empty = np.zeros_like(a)
    wc = cols_mask(img, WING_COLS)
    if kind == "front":
        wing_l = flood((14, 20), wc & (xx <= 31) & (yy <= 46))
        wing_r = flood((12, 75), wc & (xx >= 60) & (yy <= 46))
        tail = flood((48, 75), a & (xx >= 71) & (yy >= 44) & (yy <= 60) & ~wing_r)
        leg_a = flood((80, 20), a & (yy >= 66) & (xx <= 32))
        leg_d = flood((76, 40), a & (yy >= 74) & (xx >= 35) & (xx <= 47))
        leg_c = flood((75, 78), a & (yy >= 67) & (xx >= 66) & ~tail)
        leg_b = flood((84, 57), a & (yy >= 69) & (xx >= 48) & (xx <= 65) & ~leg_c)
        jaw = a & (yy >= 24) & (yy <= 27) & (xx >= 39) & (xx <= 46)
    else:
        wing_l = flood((20, 20), wc & (xx <= 40) & (yy <= 50))
        wing_r = flood((20, 80), wc & (xx >= 58) & (yy <= 50))
        tail = flood((60, 8), a & (xx <= 26) & (yy >= 48) & (yy <= 70))
        leg_a = flood((82, 15), a & (yy >= 76) & (xx <= 30) & ~tail)
        leg_d = empty
        leg_b = flood((84, 42), a & (yy >= 76) & (xx >= 34) & (xx <= 49))
        leg_c = flood((82, 55), ((a & (xx >= 50) & (yy >= 70)) | (a & (xx >= 57) & (yy >= 65))) & ~wing_r)
        jaw = empty
    parts = {"wing_l": wing_l, "wing_r": wing_r, "tail": tail, "leg_a": leg_a,
             "leg_b": leg_b, "leg_c": leg_c, "leg_d": leg_d, "jaw": jaw}
    cover = np.zeros_like(a)
    for name, m in parts.items():
        if np.any(cover & m):
            raise SystemExit(f"{kind}: {name} overlaps another part")
        cover |= m
    torso = a & ~cover
    # torso islands (spike tips / outline bits that touch the body only through
    # a wing) ride with the part they touch, so they never float
    lab = label4(torso)
    if lab.max() > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        main = int(sizes.argmax())
        for k in range(1, lab.max() + 1):
            if k == main:
                continue
            isl = lab == k
            ring = dilate(isl, 1) & ~isl
            best = max(parts, key=lambda n: int((ring & parts[n]).sum()))
            parts[best] = parts[best] | isl
            torso &= ~isl
    parts = {"torso": torso, **parts}
    if kind == "front":
        eye = is_col(img, EYE) & (yy <= 24) & (xx >= 40) & (xx <= 50)
        if not np.any(eye) or np.any(eye & ~parts["torso"]):
            raise SystemExit("the eye left the head")
    for name in ("leg_a", "leg_b", "leg_c", "tail", "wing_l", "wing_r"):
        if int(parts[name].sum()) < 40:
            raise SystemExit(f"{kind}: {name} collapsed ({int(parts[name].sum())})")
    return parts


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


# ---------------------------------------------------------------- rig
def placement(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1, (CELL_W - (x1 - x0)) // 2, FOOT_Y - (y1 - y0)


def placed_still(img, flip=False):
    y0, y1, x0, x1, dx, dy = placement(img)
    out = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    out[dy:dy + (y1 - y0), dx:dx + (x1 - x0)] = img[y0:y1, x0:x1]
    return out[:, ::-1].copy() if flip else out


def build_rig(img, masks, kind, flip):
    y0, y1, x0, x1, dst_x, dst_y = placement(img)
    sh, sw = y1 - y0, x1 - x0
    canvas = placed_still(img)
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
    body = placed["torso"]
    # joint cushions on their own layer under everything: a moved part never
    # leaves a hole, and the rest pose stays the exact still
    under = np.zeros_like(canvas)
    for name in PARTS:
        if np.any(placed[name]) and name != "jaw":
            add_socket(under, body, placed[name], canvas)
    layers["under"] = under

    pivots = {}
    for name in PARTS:
        if not np.any(placed[name]):
            pivots[name] = (0.0, 0.0)
        elif name.startswith("leg"):
            pivots[name] = top_pivot(placed[name])
        else:
            pivots[name] = contact_pivot(placed[name], body)
    opy, opx = np.nonzero(canvas[..., 3] >= 128)
    ground = int(opy.max())
    feet = {}
    for name in LEGS:
        if not np.any(placed[name]):
            continue
        ly, lx = np.nonzero(placed[name])
        low = ly >= int(ly.max()) - 1
        feet[name] = (float(np.median(lx[low])), float(ly.max()))
    # front faces left, back faces right
    face = -1.0 if kind == "front" else 1.0
    rear = max(feet.values(), key=lambda p: -face * p[0])
    # the crown top (roar origin for the back view)
    tys, txs = np.nonzero(placed["torso"])
    top_y = int(tys.min())
    crown = (float(np.median(txs[tys <= top_y + 6])), float(top_y + 6))
    if flip:
        layers = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in layers.items()}
        mx = lambda p: (CELL_W - 1 - p[0], p[1])  # noqa: E731
        pivots = {k: mx(v) for k, v in pivots.items()}
        feet = {k: mx(v) for k, v in feet.items()}
        rear = mx(rear)
        crown = mx(crown)
        face = -face
    # +1 when a positive angle raises the part's tip (tip left of the pivot)
    rise = {}
    for name in ("wing_l", "wing_r", "tail"):
        ys, xs = np.nonzero(layers[name][..., 3] >= 128)
        rise[name] = 1.0 if float(xs.mean()) < pivots[name][0] else -1.0
    strike = "leg_a" if kind == "front" else "leg_c"
    draw = ["under", "wing_l", "wing_r", "tail", "leg_c", "leg_d", "torso", "leg_a", "leg_b", "jaw"]
    return {
        "kind": kind, "flip": flip, "layers": layers, "pv": pivots, "face": face,
        "ground": ground, "feet": feet, "rear": rear, "strike": strike, "rise": rise,
        "crown": crown, "draw": draw,
    }


# ---------------------------------------------------------------- poses
def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0, bdx=0, wing=0.0, tail=0.0, jaw=0,
        leg_a=0.0, leg_b=0.0, leg_c=0.0, leg_d=0.0,
        lift_a=0, lift_b=0, lift_c=0, lift_d=0,
        strike=0.0, slift=0, brace=0, rings=0, ring_age=0, eye=0,
        slash=0, shock=0, burst=0, hurt=0.0, hdx=0, flash=0, sparks=0, spin=0,
    )
    if action == "idle":
        p["bdy"] = -int(round(max(0.0, math.sin(2 * math.pi * f / n)) * 1.0))
        p["wing"] = wave(f, n, 7.0, phase=0.0)
        p["tail"] = wave(f, n, 6.0, phase=1.0)
        p["eye"] = 1 if f in (10, 11) else 0
    elif action == "walk":
        ph = 2 * math.pi * 2 * f / n
        sw = math.sin(ph)
        # pair A = near fore + hind, pair B = middle + far fore
        p["leg_a"] = 14.0 * sw
        p["leg_c"] = 14.0 * sw
        p["leg_b"] = -14.0 * sw
        p["leg_d"] = -10.0 * sw
        swing = math.cos(ph)
        p["lift_a"] = int(round(3 * max(0.0, swing)))
        p["lift_c"] = int(round(2 * max(0.0, swing)))
        p["lift_b"] = int(round(3 * max(0.0, -swing)))
        p["lift_d"] = int(round(2 * max(0.0, -swing)))
        p["bdy"] = -1 if abs(swing) > 0.7 else 0
        p["wing"] = wave(f, n, 9.0, cycles=2.0, phase=0.0)
        p["tail"] = wave(f, n, 8.0, cycles=2.0, phase=1.2)
    elif action == "attack":
        jaw = (1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 0, 0, 0, 0)
        wing = (6, 12, 16, 18, 18, -10, -14, 20, 22, 16, 12, 9, 6, 3, 1, 0)
        tail = (3, 7, 10, 12, 12, 14, 16, 4, 2, 4, 6, 6, 4, 2, 1, 0)
        strike = (0, 0, 2, 5, 9, 18, 24, 6, 0, 0, 0, 0, 0, 0, 0, 0)
        slift = (0, 0, 1, 3, 5, 9, 12, 3, 0, 0, 0, 0, 0, 0, 0, 0)
        bdy = (0, -1, -1, -1, -1, -2, -2, 0, 1, 1, 0, 0, 0, 0, 0, 0)
        p["jaw"], p["wing"], p["tail"] = jaw[f], wing[f], tail[f]
        p["strike"], p["slift"], p["bdy"] = strike[f], slift[f], bdy[f]
        p["brace"] = (0, 0, 0, 1, 2, 4, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0)[f]
        if 1 <= f <= 5:
            p["rings"] = 3 if f >= 2 else 2
            p["ring_age"] = f - 1
        p["eye"] = 1 if 1 <= f <= 11 else 0
        p["slash"] = f - 6 if f in (7, 8) else 0
        p["shock"] = f - 7 if 8 <= f <= 12 else 0
        p["burst"] = f - 8 if 9 <= f <= 12 else 0
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 18.0 * env
        p["hdx"] = int(round(3.0 * env))
        p["jaw"] = 1 if env > 0.5 else 0
        p["wing"] = -8.0 * env
        p["tail"] = 9.0 * env
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = f if 2 <= f <= 6 else 0
    return p


# ---------------------------------------------------------------- effects
def mouth_point(rig, bdx, bdy):
    """Front: the bottom middle of the jaw. Back: the crown (no mouth drawn)."""
    if rig["kind"] == "front":
        lyr = shift(rig["layers"]["jaw"], bdx, bdy)
        ys, xs = np.nonzero(lyr[..., 3] >= 128)
        if len(xs) == 0:
            return None
        x = int(xs.min()) if rig["face"] < 0 else int(xs.max())
        return x + int(rig["face"]) * 2, int(ys.mean()) + 1
    cx, cy = rig["crown"]
    return int(cx + rig["face"] * 8 + bdx), int(cy + bdy)


def roar_rings(out, mouth, face, age, strength):
    mx, my = mouth
    cols = (CREAM, RED, PALE)
    for r_i in range(strength):
        rad = 9 + age * 5 + r_i * 6
        col = cols[min(r_i, 2)]
        for deg in range(-70, 71, 3):
            ang = math.radians(deg)
            for t in (0, 1):
                put(out, mx + face * (rad + t) * math.cos(ang), my + (rad + t) * math.sin(ang) * 0.9, col, True)
            if r_i == 1:
                put(out, mx + face * (rad + 2) * math.cos(ang), my + (rad + 2) * math.sin(ang) * 0.9, DRED, True)


def slash_arcs(out, foot, face, k):
    """Shadow Claw: three dark arcs with a red edge."""
    fx, fy = foot
    for i in range(3):
        r = 12 + i * 4 + k * 3
        for deg in range(-80, 30, 3):
            ang = math.radians(deg)
            x = fx + face * (4 + r * math.cos(ang) * 0.75)
            y = fy - 14 + r * math.sin(ang)
            put(out, x, y, (RED, CREAM, RED)[i], True)
            put(out, x + face, y, (DRED, RED, DRED)[i], True)
            put(out, x + 2 * face, y, SHADOW, True)


def ground_shock(out, foot, age, ground):
    fx, _ = foot
    for i, col in enumerate((PALE, RED, SHADOW)):
        rx = 6 + age * 5 + i * 4
        ry = max(2, rx * 0.28)
        if rx > 60:
            continue
        for deg in range(0, 360, 7):
            ang = math.radians(deg)
            x = fx + rx * math.cos(ang)
            y = ground + 1 + ry * math.sin(ang)
            if y > ground - 2:
                put(out, x, y, col, True)


def shadow_burst(out, center, size):
    """A dark orb with a red rim and red rays (Shadow Force)."""
    cx, cy = center
    r = 2 + size * 2
    for dy in range(-r - 1, r + 2):
        for dx in range(-r - 1, r + 2):
            d = math.hypot(dx, dy * 1.15)
            if d <= r - 1:
                put(out, cx + dx, cy + dy, SHADOW if (dx + dy) % 3 else SHADOW2, True)
            elif d <= r + 0.6:
                put(out, cx + dx, cy + dy, RED if size < 4 else DRED, True)
    for k in range(8):
        ang = k * math.pi / 4 + math.pi / 8 + size * 0.2
        for rr in range(r + 3, r + 3 + size * 3, 2):
            put(out, cx + rr * math.cos(ang), cy + rr * math.sin(ang) * 0.8, (RED, DRED)[rr % 2], True)


def sparks(out, frame_idx, face, box):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    d = frame_idx
    pts = ((cx + face * (8 + d * 2), y0 + 8 - d), (cx - face * (4 + d), y0 - 2 - d),
           (cx + face * 3, y0 + 18 - d))
    for sx, sy in pts:
        s = 2 if d % 2 else 1
        for t in range(-s, s + 1):
            put(out, sx + t, sy, CREAM, True)
            put(out, sx, sy + t, CREAM, True)
        put(out, sx, sy, RED, False)


def flash_white(img):
    out = img.copy()
    a = out[..., 3] >= 128
    ink = is_col(out, INK)
    out[a & ~ink, :3] = CREAM
    return out


# ---------------------------------------------------------------- synth
def bridge_diagonals(img):
    """Nearest-neighbour rotation turns 1px wisps into diagonal steps that
    touch only at corners. Fill one side of every such corner so the part
    stays 4-connected (colour from the nearest opaque neighbour)."""
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        h, w = a.shape
        need = np.zeros_like(a)
        p = np.pad(a, 1)
        c = p[1:-1, 1:-1]
        r, d, dr = p[1:-1, 2:], p[2:, 1:-1], p[2:, 2:]
        l, dl = p[1:-1, :-2], p[2:, :-2]
        # a and its down-right are set, right and down are empty -> fill right
        need |= np.pad((c & dr & ~r & ~d)[:, :-1], ((0, 0), (1, 0)))
        # a and its down-left are set, left and down are empty -> fill left
        need |= np.pad((c & dl & ~l & ~d)[:, 1:], ((0, 0), (0, 1)))
        need &= ~a
        if not np.any(need):
            break
        fill = nearest_color(out, need, a)
        out[need] = fill[need]
    return out


def part_moved(layer, pivot, ang, dx, dy):
    lyr = layer
    if abs(ang) >= 0.45:
        lyr = despeckle(bridge_diagonals(fill_pinholes(rotate_nearest(layer, ang, pivot))))
    return shift(lyr, dx, dy)


def assemble_body(rig, p):
    L, pv, face, rise = rig["layers"], rig["pv"], rig["face"], rig["rise"]
    bdx, bdy = int(p["bdx"]), int(p["bdy"])
    fwd = -face  # positive angle moves a hanging foot toward -x
    moved = {}
    for name in ("under", "torso"):
        moved[name] = shift(L[name], bdx, bdy)
    for name in ("wing_l", "wing_r"):
        # positive "wing" spreads the wing outward (tip away from the head)
        moved[name] = part_moved(L[name], pv[name], -rise[name] * p["wing"], bdx, bdy)
    moved["tail"] = part_moved(L["tail"], pv["tail"], rise["tail"] * p["tail"], bdx, bdy)
    moved["jaw"] = shift(L["jaw"], bdx, bdy + int(p["jaw"]))
    for leg, key in (("leg_a", "lift_a"), ("leg_b", "lift_b"), ("leg_c", "lift_c"), ("leg_d", "lift_d")):
        deg = fwd * p[leg]
        lift = int(p[key])
        if leg == rig["strike"]:
            deg += fwd * p["strike"]
            lift += int(p["slift"])
        elif p.get("brace"):
            deg -= fwd * p["brace"]
        moved[leg] = part_moved(L[leg], pv[leg], deg, bdx, -lift)
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in rig["draw"]:
        if name == "jaw" and rig["kind"] == "front" and int(p["jaw"]) > 0:
            # open mouth: the gap the dropped jaw leaves is black
            home = shift(L["jaw"], bdx, bdy)
            hole = (home[..., 3] >= 128) & (moved["jaw"][..., 3] < 128)
            img[hole, :3] = INK
            img[hole, 3] = 255
        paste(img, moved[name])
    if p["eye"] and rig["kind"] == "front":
        # the red eye flares: its brightest pixels turn cream
        eye = is_col(img, EYE)
        hy, hx = np.mgrid[0:CELL_H, 0:CELL_W]
        jy, jx = np.nonzero(moved["jaw"][..., 3] >= 128)
        if len(jy):
            near = (hy >= jy.min() - 7) & (hy <= jy.min()) & (np.abs(hx - jx.mean()) <= 6)
            ey, ex = np.nonzero(eye & near)
            if len(ey):
                img[ey[0], ex[0], :3] = CREAM
    return img


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    face = rig["face"]
    if p["hurt"] > 0.45 or p["hdx"]:
        base = assemble_body(rig, p)
        img = rotate_nearest(base, -face * p["hurt"], rig["rear"])
        img = shift(img, -face * p["hdx"], 0)
        img = despeckle(fill_pinholes(img))
    else:
        img = assemble_body(rig, p)
    img = snap(img, palette)
    if not effects:
        return img
    if p["flash"]:
        img = snap(flash_white(img), palette)
    if p["rings"]:
        m = mouth_point(rig, int(p["bdx"]), int(p["bdy"]))
        if m:
            roar_rings(img, m, face, p["ring_age"], p["rings"])
    foot = rig["feet"][rig["strike"]]
    if p["slash"]:
        slash_arcs(img, (foot[0], foot[1] - 2), face, p["slash"])
    if p["shock"]:
        ground_shock(img, (foot[0] + face * 3, foot[1]), p["shock"], rig["ground"])
    if p["burst"]:
        shadow_burst(img, (foot[0] + face * 14, foot[1] - 14), p["burst"])
    if p["sparks"]:
        bb = body_box(img)
        if bb:
            sparks(img, p["sparks"], face, bb)
    return snap(img, palette)


# ---------------------------------------------------------------- output
def params_hash():
    src = (Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes()
           + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes())
    return hashlib.sha1(src).hexdigest()[:12]


def stack_sheets(frames_dir):
    width = max(FRAMES.values()) * CELL_W
    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, width, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = np.array(Image.open(frames_dir / act / d / f"{f:02d}.png").convert("RGBA"))
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
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 2, 4, 6, 8, 10, 12, 14),
        "attack": (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15),
        "hurt": (0, 2, 4, 6, 8, 11),
    }
    cols = max(len(v) for v in samples.values())
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            m = tile[..., 3] >= 128
            bg[ai * CELL_H:(ai + 1) * CELL_H, si * CELL_W:(si + 1) * CELL_W][m] = tile[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """Every frame of every action, cropped, for S (front) and N (back)."""
    pad = 3
    rows = []
    for di in (0, 4):
        for act in ACTIONS:
            row = []
            band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            for f in range(FRAMES[act]):
                tile = band[:, f * CELL_W:(f + 1) * CELL_W]
                a = tile[..., 3] >= 128
                if not np.any(a):
                    continue
                ys, xs = np.nonzero(a)
                y0, y1 = max(0, int(ys.min()) - pad), min(CELL_H, int(ys.max()) + 1 + pad)
                x0, x1 = max(0, int(xs.min()) - pad), min(CELL_W, int(xs.max()) + 1 + pad)
                row.append(tile[y0:y1, x0:x1])
            rows.append(row)
    ch = max(c.shape[0] for r in rows for c in r)
    cw = max(c.shape[1] for r in rows for c in r)
    ncol = max(len(r) for r in rows)
    bg = np.full((len(rows) * ch, ncol * cw, 4), (15, 20, 31, 255), np.uint8)
    for ri, r in enumerate(rows):
        for ci, c in enumerate(r):
            y = ri * ch + (ch - c.shape[0])
            x = ci * cw + (cw - c.shape[1]) // 2
            m = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][m] = c[m]
    return Image.fromarray(bg, "RGBA")


SHEET_FILES = (
    "sheet.png", "preview.png", "preview_motion.png", "meta.json",
    "all_8dir_actions.png",
    "idle.png", "walk.png", "attack.png", "hurt.png",
    "idle_8dir.png", "walk_8dir.png", "attack_8dir.png", "hurt_8dir.png",
    "idle_S_strip.png", "walk_S_strip.png", "attack_S_strip.png", "hurt_S_strip.png",
)


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in SHEET_FILES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, SYNC_OUT / name)
            n += 1
    return n


def body_pieces(img):
    return [s for s in components(img[..., 3] >= 128) if s > 6]


def diff_count(a_img, b_img):
    d = np.any(a_img != b_img, axis=2) & ((a_img[..., 3] >= 128) | (b_img[..., 3] >= 128))
    return int(d.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    stills = {k: np.array(Image.open(SRC_DIR / f"{k}.png").convert("RGBA")) for k in ("front", "back")}
    for k, im in stills.items():
        if im.shape[:2] != (96, 96):
            raise SystemExit(f"expected 96x96 {k}.png, got {im.shape}")
    palettes = {k: extract_palette(im) for k, im in stills.items()}
    masks = {k: segment(im, k) for k, im in stills.items()}
    counts = {k: {n: int(m.sum()) for n, m in v.items()} for k, v in masks.items()}
    print("parts", json.dumps(counts), flush=True)
    # facing: the fore claw (leg_a) is left of the tail; the back tail runs left
    fa = np.nonzero(masks["front"]["leg_a"])[1].mean()
    ft = np.nonzero(masks["front"]["tail"])[1].mean()
    ba = np.nonzero(masks["back"]["tail"])[1].mean()
    bc = np.nonzero(masks["back"]["leg_c"])[1].mean()
    print("facing check: front claw x", round(fa, 1), "tail x", round(ft, 1),
          "| back tail x", round(ba, 1), "fore claw x", round(bc, 1), flush=True)
    if not (fa < ft and ba < bc):
        raise SystemExit("front.png is expected to face left and back.png right")

    rigs = {d: build_rig(stills[VIEW[d][0]], masks[VIEW[d][0]], VIEW[d][0], VIEW[d][1]) for d in DIRS}
    for d in ("W", "N"):
        print(d, "pivots", {k: tuple(round(v, 1) for v in p) for k, p in rigs[d]["pv"].items()},
              "rear", rigs[d]["rear"], flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)

    # still check for every direction
    bad = {}
    for d in DIRS:
        kind, flip = VIEW[d]
        ref = placed_still(stills[kind], flip)
        n = diff_count(synth(rigs[d], "idle", 0, palettes[kind]), ref)
        if n:
            bad[d] = n
    print("still diff", bad if bad else 0, flush=True)
    if bad:
        raise SystemExit("idle frame 0 does not match the still")

    broken = []
    for d in ("S", "N", "E"):
        kind = VIEW[d][0]
        for act in ACTIONS:
            for f in range(FRAMES[act]):
                fr = synth(rigs[d], act, f, palettes[kind], effects=False)
                pcs = body_pieces(fr)
                if len(pcs) != 1:
                    broken.append((d, act, f, pcs[:4]))
    print("piece check", broken if broken else "ok", flush=True)

    if args.samples:
        want = [("idle", "W", f) for f in (0, 4, 8, 12)]
        want += [("walk", "W", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "S", f) for f in range(16)]
        want += [("hurt", "S", f) for f in range(12)]
        want += [("walk", "N", f) for f in (0, 2, 4, 6)]
        want += [("attack", "N", f) for f in (2, 4, 6, 7, 9, 11)]
        want += [("idle", d, 0) for d in DIRS]
        tiles = [synth(rigs[d], act, f, palettes[VIEW[d][0]]) for act, d, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            m = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][m] = tile[m]
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png", flush=True)
        return
    if broken:
        raise SystemExit("a pose broke the body")

    frames_dir = work / "frames"
    state_p = work / "state.json"
    digest = params_hash()
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if args.force or state.get("hash") != digest:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": digest, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(DIRS) * sum(FRAMES.values())
    done = 0
    for act in ACTIONS:
        for d in DIRS:
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                done += 1
                if fp.exists() and fp.stat().st_size > 0 and state.get("hash") == digest:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palettes[VIEW[d][0]]), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"] = sorted(set(state.get("done", [])) | {f"{act}/{d}"})
            state_p.write_text(json.dumps(state))
            print(f"[{done}/{total}] {act}/{d}", flush=True)

    sheets = stack_sheets(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(exist_ok=True)
    preview.save(SRC_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    neutral = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(neutral[..., 3] >= 128)[0]
    sprite_h = int(nys.max() - nys.min() + 1)
    bounds = []
    for di in range(8):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        ys, xs = np.nonzero(t[..., 3] >= 128)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]

    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    union = np.unique(np.concatenate(list(palettes.values())), axis=0)
    pal = {tuple(int(v) for v in c) for c in union}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    # per-view palette check
    off_view = 0
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            pv = {tuple(int(v) for v in c) for c in palettes[VIEW[d][0]]}
            u = {tuple(int(v) for v in c) for c in np.unique(band[band[..., 3] >= 128][:, :3], axis=0)}
            off_view += len(u - pv)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]))
    clip = {}
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    still_bad = 0
    for di, d in enumerate(DIRS):
        kind, flip = VIEW[d]
        still_bad += diff_count(sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W], placed_still(stills[kind], flip))
    report = {
        "version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
        "off_palette_colors": len(off), "off_view_palette": off_view, "alphas": alphas,
        "edge_touching": sum(len(v) for v in clip.values()), "still_diff": still_bad,
        "bounds_s": bounds, "views": {d: list(VIEW[d]) for d in DIRS},
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
            else:
                report[f"gait_{act}"] = (r.stdout + r.stderr).strip()[-300:]
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "parts"}, ensure_ascii=False), flush=True)

    meta = {
        "generator": "forge_giratina_altered_actions", "version": VERSION, "key": KEY,
        "dex": 487, "form": "normal", "variant": "giratina-altered",
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "archetype": "quadruped_beast + shadow wings (six-legged ghost dragon; 4 legs drawn front, 3 back)",
        "off_palette_colors": len(off), "source": "front.png + back.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = bool(off or off_view or clip or still_bad or alphas not in ([0, 255], [0]))
    if failed:
        raise SystemExit("qa failed")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)


if __name__ == "__main__":
    main()
