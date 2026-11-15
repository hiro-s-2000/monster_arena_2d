#!/usr/bin/env python3
"""Floatzel (0419_floatzel) 8-direction action sheet from the 96x96 front still.

Measured on front.png (not copied from another species):

  Facing left. Snout tip is x=18 at y=39. Eyes sit near x=34. The yellow
  head-float and both flotation tails are to the right of the face.
  Unflipped: N NW W SW. Flipped: S SE E NE, so E faces screen-right.

  Feet: two. Near foot x=26-38 y=73-78. Far foot x=43-59 y=73-81.
  Hands: two blue forepaws (near x=20-28, far x=46-54). They are not feet.
  Flotation tails: two, sharing a rear base and not feet.
    Upper lobe x=61-71 y=45-52. Lower lobe x=60-77 y=50-72.
  Mouth: the lower snout (jaw) already on the still, y=41-43.

  idle   : both feet planted. Torso breathes 1px. The two tails wag.
  walk   : the two feet alternate (one stays planted). Tails counter.
           Hands swing a couple of pixels. No new legs.
  attack : 0-4 the jaw drops and rings sit in front of the snout.
           5-10 one-piece in-plane 360 spin (no |cos| squash).
           11-12 impact. 13-15 the jaw settles.
  hurt   : whole-body recoil toward the tails, 38 degrees max.

Palette-locked. Alpha is 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
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
NORMAL = ROOT / "monsterSprite/gen4/0419_floatzel/normal"
OUT_DIR = NORMAL / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0419_floatzel/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
# Left-facing still. Mirror these so E points screen-right.
FLIP_DIRS = {"S", "SE", "E", "NE"}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 184, 176, 156
VERSION = "floatzel_v2"
KEY = "0419_floatzel"
NEIGHBOR = "0420_cherubi"

INK = (24, 24, 24)
WHITE = (255, 255, 255)
CREAM = (246, 246, 180)
YELLOW = (255, 222, 0)


def extract_palette(img):
    opaque = img[..., 3] >= 128
    return np.unique(img[opaque][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    opaque = out[..., 3] >= 128
    if not np.any(opaque):
        out[..., 3] = 0
        return out
    flat = out[..., :3].reshape(-1, 3)
    sel = opaque.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    mapped = palette[dist.argmin(axis=1)]
    flat = flat.copy()
    flat[sel] = mapped[inv]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    return out


def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


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
    """4x nearest. 0 and 360 degrees return the same pixels. No |cos| squash."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    up = np.repeat(np.repeat(layer, 4, axis=0), 4, axis=1)
    px, py = pivot[0] * 4 + 1.5, pivot[1] * 4 + 1.5
    rad = math.radians(ang)
    co, s = math.cos(rad), math.sin(rad)
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + co * rx + s * ry).astype(np.int32)
    src_y = np.rint(py - s * rx + co * ry).astype(np.int32)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(up)
    vy, vx = np.nonzero(valid)
    out[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return out[1::4, 1::4].copy()


def despeckle(img):
    """Drop only pixels whose four neighbours are all empty."""
    out = img.copy()
    opaque = out[..., 3] >= 128
    p = np.pad(opaque, 1)
    n = (
        p[:-2, 1:-1].astype(np.int16)
        + p[2:, 1:-1]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
    )
    out[opaque & (n == 0)] = 0
    return out


def fill_pinholes(img):
    """Fill enclosed holes of 1 or 2 pixels so a spin does not punch the body."""
    opaque = img[..., 3] >= 128
    trans = ~opaque
    h, w = trans.shape
    seen = np.zeros_like(trans)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if trans[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if trans[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and trans[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    enc = trans & ~seen
    if not np.any(enc):
        return img
    seen_e = np.zeros_like(enc)
    out = img.copy()
    ys, xs = np.nonzero(opaque)
    src = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = img[ys, xs]
    for y, x in zip(*np.nonzero(enc)):
        y, x = int(y), int(x)
        if seen_e[y, x]:
            continue
        q = deque([(y, x)])
        seen_e[y, x] = True
        comp = [(y, x)]
        while q:
            cy, cx = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and enc[ny, nx] and not seen_e[ny, nx]:
                    seen_e[ny, nx] = True
                    q.append((ny, nx))
                    comp.append((ny, nx))
        if len(comp) > 2:
            continue
        for cy, cx in comp:
            d = (src[:, 0] - cy) ** 2 + (src[:, 1] - cx) ** 2
            out[cy, cx] = cols[int(d.argmin())]
            out[cy, cx, 3] = 255
    return out


def nearest_color(src, target_mask, from_mask):
    layer = np.zeros_like(src)
    if not np.any(target_mask) or not np.any(from_mask):
        return layer
    ys, xs = np.nonzero(from_mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    cols = src[ys, xs]
    ty, tx = np.nonzero(target_mask)
    pick = np.empty(len(ty), np.int32)
    step = 400
    for i in range(0, len(ty), step):
        sl = slice(i, i + step)
        d = (ty[sl, None].astype(np.int32) - pts[None, :, 0]) ** 2
        d += (tx[sl, None].astype(np.int32) - pts[None, :, 1]) ** 2
        pick[sl] = d.argmin(axis=1)
    layer[ty, tx] = cols[pick]
    return layer


def segment(img):
    """Two feet, two blue hands, two flotation tails, one lower jaw, the torso."""
    opaque = img[..., 3] >= 128
    rgb = img[..., :3]
    h, w = opaque.shape
    yy, xx = np.mgrid[0:h, 0:w]
    blue = opaque & (
        ((rgb[..., 0] == 98) & (rgb[..., 1] == 189) & (rgb[..., 2] == 230))
        | ((rgb[..., 0] == 32) & (rgb[..., 1] == 98) & (rgb[..., 2] == 164))
    )
    ink = opaque & (rgb[..., 0] == 24) & (rgb[..., 1] == 24) & (rgb[..., 2] == 24)

    foot_near = opaque & (yy >= 73) & (xx <= 40)
    foot_far = opaque & (yy >= 73) & (xx >= 41)

    # Upper flotation lobe is the detached rear tip above the cream tail.
    tail_hi = opaque & (yy >= 45) & (yy <= 52) & (xx >= 60) & ~((yy >= 50) & (xx >= 69))
    tail_lo = opaque & (yy >= 50) & (yy <= 72) & (xx >= 60) & ~tail_hi

    left_blue = blue & (xx < 36)
    hand_near = (left_blue | (ink & dilate(left_blue, 1))) & (yy >= 43) & (yy <= 58) & (xx <= 30)
    right_blue = blue & (xx >= 40)
    hand_far = (right_blue | (ink & dilate(right_blue, 1))) & (yy >= 48) & (yy <= 64) & (xx >= 44) & (xx <= 58)

    # Lower snout that already sticks out in front of the eyes.
    jaw = opaque & (xx <= 31) & (yy >= 41) & (yy <= 43)

    hand_near &= ~jaw
    hand_far &= ~(tail_hi | tail_lo | foot_near | foot_far | jaw | hand_near)
    tail_lo &= ~tail_hi
    named = {
        "foot_near": foot_near,
        "foot_far": foot_far,
        "hand_near": hand_near,
        "hand_far": hand_far,
        "tail_hi": tail_hi,
        "tail_lo": tail_lo,
        "jaw": jaw,
    }
    acc = np.zeros_like(opaque)
    for name, mask in named.items():
        if np.any(acc & mask):
            raise SystemExit(f"floatzel part overlap at {name}")
        acc |= mask
    torso = opaque & ~acc
    if int((acc | torso).sum()) != int(opaque.sum()):
        raise SystemExit("floatzel parts do not cover the still")
    named["torso"] = torso
    return named


def place_still(img):
    opaque = img[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x = (CELL_W - sw) // 2
    dst_y = FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = img[y0:y1, x0:x1]
    origin = (dst_x - x0, dst_y - y0)
    return canvas, origin


def move_mask(mask, origin):
    ox, oy = origin
    out = np.zeros((CELL_H, CELL_W), bool)
    ys, xs = np.nonzero(mask)
    out[ys + oy, xs + ox] = True
    return out


def top_pivot(mask):
    ys, xs = np.nonzero(mask)
    top = int(ys.min())
    sel = ys <= top + 1
    return (float(xs[sel].mean()), float(ys[sel].mean()))


def base_pivot(mask):
    """Left edge of a rear tail. The still faces left, so the base is -x."""
    ys, xs = np.nonzero(mask)
    left = int(xs.min())
    sel = xs <= left + 2
    return (float(xs[sel].mean()), float(ys[sel].mean()))


def build_rig(img, masks):
    placed, origin = place_still(img)
    moved = {name: move_mask(mask, origin) for name, mask in masks.items()}
    layers = {}
    for name, mask in moved.items():
        layer = np.zeros_like(placed)
        layer[mask] = placed[mask]
        layers[name] = layer

    # Belly continues under the far paw. The near paw only needs its root.
    torso_m = moved["torso"]
    under_far = moved["hand_far"]
    under_near = moved["hand_near"] & dilate(torso_m, 2)
    cushion = nearest_color(placed, under_far | under_near, torso_m)
    layers["torso"][under_far | under_near] = cushion[under_far | under_near]
    # Dark mouth under the jaw. Covered while the jaw is shut.
    layers["torso"][moved["jaw"], :3] = INK
    layers["torso"][moved["jaw"], 3] = 255

    feet = moved["foot_near"] | moved["foot_far"]
    ankle = torso_m & dilate(feet, 1)
    tails = moved["tail_hi"] | moved["tail_lo"]
    hip = tails & dilate(torso_m, 1)
    bridge = moved["tail_hi"] & dilate(moved["tail_lo"], 1)
    socket = np.zeros_like(placed)
    socket[ankle] = placed[ankle]
    hip_fill = nearest_color(placed, hip | bridge, torso_m | moved["tail_lo"])
    socket[hip | bridge] = hip_fill[hip | bridge]

    opaque = placed[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    return {
        "layers": layers,
        "socket": socket,
        "placed": placed,
        "moved": moved,
        "tail_hi_pivot": base_pivot(moved["tail_hi"]),
        "tail_lo_pivot": base_pivot(moved["tail_lo"]),
        "foot_near_pivot": top_pivot(moved["foot_near"]),
        "foot_far_pivot": top_pivot(moved["foot_far"]),
        "centroid": (float(xs.mean()), float(ys.mean())),
        "foot_pivot": (float(xs.mean()), float(ys.max())),
    }


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2.0 * math.pi * cycles * (f / n)
    return amp * (math.sin(ph + phase) - math.sin(phase))


def step_lift(f, n, phase):
    return max(0.0, wave(f, n, 1.0, phase=phase, cycles=2.0))


def pose(action, f):
    n = FRAMES[action]
    p = {
        "bdy": 0,
        "thi": 0.0, "tlo": 0.0,
        "ndx": 0, "ndy": 0, "nang": 0.0,
        "fdx": 0, "fdy": 0, "fang": 0.0,
        "hndx": 0, "hfdx": 0,
        "jdx": 0, "jdy": 0, "rings": 0, "impact": 0,
        "spin": 0.0, "hurt": 0.0, "hdx": 0, "sweat": 0,
    }
    if action == "idle":
        p["bdy"] = -int(round(wave(f, n, 1.15)))
        p["thi"] = wave(f, n, 6.0, phase=0.4)
        p["tlo"] = wave(f, n, 5.0, phase=1.3)
    elif action == "walk":
        p["bdy"] = -int(round(wave(f, n, 1.0, cycles=2.0)))
        p["thi"] = wave(f, n, 8.0, phase=0.5, cycles=2.0)
        p["tlo"] = wave(f, n, 7.0, phase=1.6, cycles=2.0)
        near = step_lift(f, n, 0.0)
        far = step_lift(f, n, math.pi)
        # Facing left: a lifted foot reaches toward the snout (negative x).
        p["ndy"] = -int(round(3.0 * near))
        p["ndx"] = -int(round(2.0 * near))
        p["nang"] = 20.0 * near
        p["fdy"] = -int(round(3.0 * far))
        p["fdx"] = -int(round(2.0 * far))
        p["fang"] = 20.0 * far
        p["hndx"] = int(round(wave(f, n, 2.0, cycles=2.0, phase=math.pi)))
        p["hfdx"] = int(round(wave(f, n, 2.0, cycles=2.0, phase=0.0)))
    elif action == "attack":
        jaw_dy = (0, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
        jaw_dx = (0, -1, -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        rings = (0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        p["jdy"] = jaw_dy[f]
        p["jdx"] = jaw_dx[f]
        p["rings"] = rings[f]
        if 5 <= f <= 10:
            p["spin"] = 360.0 * (f - 4) / 6.0
        if f in (11, 12):
            p["impact"] = 1
    else:
        # Positive angle swings the head toward +x, which is the tail side.
        angles = (0, 12, 24, 32, 38, 34, 24, 14, 8, 4, 2, 0)
        shifts = (0, 1, 2, 3, 4, 3, 2, 1, 1, 0, 0, 0)
        p["hurt"] = float(angles[f])
        p["hdx"] = shifts[f]
        p["sweat"] = 1 if 2 <= f <= 5 else 0
    return p


def paste(dst, layer):
    mask = layer[..., 3] >= 128
    dst[mask] = layer[mask]


def put_empty(img, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if 1 <= x < w - 1 and 1 <= y < h - 1 and img[y, x, 3] == 0:
        img[y, x, :3] = col
        img[y, x, 3] = 255


def forward_anchor(img):
    """Left edge of the snout. The tails and the far paw are not the front."""
    opaque = img[..., 3] >= 128
    ys, xs = np.nonzero(opaque)
    if len(xs) == 0:
        return None
    y1 = int(ys.min() + 0.48 * (ys.max() - ys.min()))
    sel = ys <= y1
    x = int(xs[sel].min())
    y = int(np.median(ys[sel][xs[sel] <= x + 1]))
    return x, y


def roar_rings(img, mouth, strength):
    """Sparse arcs in the empty space in front of the snout."""
    if mouth is None or strength <= 0:
        return img
    mx, my = mouth
    cols = (WHITE, CREAM, YELLOW)
    radii = (6, 10, 14)[:strength]
    for i, rad in enumerate(radii):
        col = cols[i]
        for deg in range(-54, 55, 18):
            ang = math.radians(deg)
            put_empty(img, mx - 3 - rad * math.cos(ang), my + rad * math.sin(ang) * 0.62, col)
    return img


def impact_dots(img, mouth):
    if mouth is None:
        return img
    mx, my = mouth
    for k in range(5):
        ang = (k - 2) * 0.45
        put_empty(img, mx - 6 - 4 * math.cos(ang), my + 6 * math.sin(ang), WHITE)
        put_empty(img, mx - 10 - 3 * math.cos(ang), my + 9 * math.sin(ang), YELLOW)
    return img


def sweat(img, anchor):
    if anchor is None:
        return img
    ax, ay = anchor
    for dx, dy in ((6, -8), (9, -5), (4, -12), (12, -9)):
        put_empty(img, ax + dx, ay + dy, WHITE)
    return img


def compose(rig, p):
    layers = rig["layers"]
    bdy = p["bdy"]
    body = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    paste(body, rig["socket"])
    paste(body, shift(layers["torso"], 0, bdy))
    for name, ang, pivot in (
        ("tail_hi", p["thi"], rig["tail_hi_pivot"]),
        ("tail_lo", p["tlo"], rig["tail_lo_pivot"]),
    ):
        tail = rotate_nearest(layers[name], ang, pivot)
        if abs(ang) >= 0.45:
            tail = despeckle(tail)
        paste(body, shift(tail, 0, bdy))
    far = rotate_nearest(layers["foot_far"], p["fang"], rig["foot_far_pivot"])
    if abs(p["fang"]) >= 0.45:
        far = despeckle(far)
    paste(body, shift(far, p["fdx"], p["fdy"]))
    paste(body, shift(layers["hand_far"], p["hfdx"], bdy))
    near = rotate_nearest(layers["foot_near"], p["nang"], rig["foot_near_pivot"])
    if abs(p["nang"]) >= 0.45:
        near = despeckle(near)
    paste(body, shift(near, p["ndx"], p["ndy"]))
    paste(body, shift(layers["hand_near"], p["hndx"], bdy))
    paste(body, shift(layers["jaw"], p["jdx"], bdy + p["jdy"]))
    return body


def finish_body(img, pivot, angle):
    turned = rotate_nearest(img, angle, pivot)
    ang = angle % 360.0
    if ang < 0.45:
        # 360 lands on the still. Do not fill the snout's original 2px gap.
        return turned
    turned = despeckle(turned)
    return fill_pinholes(turned)


def synth(rig, action, f, palette, flip):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)

    if p["spin"] > 0.45:
        img = finish_body(rig["placed"], rig["centroid"], p["spin"])
    elif p["hurt"] > 0.45 or p["hdx"]:
        leaned = finish_body(rig["placed"], rig["foot_pivot"], p["hurt"])
        img = shift(leaned, p["hdx"], 0)
    else:
        img = compose(rig, p)
    img = snap(img, palette)
    if p["rings"]:
        img = roar_rings(img, forward_anchor(img), p["rings"])
    if p["impact"]:
        img = impact_dots(img, forward_anchor(img))
    if p["sweat"]:
        opaque = rig["placed"][..., 3] >= 128
        ys, xs = np.nonzero(opaque)
        top = int(ys.min())
        band = xs[ys <= top + 6]
        img = sweat(img, (int(np.median(band)) + p["hdx"], top))
    img = snap(img, palette)
    if flip:
        img = img[:, ::-1].copy()
    return img


def params_hash():
    src = Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes() + (NORMAL / "front.png").read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
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
    samples = {
        "idle": (0, 4, 8, 12),
        "walk": (0, 4, 8, 12),
        "attack": (2, 4, 7, 11),
        "hurt": (2, 4, 6, 9),
    }
    cols = 4
    bg = np.full((len(ACTIONS) * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, f in enumerate(samples[act]):
            tile = sheets[act][0:CELL_H, f * CELL_W:(f + 1) * CELL_W]
            y, x = ai * CELL_H, si * CELL_W
            mask = tile[..., 3] >= 128
            bg[y:y + CELL_H, x:x + CELL_W][mask] = tile[mask]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    pad = 6
    crops = []
    for act in ACTIONS:
        # East row is index 2. It is the mirrored still, nose to screen-right.
        row = []
        for f in range(FRAMES[act]):
            tile = sheets[act][2 * CELL_H:3 * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            opaque = tile[..., 3] >= 128
            if not np.any(opaque):
                continue
            ys, xs = np.nonzero(opaque)
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
            mask = c[..., 3] >= 128
            bg[y:y + c.shape[0], x:x + c.shape[1]][mask] = c[mask]
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    """Copy the sheet tree except _work."""
    if SYNC_OUT.exists():
        for fp in SYNC_OUT.rglob("*"):
            if fp.is_file() and "_work" not in fp.relative_to(SYNC_OUT).parts:
                pass
    n = 0
    for fp in OUT_DIR.rglob("*"):
        if not fp.is_file():
            continue
        rel = fp.relative_to(OUT_DIR)
        if rel.parts[0] == "_work":
            continue
        dest = SYNC_OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, dest)
        n += 1
    return n


def sprite_height(sheets):
    neutral = sheets["idle"][:CELL_H, :CELL_W]
    ys = np.nonzero(neutral[..., 3] >= 128)[0]
    return int(ys.max() - ys.min() + 1)


def south_attack_ratios(sheets):
    row = sheets["attack"][:CELL_H]
    counts = []
    for f in range(16):
        tile = row[:, f * CELL_W:(f + 1) * CELL_W]
        counts.append(int((tile[..., 3] >= 128).sum()))
    base = counts[0]
    ratios = [c / base for c in counts]
    return counts, ratios


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


def qa_report(sheets, master, palette, digest, counts):
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] >= 128][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = set(int(v) for v in np.unique(master[..., 3]))
    clip = []
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                c = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] >= 128
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
    atk_counts, ratios = south_attack_ratios(sheets)
    spin_parts = []
    for f in range(5, 11):
        tile = sheets["attack"][:CELL_H, f * CELL_W:(f + 1) * CELL_W]
        sizes = components(tile[..., 3] >= 128)
        spin_parts.append(sizes[:3])
    return {
        "version": VERSION,
        "hash": digest,
        "parts": counts,
        "sprite_h": sprite_height(sheets),
        "off_palette": len(off),
        "off_colors": [list(c) for c in off[:8]],
        "alphas": sorted(alphas),
        "edge_touching": clip,
        "attack_counts": atk_counts,
        "attack_ratios": [round(r, 4) for r in ratios],
        "attack_ratio_min": min(ratios),
        "attack_ratio_max": max(ratios),
        "spin_components": spin_parts,
    }


def _entry_bounds(text, key):
    token = f'"{key}":'
    if token not in text:
        return None
    i = text.index(token) + len(token)
    while text[i] != "{":
        i += 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return i, j + 1
    raise SystemExit(f"dir8 entry braces did not close for {key}")


def publish_dir8(sprite_h):
    """Replace only 0419_floatzel. The rest of dir8.js stays byte-for-byte."""
    entry = {
        "dex": 419,
        "form": "normal",
        "variant": "floatzel",
        "gen": "gen4",
        "dir": "monsterSprite/gen4/0419_floatzel/normal",
        "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir",
        "nf": 16,
        "cw": CELL_W,
        "ch": CELL_H,
        "w": CELL_W,
        "h": int(sprite_h),
        "fy": FOOT_Y,
        "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055],
        "raw": 1,
        "src": "agent-sprite-forge",
        "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    key = f'"{KEY}":'
    DIR8_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DIR8_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed; not publishing")
        neighbor = NEIGHBOR if f'"{NEIGHBOR}":' in text else "0001_bulbasaur"
        before = _entry_bounds(text, neighbor)
        if before is None:
            raise SystemExit(f"neighbor {neighbor} missing; not publishing")
        neighbor_blob = text[before[0]:before[1]]
        if key in text:
            i, end = _entry_bounds(text, KEY)
            text = text[:i] + blob + text[end:]
        else:
            text = text.replace(prefix + "{", prefix + "{" + key + blob + ",", 1)
        after = _entry_bounds(text, neighbor)
        if after is None or text[after[0]:after[1]] != neighbor_blob:
            raise SystemExit(f"neighbor {neighbor} changed; not publishing")
        json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if text.count(f'"v":"{VERSION}"') != 1:
            raise SystemExit("dir8 version stamp collided")
        DIR8.write_text(text)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return neighbor


def update_queue(status):
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(QUEUE.read_text())
        found = False
        for item in data["items"]:
            if item.get("id") == KEY:
                item["status"] = status
                item.pop("claim", None)
                found = True
                break
        if not found:
            raise SystemExit("queue item 0419_floatzel missing")
        QUEUE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    front_path = NORMAL / "front.png"
    front = np.array(Image.open(front_path).convert("RGBA"))
    if front.shape[0] != 96 or front.shape[1] != 96:
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    palette = extract_palette(front)
    masks = segment(front)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    print("parts", counts, "palette", len(palette), flush=True)
    rig = build_rig(front, masks)

    rest = synth(rig, "idle", 0, palette, flip=False)
    if not np.array_equal(rest, rig["placed"]):
        differ = np.any(rest != rig["placed"], axis=2)
        raise SystemExit(f"idle frame 0 does not match the still ({int(differ.sum())} pixels)")
    flipped = synth(rig, "idle", 0, palette, flip=True)
    if not np.array_equal(flipped, rig["placed"][:, ::-1]):
        raise SystemExit("flipped idle frame 0 is not the mirrored still")
    print("rest-pose pixels changed 0", flush=True)
    # Face x is the snout, left of the eyes. Tails stay on the right.
    op = rig["placed"][..., 3] >= 128
    ys, xs = np.nonzero(op)
    print(
        f"placed face_x {int(xs.min())} eye_side centroid_x {rig['centroid'][0]:.1f} "
        f"tail_hi_pivot {rig['tail_hi_pivot'][0]:.1f} tail_lo_pivot {rig['tail_lo_pivot'][0]:.1f}",
        flush=True,
    )

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros((*front.shape[:2], 4), np.uint8)
    part_cols = {
        "torso": (170, 170, 170),
        "tail_hi": (220, 40, 40),
        "tail_lo": (230, 150, 40),
        "jaw": (220, 40, 180),
        "hand_near": (40, 140, 255),
        "hand_far": (20, 70, 180),
        "foot_near": (40, 190, 70),
        "foot_far": (16, 110, 40),
    }
    for name, col in part_cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    Image.fromarray(np.repeat(np.repeat(part, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        want = [("idle", False, f) for f in (0, 4, 8, 12)]
        want += [("walk", False, f) for f in (0, 2, 4, 6, 8, 12)]
        want += [("attack", False, f) for f in (0, 2, 3, 5, 7, 9, 11, 15)]
        want += [("hurt", False, f) for f in (0, 2, 4, 8)]
        tiles = [synth(rig, act, f, palette, flip) for act, flip, f in want]
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            mask = tile[..., 3] >= 128
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W][mask] = tile[mask]
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
                if fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rig, act, f, palette, d in FLIP_DIRS), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(parents=True, exist_ok=True)
    preview.save(OUT_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    report = qa_report(sheets, master, palette, digest, counts)
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    ratios = report["attack_ratios"]
    print("SOUTH attack opaque ratios:", " ".join(f"{r:.4f}" for r in ratios), flush=True)
    print(
        f"SOUTH attack opaque min {report['attack_ratio_min']:.4f} max {report['attack_ratio_max']:.4f}",
        flush=True,
    )
    print(
        json.dumps(
            {k: report[k] for k in ("off_palette", "alphas", "edge_touching", "sprite_h", "spin_components")},
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"cell {CELL_W}x{CELL_H} fy {FOOT_Y} h {report['sprite_h']}", flush=True)

    meta = {
        "generator": "forge_floatzel_actions",
        "version": VERSION,
        "key": KEY,
        "dex": 419,
        "archetype": "biped_flotation_otter",
        "feet": 2,
        "flotation_tails": 2,
        "hands": 2,
        "facing": "left",
        "unflipped": ["N", "NW", "W", "SW"],
        "flipped": ["S", "SE", "E", "NE"],
        "cell_w": CELL_W,
        "cell_h": CELL_H,
        "foot_y": FOOT_Y,
        "sprite_h": report["sprite_h"],
        "frames": FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "palette_colors": int(len(palette)),
        "off_palette_colors": report["off_palette"],
        "attack_ratio_min": report["attack_ratio_min"],
        "attack_ratio_max": report["attack_ratio_max"],
        "source": "front.png 96x96",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = (
        report["off_palette"] != 0
        or set(report["alphas"]) != {0, 255}
        or report["edge_touching"]
        or report["attack_ratio_min"] < 0.90
        or report["attack_ratio_max"] > 1.08
    )
    # Spin must keep the body as one piece. Rings are not on these frames.
    for sizes in report["spin_components"]:
        if not sizes or sizes[0] < int(0.9 * counts["torso"]):
            failed = True
        if len(sizes) > 1 and sizes[1] > 24:
            failed = True
    if failed:
        print("QA failed", flush=True)
        if args.publish:
            update_queue("needs_fix")
            raise SystemExit("refusing to publish a failed sheet")
        return

    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
        print("arena sheet bytes", (SYNC_OUT / "sheet.png").stat().st_size, flush=True)
    if args.publish:
        neighbor = publish_dir8(report["sprite_h"])
        update_queue("done")
        print("published", VERSION, "neighbor", neighbor, flush=True)


if __name__ == "__main__":
    main()
