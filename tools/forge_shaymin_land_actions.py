#!/usr/bin/env python3
"""Shaymin Land Forme (0492) normal 8-direction action sheet from this form's
own front.png / back.png only (no pokeAnim, no image generation).

Measured on normal/front.png: 96x96, opaque bbox x27-59 y33-63 (33x31),
15 colours, alpha 0/255, one piece.

Body plan from the pixels: a small low quadruped hedgehog.
  face     white/grey head at the far left x27-36 y45-55, ONE green eye at
           (34,48). No mouth is drawn, so nothing opens. Faces screen-left.
  grass    the light-green fur cap x28-59 y33-48 (yellow flecks). Its top
           (y<=43) is the `crest` that flares for the roar and whips forward.
  flower   the pink five-petal flower on the cheek x35-49 y43-56 with its
           ink seed dots. It spins in place during the attack (the body
           never spins).
  leaves   dark-green leaves behind the flower x44-58 y47-60 (body).
  paws     three short grey paws on y56-63:
             fore_far  dark paw x35-40,
             fore_near light paw x40-45 (sole y63),
             hind      x46-56 (the two hind toes read as one paw block).
back.png (same palette): grass back, flower and face at the right x52-64,
faces up-right. Paws: paw_l x36-49 y59-61, paw_r x53-58 y56-61.

Directions: front still faces left -> W SW unflipped, S SE E mirrored
(face screen-right). back still faces up-right -> NE N unflipped, NW mirrored.

  idle   : feet planted, 1px breath of the upper body, grass sway, flower nod.
           Frame 0 and loop end = still.
  walk   : two steps, fore_near+hind against fore_far, lift 2px and step,
           body bob, grass sway.
  attack : 0-4 roar: the grass crest bristles up, the flower starts to spin,
           green/yellow roar rings from the face, seed sparks gather.
           5-10 Seed Flare: a small forward pounce (integer px), the near
           forepaw stomps, the crest whips forward, the flower spins fast
           and fires a seed-flare beam. 11-12 burst. 13-15 recover.
           No in-plane whole-body rotation.
  hurt   : 12 frames, white flash, one-piece recoil about the sole
           (<= 30 deg), knock back, petals fly, recover.
The face/eye only moves by integer pixels.

Palette-locked to this still. front/back png and gif are not written.
Resumable via agent-sprite-forge/_work. --force clears only this form's frames.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_shaymin_land_actions.py [--samples] [--publish]
"""

import argparse
import fcntl
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
SRC_DIR = ROOT / "monsterSprite/gen4/0492_shaymin-land/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0492_shaymin-land/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
VIEW = {
    "S": ("front", True), "SE": ("front", True), "E": ("front", True),
    "NE": ("back", False), "N": ("back", False), "NW": ("back", True),
    "W": ("front", False), "SW": ("front", False),
}
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "shaymin-land_v1"
KEY = "0492_shaymin-land"
DEX = 492

C_G0 = (172, 222, 98)    # light grass
C_K = (16, 16, 16)
C_P = (238, 139, 164)    # pink petal
C_L3 = (57, 148, 65)     # leaf green (also the eye)
C_GR = (189, 197, 213)   # light grey
C_G5 = (90, 115, 65)     # grass outline
C_W = (255, 255, 255)
C_G7 = (139, 172, 98)    # grass mid
C_PD = (131, 74, 106)    # dark pink
C_Y = (255, 238, 123)    # yellow fleck
C_DG = (98, 90, 106)     # dark grey
C_MG = (139, 139, 172)   # mid grey
C_LC = (49, 98, 57)
C_LD = (41, 65, 41)
C_PM = (205, 106, 139)

GRASS = (C_G0, C_G5, C_G7, C_Y)
PINK = (C_P, C_PD, C_PM)
GREY = (C_GR, C_DG, C_MG, C_K)

FRONT_BBOX = (27, 59, 33, 63)
BACK_BBOX = (30, 64, 33, 61)
FRONT_EYE = (34, 48)


# ---------------------------------------------------------------- helpers
def dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def labels(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        comp = np.zeros_like(mask)
        q = deque([(y, x)])
        seen[y, x] = True
        while q:
            cy, cx = q.popleft()
            comp[cy, cx] = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(comp)
    out.sort(key=lambda m: -int(m.sum()))
    return out


def component_sizes(mask):
    return [int(m.sum()) for m in labels(mask)]


def flood(seeds, allow):
    h, w = allow.shape
    seen = np.zeros_like(allow)
    q = deque()
    for y, x in seeds:
        if allow[y, x] and not seen[y, x]:
            seen[y, x] = True
            q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allow[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def is_col(img, cols):
    m = np.zeros(img.shape[:2], bool)
    for c in cols:
        m |= np.all(img[..., :3] == np.array(c, np.uint8), axis=-1)
    return m & (img[..., 3] >= 128)


def extract_palette(img):
    a = img[..., 3] >= 128
    return np.unique(img[a][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    a = out[..., 3] >= 128
    if not np.any(a):
        out[..., 3] = 0
        return out
    flat = out[..., :3].reshape(-1, 3).copy()
    sel = a.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    flat[sel] = palette[dist.argmin(axis=1)][inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
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
    """4x nearest in-plane rotation (positive = clockwise on screen)."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.05 or not np.any(layer[..., 3] >= 128):
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


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def nearest_fill(img, hole, source):
    """Give each hole pixel the colour of the nearest source pixel (BFS)."""
    out = img.copy()
    h, w = hole.shape
    filled = source.copy()
    q = deque((int(y), int(x)) for y, x in zip(*np.nonzero(source & dilate(hole, 1))))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and hole[ny, nx] and not filled[ny, nx]:
                filled[ny, nx] = True
                out[ny, nx] = out[y, x]
                q.append((ny, nx))
    return out


def body_box(img):
    ys, xs = np.nonzero(img[..., 3] >= 128)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def touches_edge(tile):
    c = tile[..., 3] >= 128
    if not np.any(c):
        return False
    return bool(c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any())


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 3 <= x < w - 3 and 3 <= y < h - 3 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def put_over(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 3 <= x < w - 3 and 3 <= y < h - 3:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    ph = 2 * math.pi * cycles * f / n
    return amp * (math.sin(ph + phase) - math.sin(phase))


# ---------------------------------------------------------------- segmentation
def check_still(img, bbox, name):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    got = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    if got != bbox:
        raise SystemExit(f"{name} bbox {got} != measured {bbox}")
    if len(component_sizes(a)) != 1:
        raise SystemExit(f"{name} still is not one piece")


def segment(img, kind):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    opaque = img[..., 3] >= 128
    grass = is_col(img, GRASS)
    pink = is_col(img, PINK)
    grey = is_col(img, GREY)
    ink = is_col(img, (C_K,))

    # flower: pink plus the ink seed dots / outline that touch pink on 2+ sides
    p = np.pad(pink, 1)
    npink = p[:-2, 1:-1].astype(np.int8) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    if kind == "front":
        fbox = (xx >= 35) & (xx <= 50) & (yy >= 42) & (yy <= 56)
        crest = grass & (yy <= 43)
        fore_far = flood([(60, 37)], grey & (yy >= 57) & (xx <= 40))
        fore_near = flood([(62, 42)], grey & (yy >= 56) & (xx >= 40) & (xx <= 45) & ~fore_far)
        hind = flood([(61, 52), (62, 49)], grey & (yy >= 56) & (xx >= 46) & (xx <= 56))
        paws = {"fore_far": fore_far, "fore_near": fore_near, "hind": hind}
    else:
        fbox = (xx >= 51) & (xx <= 64) & (yy >= 41) & (yy <= 58)
        crest = grass & (yy <= 40)
        paw_l = flood([(60, 43)], grey & (yy >= 59) & (xx >= 40) & (xx <= 48))
        paw_r = flood([(59, 55), (57, 58)], grey & (yy >= 56) & (xx >= 53))
        paws = {"paw_l": paw_l, "paw_r": paw_r}
    flower = fbox & (pink | (ink & (npink >= 2)))
    parts = {"crest": crest, "flower": flower}
    parts.update(paws)
    used = np.zeros_like(opaque)
    for k, m in parts.items():
        if int((m & used).sum()):
            raise SystemExit(f"{kind}: part overlap at {k}")
        used |= m
    parts["body"] = opaque & ~used

    # pivots in still coordinates
    fy, fx = np.nonzero(flower)
    cy, cx = np.nonzero(crest)
    low = cy >= cy.max() - 1
    piv = {
        "flower": (float(fx.mean()), float(fy.mean())),
        # crest hinge at the rear base of the cap (front faces left, back right)
        "crest": (float(cx[low].max()) - 3.0, float(cy.max())) if kind == "front"
        else (float(cx[low].min()) + 3.0, float(cy.max())),
        "crest_mid": (float(cx[low].mean()), float(cy.max())),
    }
    info = {k: int(v.sum()) for k, v in parts.items()}
    for k in paws:
        if info[k] < 6:
            raise SystemExit(f"{kind}: paw {k} collapsed ({info[k]})")
        if len(component_sizes(paws[k])) != 1:
            raise SystemExit(f"{kind}: paw {k} not one piece")
    if info["flower"] < 60 or info["crest"] < 120:
        raise SystemExit(f"{kind}: flower/crest collapsed {info}")
    if kind == "front" and not parts["body"][FRONT_EYE[1], FRONT_EYE[0]]:
        raise SystemExit("eye left the body")
    return {"parts": parts, "pivots": piv, "paws": tuple(paws)}, info


def build_rig(img, seg, flip, kind):
    a = img[..., 3] >= 128
    ys, xs = np.nonzero(a)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    sw, sh = x1 - x0, y1 - y0
    dx, dy = (CELL_W - sw) // 2, FOOT_Y - sh
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dy:dy + sh, dx:dx + sw] = img[y0:y1, x0:x1]

    def place(m):
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dy:dy + sh, dx:dx + sw] = m[y0:y1, x0:x1]
        return pm

    def pt(p):
        return (p[0] - x0 + dx, p[1] - y0 + dy)

    pm = {k: place(v) for k, v in seg["parts"].items()}
    layers = {}
    for k, m in pm.items():
        lyr = np.zeros_like(canvas)
        lyr[m] = canvas[m]
        layers[k] = lyr
    body = layers["body"]
    # paws: keep their top two rows on the body so a lifted paw leaves no hole
    yy = np.arange(CELL_H)[:, None]
    for k in seg["paws"]:
        m = pm[k]
        top = int(np.nonzero(m)[0].min())
        cush = m & (yy <= top + 1)
        body[cush] = canvas[cush]
    # crest underlay
    cm = pm["crest"]
    # the whole cap stays underneath: a bristling crest shows thicker fur, never a hole
    body[cm] = canvas[cm]
    # flower: the hole under the spinning flower takes the nearest leaf/fur colour
    body_src = body[..., 3] >= 128
    body = nearest_fill(body, pm["flower"], body_src & ~pm["flower"])
    body[pm["flower"], 3] = 255
    layers["body"] = body

    piv = {k: pt(v) for k, v in seg["pivots"].items()}
    bys, bxs = np.nonzero(a[y0:y1, x0:x1])
    sole_x = bxs[bys >= bys.max() - 1] + dx
    piv["sole"] = (float(np.median(sole_x)), float(FOOT_Y - 1))
    piv["center"] = (float(bxs.mean() + dx), float(bys.mean() + dy))
    if kind == "front":
        face = (float(dx - x0 + 26), float(dy - y0 + 50))  # just ahead of the face
        face_dir = -1.0
    else:
        face = (float(dx - x0 + 66), float(dy - y0 + 46))
        face_dir = 1.0
    piv["face"] = face
    piv["paw_tops"] = {k: int(np.nonzero(pm[k])[0].min()) for k in seg["paws"]}
    rest = canvas.copy()
    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        for k in ("flower", "crest", "crest_mid", "sole", "center", "face"):
            px, py = piv[k]
            piv[k] = (CELL_W - 1 - px, py)
        rest = rest[:, ::-1].copy()
        face_dir = -face_dir
    return {"layers": layers, "piv": piv, "facing": face_dir, "rest": rest,
            "kind": kind, "paws": seg["paws"]}


# ---------------------------------------------------------------- poses
def pose(action, f):
    n = FRAMES[action]
    p = dict(bdy=0, bdx=0, crest=0.0, flower=0.0, fdy=0, lift={}, step={},
             rings=0, ring_age=0, charge=0, beam=0, burst=0, dust=0,
             hurt=0.0, knock=0, flash=0, petals=0, spin=0.0)
    if action == "idle":
        br = wave(f, n, 1.0)
        p["bdy"] = -1 if br > 0.5 else 0
        p["crest"] = wave(f, n, 4.0, phase=0.4)
        p["flower"] = wave(f, n, 6.0, phase=1.1)
    elif action == "walk":
        s = math.sin(2 * math.pi * 2 * f / n)
        a_l = int(round(2 * max(0.0, s)))
        b_l = int(round(2 * max(0.0, -s)))
        p["lift"] = {"A": a_l, "B": b_l}
        p["step"] = {"A": 1 if a_l >= 1 else 0, "B": 1 if b_l >= 1 else 0}
        p["bdy"] = -1 if abs(s) > 0.7 else 0
        p["crest"] = wave(f, n, 5.0, cycles=2.0)
        p["flower"] = wave(f, n, 8.0, phase=0.0, cycles=2.0)
    elif action == "attack":
        # 0 still | 1-4 roar | 5-10 seed flare pounce | 11-12 burst | 13-15 recover
        p["crest"] = [0, -6, -11, -13, -10, -2, 4, 7, 7, 6, 4, 3, 2, 1, 0, 0][f]
        spin = [0, 20, 50, 90, 140, 200, 270, 345, 420, 495, 570, 630, 680, 705, 718, 720][f]
        p["flower"] = float(spin)
        p["spin"] = 0.0
        p["rings"] = [0, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ring_age"] = f
        p["charge"] = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["bdy"] = [0, 0, 1, 1, 1, -2, -3, -2, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["bdx"] = [0, 0, -1, -1, -1, 1, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0][f]
        lift = [0, 0, 0, 0, 0, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["lift"] = {"A": lift, "B": 0}
        p["step"] = {"A": 1 if 5 <= f <= 7 else 0, "B": 0}
        p["beam"] = f - 6 if 7 <= f <= 10 else 0
        p["burst"] = f - 10 if f in (11, 12) else 0
        p["dust"] = f - 7 if f in (8, 9, 10) else 0
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 30.0 * env if f >= 2 else 0.0
        p["knock"] = int(round(3 * env)) if f >= 1 else 0
        p["flash"] = 1 if f == 1 else 0
        p["petals"] = f if 2 <= f <= 7 else 0
        p["crest"] = 10.0 * env
        p["flower"] = 20.0 * env
    return p


def assemble_pose(rig, p, airborne):
    L, piv, fc = rig["layers"], rig["piv"], rig["facing"]
    bdx, bdy = p["bdx"] * fc, p["bdy"]
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    # paws: group A = near fore + hind (front) / right paw (back); B = far fore / left paw
    if rig["kind"] == "front":
        groups = {"fore_far": "B", "fore_near": "A", "hind": "A"}
        order = ("fore_far", None, "hind", "fore_near")
    else:
        groups = {"paw_l": "B", "paw_r": "A"}
        order = (None, "paw_l", "paw_r")
    paw_imgs = {}
    for k, g in groups.items():
        lift = p["lift"].get(g, 0)
        step = p["step"].get(g, 0)
        ddy = -lift + (bdy if airborne else 0)
        paw_imgs[k] = shift(L[k], bdx + fc * step, ddy)
    body = shift(L["body"], bdx, bdy)
    crest_ang = fc * p["crest"]
    crest = shift(rotate_nearest(L["crest"], crest_ang, piv["crest"]), bdx, bdy)
    flower = shift(rotate_nearest(L["flower"], -fc * p["flower"], piv["flower"]), bdx, bdy + p["fdy"])
    for k in order:
        if k is None:
            paste(img, body)
            paste(img, crest)
        else:
            paste(img, paw_imgs[k])
    paste(img, flower)
    return img


def white_flash(img):
    out = img.copy()
    a = out[..., 3] >= 128
    out[a, :3] = C_W
    # keep the outline readable
    ink = a & np.all(img[..., :3] == np.array(C_K, np.uint8), axis=-1)
    out[ink, :3] = C_K
    return out


# ---------------------------------------------------------------- effects
def roar_rings(out, rig, p):
    fx, fy = rig["piv"]["face"]
    fc = rig["facing"]
    fx += p["bdx"] * fc
    fy += p["bdy"]
    cols = (C_W, C_Y, C_G0)
    for i in range(p["rings"]):
        rad = 4 + 3 * p["ring_age"] + i * 5
        col = cols[i % 3]
        for deg in range(-60, 61, 12):
            ang = math.radians(deg)
            put_empty(out, fx + fc * rad * math.cos(ang), fy + rad * math.sin(ang) * 0.8, col)


def charge_sparks(out, rig, p):
    cx, cy = rig["piv"]["flower"]
    cx += p["bdx"] * rig["facing"]
    cy += p["bdy"]
    k = p["charge"]
    for i in range(6):
        ang = math.radians(i * 60 + k * 20)
        r = 18 - 4 * k
        col = (C_Y, C_W, C_G0)[i % 3]
        put_empty(out, cx + r * math.cos(ang), cy + r * math.sin(ang) * 0.8, col)
        put_empty(out, cx + (r + 2) * math.cos(ang), cy + (r + 2) * math.sin(ang) * 0.8, col)


def seed_beam(out, rig, p):
    cx, cy = rig["piv"]["flower"]
    fc = rig["facing"]
    cx += p["bdx"] * fc
    cy += p["bdy"]
    face_x = rig["piv"]["face"][0] + p["bdx"] * fc
    k = p["beam"]
    reach = 12 + 9 * k
    for d in range(3, reach):
        x = face_x + fc * d
        spread = 1 + d // 12
        put_empty(out, x, cy, C_W)
        if d % 2 == k % 2:
            put_empty(out, x, cy - 1, C_Y)
            put_empty(out, x, cy + 1, C_Y)
        h = (d * 7 + k * 13) % 11
        if h < 4:
            o = spread + 1 + (h % 2)
            put_empty(out, x, cy - o if h < 2 else cy + o, C_G0 if h % 2 else C_Y)
    # seeds riding the beam tip
    tip = face_x + fc * reach
    for j, (ox, oy) in enumerate(((0, 0), (-2, -2), (-2, 2), (-4, 0), (-5, -3), (-5, 3))):
        put_empty(out, tip + fc * ox, cy + oy, (C_W, C_Y, C_Y, C_G0, C_G0, C_G0)[j])


def seed_burst(out, rig, p):
    fc = rig["facing"]
    face_x = rig["piv"]["face"][0] + p["bdx"] * fc
    cy = rig["piv"]["flower"][1] + p["bdy"]
    bx = face_x + fc * 46
    r0 = 4 + 5 * p["burst"]
    for i in range(16):
        ang = math.radians(i * 22.5)
        for rr, col in ((r0, C_W), (r0 + 3, C_Y), (r0 + 6, C_G0)):
            put_empty(out, bx + rr * math.cos(ang), cy + rr * math.sin(ang) * 0.8, col)
    # five-petal flash in the middle
    for i in range(5):
        ang = math.radians(i * 72 - 90 + 30 * p["burst"])
        for rr in range(1, 4):
            put_empty(out, bx + rr * math.cos(ang), cy + rr * math.sin(ang), C_P)
    put_empty(out, bx, cy, C_W)


def stomp_dust(out, rig, p):
    fc = rig["facing"]
    sx = rig["piv"]["sole"][0] + p["bdx"] * fc
    y = FOOT_Y - 1
    k = p["dust"]
    for i in range(3):
        for side in (-1, 1):
            put_empty(out, sx + side * (10 + 3 * k + 2 * i), y - i - (k - 1), C_GR)


def petals(out, rig, p):
    fc = rig["facing"]
    cx, cy = rig["piv"]["flower"]
    k = p["petals"]
    for i in range(4):
        ang = math.radians(-120 + i * 40)
        r = 6 + 3 * k
        x = cx - fc * r * abs(math.cos(ang)) * 0.6 - fc * 2 * k
        y = cy - 10 + r * math.sin(ang) * 0.5 + k * k * 0.25
        put_empty(out, x, y, C_P)
        put_empty(out, x + 1, y, C_PM)
    # hit sparks at the face
    fx, fy = rig["piv"]["face"]
    if k <= 4:
        for i in range(5):
            ang = math.radians(i * 72 + 20 * k)
            put_empty(out, fx + fc * 2 + (3 + k) * math.cos(ang), fy - 4 + (3 + k) * math.sin(ang), C_W if i % 2 else C_Y)


def synth(rigs, d, action, f, palette, with_fx=True):
    rig = rigs[d]
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    fc = rig["facing"]
    airborne = action == "attack" and p["bdy"] < 0
    img = assemble_pose(rig, p, airborne)
    if action == "hurt" and (p["hurt"] > 0.45 or p["knock"]):
        if p["hurt"] > 0.45:
            img = rotate_nearest(img, -fc * p["hurt"], (rig["piv"]["sole"][0], rig["piv"]["sole"][1]))
            img = despeckle(img)
        img = shift(img, -fc * p["knock"], 0)
    if p["flash"]:
        img = white_flash(img)
    img = snap(img, palette)
    if not with_fx:
        return img
    if p["rings"]:
        roar_rings(img, rig, p)
    if p["charge"]:
        charge_sparks(img, rig, p)
    if p["beam"]:
        seed_beam(img, rig, p)
    if p["burst"]:
        seed_burst(img, rig, p)
    if p["dust"]:
        stomp_dust(img, rig, p)
    if p["petals"]:
        petals(img, rig, p)
    return snap(img, palette)


# ---------------------------------------------------------------- outputs
def params_hash():
    src = (Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes()
           + (SRC_DIR / "front.png").read_bytes() + (SRC_DIR / "back.png").read_bytes())
    return hashlib.sha1(src).hexdigest()[:12]


def assemble(frames_dir):
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


CROP = (CELL_W // 2 - 64, FOOT_Y - 64, CELL_W // 2 + 64, FOOT_Y + 8)


def preview_image(sheets):
    picks = [("idle", 0), ("walk", 4), ("walk", 12), ("attack", 3), ("attack", 6),
             ("attack", 9), ("attack", 11), ("hurt", 1), ("hurt", 5)]
    cw, ch = CROP[2] - CROP[0], CROP[3] - CROP[1]
    sc = 2
    bg = np.full((8 * ch * sc, len(picks) * cw * sc, 4), (15, 20, 31, 255), np.uint8)
    for di in range(8):
        for pi, (act, f) in enumerate(picks):
            tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            t = tile[CROP[1]:CROP[3], CROP[0]:CROP[2]]
            t = np.repeat(np.repeat(t, sc, 0), sc, 1)
            m = t[..., 3] >= 128
            bg[di * ch * sc:(di + 1) * ch * sc, pi * cw * sc:(pi + 1) * cw * sc][m] = t[m]
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets, row="E", scale=2):
    ri = DIRS.index(row)
    cw, ch = (CROP[2] - CROP[0]) * scale, (CROP[3] - CROP[1]) * scale
    cols = 8
    rows = sum(math.ceil(FRAMES[a] / cols) for a in ACTIONS)
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for act in ACTIONS:
        for f in range(FRAMES[act]):
            rr, cc = divmod(f, cols)
            tile = sheets[act][ri * CELL_H:(ri + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
            t = tile[CROP[1]:CROP[3], CROP[0]:CROP[2]]
            t = np.repeat(np.repeat(t, scale, 0), scale, 1)
            m = t[..., 3] >= 128
            bg[(r + rr) * ch:(r + rr + 1) * ch, cc * cw:(cc + 1) * cw][m] = t[m]
        r += math.ceil(FRAMES[act] / cols)
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.iterdir():
        if fp.is_file():
            if fp.name.startswith(("front", "back")):
                raise SystemExit(f"refusing to copy {fp.name}")
            shutil.copy2(fp, SYNC_OUT / fp.name)
            n += 1
    return n


def with_flock(lock_path, fn):
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def replace_key_value(text, key, new_val, start_at=0):
    token = '"' + key + '":'
    start = text.find(token, start_at)
    if start < 0 or text.find(token, start + len(token)) != -1:
        raise SystemExit(f"key {key} missing or duplicated")
    val_at = start + len(token)
    _old, end = json.JSONDecoder().raw_decode(text, val_at)
    return text[:val_at] + new_val + text[end:]


def publish_dir8(sprite_h):
    entry = {
        "dex": DEX, "form": "normal", "variant": "shaymin-land", "gen": "gen4",
        "dir": "monsterSprite/gen4/0492_shaymin-land/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    new_val = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js prefix changed")
        out = replace_key_value(text, KEY, new_val)
        before = json.loads(text[len(prefix):].strip().rstrip(";"))
        after = json.loads(out[len(prefix):].strip().rstrip(";"))
        if after[KEY] != entry or list(after) != list(before):
            raise SystemExit("dir8 edit failed")
        for k in before:
            if k != KEY and before[k] != after[k]:
                raise SystemExit(f"dir8 neighbour changed: {k}")
        tmp = DIR8.with_suffix(".js.tmp")
        tmp.write_text(out)
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)


def publish_bounds(sheets):
    vals = []
    for di in range(8):
        tile = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        x0, y0, x1, y1 = body_box(tile)
        vals += [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    new_val = json.dumps(vals, separators=(",", ":"))
    text = BOUNDS.read_text()
    s_at = text.find('"s":{')
    if s_at < 0:
        raise SystemExit("sprite_bounds has no s")
    out = replace_key_value(text, KEY, new_val, s_at)
    tmp = BOUNDS.with_suffix(".js.tmp")
    tmp.write_text(out)
    tmp.replace(BOUNDS)
    return vals


def update_queue(status):
    def edit():
        raw = QUEUE.read_text(encoding="utf-8")
        i = raw.find('"id": "' + KEY + '",')
        if i < 0:
            raise SystemExit("queue item missing")
        j = raw.find('"status": "', i)
        k = raw.find('"', j + len('"status": "'))
        out = raw[:j] + '"status": "' + status + raw[k:]
        json.loads(out)
        tmp = QUEUE.with_suffix(".json.tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


# ---------------------------------------------------------------- main
def load_rigs():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    for nm, im in (("front", front), ("back", back)):
        if im.shape[:2] != (96, 96):
            raise SystemExit(f"{nm} is not 96x96")
        if set(np.unique(im[..., 3]).tolist()) - {0, 255}:
            raise SystemExit(f"{nm} alpha is not binary")
    check_still(front, FRONT_BBOX, "front")
    check_still(back, BACK_BBOX, "back")
    palette = extract_palette(front)
    fp = {tuple(int(v) for v in c) for c in palette}
    bp = {tuple(int(v) for v in c) for c in extract_palette(back)}
    if not bp <= fp:
        raise SystemExit("back uses colours outside the front palette")
    fx = set(GRASS) | set(PINK) | set(GREY) | {C_W, C_L3, C_LC, C_LD}
    if not fx <= fp:
        raise SystemExit("fx colours are not in the front palette")
    fseg, finfo = segment(front, "front")
    bseg, binfo = segment(back, "back")
    rigs = {}
    for kind, img, seg in (("front", front, fseg), ("back", back, bseg)):
        for flip in (False, True):
            rigs[(kind, flip)] = build_rig(img, seg, flip, kind)
    for d in DIRS:
        rigs[d] = rigs[VIEW[d]]
    return front, back, palette, fseg, bseg, finfo, binfo, rigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="after visual approval: dir8 + sprite_bounds + queue done")
    args = ap.parse_args()

    front, back, palette, fseg, bseg, finfo, binfo, rigs = load_rigs()
    print("front parts", finfo, "back parts", binfo, "palette", len(palette), flush=True)

    for d in DIRS:
        for act in ACTIONS:
            r0 = synth(rigs, d, act, 0, palette, with_fx=False)
            if np.any(r0 != rigs[d]["rest"]):
                diff = int(np.any(r0 != rigs[d]["rest"], axis=2).sum())
                raise SystemExit(f"{act} frame 0 differs from the still ({d}): {diff}px")
        for act in ("idle", "walk"):
            rn = synth(rigs, d, act, FRAMES[act], palette, with_fx=False)
            if np.any(rn != rigs[d]["rest"]):
                raise SystemExit(f"{act} loop end is not the still ({d})")
    if not (rigs["E"]["facing"] > 0 and rigs["W"]["facing"] < 0
            and rigs["NE"]["facing"] > 0 and rigs["NW"]["facing"] < 0
            and rigs["E"]["piv"]["face"][0] > CELL_W / 2 and rigs["W"]["piv"]["face"][0] < CELL_W / 2):
        raise SystemExit("E must face screen-right and W screen-left")
    print("still diff 0 for all dirs; E faces right, W faces left", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pcols = {"body": (60, 120, 220), "crest": (250, 220, 60), "flower": (250, 90, 200),
             "fore_far": (200, 40, 40), "fore_near": (40, 220, 220), "hind": (60, 220, 60),
             "paw_l": (200, 40, 40), "paw_r": (60, 220, 60)}
    for nm, seg in (("front", fseg), ("back", bseg)):
        part = np.zeros((96, 96, 4), np.uint8)
        for name, msk in seg["parts"].items():
            part[msk, :3] = pcols[name]
            part[msk, 3] = 255
        for name, (px, py) in seg["pivots"].items():
            part[int(round(py)), int(round(px))] = (255, 0, 0, 255)
        Image.fromarray(np.repeat(np.repeat(part, 6, 0), 6, 1), "RGBA").save(work / f"parts_{nm}.png")

    if args.samples:
        want = [("idle", "E", f) for f in (0, 4, 8, 12)]
        want += [("walk", "E", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        want += [("attack", "E", f) for f in range(16)]
        want += [("attack", "N", f) for f in range(0, 16, 2)]
        want += [("hurt", "E", f) for f in range(0, 12, 1)][:8]
        want += [("idle", d, 0) for d in DIRS]
        want += [("walk", "N", f) for f in (0, 2, 4, 6, 8, 10, 12, 14)]
        tiles = []
        for act, d, f in want:
            bare = synth(rigs, d, act, f, palette, with_fx=False)
            tile = synth(rigs, d, act, f, palette)
            sizes = component_sizes(bare[..., 3] >= 128)
            print(f"sample {act} {d} {f:02d} comps={sizes[:4]} edge={touches_edge(tile)}", flush=True)
            tiles.append(tile[CROP[1]:CROP[3], CROP[0]:CROP[2]])
        cols = 8
        th, tw = tiles[0].shape[:2]
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * th, cols * tw, 4), (15, 20, 31, 255), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            m = t[..., 3] >= 128
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw][m] = t[m]
        sheet = np.repeat(np.repeat(sheet, 3, 0), 3, 1)
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
                Image.fromarray(synth(rigs, d, act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            state["done"].append(key)
            state_p.write_text(json.dumps(state))
            print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    preview = preview_image(sheets)
    preview.save(OUT_DIR / "preview.png")
    actions_dir = SRC_DIR / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    preview.save(actions_dir / "preview.png")
    motion_contact(sheets, "E").save(OUT_DIR / "preview_motion.png")
    motion_contact(sheets, "N").save(work / "review_N.png")
    motion_contact(sheets, "W").save(work / "review_W.png")

    idle0 = sheets["idle"][:CELL_H, :CELL_W]
    bb = body_box(idle0)
    sprite_h = bb[3] - bb[1] + 1
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    pal = {tuple(int(v) for v in c) for c in palette}
    opaque = master[..., 3] >= 128
    used = {tuple(int(v) for v in c) for c in np.unique(master[opaque][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = sorted(int(v) for v in np.unique(master[..., 3]).tolist())
    edge = []
    split = []
    still_diff = 0
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                if touches_edge(tile):
                    edge.append(f"{act}:{d}:{f}")
                bare = synth(rigs, d, act, f, palette, with_fx=False)
                sizes = component_sizes(bare[..., 3] >= 128)
                if len([s for s in sizes if s >= 12]) != 1 or sizes[0] < 0.95 * sum(sizes):
                    split.append({"a": act, "d": d, "f": f, "sizes": sizes[:5]})
            t0 = synth(rigs, d, act, 0, palette, with_fx=False)
            still_diff += int(np.any(t0 != rigs[d]["rest"], axis=2).sum())
            if act != "attack":
                t0 = sheets[act][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
                still_diff += int(np.any(t0 != rigs[d]["rest"], axis=2).sum())
    report = {
        "version": VERSION, "hash": digest, "front_parts": finfo, "back_parts": binfo,
        "sprite_h": int(sprite_h), "off_palette": len(off), "alphas": alphas,
        "edge_touching": len(edge), "edge_list": edge[:20], "still_diff": still_diff,
        "split": split[:20], "split_n": len(split), "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_E"
            if od.exists():
                shutil.rmtree(od)
            subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "E"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True,
            )
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
            else:
                report[f"gait_{act}"] = "no report"
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False)[:2000], flush=True)
    meta = {
        "generator": "forge_shaymin_land_actions", "version": VERSION, "key": KEY, "dex": DEX,
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(sprite_h),
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "views": {d: {"still": VIEW[d][0], "mirrored": VIEW[d][1]} for d in DIRS},
        "still_faces": "front.png faces screen-left (face x27-36); back.png faces up-right",
        "palette_colors": int(len(palette)), "off_palette_colors": len(off),
        "source": "front.png + back.png 96x96", "gifs_untouched": True,
        "archetype": "quadruped_beast (small hedgehog, 3 drawn paws)",
        "roar": "no mouth: grass crest bristles, flower starts spinning, rings from the face",
        "attack": "Seed Flare: integer pounce + forepaw stomp + crest whip + spinning flower beam, burst (no whole-body spin)",
        "layout": "actions_8dir", "sheetDir": "agent-sprite-forge", "raw": 1,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = (report["off_palette"] or report["edge_touching"] or report["still_diff"]
              or report["split_n"] or alphas != [0, 255])
    if failed:
        print("QA FAILED", flush=True)
        return
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        vals = publish_bounds(sheets)
        update_queue("done")
        print("published dir8", VERSION, "h", sprite_h, "bounds", vals[:4], flush=True)


if __name__ == "__main__":
    main()
