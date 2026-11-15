#!/usr/bin/env python3
"""Gliscor (0472) normal 8-direction action sheet from this form's own front.png.

Measured on normal/front.png (96x96, alpha already 0 or 255, 14 colours,
one 4-connected piece). bbox x12-84 y20-76 (73x57).

Body plan (from the pixels, nothing invented):
  - Head x34-62 y24-43, two yellow eyes (x38-40 / x47-50, y35-38).
    The mouth is already open: white fangs and a pink/red inside at
    x40-51 y40-42. The lower lip row y42 is the jaw that drops on the roar.
  - Two large pincers: left claw x12-33 y20-41, right claw x62-84 y22-46,
    each on a short forearm with a red wrist joint.
  - Two grey bat wings (colours (65,65,82) and (106,106,123)):
    left x28-47 y36-60, right x48-84 y39-64.
  - A segmented tail curling to the bottom left with the stinger tip
    x12-28 y59-70. Feet are tiny and hang; Gliscor glides, it does not walk.
  Archetype: avian hover glider with pincers (no planted feet).

The face looks to screen-left (eyes and mouth sit left of the head centre,
tail to the left), so N NW W SW keep the still and S SE E NE are mirrored.

  idle   : hover bob, wing flutter, claws breathe, tail sways. Frame 0 = still.
  walk   : glide: big wing beats twice per loop, claws alternate, tail swings.
  attack : 0-4 roar (claws flare up/out, existing jaw drops, roar rings);
           5-6 hold that cock; 7-10 claws snap down into the X-scissor
           (same 14 deg / wing 3px as the impact). Body stays upright.
           11-12 slash hold + poison sparks; 13-15 recover to the still.
  hurt   : 12 frames, white flash, one-piece recoil <= 26 degrees, recover.

Palette-locked to front.png. Alpha 0 or 255. front/back png and gif are not written.
"""

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
SRC_DIR = ROOT / "monsterSprite/gen4/0472_gliscor/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0472_gliscor/normal/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

KEY = "0472_gliscor"
VERSION = "gliscor_v2"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"N", "NW", "W", "SW"}  # still faces screen-left
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
HOVER_GAP = 16
MAX_RECOIL = 26.0

INK = (16, 16, 16)
WHITE = (255, 255, 255)
PALE = (180, 164, 255)
LIGHT = (156, 139, 238)
MIDP = (115, 123, 197)
RED = (222, 57, 106)
PINK = (197, 82, 148)
MOUTH = (115, 8, 0)
YELLOW = (255, 222, 0)
WING_COLS = ((65, 65, 82), (106, 106, 123))
PARTS = ("wing_l", "wing_r", "tail", "core", "jaw", "claw_l", "claw_r")


# ---------------------------------------------------------------- helpers
def components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, bool)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        n = 0
        while q:
            cy, cx = q.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        sizes.append(n)
    sizes.sort(reverse=True)
    return sizes


def is_col(img, col):
    return (img[..., 3] == 255) & np.all(img[..., :3] == np.array(col, np.uint8), axis=-1)


def extract_palette(img):
    return np.unique(img[img[..., 3] >= 128][:, :3], axis=0)


def snap(img, palette):
    out = img.copy()
    op = out[..., 3] >= 128
    if not np.any(op):
        return np.zeros_like(out)
    flat = out[..., :3].reshape(-1, 3).copy()
    sel = op.reshape(-1)
    uniq, inv = np.unique(flat[sel], axis=0, return_inverse=True)
    diff = uniq.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None, :, :]
    mapped = palette[np.einsum("ijk,ijk->ij", diff, diff).argmin(axis=1)]
    flat[sel] = mapped[inv.reshape(-1)]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(op, 255, 0).astype(np.uint8)
    out[~op] = 0
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
    """4x nearest in-plane rotation (inverse map). Multiples of 360 stay exact."""
    ang = angle_deg % 360.0
    if ang > 180.0:
        ang -= 360.0
    if abs(ang) < 0.45 or not np.any(layer[..., 3] >= 128):
        return layer.copy()
    s = 4
    up = np.repeat(np.repeat(layer, s, axis=0), s, axis=1)
    px, py = pivot[0] * s + 1.5, pivot[1] * s + 1.5
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rad = math.radians(ang)
    c, sn = math.cos(rad), math.sin(rad)
    rx, ry = xx - px, yy - py
    src_x = np.rint(px + c * rx + sn * ry).astype(np.int32)
    src_y = np.rint(py - sn * rx + c * ry).astype(np.int32)
    ok = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    outu = np.zeros_like(up)
    vy, vx = np.nonzero(ok)
    outu[vy, vx] = up[src_y[vy, vx], src_x[vy, vx]]
    return outu[s // 2::s, s // 2::s].copy()


def wing_shear(layer, shoulder_x, side, lift, span):
    """Flap: each column moves vertically by lift * (distance from shoulder / span).
    Inverse mapped per column, so the membrane never tears. side=-1 wing extends to -x."""
    if lift == 0:
        return layer.copy()
    out = np.zeros_like(layer)
    w = layer.shape[1]
    for x in range(w):
        d = (x - shoulder_x) * side
        if d <= 0:
            out[:, x] = layer[:, x]
            continue
        dy = int(round(lift * min(1.0, d / span)))
        if dy == 0:
            out[:, x] = layer[:, x]
        else:
            out[:, x] = shift(layer[:, x:x + 1], 0, dy)[:, 0]
    return out


def despeckle(img):
    out = img.copy()
    a = out[..., 3] >= 128
    pad = np.pad(a, 1, constant_values=False)
    n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


def fill_pinholes(img):
    out = img.copy()
    for _ in range(2):
        a = out[..., 3] >= 128
        pad = np.pad(a, 1, constant_values=False)
        n = pad[:-2, 1:-1].astype(np.int16) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        hole = (~a) & (n == 4)
        if not np.any(hole):
            break
        out[hole] = np.roll(out, 1, axis=0)[hole]
    return out


def paste(dst, layer):
    m = layer[..., 3] >= 128
    dst[m] = layer[m]


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 3 <= x < w - 3 and 3 <= y < h - 3 and out[y, x, 3] < 128:
        out[y, x, :3] = col
        out[y, x, 3] = 255


# ---------------------------------------------------------------- segmentation
def segment(img):
    op = img[..., 3] == 255
    yy, xx = np.mgrid[0:96, 0:96]
    ink = is_col(img, INK)
    wing = np.zeros_like(op)
    for c in WING_COLS:
        wing |= is_col(img, c)
    solid = op & ~ink
    lab = np.full((96, 96), -1, np.int16)
    idx = {n: i for i, n in enumerate(PARTS)}
    lab[solid] = idx["core"]
    lab[solid & wing & (xx <= 47) & (yy <= 57)] = idx["wing_l"]
    lab[solid & wing & (xx >= 48)] = idx["wing_r"]
    tail = solid & ~wing & (((yy >= 60) & (xx <= 50)) | ((yy >= 58) & (xx <= 30)))
    lab[tail] = idx["tail"]
    claw_l = solid & ~wing & (((xx <= 31) & (yy <= 41)) | ((xx <= 33) & (yy <= 37)))
    lab[claw_l] = idx["claw_l"]
    claw_r = solid & ~wing & (((xx >= 63) & (yy <= 40)) | ((xx >= 66) & (yy <= 46)))
    lab[claw_r] = idx["claw_r"]
    jaw = solid & (yy == 42) & (xx >= 41) & (xx <= 49)
    lab[jaw] = idx["jaw"]
    # Ink outline joins the part most of its solid 8-neighbours belong to.
    todo = ink & op
    for _ in range(6):
        ys, xs = np.nonzero(todo & (lab < 0))
        if len(ys) == 0:
            break
        new = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            votes = {}
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 96 and 0 <= nx < 96 and lab[ny, nx] >= 0:
                        v = int(lab[ny, nx])
                        votes[v] = votes.get(v, 0) + (2 if dy == 0 or dx == 0 else 1)
            if votes:
                best = max(votes.items(), key=lambda kv: (kv[1], kv[0] == idx["core"]))[0]
                new.append((y, x, best))
        for y, x, v in new:
            lab[y, x] = v
    lab[op & (lab < 0)] = idx["core"]
    masks = {n: lab == i for n, i in idx.items()}
    cover = np.zeros_like(op)
    for m in masks.values():
        cover |= m
    if int((cover != op).sum()):
        raise SystemExit("masks do not partition the still")
    if len(components(op)) != 1:
        raise SystemExit("still is not one piece")
    for n, lo in (("claw_l", 250), ("claw_r", 250), ("wing_l", 100), ("wing_r", 250), ("tail", 200), ("jaw", 6)):
        if int(masks[n].sum()) < lo:
            raise SystemExit(f"{n} mask collapsed: {int(masks[n].sum())}")
    return masks


def measure(img):
    op = img[..., 3] == 255
    ys, xs = np.nonzero(op)
    eye = is_col(img, YELLOW) | is_col(img, (172, 148, 0))
    ey, ex = np.nonzero(eye)
    gy, gx = np.mgrid[0:96, 0:96]
    fang = is_col(img, WHITE) & (gy >= 39) & (gy <= 42) & (gx >= 38) & (gx <= 52)
    fy, fx = np.nonzero(fang)
    head_cx = 48.5  # head x34-62 measured from the horns
    info = {
        "opaque": int(op.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "eye_x": round(float(ex.mean()), 2),
        "mouth_x": round(float(fx.mean()), 2),
        "mouth_y": round(float(fy.mean()), 2),
        "head_cx": head_cx,
    }
    info["facing"] = "left" if info["eye_x"] < head_cx and info["mouth_x"] < head_cx else "right"
    if info["facing"] != "left":
        raise SystemExit(f"expected a left-looking face, got {info}")
    return info


# ---------------------------------------------------------------- rig
PIV = {
    "claw_l": (32, 41), "claw_r": (64, 46), "tail": (47, 61),
    "wing_l_x": 41, "wing_r_x": 57, "mouth": (45, 41),
}


def placement(img):
    ys, xs = np.nonzero(img[..., 3] == 255)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    dx = (CELL_W - (x1 - x0)) // 2 - x0
    dy = FOOT_Y - HOVER_GAP - (y1 - 1)
    return dx, dy


def place(img, dx, dy):
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dy:dy + 96, dx:dx + 96] = img
    return canvas


def build_rig(front, back, masks, flip):
    dx, dy = placement(front)
    still = place(front, dx, dy)
    layers = {}
    for n, m in masks.items():
        lyr = np.zeros_like(front)
        lyr[m] = front[m]
        layers[n] = place(lyr, dx, dy)
    # Back view, centred on the same centroid.
    fys, fxs = np.nonzero(front[..., 3] == 255)
    bys, bxs = np.nonzero(back[..., 3] == 255)
    bdx = int(round(fxs.mean() - bxs.mean()))
    bdy = int(round(fys.mean() - bys.mean()))
    backc = place(shift(back, bdx, bdy), dx, dy)
    piv = {}
    for k, v in PIV.items():
        if isinstance(v, tuple):
            piv[k] = (v[0] + dx, v[1] + dy)
        else:
            piv[k] = v + dx
    oys, oxs = np.nonzero(still[..., 3] == 255)
    piv["center"] = (float(oxs.mean()), float(oys.mean()))
    facing = -1
    if flip:
        W = CELL_W - 1
        still = still[:, ::-1].copy()
        backc = backc[:, ::-1].copy()
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        for k, v in list(piv.items()):
            piv[k] = (W - v[0], v[1]) if isinstance(v, tuple) else W - v
        facing = 1
    return {"still": still, "back": backc, "layers": layers, "piv": piv, "facing": facing}


# ---------------------------------------------------------------- poses
def wave(amp, cycles, f, n, phase=0.0):
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


def pose(action, f):
    p = dict(dx=0, dy=0, wl=0, wr=0, cl=0.0, cr=0.0, cdy_l=0, cdy_r=0, tail=0.0, jaw=0,
             spin=0.0, view="front", lift=0, hurt=0.0, flash=False,
             rings=0, ring_age=0, trail=0, slash=0, sparks=0)
    n = FRAMES[action]
    if action == "idle":
        p["dy"] = int(round(wave(1.5, 1, f, n)))
        p["wl"] = p["wr"] = int(round(wave(-2.0, 2, f, n)))
        p["cdy_l"] = p["cdy_r"] = int(round(wave(-1.0, 1, f, n, 0.0)))
        p["tail"] = wave(4.0, 1, f, n, 0.6)
    elif action == "walk":
        p["dy"] = int(round(wave(2.0, 2, f, n)))
        p["wl"] = p["wr"] = int(round(wave(-6.0, 2, f, n)))
        p["cl"] = wave(7.0, 1, f, n)
        p["cr"] = wave(7.0, 1, f, n)
        p["cdy_l"] = int(round(wave(-1.4, 1, f, n)))
        p["cdy_r"] = int(round(wave(1.4, 1, f, n)))
        p["tail"] = wave(7.0, 1, f, n, 1.2)
        p["dx"] = int(round(wave(1.0, 1, f, n)))
    elif action == "attack":
        if f <= 4:
            k = (0.0, 0.6, 1.0, 1.0, 0.75)[f]
            p["cl"] = -24.0 * k  # claws flare up and outward
            p["cr"] = -24.0 * k
            p["cdy_l"] = p["cdy_r"] = -int(round(2 * k))
            p["wl"] = p["wr"] = -int(round(5 * k))
            p["jaw"] = (0, 1, 2, 2, 1)[f]
            p["dy"] = -int(round(2 * k))
            p["tail"] = 9.0 * k
            p["rings"] = (0, 1, 2, 3, 2)[f]
            p["ring_age"] = f
        elif f <= 10:
            # X-Scissor with the claws and wings. The body stays upright.
            # Angles are the roar cock (-24 / wing -5px) and the impact
            # snap (14 / wing 3px) already used on frames 0-4 and 11-12.
            if f <= 6:
                p["cl"] = p["cr"] = -24.0
                p["cdy_l"] = p["cdy_r"] = -2
                p["wl"] = p["wr"] = -5
                p["jaw"] = 2
                p["tail"] = 9.0
            else:
                p["cl"] = p["cr"] = 14.0
                p["wl"] = p["wr"] = 3
                p["jaw"] = 2
                p["tail"] = 9.0
                p["slash"] = 1
        elif f <= 12:
            p["dx"] = (8, 5)[f - 11]
            p["cl"] = p["cr"] = (14.0, 8.0)[f - 11]
            p["wl"] = p["wr"] = (3, 2)[f - 11]
            p["slash"] = f - 10
        else:
            p["dx"] = (3, 1, 0)[f - 13]
            p["dy"] = (-1, 0, 0)[f - 13]
            p["cl"] = p["cr"] = (4.0, 1.5, 0.0)[f - 13]
            p["wl"] = p["wr"] = (-1, 0, 0)[f - 13]
    else:
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = MAX_RECOIL * env
        p["dx"] = -int(round(6 * env))  # knocked backward
        p["dy"] = -int(round(2 * env))
        p["flash"] = f in (1, 2)
        p["sparks"] = 1 if 1 <= f <= 5 else 0
    if p["hurt"] > 38.0:
        raise SystemExit("recoil exceeds 38")
    return p


# ---------------------------------------------------------------- compose
def compose(rig, p):
    L, P, fc = rig["layers"], rig["piv"], rig["facing"]
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    # wings behind: left wing extends to screen-left in the unflipped still
    sl = -1 if fc < 0 else 1
    paste(img, wing_shear(L["wing_l"], P["wing_l_x"], sl, p["wl"], 18))
    paste(img, wing_shear(L["wing_r"], P["wing_r_x"], -sl, p["wr"], 26))
    paste(img, rotate_nearest(L["tail"], -fc * p["tail"], P["tail"]))
    paste(img, L["core"])
    if p["jaw"]:
        home = L["jaw"]
        drop = shift(home, 0, p["jaw"])
        hole = (home[..., 3] == 255) & (drop[..., 3] < 255)
        img[hole, :3] = MOUTH
        img[hole, 3] = 255
        # the inside of the open mouth between lip and chin
        for k in range(1, p["jaw"]):
            mid = shift(home, 0, k)
            m = mid[..., 3] == 255
            img[m, :3] = MOUTH
            img[m, 3] = 255
        paste(img, drop)
    else:
        paste(img, L["jaw"])
    # claws: the outer claw (screen-left in the still) is claw_l.
    # positive angle = tip swings down/inward, negative = flare up/out.
    paste(img, shift(rotate_nearest(L["claw_l"], fc * p["cl"], P["claw_l"]), 0, p["cdy_l"]))
    paste(img, shift(rotate_nearest(L["claw_r"], -fc * p["cr"], P["claw_r"]), 0, p["cdy_r"]))
    return img


def flash(img):
    out = img.copy()
    a = out[..., 3] == 255
    ink = np.all(out[..., :3] == np.array(INK, np.uint8), axis=-1)
    out[a & ~ink, :3] = WHITE
    return out


def roar_rings(img, origin, fc, age, strength):
    ox, oy = origin
    cols = (WHITE, PALE, RED)
    for ri in range(strength):
        rad = 13 + age * 3 + ri * 6
        for deg in range(-60, 61, 10):
            a = math.radians(deg)
            put_empty(img, ox + fc * rad * math.cos(a), oy + rad * math.sin(a) * 0.7, cols[ri % 3])


def spin_trail(img, center, rad, ang_deg, fc):
    cx, cy = center
    # two short speed arcs trailing the claws, brightest nearest the body
    for base in (0.0, 180.0):
        for k in range(22):
            a = math.radians(ang_deg + base + fc * (12 + k * 3))
            col = WHITE if k < 7 else (PALE if k < 15 else LIGHT)
            put_empty(img, cx + rad * math.cos(a), cy + rad * math.sin(a), col)
            if k < 12:
                put_empty(img, cx + (rad - 3) * math.cos(a), cy + (rad - 3) * math.sin(a), PALE)


def x_scissor(img, center, fc, stage):
    """Two crossing slash streaks just ahead of the claws (X-Scissor) + poison sparks."""
    cx, cy = center
    ox = cx + fc * (46 if stage == 1 else 50)
    oy = cy - 6
    half = 13 if stage == 1 else 16
    for t in range(-half, half + 1):
        w = 1 if abs(t) < half - 3 else 0
        for u in range(-w, w + 1):
            put_empty(img, ox + t + u, oy + t, WHITE if u == 0 else PALE)
            put_empty(img, ox + t + u, oy - t, WHITE if u == 0 else PALE)
    for dx, dy, col in ((-6, -15, RED), (8, -12, PINK), (12, 9, RED), (-9, 13, PINK), (16, 0, YELLOW),
                        (-15, 2, PINK), (3, 18, RED), (0, -19, PINK)):
        s = 1.0 if stage == 1 else 1.35
        put_empty(img, ox + fc * dx * s, oy + dy * s, col)


def hurt_sparks(img, fc, f):
    ys, xs = np.nonzero(img[..., 3] == 255)
    if len(xs) == 0:
        return
    top = int(ys.min())
    x = int(xs.max() - 10 if fc < 0 else xs.min() + 10)
    for dx, dy, col in ((0, -5, WHITE), (-4, -3, PALE), (4, -3, PALE), (-7, 1, WHITE), (7, 1, WHITE)):
        put_empty(img, x + dx * (1 + f * 0.2), top + 6 + dy * (1 + f * 0.2), col)


def synth(rig, action, f, palette, effects=True):
    p = pose(action, f)
    fc, P = rig["facing"], rig["piv"]
    if action == "attack" and p["spin"] > 0.5:
        view = rig["back"] if p["view"] == "back" else rig["still"]
        ang = -fc * p["spin"]
        img = rotate_nearest(view, ang, P["center"])
        img = shift(img, fc * p["dx"], -p["lift"])
        img = despeckle(fill_pinholes(img))
    elif action == "hurt" and p["hurt"] > 0.45:
        img = rotate_nearest(rig["still"], fc * p["hurt"], P["center"])
        img = shift(img, fc * p["dx"], p["dy"])
        img = despeckle(fill_pinholes(img))
    else:
        img = shift(compose(rig, p), fc * p["dx"], p["dy"])
        if action != "idle" or p["tail"]:
            img = despeckle(img)  # parts were rotated; drop lone pixels only
    img = snap(img, palette)
    if p["flash"]:
        img = flash(img)
    body = img.copy()
    if effects:
        mx, my = P["mouth"]
        if p["rings"]:
            roar_rings(img, (mx + fc * p["dx"] + fc * 2, my + p["dy"] + 2), fc, p["ring_age"], p["rings"])
        if p["trail"]:
            c = (P["center"][0] + fc * p["dx"], P["center"][1] - p["lift"])
            spin_trail(img, c, 46, -fc * p["spin"] + 90, fc)
        if p["slash"]:
            x_scissor(img, (P["center"][0] + fc * p["dx"], P["center"][1]), fc, p["slash"])
        if p["sparks"]:
            hurt_sparks(img, fc, f)
        img = snap(img, palette)
    return img, body


# ---------------------------------------------------------------- outputs
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


def preview_image(sheets):
    x0, x1, y0, y1 = 12, CELL_W - 12, 24, CELL_H - 4
    cw, ch = x1 - x0, y1 - y0
    rows = len(ACTIONS) * 8
    cols = max(FRAMES.values())
    bg = np.full((rows * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    r = 0
    for act in ACTIONS:
        sh = sheets[act]
        for di in range(8):
            for f in range(FRAMES[act]):
                tile = sh[di * CELL_H + y0:di * CELL_H + y1, f * CELL_W + x0:f * CELL_W + x1]
                m = tile[..., 3] == 255
                bg[r * ch:(r + 1) * ch, f * cw:(f + 1) * cw][m] = tile[m]
                bg[r * ch:(r + 1) * ch, f * cw] = (40, 48, 64, 255)
            r += 1
    return Image.fromarray(bg, "RGBA")


def motion_contact(sheets):
    """S and W rows of each action at 2x."""
    rows = []
    for act in ACTIONS:
        for di in (0, 6):
            band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            rows.append([band[:, f * CELL_W:(f + 1) * CELL_W] for f in range(FRAMES[act])])
    allm = np.zeros((CELL_H, CELL_W), bool)
    for row in rows:
        for t in row:
            allm |= t[..., 3] == 255
    ys, xs = np.nonzero(allm)
    pad = 4
    yy0, yy1 = max(0, ys.min() - pad), min(CELL_H, ys.max() + 1 + pad)
    xx0, xx1 = max(0, xs.min() - pad), min(CELL_W, xs.max() + 1 + pad)
    ch, cw = (yy1 - yy0) * 2, (xx1 - xx0) * 2
    cols = max(FRAMES.values())
    bg = np.full((len(rows) * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(rows):
        for ci, t in enumerate(row):
            c = np.repeat(np.repeat(t[yy0:yy1, xx0:xx1], 2, 0), 2, 1)
            m = c[..., 3] == 255
            bg[ri * ch:(ri + 1) * ch, ci * cw:(ci + 1) * cw][m] = c[m]
            bg[ri * ch:(ri + 1) * ch, ci * cw] = (40, 48, 64, 255)
    return Image.fromarray(bg, "RGBA")


def sync_outputs():
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in OUT_DIR.rglob("*"):
        if not fp.is_file():
            continue
        rel = fp.relative_to(OUT_DIR)
        if rel.parts[0] == "_work":
            continue
        if rel.name.startswith(("front", "back")):
            raise SystemExit(f"refusing to copy {rel}")
        dest = SYNC_OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, dest)
        n += 1
    return n


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _object_span(text, key):
    token = f'"{key}":'
    start = text.find(token)
    if start < 0 or text.find(token, start + 1) >= 0:
        raise SystemExit(f"{key} not found exactly once")
    val = start + len(token)
    if text[val] != "{":
        raise SystemExit(f"{key} value is not an object")
    depth = 0
    for j in range(val, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return start, val, j + 1
    raise SystemExit(f"{key} object did not close")


def publish_dir8(sprite_h):
    entry = {
        "dex": 472, "form": "normal", "variant": "gliscor", "gen": "gen4",
        "dir": "monsterSprite/gen4/0472_gliscor/normal", "sheetDir": "agent-sprite-forge",
        "layout": "actions_8dir", "nf": 16, "cw": CELL_W, "ch": CELL_H, "w": CELL_W,
        "h": int(sprite_h), "fy": FOOT_Y, "an": [16, 16, 16, 12],
        "ad": [0.08, 0.062, 0.05, 0.055], "raw": 1, "src": "agent-sprite-forge", "v": VERSION,
    }
    blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def edit():
        text = DIR8.read_text()
        prefix = "window.ARENA_DIR8 = "
        if not text.startswith(prefix):
            raise SystemExit("dir8.js header changed")
        before = json.loads(text[len(prefix):].rstrip().rstrip(";"))
        if KEY in before:
            _s, val, end = _object_span(text, KEY)
            new_text = text[:val] + blob + text[end:]
        else:
            body_end = text.rstrip().rstrip(";").rstrip()
            if not body_end.endswith("}"):
                raise SystemExit("dir8 body does not end with }")
            cut = len(body_end) - 1
            new_text = text[:cut] + f',"{KEY}":' + blob + text[cut:]
        parsed = json.loads(new_text[len(prefix):].rstrip().rstrip(";"))
        if parsed[KEY] != entry:
            raise SystemExit("dir8 entry did not parse back")
        for k, v in before.items():
            if k != KEY and parsed.get(k) != v:
                raise SystemExit(f"dir8 neighbour {k} changed")
        tmp = DIR8.with_suffix(".js.gliscor.tmp")
        tmp.write_text(new_text)
        tmp.replace(DIR8)

    with_flock(DIR8_LOCK, edit)


def publish_bounds(bounds):
    text = BOUNDS.read_text()
    token = f'"{KEY}":['
    i = text.find('"s":{')
    if i < 0:
        raise SystemExit("sprite_bounds has no s map")
    j = text.find(token, i)
    if j < 0:
        raise SystemExit("sprite_bounds has no s entry for gliscor")
    k = text.index("]", j)
    new = text[:j] + token + ",".join(str(int(v)) for v in bounds) + "]" + text[k + 1:]
    tmp = BOUNDS.with_suffix(".js.gliscor.tmp")
    tmp.write_text(new)
    tmp.replace(BOUNDS)


def update_queue(status, note=None):
    def edit():
        raw = QUEUE.read_text()
        marker = f'"id": "{KEY}"'
        if raw.count(marker) != 1:
            raise SystemExit("queue id count")
        idx = raw.index(marker)
        start = raw.rfind("{", 0, idx)
        depth, end = 0, None
        for j in range(start, len(raw)):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        block = raw[start:end]
        block2, n = re.subn(r'"status": "[^"]*"', f'"status": "{status}"', block, count=1)
        if n != 1:
            raise SystemExit("queue status not found")
        if note:
            block2 = block2.replace('"notes": "', '"notes": "' + note.replace('"', "'") + " | ", 1)
        new = raw[:start] + block2 + raw[end:]
        old = json.loads(raw)
        data = json.loads(new)
        for a, b in zip(old["items"], data["items"]):
            if a.get("id") != KEY and a != b:
                raise SystemExit("queue neighbour changed")
        tmp = QUEUE.with_suffix(".json.gliscor.tmp")
        tmp.write_text(new)
        tmp.replace(QUEUE)

    with_flock(QUEUE_LOCK, edit)


def differ(a, b):
    return int((np.any(a != b, axis=2) & ((a[..., 3] == 255) | (b[..., 3] == 255))).sum())


def params_hash():
    return hashlib.sha1(Path(__file__).read_bytes() + (SRC_DIR / "front.png").read_bytes()
                        + (SRC_DIR / "back.png").read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="after a visually checked pass: dir8 key, sprite_bounds s[], queue done")
    args = ap.parse_args()

    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    back = np.array(Image.open(SRC_DIR / "back.png").convert("RGBA"))
    if front.shape[:2] != (96, 96) or back.shape[:2] != (96, 96):
        raise SystemExit("expected 96x96 front/back")
    for im in (front, back):
        if sorted(int(v) for v in np.unique(im[..., 3])) != [0, 255]:
            raise SystemExit("alpha is not binary")
    palette = extract_palette(front)
    masks = segment(front)
    info = measure(front)
    print("info", info, "palette", len(palette), flush=True)
    print("parts", {k: int(v.sum()) for k, v in masks.items()}, flush=True)

    rigs = {d: build_rig(front, back, masks, flip=d not in UNFLIPPED) for d in DIRS}
    for act in ACTIONS:
        for d in DIRS:
            d0 = differ(synth(rigs[d], act, 0, palette)[0], rigs[d]["still"])
            if d0:
                raise SystemExit(f"{act} {d} frame 0 differs from the still by {d0}")
    print("still diff 0 (all actions, all dirs)", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pcol = {"wing_l": (90, 90, 90), "wing_r": (160, 160, 160), "tail": (40, 160, 90),
            "core": (120, 110, 220), "jaw": (255, 60, 60), "claw_l": (255, 160, 40), "claw_r": (60, 200, 240)}
    pm = np.zeros((96, 96, 4), np.uint8)
    for n, c in pcol.items():
        pm[masks[n], :3] = c
        pm[masks[n], 3] = 255
    Image.fromarray(np.repeat(np.repeat(pm, 4, 0), 4, 1), "RGBA").save(work / "parts.png")

    body_report = {}
    for act in ACTIONS:
        for f in range(FRAMES[act]):
            _img, body = synth(rigs["W"], act, f, palette)
            sizes = components(body[..., 3] == 255)
            if len(sizes) > 1:
                body_report[f"{act}{f:02d}"] = sizes[:4]
    print("multi-component bodies (W)", body_report or "none", flush=True)

    if args.samples:
        want = [(a, d) for a in ACTIONS for d in ("W", "S")]
        sheet = np.full((len(want) * CELL_H, 16 * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for r, (act, d) in enumerate(want):
            for f in range(FRAMES[act]):
                tile = synth(rigs[d], act, f, palette)[0]
                m = tile[..., 3] == 255
                sheet[r * CELL_H:(r + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W][m] = tile[m]
                sheet[r * CELL_H:(r + 1) * CELL_H, f * CELL_W] = (40, 48, 64, 255)
        Image.fromarray(sheet, "RGBA").save(work / "samples.png")
        for act in ACTIONS:
            row = []
            for f in range(FRAMES[act]):
                t = synth(rigs["W"], act, f, palette)[0][20:CELL_H - 4, 8:CELL_W - 8]
                bgt = np.full(t.shape, (15, 20, 31, 255), np.uint8)
                m = t[..., 3] == 255
                bgt[m] = t[m]
                bgt[:, 0] = (40, 48, 64, 255)
                row.append(bgt)
            z = np.concatenate(row, axis=1)
            z = np.repeat(np.repeat(z, 2, 0), 2, 1)
            Image.fromarray(z, "RGBA").save(work / f"samples_W_{act}.png")
        print("samples", work, flush=True)
        return

    frames_dir = work / "frames"
    state_p = work / "state.json"
    if args.force and frames_dir.exists():
        shutil.rmtree(frames_dir)
    digest = params_hash()
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    if state.get("hash") != digest:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        state = {"hash": digest, "done": []}
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(DIRS) * len(ACTIONS)
    for act in ACTIONS:
        for d in DIRS:
            key = f"{act}/{d}"
            od = frames_dir / act / d
            od.mkdir(parents=True, exist_ok=True)
            for f in range(FRAMES[act]):
                fp = od / f"{f:02d}.png"
                if key in state["done"] and fp.exists() and fp.stat().st_size > 0:
                    continue
                tmp = od / f"{f:02d}.tmp.png"
                Image.fromarray(synth(rigs[d], act, f, palette)[0], "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
                state_p.write_text(json.dumps(state))
            print(f"[{state['done'].index(key) + 1}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    prev = preview_image(sheets)
    prev.save(OUT_DIR / "preview.png")
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    prev.save(OUT_DIR / "actions" / "preview.png")
    motion_contact(sheets).save(OUT_DIR / "preview_motion.png")

    bounds = []
    for di in range(8):
        t = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W]
        ys, xs = np.nonzero(t[..., 3] == 255)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    sprite_h = bounds[3]

    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] == 255][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    clip = []
    still_diff = 0
    for act in ACTIONS:
        sh = sheets[act]
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                a = sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] == 255
                if a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any():
                    clip.append(f"{act}:{d}:{f}")
            t0 = sh[di * CELL_H:(di + 1) * CELL_H, :CELL_W]
            still_diff += differ(t0, rigs[d]["still"])
    spin = []
    for d in DIRS:
        for f in range(5, 11):
            _img, body = synth(rigs[d], "attack", f, palette)
            sizes = components(body[..., 3] == 255)
            spin.append(round(sizes[0] / sum(sizes), 3))
    hurt_main = []
    for f in range(12):
        _img, body = synth(rigs["W"], "hurt", f, palette)
        sizes = components(body[..., 3] == 255)
        hurt_main.append(round(sizes[0] / sum(sizes), 3))
    report = {
        "version": VERSION, "hash": digest, "sprite_h": sprite_h, "bounds": bounds,
        "off_palette": len(off), "alphas": alphas, "edge_touching": len(clip), "clip": clip[:20],
        "still_diff": still_diff, "spin_main_min": min(spin), "hurt_main_min": min(hurt_main),
        "cell": [CELL_W, CELL_H, FOOT_Y],
    }
    if not args.skip_qa and GAIT_LOOP.exists():
        qa = work / "qa"
        qa.mkdir(parents=True, exist_ok=True)
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FORGE / ".venv/bin/python"), "-I", str(GAIT_LOOP), "select",
                                "--frames-dir", str(frames_dir / act / "S"), "--fps", "12",
                                "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                               capture_output=True, text=True)
            rep = od / "loop-report.json"
            report[f"gait_{act}"] = (json.loads(rep.read_text()).get("recommendation")
                                     if rep.exists() else (r.stderr or r.stdout)[-300:])
    ok = (not off and alphas in ([0, 255], [0]) and not clip and still_diff == 0
          and min(spin) >= 0.99 and min(hurt_main) >= 0.99)
    report["ok"] = ok
    (work / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    meta = {
        "generator": "forge_gliscor_actions", "version": VERSION, "key": KEY, "dex": 472,
        "variant": "gliscor", "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y,
        "sprite_h": sprite_h, "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "facing": info["facing"],
        "parts": {k: int(v.sum()) for k, v in masks.items()},
        "archetype": "avian_hover_pincer", "feet": "hanging (glider)",
        "source": "front.png 96x96 (+ back.png kept for the rig, unused on attack)",
        "gifs_untouched": True,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print("QA", json.dumps(report, ensure_ascii=False), flush=True)
    if not ok:
        raise SystemExit("qa failed; not publishing")
    if not args.no_sync:
        print("synced files", sync_outputs(), flush=True)
    if args.publish:
        publish_dir8(sprite_h)
        publish_bounds(bounds)
        update_queue("done")
        print("published", VERSION, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
