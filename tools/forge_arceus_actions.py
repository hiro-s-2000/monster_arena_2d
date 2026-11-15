#!/usr/bin/env python3
"""Arceus normal (0493) 8-direction action sheet from this form's front.png only.

Measured on normal/front.png: 96x96, 2326 opaque, bbox x14-81 y11-83 (68x73),
15 colours, alpha already 0 or 255. Faces screen-left: the snout with its gold
mark is x14-17 y20-25 and the green eye (red pupil x20-21 y24-25) is x20-25
y24-28. Unflipped rows are N NW W SW; S SE E NE are mirrored so E looks right.

Quadruped with the golden cross-wheel round its waist.
  head      x14-40 y11-31. Two ear spikes x20-24 y11-16 and x29-31 y13-16.
  chin      x17-25 y30-31, hangs over empty pixels (y32-33 free). This is the
            only jaw. Roar drops it 1-2 px and shows the dark mouth behind.
  crest     back-swept head plate x44-61 y17-26 (rooted on the head).
  near fore x20-33 y58-83, gold hoof tip x20-25 y72-83.
  far fore  right of the dark crease (x31 at y70 -> x27 at y77), x28-33
            y70-79, gold heel x29-31 y75-79. Mostly hidden behind the near fore.
  far hind  dark grey x48-59 y61-80, gold hoof x55-58 y73-80, behind the wheel.
  near hind x62-72 y60-83, gold hoof x65-71 y73-83.
  tail      white spike x63-81 y40-58.
  wheel     gold + green gems, x34-69 y30-75. Four gems at about (41,36),
            (61,38), (38,66), (58,67). Not moved, only lit with palette steps.
No limbs, mouth or wings are invented.

  idle   : feet planted. Breath lifts the trunk 1px, the crest and tail sway,
           the gems pulse. Frame 0 and the loop end equal the still.
  walk   : diagonal trot, two steps. Near fore + far hind, then far fore + near
           hind. The near legs swing about the shoulder / hip.
  attack : (2026-11-04 rule: no flat whole-body spin) 0 still, 1-4 roar: chin
           drops, roar rings, crest and tail flare, the wheel lights up.
           5-10 rear the near fore up (swing up to 30 deg) while judgement rays
           leave the four gems, 9-10 stamp down, 11-12 ground shockwave and
           light burst, 13-15 settle.
  hurt   : 12 frames. White flash, one-piece recoil about the rear hoof
           (34 deg max), sparks, recover.
Head moves by integer pixels only, so the face and eye stay intact.

  <agent-sprite-forge>/.venv/bin/python -I tools/forge_arceus_actions.py [--samples|--force|--no-sync|--skip-qa|--publish]
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
SRC_DIR = ROOT / "monsterSprite/gen4/0493_arceus/normal"
OUT_DIR = SRC_DIR / "agent-sprite-forge"
SYNC_OUT = Path(
    "/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen4/0493_arceus/normal/agent-sprite-forge"
)
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"
DIR8 = ROOT / "data/dir8.js"
DIR8_LOCK = ROOT / "data/dir8.lock"
BOUNDS = ROOT / "data/sprite_bounds.js"
QUEUE = ROOT / "monsterSprite/gen4/_forge_queue.json"
QUEUE_LOCK = ROOT / "monsterSprite/gen4/_forge_queue.lock"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
UNFLIPPED = {"N", "NW", "W", "SW"}  # still faces left
ACTIONS = ["idle", "walk", "attack", "hurt"]
FRAMES = {"idle": 16, "walk": 16, "attack": 16, "hurt": 12}
CELL_W, CELL_H, FOOT_Y = 176, 168, 148
VERSION = "arceus_v1"
KEY = "0493_arceus"

INK = (16, 16, 16)
WHITE = (230, 230, 238)
LIGHT = (180, 180, 189)
GREY = (131, 131, 139)
G_DARK = (131, 115, 0)
G_MID = (189, 164, 24)
G_HI = (222, 197, 57)
G_TOP = (255, 238, 90)
GEM_D = (0, 156, 0)
GEM_L = (0, 230, 0)
RED = (197, 0, 57)
GOLD = (G_DARK, G_MID, G_HI, G_TOP)
GEMS = (GEM_D, GEM_L)

PARTS = ("body", "fore_near", "fore_far", "hind_far", "hind_near", "tail", "crest", "jaw")
DRAW_ORDER = ("hind_far", "fore_far", "tail", "body", "crest", "jaw", "hind_near", "fore_near")


# ---------------------------------------------------------------- helpers
def colour_mask(img, colour):
    return (img[..., 3] == 255) & np.all(img[..., :3] == np.array(colour, np.uint8), axis=-1)


def any_colour(img, cols):
    m = np.zeros(img.shape[:2], bool)
    for c in cols:
        m |= colour_mask(img, c)
    return m


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


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    sizes = []
    for y, x in zip(*[a.tolist() for a in np.nonzero(mask)]):
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
    if abs(ang) < 0.05 or not np.any(layer[..., 3] == 255):
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
    a = out[..., 3] == 255
    p = np.pad(a, 1)
    n = p[:-2, 1:-1].astype(np.int16) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    out[a & (n == 0)] = 0
    return out


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
    flat[sel] = palette[dist.argmin(axis=1)][inv.ravel()]
    out[..., :3] = flat.reshape(out.shape[:2] + (3,))
    out[..., 3] = np.where(a, 255, 0).astype(np.uint8)
    out[~a] = 0
    return out


def put_empty(out, x, y, col):
    x, y = int(round(x)), int(round(y))
    h, w = out.shape[:2]
    if 2 <= x < w - 2 and 2 <= y < h - 2 and out[y, x, 3] == 0:
        out[y, x, :3] = col
        out[y, x, 3] = 255


def paste(dst, layer):
    m = layer[..., 3] == 255
    dst[m] = layer[m]


def wave(f, n, amp, phase=0.0, cycles=1.0):
    """Zero on frame 0 and on the frame after the loop."""
    return amp * (math.sin(2 * math.pi * cycles * f / n + phase) - math.sin(phase))


# ---------------------------------------------------------------- segmentation
def segment(img):
    h, w = img.shape[:2]
    opaque = img[..., 3] == 255
    yy, xx = np.mgrid[0:h, 0:w]
    gold = any_colour(img, GOLD)
    gem = any_colour(img, GEMS)

    jaw = opaque & (xx >= 17) & (xx <= 25) & (yy >= 30) & (yy <= 31)

    crest = opaque & (xx >= 44) & (yy <= 26)

    tail_allow = opaque & (xx >= 63) & (yy >= 38) & (yy <= 58) & ~gold & ~gem
    tail = flood([(42, 78)], tail_allow)

    fore = opaque & (yy >= 58) & (xx <= 33)
    crease = 27 + (77 - yy) * 0.6
    fore_far = fore & (yy >= 71) & (xx > crease)
    fore_near = fore & ~fore_far

    hind_far = opaque & (xx >= 47) & (xx <= 59) & (yy >= 61) & ~gold & ~gem
    hind_far |= opaque & gold & (xx >= 54) & (xx <= 58) & (yy >= 73)
    # keep only the dark leg reachable from its shin
    hind_far = flood([(64, 51), (70, 52), (76, 57)], hind_far)

    hind_near = opaque & (xx >= 62) & (yy >= 60) & ~((gold | gem) & (yy < 73))
    hind_near &= ~hind_far

    taken = jaw | crest | tail | fore_near | fore_far | hind_far | hind_near
    body = opaque & ~taken
    wheel = body & (gold | gem) & (xx >= 34)
    masks = {
        "body": body, "fore_near": fore_near, "fore_far": fore_far,
        "hind_far": hind_far, "hind_near": hind_near, "tail": tail,
        "crest": crest, "jaw": jaw,
    }
    return masks, wheel


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(mask.sum()))


# ---------------------------------------------------------------- rig
def top_rows(mask, rows):
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return np.zeros_like(mask)
    yy = np.arange(mask.shape[0])[:, None]
    return mask & (yy <= int(ys.min()) + rows - 1)


def left_cols(mask, cols):
    xs = np.nonzero(mask)[1]
    if len(xs) == 0:
        return np.zeros_like(mask)
    xx = np.arange(mask.shape[1])[None, :]
    return mask & (xx <= int(xs.min()) + cols - 1)


def build_rig(img, masks, wheel, flip):
    a = img[..., 3] == 255
    ys, xs = np.nonzero(a)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dx = (CELL_W - sw) // 2
    dy = FOOT_Y - sh
    off = (dx - x0, dy - y0)

    def place(m):
        pm = np.zeros((CELL_H, CELL_W), bool)
        pm[dy:dy + sh, dx:dx + sw] = m[y0:y1, x0:x1]
        return pm

    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    canvas[dy:dy + sh, dx:dx + sw] = img[y0:y1, x0:x1]
    placed = {k: place(v) for k, v in masks.items()}
    pwheel = place(wheel)

    cushion = (
        top_rows(placed["fore_near"], 4) | top_rows(placed["hind_near"], 3)
        | top_rows(placed["hind_far"], 3) | left_cols(placed["tail"], 2)
        | left_cols(placed["crest"], 2)
    )
    layers = {}
    for name in PARTS:
        lyr = np.zeros_like(canvas)
        lyr[placed[name]] = canvas[placed[name]]
        layers[name] = lyr
    layers["body"][cushion] = canvas[cushion]
    # dark mouth that shows only when the chin drops
    mouth = np.zeros_like(canvas)
    jm = placed["jaw"]
    jys, jxs = np.nonzero(jm)
    for x in range(int(jxs.min()), int(jxs.max()) + 1):
        col = jm[:, x]
        top = int(np.nonzero(col)[0].min())
        mouth[top, x, :3] = INK if x in (int(jxs.min()), int(jxs.max())) else RED
        mouth[top, x, 3] = 255
        mouth[top + 1, x, :3] = INK
        mouth[top + 1, x, 3] = 255
    layers["mouth"] = mouth

    def pt(x, y):
        return (x + off[0], y + off[1])

    piv = {
        "fore_near": pt(28.0, 59.0),
        "hind_near": pt(66.0, 61.0),
        "tail": pt(64.0, 53.0),
        "crest": pt(44.0, 22.0),
        "sole": pt(68.0, 83.0),
        "mouth": pt(13.0, 31.0),
        "wheel": pt(52.0, 54.0),
        "hoof": pt(21.0, 83.0),
    }
    gems_still = [(41, 36), (61, 38), (38, 66), (58, 67)]
    gems = [pt(x, y) for x, y in gems_still]
    if flip:
        layers = {k: v[:, ::-1].copy() for k, v in layers.items()}
        canvas = canvas[:, ::-1].copy()
        pwheel = pwheel[:, ::-1].copy()
        piv = {k: (CELL_W - 1 - x, y) for k, (x, y) in piv.items()}
        gems = [(CELL_W - 1 - x, y) for x, y in gems]
    facing = 1.0 if flip else -1.0  # screen x direction of the snout
    return {"layers": layers, "canvas": canvas, "wheel": pwheel, "piv": piv,
            "gems": gems, "facing": facing}


# ---------------------------------------------------------------- poses
def pose(action, f):
    n = FRAMES[action]
    p = dict(
        bdy=0, jaw=0,
        fn_ang=0.0, fn_lift=0, fn_dx=0,
        ff_lift=0, ff_dx=0,
        hf_lift=0, hf_dx=0,
        hn_ang=0.0, hn_lift=0, hn_dx=0,
        tail=0.0, crest=0.0, glow=0,
        rings=0, ring_age=0, rays=0, quake=0, burst=0,
        hurt=0.0, knock=0, flash=0, sparks=0,
    )
    if action == "idle":
        b = wave(f, n, 1.0, phase=-math.pi / 2)  # 0 at f0, peaks mid loop
        p["bdy"] = -1 if b > 0.9 else 0
        p["tail"] = wave(f, n, 6.0)
        p["crest"] = wave(f, n, 3.5, phase=0.9)
        p["glow"] = 1 if 6 <= f <= 10 else 0
    elif action == "walk":
        s = math.sin(2 * math.pi * 2 * f / n)
        a_ = max(0.0, s)   # near fore + far hind
        b_ = max(0.0, -s)  # far fore + near hind
        p["bdy"] = -1 if abs(s) > 0.7 else 0
        p["fn_lift"] = int(round(3 * a_))
        p["fn_ang"] = 10.0 * s
        p["hf_lift"] = int(round(2 * a_))
        p["hf_dx"] = int(round(2 * s))
        p["ff_lift"] = int(round(2 * b_))
        p["ff_dx"] = int(round(-2 * s))
        p["hn_lift"] = int(round(3 * b_))
        p["hn_ang"] = -9.0 * s
        p["tail"] = 7.0 * math.sin(2 * math.pi * 2 * f / n + 0.8) - 7.0 * math.sin(0.8)
        p["crest"] = 3.0 * s
    elif action == "attack":
        p["jaw"] = [0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0][f]
        p["rings"] = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        p["ring_age"] = f
        p["crest"] = [0, -5, -9, -10, -8, -4, -3, -3, -3, -3, -2, 0, 0, 0, 0, 0][f]
        p["tail"] = [0, -6, -10, -12, -10, -6, -4, -4, -4, -4, -2, 4, 2, 0, 0, 0][f]
        p["glow"] = [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 0][f]
        # near fore rears up and stamps
        p["fn_ang"] = [0, 0, 0, 0, 4, 14, 24, 30, 30, 18, 4, 0, 0, 0, 0, 0][f]
        p["fn_lift"] = [0, 0, 0, 0, 1, 3, 5, 6, 6, 4, 1, 0, 0, 0, 0, 0][f]
        p["bdy"] = [0, 0, 0, 0, 0, -1, -1, -2, -2, -1, 0, 1, 0, 0, 0, 0][f]
        p["rays"] = [0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 2, 0, 0, 0, 0, 0][f]
        p["quake"] = {11: 1, 12: 2, 13: 3}.get(f, 0)
        p["burst"] = 1 if f in (11, 12) else 0
    else:  # hurt
        env = math.sin(math.pi * f / 11.0)
        p["hurt"] = 34.0 * env
        p["knock"] = int(round(3 * env))
        p["flash"] = 1 if f == 1 else 0
        p["sparks"] = 1 if 2 <= f <= 6 else 0
        p["jaw"] = 1 if 2 <= f <= 6 else 0
        p["tail"] = 8.0 * env
        p["crest"] = 6.0 * env
    return p


GLOW1 = {G_DARK: G_MID, G_MID: G_HI, G_HI: G_TOP, GEM_D: GEM_L}
GLOW2 = {G_DARK: G_MID, G_MID: G_TOP, G_HI: G_TOP, GEM_D: GEM_L, GEM_L: WHITE}


def light_wheel(img, wheel, level, bdy):
    if level <= 0:
        return img
    wm = shift(wheel[..., None].repeat(4, 2).astype(np.uint8) * 255, 0, bdy)[..., 3] == 255
    table = GLOW1 if level == 1 else GLOW2
    out = img.copy()
    for src, dst in table.items():
        m = wm & colour_mask(img, src)
        out[m, :3] = dst
    return out


def assemble_pose(rig, p):
    L, piv, fc = rig["layers"], rig["piv"], rig["facing"]
    bdy = p["bdy"]
    parts = {}
    parts["body"] = shift(L["body"], 0, bdy)
    parts["tail"] = shift(rotate_nearest(L["tail"], -fc * p["tail"], piv["tail"]), 0, bdy)
    parts["crest"] = shift(rotate_nearest(L["crest"], -fc * p["crest"], piv["crest"]), 0, bdy)
    parts["jaw"] = shift(L["jaw"], 0, bdy + p["jaw"])
    # Legs: positive angle swings the hoof forward (toward the snout).
    fn = rotate_nearest(L["fore_near"], -fc * p["fn_ang"], piv["fore_near"]) if p["fn_ang"] else L["fore_near"]
    parts["fore_near"] = shift(fn, fc * p["fn_dx"], -p["fn_lift"])
    parts["fore_far"] = shift(L["fore_far"], fc * p["ff_dx"], -p["ff_lift"])
    parts["hind_far"] = shift(L["hind_far"], fc * p["hf_dx"], -p["hf_lift"])
    hn = rotate_nearest(L["hind_near"], -fc * p["hn_ang"], piv["hind_near"]) if p["hn_ang"] else L["hind_near"]
    parts["hind_near"] = shift(hn, fc * p["hn_dx"], -p["hn_lift"])
    img = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for name in DRAW_ORDER:
        if name == "jaw" and p["jaw"]:
            paste(img, shift(L["mouth"], 0, bdy))
        paste(img, parts[name])
    # Wheel sits in front of the near hind (as in the still).
    wm = shift(rig["wheel"][..., None].repeat(4, 2).astype(np.uint8) * 255, 0, bdy)[..., 3] == 255
    body_s = parts["body"]
    img[wm] = body_s[wm]
    img = light_wheel(img, rig["wheel"], p["glow"], bdy)
    return img


# ---------------------------------------------------------------- effects
def roar_rings(out, mouth, facing, age, strength):
    mx, my = mouth
    cols = (G_TOP, WHITE, G_HI)
    for i in range(strength):
        rad = 4 + age * 2 + i * 5
        col = cols[i % 3]
        for deg in range(-40, 41, 8):
            ang = math.radians(deg)
            put_empty(out, mx + facing * rad * math.cos(ang), my + rad * math.sin(ang) * 0.8, col)
    return out


def judgement_rays(out, gems, centre, level, bdy):
    cx, cy = centre
    for gx, gy in gems:
        gy += bdy
        vx, vy = gx - cx, gy - cy
        d = math.hypot(vx, vy) or 1.0
        ux, uy = vx / d, vy / d
        for k in range(level * 3):
            r = 5 + k * 2
            col = (G_TOP, WHITE, GEM_L)[k % 3]
            put_empty(out, gx + ux * r, gy + uy * r, col)
            # side sparks make a four-point star
            if k == level * 3 - 1:
                put_empty(out, gx + ux * r - uy * 2, gy + uy * r + ux * 2, G_TOP)
                put_empty(out, gx + ux * r + uy * 2, gy + uy * r - ux * 2, G_TOP)
    return out


def quake(out, hoof, facing, stage):
    hx, hy = hoof
    hx += facing * 2
    rad = 6 + stage * 6
    for t in range(-12, 13):
        ang = math.radians(t * 7.5)
        x = hx + rad * math.sin(ang)
        y = hy - abs(math.cos(ang)) * (2 + stage)
        if y <= hy:
            col = (G_TOP, WHITE, G_HI)[(t + stage) % 3]
            put_empty(out, x, y, col)
    # dust ahead of the hoof
    for k in range(stage + 1):
        put_empty(out, hx + facing * (rad + 2 + k * 2), hy - 1 - k, LIGHT)
        put_empty(out, hx - facing * (rad + 2 + k * 2), hy - 1 - k, GREY)
    return out


def light_burst(out, hoof, facing):
    hx, hy = hoof
    cx, cy = hx + facing * 4, hy - 8
    for k in range(8):
        ang = k * math.pi / 4
        for r in (4, 7, 10):
            put_empty(out, cx + r * math.cos(ang), cy + r * math.sin(ang) * 0.8,
                      (WHITE, G_TOP, GEM_L)[(k + r) % 3])
    return out


def sparks(out, facing, f):
    ys, xs = np.nonzero(out[..., 3] == 255)
    if len(xs) == 0:
        return out
    y = int(ys.min()) + 12
    x = int(xs.min() + 10 if facing < 0 else xs.max() - 10)
    for i, (dx, dy) in enumerate(((0, 0), (-4, 3), (3, -3), (-2, -5), (5, 4))):
        if (i + f) % 2 == 0:
            put_empty(out, x + facing * (-6 + dx), y + dy, (WHITE, G_TOP)[i % 2])
    return out


def flash(img):
    out = img.copy()
    a = out[..., 3] == 255
    p = np.pad(a, 1)
    rim = a & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    out[a, :3] = WHITE
    out[rim, :3] = LIGHT
    return out


def synth(rig, action, f, palette):
    p = pose(action, f)
    p = gen4_attack_strike.finish(p, action, f, __file__)
    fc = rig["facing"]
    piv = rig["piv"]
    if action == "hurt" and (p["hurt"] > 0.45 or p["knock"]):
        base = assemble_pose(rig, dict(p, hurt=0.0))
        img = rotate_nearest(base, -fc * p["hurt"], piv["sole"])
        img = shift(img, -fc * p["knock"], 0)
        img = despeckle(img)
    else:
        img = assemble_pose(rig, p)
    img = snap(img, palette)
    if p["flash"]:
        img = flash(img)
    if p["rings"]:
        m = piv["mouth"]
        img = roar_rings(img, (m[0], m[1] + p["bdy"]), fc, p["ring_age"], p["rings"])
    if p["rays"]:
        img = judgement_rays(img, rig["gems"], piv["wheel"], p["rays"], p["bdy"])
    if p["quake"]:
        img = quake(img, piv["hoof"], fc, p["quake"])
    if p["burst"]:
        img = light_burst(img, piv["hoof"], fc)
    if p["sparks"]:
        img = sparks(img, fc, f)
    return snap(img, palette)


# ---------------------------------------------------------------- sheets
def params_hash():
    src = (Path(__file__).read_bytes() + Path(__file__).with_name("gen4_attack_strike.py").read_bytes()
           + (SRC_DIR / "front.png").read_bytes())
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


def contact(tiles_rows, scale=2, pad=4):
    """tiles_rows: list of lists of RGBA tiles. Crops to the union box."""
    union = None
    for row in tiles_rows:
        for t in row:
            ys, xs = np.nonzero(t[..., 3] == 255)
            if len(xs) == 0:
                continue
            b = (xs.min(), ys.min(), xs.max(), ys.max())
            union = b if union is None else (min(union[0], b[0]), min(union[1], b[1]),
                                             max(union[2], b[2]), max(union[3], b[3]))
    x0, y0 = max(0, union[0] - pad), max(0, union[1] - pad)
    x1, y1 = min(CELL_W, union[2] + 1 + pad), min(CELL_H, union[3] + 1 + pad)
    cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    cols = max(len(r) for r in tiles_rows)
    bg = np.full((len(tiles_rows) * ch, cols * cw, 4), (15, 20, 31, 255), np.uint8)
    for ri, row in enumerate(tiles_rows):
        for ci, t in enumerate(row):
            up = np.repeat(np.repeat(t[y0:y1, x0:x1], scale, 0), scale, 1)
            m = up[..., 3] == 255
            bg[ri * ch:(ri + 1) * ch, ci * cw:(ci + 1) * cw][m] = up[m]
            bg[ri * ch:(ri + 1) * ch, ci * cw, :3] = (40, 48, 64)
    return Image.fromarray(bg, "RGBA")


def preview_image(sheets):
    rows = []
    for act in ACTIONS:
        band = sheets[act][:CELL_H]
        tiles = [band[:, f * CELL_W:(f + 1) * CELL_W] for f in range(FRAMES[act])]
        rows.append(tiles[:8])
        rows.append(tiles[8:])
    return contact(rows, scale=2)


def motion_image(sheets):
    rows = []
    for act in ACTIONS:
        for di in (2, 6):  # E and W
            band = sheets[act][di * CELL_H:(di + 1) * CELL_H]
            rows.append([band[:, f * CELL_W:(f + 1) * CELL_W] for f in range(FRAMES[act])])
    return contact(rows, scale=2)


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


def with_flock(lock_path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def replace_value(text, start_at, key):
    token = '"' + key + '":'
    pos = text.find(token, start_at)
    if pos < 0 or text.find(token, pos + len(token)) != -1:
        raise SystemExit(f"key {key} not unique")
    val_at = pos + len(token)
    _old, end = json.JSONDecoder().raw_decode(text, val_at)
    return val_at, end


def publish_dir8(sprite_h):
    entry = {
        "dex": 493, "form": "normal", "variant": "arceus", "gen": "gen4",
        "dir": "monsterSprite/gen4/0493_arceus/normal", "sheetDir": "agent-sprite-forge",
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
        a, b = replace_value(text, 0, KEY)
        out = text[:a] + new_val + text[b:]
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
        ys, xs = np.nonzero(tile[..., 3] == 255)
        vals += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    new_val = json.dumps(vals, separators=(",", ":"))
    text = BOUNDS.read_text()
    s_at = text.find('"s":{')
    if s_at < 0:
        raise SystemExit("sprite_bounds has no s table")
    a, b = replace_value(text, s_at, KEY)
    out = text[:a] + new_val + text[b:]
    tmp = BOUNDS.with_suffix(".js.tmp")
    tmp.write_text(out)
    tmp.replace(BOUNDS)
    return vals


def update_queue(status, note=None):
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
def opaque_stats(tile, base):
    c = tile[..., 3] == 255
    n = int(c.sum())
    sizes = components(c)
    edge = bool(c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any())
    return {"opaque": n, "ratio": n / base if base else 0.0, "parts": len(sizes),
            "main": sizes[0] / n if n else 0.0, "edge": edge}


def load():
    front = np.array(Image.open(SRC_DIR / "front.png").convert("RGBA"))
    if front.shape[:2] != (96, 96):
        raise SystemExit(f"expected 96x96 front.png, got {front.shape}")
    if set(np.unique(front[..., 3]).tolist()) - {0, 255}:
        raise SystemExit("front alpha is not binary")
    palette = extract_palette(front)
    pal = {tuple(int(v) for v in c) for c in palette}
    for c in (INK, WHITE, LIGHT, GREY, RED) + GOLD + GEMS:
        if c not in pal:
            raise SystemExit(f"fx colour {c} not in palette")
    masks, wheel = segment(front)
    return front, palette, masks, wheel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="after visual approval: sync + dir8 + sprite_bounds + queue done")
    args = ap.parse_args()

    front, palette, masks, wheel = load()
    opaque = front[..., 3] == 255
    covered = np.zeros_like(opaque)
    overlap = 0
    for name in PARTS:
        overlap += int((covered & masks[name]).sum())
        covered |= masks[name]
    counts = {k: int(masks[k].sum()) for k in PARTS}
    counts["missing"] = int((opaque & ~covered).sum())
    counts["overlap"] = overlap
    counts["wheel"] = int(wheel.sum())
    print("parts", counts, "palette", len(palette), flush=True)
    for k in PARTS:
        print(" ", k, bbox_of(masks[k]), "pieces", components(masks[k])[:4], flush=True)
    if counts["missing"] or overlap:
        raise SystemExit("segmentation does not partition the still")

    rigs = {d: build_rig(front, masks, wheel, flip=d not in UNFLIPPED) for d in DIRS}
    for d in DIRS:
        for act in ACTIONS:
            r0 = synth(rigs[d], act, 0, palette)
            diff = int(np.any(r0 != rigs[d]["canvas"], axis=2).sum())
            if diff:
                raise SystemExit(f"{act} frame 0 differs from the still ({d}): {diff}px")
    print("still diff 0 for all dirs/actions", flush=True)

    work = OUT_DIR / "_work"
    work.mkdir(parents=True, exist_ok=True)
    part = np.zeros_like(front)
    cols = {"body": (90, 140, 220), "fore_near": (220, 60, 60), "fore_far": (250, 150, 150),
            "hind_far": (240, 200, 40), "hind_near": (40, 180, 90), "tail": (160, 90, 220),
            "crest": (255, 140, 40), "jaw": (255, 0, 255)}
    for name, col in cols.items():
        part[masks[name], :3] = col
        part[masks[name], 3] = 255
    part[wheel, :3] = (250, 240, 120)
    Image.fromarray(np.repeat(np.repeat(part, 5, 0), 5, 1), "RGBA").save(work / "parts.png")

    if args.samples:
        rows = []
        for d in ("W", "E"):
            for act in ACTIONS:
                rows.append([synth(rigs[d], act, f, palette) for f in range(FRAMES[act])])
        contact(rows, scale=2).save(work / "samples.png")
        rows = [[synth(rigs[d], "idle", 0, palette) for d in DIRS],
                [synth(rigs[d], "attack", 7, palette) for d in DIRS]]
        contact(rows, scale=2).save(work / "samples_dirs.png")
        print("samples", work / "samples.png", flush=True)
        return

    frames_dir = work / "frames"
    state_p = work / "state.json"
    digest = params_hash()
    if args.force and frames_dir.exists():
        shutil.rmtree(frames_dir)
    state = json.loads(state_p.read_text()) if state_p.exists() and not args.force else {}
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
                Image.fromarray(synth(rigs[d], act, f, palette), "RGBA").save(tmp)
                tmp.replace(fp)
            if key not in state["done"]:
                state["done"].append(key)
                state_p.write_text(json.dumps(state))
                print(f"[{len(state['done'])}/{total}] {key}", flush=True)

    sheets = assemble(frames_dir)
    master = write_outputs(sheets)
    prev = preview_image(sheets)
    prev.save(OUT_DIR / "preview.png")
    (SRC_DIR / "actions").mkdir(exist_ok=True)
    prev.save(SRC_DIR / "actions" / "preview.png")
    motion_image(sheets).save(OUT_DIR / "preview_motion.png")

    s0 = sheets["idle"][:CELL_H, :CELL_W]
    nys = np.nonzero(s0[..., 3] == 255)[0]
    sprite_h = int(nys.max() - nys.min() + 1)
    pal = {tuple(int(v) for v in c) for c in palette}
    used = {tuple(int(v) for v in c) for c in np.unique(master[master[..., 3] == 255][:, :3], axis=0)}
    off = sorted(used - pal)
    alphas = [int(v) for v in np.unique(master[..., 3])]
    clip = []
    base = int((s0[..., 3] == 255).sum())
    south = {}
    for act in ACTIONS:
        south[act] = []
        for di, d in enumerate(DIRS):
            for f in range(FRAMES[act]):
                tile = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W]
                st = opaque_stats(tile, base)
                if st["edge"]:
                    clip.append(f"{act}:{d}:{f}")
                if d == "S":
                    south[act].append({"f": f, "ratio": round(st["ratio"], 4),
                                       "parts": st["parts"], "main": round(st["main"], 4)})
    for act in ACTIONS:
        print("S", act, "ratio", " ".join(f"{r['ratio']:.2f}" for r in south[act]),
              "main", " ".join(f"{r['main']:.2f}" for r in south[act]), flush=True)
    print("sprite_h", sprite_h, "off", len(off), "alphas", alphas, "clip", len(clip), clip[:6], flush=True)
    report = {"version": VERSION, "hash": digest, "parts": counts, "sprite_h": sprite_h,
              "off_palette_colors": len(off), "alphas": alphas, "edge_touching": len(clip),
              "still_diff": 0, "south": south}
    qa = work / "qa"
    qa.mkdir(exist_ok=True)
    if not args.skip_qa and GAIT_LOOP.exists():
        py = str(FORGE / ".venv/bin/python")
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run(
                [py, "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / act / "S"),
                 "--fps", "12", "--output-dir", str(od), "--kind", "hover", "--no-aids"],
                capture_output=True, text=True)
            rep = od / "loop-report.json"
            if rep.exists():
                report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation")
            else:
                report[f"gait_{act}"] = (r.stdout + r.stderr).strip()[-300:]
            print("gait", act, report[f"gait_{act}"], flush=True)
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    meta = {
        "generator": "forge_arceus_actions", "version": VERSION, "key": KEY, "dex": 493,
        "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h,
        "frames": FRAMES, "directions": DIRS, "actions": ACTIONS,
        "unflipped": sorted(UNFLIPPED), "palette_colors": int(len(palette)),
        "off_palette_colors": len(off), "source": "front.png 96x96", "gifs_untouched": True,
        "facing": "left", "archetype": "quadruped_beast (cross-wheel)",
        "attack": "chin roar + wheel light -> foreleg rear and stamp + judgement rays + quake",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    ok = not off and alphas == [0, 255] and not clip and sprite_h > 0
    if not ok:
        print("QA FAIL", flush=True)
        raise SystemExit(1)
    print("QA OK", flush=True)
    if args.publish:
        if not args.no_sync:
            print("synced files", sync_outputs(), flush=True)
        publish_dir8(sprite_h)
        print("bounds", publish_bounds(sheets), flush=True)
        update_queue("done")
        print("published", KEY, VERSION, "h", sprite_h, flush=True)


if __name__ == "__main__":
    main()
