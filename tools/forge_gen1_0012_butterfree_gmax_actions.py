#!/usr/bin/env python3
"""Gigantamax Butterfree (0012_butterfree-gmax): 8-direction stills + 8-direction x 4-action sheet.

Batch md/gen1-forge.md (2026-11-04), same flow as tools/forge_raichu_mega_x_actions.py:
  art/ref_front.png / ref_back.png  : official front.png / back.png x8 nearest (references for the Antigravity CLI)
  art/agy_prompt.md                 : morphological prompt (no proper nouns, exact part counts and colours)
  art/out/keys_2x2.png              : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png             : SW (3/4 front, faces down-left)
  Mirrors: SE = SW, E = W, NW = NE.

Pipeline: key the white backdrop (border-connected only: the glowing white wing panes stay) -> one game scale
(S still height = opaque height of the official front.png; SW is matched to the same height) -> agent-sprite-forge
palette_tool build / lock / apply (+2 reserved glow colours for the scale dust) -> part rig (4 wings: two huge
upper wings and two lower wings, each rotating about its root at the thorax; the body with head, antennae and
feet) -> poses at 4x nearest, sampled back to 1x -> QC -> agent-sprite-forge/ -> sync.

The creature hovers (it has two tiny feet but never stands on them).
Actions (16 frames each):
  idle   : hover bob, slow wing beats (upper + lower wing of a side beat as one unit), feet dangle, a few glowing
           scales drift off the wings.
  walk   : forward glide: faster, wider wing beats, body dips on each down-stroke, trailing scale dust.
  attack : wing gust - both wings rise high and back (wind-up, body tilts back), then sweep down hard with a lunge;
           a burst of glowing scale powder and wind arcs flies forward from the wings -> settle.
  hurt   : white flash, knocked back and tilted (<= 14 deg), wings crumple shut a little, recover.

The official front / back png / gif are never written.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0012_butterfree_gmax_actions.py [--force] [--no-sync]
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
KEY = "0012_butterfree-gmax"
REL = "monsterSprite/gen1/0012_butterfree/gigantamax/butterfree-gmax"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0012_butterfree_gmax_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 232, 140, 124
UP = 4
GLOW_HI, GLOW_LO = (255, 246, 190), (232, 120, 214)     # reserved: the glittering scale powder / gust only
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))
    ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# body = polygon around head + antennae + thorax + abdomen + feet (later polygons win: feet).
# Everything else is wing: split into left / right at split_x, upper / lower below the line low_y (x-dependent).
# piv: wing roots at the thorax. fwd: screen direction of "forward" (glide, gust).
RIG = {
    "S": dict(
        body=[(64, 46), (92, 46), (95, 60), (93, 78), (87, 80), (70, 80), (63, 78), (61, 60)],
        ant=[(60, 29), (98, 29), (98, 41), (85, 41), (84, 48), (72, 48), (71, 41), (60, 41)],
        feet=[(69, 74), (88, 74), (88, 89), (69, 89)],
        low={"wing_ll": [(70, 60), (40, 76), (36, 93), (72, 93)],
             "wing_lr": [(86, 60), (116, 76), (120, 93), (84, 93)]},
        split_x=78,
        draw=["wing_ll", "wing_lr", "wing_ul", "wing_ur", "feet", "body"],
        pivot={"wing_ul": (68, 60), "wing_ur": (88, 60), "wing_ll": (70, 68), "wing_lr": (86, 68),
               "feet": (78, 76), "body": (78, 64)},
        side={"wing_ul": 1, "wing_ll": 1, "wing_ur": -1, "wing_lr": -1},
        body_x=78, fwd=(0.0, 1.0)),
    "N": dict(
        body=[(68, 41), (89, 41), (94, 52), (93, 68), (87, 74), (87, 78), (71, 78), (71, 74), (65, 68), (64, 52)],
        ant=[(61, 26), (97, 26), (97, 37), (85, 37), (85, 45), (73, 45), (73, 37), (61, 37)],
        feet=[(71, 70), (87, 70), (87, 84), (71, 84)],
        low={"wing_ll": [(68, 58), (42, 70), (40, 92), (74, 92)],
             "wing_lr": [(89, 58), (115, 70), (117, 92), (83, 92)]},
        split_x=78.5,
        draw=["wing_ll", "wing_lr", "wing_ul", "wing_ur", "feet", "body"],
        pivot={"wing_ul": (68, 56), "wing_ur": (89, 56), "wing_ll": (70, 66), "wing_lr": (87, 66),
               "feet": (79, 72), "body": (79, 60)},
        side={"wing_ul": 1, "wing_ll": 1, "wing_ur": -1, "wing_lr": -1},
        body_x=79, fwd=(0.0, -1.0)),
    "W": dict(
        body=[(8, 48), (28, 48), (36, 58), (41, 68), (39, 80), (27, 81), (12, 73), (8, 60)],
        ant=[(0, 31), (23, 31), (25, 50), (14, 50), (0, 45)],
        feet=[(25, 76), (43, 76), (43, 93), (25, 93)],
        low={"wing_l": [(36, 72), (50, 66), (69, 72), (69, 93), (36, 93)]},
        split_x=None,
        draw=["wing_l", "wing_u", "feet", "body"],
        pivot={"wing_u": (36, 62), "wing_l": (38, 70), "feet": (32, 78), "body": (25, 64)},
        side={"wing_u": -1, "wing_l": -1},
        body_x=34, fwd=(-1.0, 0.0)),
    "SW": dict(
        body=[(30, 50), (50, 50), (58, 58), (59, 72), (56, 82), (44, 84), (33, 74), (29, 62)],
        ant=[(20, 37), (31, 35), (44, 32), (45, 56), (34, 58), (25, 47), (20, 45)],
        feet=[(40, 76), (56, 76), (56, 92), (40, 92)],
        low={"wing_ll": [(20, 76), (38, 68), (42, 70), (42, 93), (16, 93)],
             "wing_lr": [(56, 74), (70, 80), (88, 80), (94, 93), (54, 93)]},
        split_x=47,
        hooks=[[(22, 39), (32, 39), (32, 45), (22, 45)], [(32, 32), (42, 32), (42, 38), (32, 38)]],
        draw=["wing_ll", "wing_ul", "wing_lr", "wing_ur", "feet", "body"],
        pivot={"wing_ul": (40, 60), "wing_ur": (55, 62), "wing_ll": (40, 72), "wing_lr": (55, 74),
               "feet": (48, 78), "body": (45, 66)},
        side={"wing_ul": 1, "wing_ll": 1, "wing_ur": -1, "wing_lr": -1},
        body_x=48, fwd=(-0.9, 0.45)),
    "NE": dict(
        body=[(50, 39), (68, 39), (71, 48), (66, 56), (62, 64), (60, 74), (48, 74), (46, 62), (47, 50)],
        ant=[(57, 23), (77, 23), (77, 36), (68, 41), (58, 41)],
        feet=[(44, 70), (64, 70), (64, 87), (44, 87)],
        low={"wing_ll": [(22, 64), (48, 58), (48, 87), (20, 87)],
             "wing_lr": [(62, 62), (80, 62), (92, 68), (92, 87), (60, 87)]},
        split_x=57,
        hooks=[[(58, 23), (70, 23), (70, 33), (58, 33)], [(70, 27), (80, 27), (80, 33), (73, 35)]],
        draw=["wing_ll", "wing_ul", "wing_lr", "wing_ur", "feet", "body"],
        pivot={"wing_ul": (50, 50), "wing_ur": (64, 50), "wing_ll": (50, 62), "wing_lr": (62, 62),
               "feet": (54, 72), "body": (56, 58)},
        side={"wing_ul": 1, "wing_ll": 1, "wing_ur": -1, "wing_lr": -1},
        body_x=56, fwd=(0.9, -0.45)),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 232
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 205) & (np.ptp(a, axis=2) < 18)   # light fringe
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 3000])] = 0
    return out


def make_stills(dst: Path, target_h):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    keyed = {v: key_white(r) for v, r in src.items()}
    bb = {v: Image.fromarray(a, "RGBA").getbbox() for v, a in keyed.items()}
    grid = target_h / (bb["S"][3] - bb["S"][1])          # S still height = official front height
    scale = {v: grid for v in src}
    scale["SW"] = target_h / (bb["SW"][3] - bb["SW"][1])
    dst.mkdir(parents=True, exist_ok=True)
    for v, a in keyed.items():
        im = Image.fromarray(a, "RGBA").crop(bb[v])
        sm = im.resize((round(im.width * scale[v]), round(im.height * scale[v])), Image.BOX)
        b = np.array(sm)
        b[..., 3] = np.where(b[..., 3] >= 128, 255, 0)
        b[b[..., 3] == 0] = 0
        Image.fromarray(b, "RGBA").save(dst / f"{v}.png")


def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


# ---------------------------------------------------------------- 3. rig
def poly_mask(p, shape):
    m = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(m).polygon(p, fill=255)
    return np.array(m) > 0


def wingish(still):
    """Wing colours: greens, mints and the glowing white panes (the body is purple / red / blue / grey)."""
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    return ((g > r + 25) & (b < g + 20)) | ((np.minimum(np.minimum(r, g), b) > 215))


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    shape = op.shape
    names = R["draw"]
    idx = {n: i + 1 for i, n in enumerate(names)}
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    dark = op & (np.maximum(np.maximum(r, g), b) < 125)
    wing_c = op & wingish(still)
    xx = np.mgrid[0:shape[0], 0:shape[1]][1]
    lab = np.zeros(shape, int)
    if R["split_x"] is None:
        lab[op] = idx["wing_u"]
    else:
        lab[op & (xx < R["split_x"])] = idx["wing_ul"]
        lab[op & (xx >= R["split_x"])] = idx["wing_ur"]
    for n, p in R["low"].items():
        lab[poly_mask(p, shape) & op & (lab != 0)] = idx[n]
    # antennae: the dark stalks and the curled tips right at them (not the wing pane behind)
    am = poly_mask(R["ant"], shape) & op
    if R.get("hooks") is None:          # antennae over the backdrop: the whole polygon
        lab[am] = idx["body"]
    else:                               # antennae over a wing: the stalks (+2 px) and the curled tips
        near = ndi.binary_dilation(am & dark, iterations=2) & am
        for hp in R["hooks"]:
            near |= poly_mask(hp, shape) & op
        lab[near] = idx["body"]
    bm = poly_mask(R["body"], shape)
    lab[bm & op] = idx["body"]
    # grey wing roots / dark outline right at the body stay on the body (they are not wing pane colours)
    lab[op & ~wing_c & (ndi.distance_transform_edt(~bm) <= 5) & (lab != idx["body"])] = idx["body"]
    fm = poly_mask(R["feet"], shape) & op & ~wing_c
    fm &= ~poly_mask(R["body"], shape) | (np.mgrid[0:shape[0], 0:shape[1]][0] >= R["pivot"]["feet"][1])
    lab[fm] = idx["feet"]
    masks = {n: lab == idx[n] for n in names}

    layers = {}
    for i, n in enumerate(names):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in names[i + 1:]:
            above |= masks[n2]
        cushion = 7 if n.startswith("wing") else (3 if n == "feet" else 2)
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion) & op
        if seam.any() and m.any():
            fillable = m & (np.maximum(np.maximum(r, g), b) > 90)
            if n.startswith("wing"):
                fillable &= (np.minimum(np.minimum(r, g), b) > 150)        # pane colours, not the green rim
            if not fillable.any():
                fillable = m
            _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
            sy, sx = np.nonzero(seam)
            lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
            lyr[sy, sx, 3] = 255
        layers[n] = lyr
    return layers, masks


def place_layer(layer, view):
    """View-sized layer on the cell: body centre on the cell centre, lowest pixel on FOOT_Y."""
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - RIG[view]["body_x"]
    oy = FOOT_Y - (h - 1)
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(h, CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
def wave(amp, cycles, f, phase=0.0):
    """0 at frame 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """flap = upper-wing beat angle (+ = wings rise / close up), flap_lo = lower wings.
    body / feet: [dx_fwd, du]; feet_rot; g = [forward px, up px, roll deg]."""
    P = {"flap": 0.0, "flap_lo": 0.0, "body_up": 0.0, "feet": 0.0, "g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        P["flap"] = wave(7.0, 2, f)
        P["flap_lo"] = wave(4.0, 2, f, -0.0) * 0.5 + wave(2.0, 2, f, 0.9)
        P["body_up"] = round(-wave(1.5, 2, f, 0.5) * 1.0)
        P["feet"] = wave(5.0, 1, f, 0.3)
        P["g"] = [0.0, round(wave(2.0, 1, f)), 0.0]
        if 2 <= f <= 13:
            fx.append(("drift", f))
    elif action == "walk":
        P["flap"] = wave(12.0, 2, f)
        P["flap_lo"] = wave(8.0, 2, f, 0.7)
        P["body_up"] = round(wave(1.5, 2, f, 1.2))
        P["feet"] = 8.0 + wave(4.0, 2, f, 0.6)
        P["g"] = [wave(2.0, 1, f), 3.0 + round(wave(2.0, 2, f, -0.8)), 4.0]
        fx.append(("trail", f))
    elif action == "attack":
        P["flap"] = [0, 5, 11, 15, 17, 7, -10, -14, -13, -10, -6, -2, 2, 2, 1, 0][f]
        P["flap_lo"] = [0, 4, 9, 12, 14, 9, -6, -12, -12, -9, -5, -2, 1, 1, 0, 0][f]
        P["g"] = [[0, -1, -3, -4, -5, -1, 5, 9, 10, 9, 7, 5, 3, 2, 1, 0][f],
                  [0, 1, 2, 4, 5, 4, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0][f],
                  [0, -2, -4, -6, -7, -3, 4, 8, 8, 6, 4, 3, 2, 1, 0, 0][f]]
        P["body_up"] = [0, 0, 1, 1, 1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0][f]
        P["feet"] = [0, -3, -6, -8, -9, -4, 6, 10, 10, 8, 6, 4, 2, 1, 0, 0][f]
        if 2 <= f <= 4:
            fx.append(("gather", f))
        if 6 <= f <= 12:
            fx.append(("gust", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-7.0 * env + jit, 2.0 * env, -12.0 * env]
        P["flap"] = 10.0 * env
        P["flap_lo"] = 7.0 * env
        P["feet"] = -10.0 * env
        P["body_up"] = round(1.0 * env)
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 9:
            fx.append(("shed", f))
    return P, fx


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
    """img4 at 4x; rotate (+ = clockwise on screen) about piv (1x coords)."""
    if abs(ang) < 0.3:
        return img4
    im = Image.fromarray(img4, "RGBA").rotate(-ang, resample=Image.NEAREST, center=(piv[0] * UP + UP / 2, piv[1] * UP + UP / 2))
    return np.array(im)


def shift4(img4, dx, dy):
    dx, dy = int(round(dx * UP)), int(round(dy * UP))
    if not dx and not dy:
        return img4
    out = np.zeros_like(img4)
    h, w = img4.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = img4[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    face = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    side_view = R["split_x"] is None
    offs = {}
    for n in R["draw"]:
        rot, dx, dy = 0.0, 0.0, -P["body_up"]
        up_n = n
        if n.startswith("wing"):
            # upper and lower wing of one side beat as one unit about the upper root, so the hidden overlap
            # between them never opens
            a = P["flap"]
            if side_view:
                a *= 0.55              # side view: the beat swings the upright wings back and forth
            rot = a * R["side"][n]
            up_n = {"wing_ll": "wing_ul", "wing_lr": "wing_ur", "wing_l": "wing_u"}.get(n, n)
        elif n == "feet":
            rot = P["feet"] * face * (1.0 if abs(fwx) > 0.3 else 0.0)
            if abs(fwx) <= 0.3:
                dy += round(P["feet"] / 10.0)       # front / back: the feet swing toward the camera
        offs[n] = (dx, dy)
        l4 = rot4(up4(cells[n]), rot, piv[up_n] if n.startswith("wing") else piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, roll = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.5                                    # toward / away from the camera: a short dart, not a slide
    if abs(roll) > 0.3:
        c = (CELL_W // 2, FOOT_Y - 40)
        sgn = face if abs(fwx) > 0.3 else 1.0
        canvas = rot4(canvas, roll * sgn * (1.0 if abs(fwx) > 0.3 else 0.5), c)
    gx, gy = round(gf * fwx), round(gf * fwy - gu)
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, offs, (gx, gy)


# ---------------------------------------------------------------- effects (reserved glow colours only)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def scale_dot(a, x, y, big=True):
    """A glowing wing scale: a small diamond."""
    put(a, x, y, GLOW_HI)
    if big:
        for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            put(a, x + ddx, y + ddy, GLOW_LO)


def arc(a, cx, cy, r, a0, a1, col, step=None):
    k = max(6, int(abs(a1 - a0) * r))
    for j in range(k + 1):
        t = a0 + (a1 - a0) * j / k
        put(a, cx + math.cos(t) * r, cy + math.sin(t) * r, col)


def ellipse_arc(a, cx, cy, rx, ry, a0, a1, col):
    k = max(8, int(abs(a1 - a0) * max(rx, ry)))
    for j in range(k + 1):
        t = a0 + (a1 - a0) * j / k
        put(a, cx + math.cos(t) * rx, cy + math.sin(t) * ry, col)


def draw_fx(a, fx, ctx):
    fwx, fwy = ctx["fwd"]
    cx, cy = ctx["centre"]
    wings = ctx["wings"]               # sample points on the wings (cell coords at rest + global offset)
    for kind, k in fx:
        if kind == "drift":            # idle: a few scales float off the wing edges and sink
            for i, (wx, wy) in enumerate(wings[:3]):
                t0 = 2 + i * 3
                if t0 <= k <= t0 + 6:
                    u = k - t0
                    scale_dot(a, wx + math.sin(u * 0.9 + i) * 2, wy + u * 2.0, big=u < 4)
        elif kind == "trail":          # walk: scale dust trails behind the glide
            for i, (wx, wy) in enumerate(wings[:4]):
                u = (k + i * 4) % 8
                bx = -fwx if abs(fwx) > 0.3 else (1 if wx > cx else -1) * 0.6
                by = -fwy * 0.6 + 0.5
                scale_dot(a, wx + bx * u * 2.2, wy + by * u * 2.0, big=u < 3)
        elif kind == "gather":         # attack wind-up: scales gather along the raised wings
            for i, (wx, wy) in enumerate(wings):
                ang = i * 1.7 + k
                r = 9 - 2 * k
                scale_dot(a, wx + math.cos(ang) * r, wy + math.sin(ang) * r, big=(i + k) % 2 == 0)
        elif kind == "gust":           # the down-stroke throws a gust: wind crescents + a spray of glittering scales
            u = k - 6
            if abs(fwx) > 0.3:         # side / diagonal: forward
                base = math.atan2(fwy, fwx)
                for j in range(3):
                    v = u - j * 2
                    if 0 <= v <= 4:
                        d = 26 + v * 9
                        rr = 9 + v * 2 + j * 2
                        hx, hy = cx + fwx * d, cy + fwy * d + (j - 1) * 9
                        arc(a, hx - fwx * rr, hy - fwy * rr, rr, base - 0.85, base + 0.85, GLOW_LO)
                        arc(a, hx - fwx * (rr + 1), hy - fwy * (rr + 1), rr + 1, base - 0.7, base + 0.7, GLOW_HI)
                for i in range(12):
                    sp = 0.7 + (i % 4) * 0.25
                    lat = ((i * 7) % 11 - 5) * 0.11
                    d = 14 + (u + 1) * 6 * sp
                    px = cx + fwx * d - fwy * lat * d
                    py = cy + fwy * d + fwx * lat * d + (i % 3) * 4 - 8
                    scale_dot(a, px, py, big=(i + u) % 3 != 0)
            else:                      # front / back: the gust rolls out in wide flat crescents under the body
                for j in range(3):
                    v = u - j * 2
                    if 0 <= v <= 4:
                        rx, ry = 20 + v * 8, 6 + 2 * v
                        ey = cy + 20 + v * 2
                        ellipse_arc(a, cx, ey - ry, rx, ry, 0.35, math.pi - 0.35, GLOW_LO)
                        ellipse_arc(a, cx, ey - ry + 1, rx + 1, ry, 0.5, math.pi - 0.5, GLOW_HI)
                for i in range(12):
                    sdir = -1 if i % 2 else 1
                    d = 16 + (u + 1) * 6 * (0.6 + (i % 4) * 0.22)
                    scale_dot(a, cx + sdir * d, cy + 4 + (i // 2) * 3 + math.sin(u + i) * 2, big=(i + u) % 3 != 0)
        elif kind == "shed":           # hurt: scales knocked loose fall away
            for i, (wx, wy) in enumerate(wings):
                u = k - 2
                scale_dot(a, wx + (i - 2) * u * 0.8, wy + u * 2.5 + i, big=u < 4)


def flash(a, col):
    out = a.copy()
    out[a[..., 3] > 0, :3] = col
    return out


def nearest(cols, target):
    arr = np.array(sorted(cols), float)
    return tuple(int(v) for v in arr[np.argmin(((arr - np.array(target, float)) ** 2).sum(1))])


def wing_points(view, masks, ox, oy):
    """5 fixed sample points on the wing panes (cell coords at rest)."""
    pts = []
    for n in RIG[view]["draw"]:
        if not n.startswith("wing"):
            continue
        ys, xs = np.nonzero(masks[n])
        if not len(ys):
            continue
        i = int(np.argmin(ys + 0 * xs))          # the wing's top
        j = len(ys) // 2
        pts.append((xs[i] + ox, ys[i] + oy + 6))
        pts.append((xs[j] + ox, ys[j] + oy))
    order = [0, 2, 1, 3, 4, 5, 6, 7]
    return [pts[i] for i in order if i < len(pts)][:6]


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    return h.hexdigest()[:12]


def stills_src_hash():
    import inspect
    h = hashlib.sha1((inspect.getsource(key_white) + inspect.getsource(make_stills)).encode())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--stills-only", action="store_true")
    args = ap.parse_args()

    work = OUT_DIR / "_work"
    h = state_hash()
    st_p = work / "state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    if args.force or st.get("hash") != h:
        keep_stills = (not args.force) and st.get("stills_src") == stills_src_hash() and (work / "stills").exists()
        if work.exists():
            for d in work.iterdir():
                if keep_stills and d.name in ("stills_raw", "stills", "palette", "palette_lock"):
                    continue
                shutil.rmtree(d) if d.is_dir() else d.unlink()
        st = {"hash": h, "stills_src": stills_src_hash(), "done": ["stills"] if keep_stills else []}
    work.mkdir(parents=True, exist_ok=True)
    save_st = lambda: st_p.write_text(json.dumps(st))
    target_h = official_height()

    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw, target_h)
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % GLOW_HI, "#%02x%02x%02x" % GLOW_LO,
                  "--balance", "--name", "butterfree-gmax", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    pj = json.loads((lock_dir / "palette.json").read_text())
    cols = set()
    for c in pj.get("colors", []):
        hx = (c["hex"] if isinstance(c, dict) else c).lstrip("#")
        cols.add((int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)))
    body_cols = cols - {GLOW_HI, GLOW_LO}
    white = nearest(body_cols, (255, 255, 255))

    rig = {}
    dbg = work / "rig"
    dbg.mkdir(exist_ok=True)
    for v in RIG:
        layers, masks = build_rig(v, still[v])
        cells, piv = {}, {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v)
        for n, p in RIG[v]["pivot"].items():
            piv[n] = (p[0] + ox, p[1] + oy)
        rig[v] = (cells, piv, (ox, oy), masks)
        pal = [(200, 60, 60), (240, 200, 40), (60, 120, 220), (40, 180, 90), (180, 80, 200), (240, 140, 40), (90, 220, 220)]
        lab = np.zeros(still[v].shape[:2] + (3,), np.uint8)
        for i, (n, m) in enumerate(masks.items()):
            lab[m] = pal[i % len(pal)]
        Image.fromarray(np.concatenate([still[v][..., :3], lab], 1)).resize(((still[v].shape[1] * 2) * 4, still[v].shape[0] * 4), Image.NEAREST).save(dbg / f"{v}.png")
    if args.stills_only:
        return

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, (ox, oy), masks = rig[v]
        wpts = wing_points(v, masks, ox, oy)
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, offs, (gx, gy) = compose(v, cells, piv, act, f)
                ctx = {"fwd": RIG[v]["fwd"], "centre": (CELL_W / 2 + gx, piv["body"][1] + gy),
                       "wings": [(x + gx, y + gy) for x, y in wpts]}
                if any(k == "flash" for k, _ in fx):
                    a = flash(a, white)
                draw_fx(a, fx, ctx)
                Image.fromarray(a, "RGBA").save(od / f"{f:02d}.png")
            st["done"].append(key); save_st()
            print("frames", key, flush=True)

    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, N * CELL_W, 4), np.uint8)
        for di, d in enumerate(DIRS):
            v, mirror = DIR_SRC[d]
            for f in range(N):
                a = np.array(Image.open(frames_dir / v / act / f"{f:02d}.png").convert("RGBA"))
                if mirror:
                    a = a[:, ::-1]
                sh[di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W] = a
        sheets[act] = sh
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for act in ACTIONS:
        im = Image.fromarray(sheets[act], "RGBA")
        im.save(OUT_DIR / f"{act}.png"); im.save(OUT_DIR / f"{act}_8dir.png")
        Image.fromarray(sheets[act][:CELL_H], "RGBA").save(OUT_DIR / f"{act}_S_strip.png")
    master = np.concatenate([sheets[a] for a in ACTIONS], axis=0)
    Image.fromarray(master, "RGBA").save(OUT_DIR / "sheet.png")
    Image.fromarray(master, "RGBA").save(OUT_DIR / "all_8dir_actions.png")
    pv = np.full((8 * CELL_H, 16 * CELL_W, 4), (15, 20, 31, 255), np.uint8)
    for ai, act in enumerate(ACTIONS):
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (3, 6, 8, 11)):
            src = sheets[act][:, fc * CELL_W:(fc + 1) * CELL_W]
            mm = src[..., 3] > 0
            pv[:, (ai * 4 + si) * CELL_W:(ai * 4 + si + 1) * CELL_W][mm] = src[mm]
    Image.fromarray(pv, "RGBA").save(OUT_DIR / "preview.png")

    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(FORM_DIR / "8dir_stills.png")

    # 5. QC
    qa = work / "qa"; qa.mkdir(exist_ok=True)
    aa = master[..., 3] > 0
    used = {tuple(int(v) for v in c) for c in np.unique(master[aa][:, :3], axis=0)}
    off = sorted(used - cols)
    clip = {}
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(N):
                c = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] > 0
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    still_diff = {}
    for v in RIG:
        ref, _ = place_layer(still[v], v)
        fr = np.array(Image.open(frames_dir / v / "idle" / "00.png").convert("RGBA"))
        still_diff[v] = int((np.abs(ref.astype(int) - fr.astype(int)).sum(2) > 0).sum())
    alpha_ok = bool(np.isin(master[..., 3], [0, 255]).all())
    sprite_h_S = int(np.ptp(np.nonzero(idle0[0][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "edge_touching_frames": clip, "still_diff_px": still_diff, "alpha_binary": alpha_ok,
              "sprite_h_S": sprite_h_S, "official_front_h": target_h}
    if not args.skip_qa and GAIT_LOOP.exists():
        for act, kind in (("idle", "idle"), ("walk", "gait")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FPY), "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / "S" / act), "--fps", "13",
                                "--output-dir", str(od), "--kind", kind, "--no-aids"], capture_output=True, text=True)
            rep = od / "loop-report.json"
            report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr.strip().splitlines() or ["no report"])[-1]
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    bounds = []
    for di in range(8):
        c = idle0[di][..., 3] > 0
        ys, xs = np.nonzero(c)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    meta = {"generator": "forge_gen1_0012_butterfree_gmax_actions", "version": VERSION, "key": KEY, "dex": 12,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h_S, "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) or k in ("still_diff_px", "edge_touching_frames") else (v.get("policy") if "policy" in v else len(v))) for k, v in report.items()}, ensure_ascii=False))

    if not args.no_sync and SYNC_DIR.parent.exists():
        if SYNC_DIR.exists():
            shutil.rmtree(SYNC_DIR)
        SYNC_DIR.mkdir(parents=True)
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, SYNC_DIR / fp.name)
        print("synced ->", SYNC_DIR)


if __name__ == "__main__":
    main()
