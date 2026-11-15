#!/usr/bin/env python3
"""Butterfree (0012_butterfree): 8-direction x 4-action sheet (gen1 forge batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py (structure only; rig, colours and attack are this form's own):
  1. The official front.png / back.png (x8 nearest -> art/ref_front.png, art/ref_back.png) were given to the
     Antigravity CLI (agy, image generation) with art/agy_prompt.md (no proper nouns, exact part counts):
       art/out/keys_2x2.png  : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
       art/out/sw_single.png : SW (3/4 front, faces down-left)
     The first try drew a different creature (kept in art/try1); the retry (art/out) is the one used.
     The other three are mirrors: SE = SW, E = W, NW = NE.
  2. Key the white backdrop (border-connected pure white only: the pale lavender wings stay), area-downscale.
     S is scaled so its height = the official front.png opaque height (51 px); W / NE / N share the grid scale;
     SW is matched to the same height.
  3. agent-sprite-forge palette_tool: build (30 colours + 2 reserved wind-FX colours) -> lock -> apply.
  4. Anatomical rig per view: four wings (two large upper, two small lower; pivot at the wing root on the thorax),
     head with the two antennae and the two red eyes, body (thorax + abdomen + two teal hands + two blue feet).
     A flying insect: no ground contact, it hovers. Posed at 4x (nearest), back to 1x.
     A wing beat = rotation about the wing root + horizontal foreshortening toward the root (nearest, no new colours).
     Actions (16 frames): idle (hover bob, two slow wing beats), walk (flight: faster, deeper wing beats, forward
     tilt, bob on the down-stroke), attack (gust: wings rise high and the body pulls back -> one hard down-stroke
     and lunge -> wind arcs + scale dust blow forward -> recover), hurt (white flash, recoil, wings flare, dust).
  5. QC: off-palette, edge-touch, still diff (frame 0 of every action = the rest pose), gait_loop.py idle / walk.

The official front / back png / gif are never touched. Writes 8dir_stills.png and agent-sprite-forge/ in the form
folder and syncs agent-sprite-forge/ (files only, not _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0012_butterfree_actions.py [--force]
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
KEY = "0012_butterfree"
REL = "monsterSprite/gen1/0012_butterfree/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0012_butterfree_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
OFFICIAL_H = 51                      # opaque height of the official front.png
GRID_H_S = 347                       # S height in the generated grid (px)
SW_H = 575                           # SW height in sw_single.png (px)
CELL_W, CELL_H, FOOT_Y = 144, 100, 86
UP = 4
FX_HI, FX_LO = (236, 250, 255), (120, 176, 236)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label: later polygons win. draw: back to front. sgn: rotation sign that RAISES the wing tip.
# kind: up = large upper wing, lo = small lower wing. fwd = screen direction of forward.
RIG = {
    "S": dict(
        poly={"wing_ul": [(0, 0), (27, 0), (27, 31), (16, 36), (0, 36)],
              "wing_ll": [(4, 32), (26, 30), (26, 51), (4, 51)],
              "wing_ur": [(33, 0), (58, 0), (58, 34), (42, 34), (33, 30)],
              "wing_lr": [(36, 31), (54, 31), (54, 51), (36, 51)],
              "body": [(24, 29), (40, 29), (41, 49), (23, 49)],
              "head": [(24, 0), (36, 0), (36, 14), (43, 18), (42, 32), (22, 32), (21, 18), (25, 14)]},
        label=["wing_ul", "wing_ll", "wing_ur", "wing_lr", "body", "head"],
        draw=["wing_ll", "wing_lr", "wing_ul", "wing_ur", "body", "head"],
        pivot={"wing_ul": (26, 27), "wing_ll": (27, 36), "wing_ur": (38, 27), "wing_lr": (37, 37),
               "body": (32, 40), "head": (32, 30)},
        sgn={"wing_ul": 1, "wing_ll": 1, "wing_ur": -1, "wing_lr": -1},
        limb_box=(17, 24, 41, 51), body_x=32, fwd=(0.0, 1.0), back=False),
    "SW": dict(
        poly={"wing_ul": [(0, 0), (18, 0), (18, 20), (12, 32), (0, 34)],
              "wing_ll": [(8, 31), (18, 31), (22, 34), (22, 51), (8, 51)],
              "wing_ur": [(30, 0), (58, 0), (58, 36), (31, 32)],
              "wing_lr": [(30, 32), (52, 32), (52, 51), (33, 51)],
              "body": [(16, 30), (31, 30), (33, 51), (16, 51)],
              "head": [(17, 0), (31, 0), (31, 18), (31, 32), (11, 32), (10, 20), (17, 17)]},
        label=["wing_ul", "wing_ll", "wing_ur", "wing_lr", "body", "head"],
        draw=["wing_ll", "wing_lr", "wing_ul", "wing_ur", "body", "head"],
        pivot={"wing_ul": (18, 27), "wing_ll": (19, 36), "wing_ur": (30, 27), "wing_lr": (31, 37),
               "body": (24, 40), "head": (23, 31)},
        sgn={"wing_ul": 1, "wing_ll": 1, "wing_ur": -1, "wing_lr": -1},
        limb_box=(12, 26, 35, 51), body_x=24, fwd=(-0.9, 0.45), back=False),
    "W": dict(
        poly={"wing_ul": [(19, 0), (35, 0), (30, 20), (27, 33), (20, 32)],
              "wing_ur": [(29, 0), (57, 0), (57, 40), (30, 38), (27, 32)],
              "wing_lr": [(27, 37), (55, 37), (55, 53), (31, 53)],
              "body": [(12, 32), (28, 31), (32, 44), (32, 53), (14, 53), (9, 40)],
              "head": [(0, 4), (18, 4), (26, 20), (26, 35), (5, 36), (5, 22)]},
        label=["wing_ul", "wing_ur", "wing_lr", "body", "head"],
        draw=["wing_ul", "wing_lr", "wing_ur", "body", "head"],
        pivot={"wing_ul": (26, 32), "wing_ur": (28, 34), "wing_lr": (29, 40), "body": (21, 42), "head": (18, 34)},
        sgn={"wing_ul": -1, "wing_ur": -1, "wing_lr": -1},
        limb_box=(8, 34, 33, 53), body_x=20, fwd=(-1.0, 0.0), back=False),
    "NE": dict(
        poly={"wing_ul": [(0, 0), (29, 0), (29, 29), (14, 34), (0, 34)],
              "wing_ll": [(8, 30), (28, 28), (28, 48), (8, 48)],
              "wing_ur": [(37, 0), (63, 0), (63, 34), (37, 29)],
              "wing_lr": [(37, 28), (52, 28), (52, 48), (37, 48)],
              "body": [(27, 16), (38, 16), (38, 48), (26, 48)],
              "head": [(28, 0), (47, 0), (40, 15), (37, 21), (28, 21), (28, 14)]},
        label=["wing_ul", "wing_ll", "wing_ur", "wing_lr", "body", "head"],
        draw=["wing_ll", "wing_lr", "wing_ul", "wing_ur", "body", "head"],
        pivot={"wing_ul": (28, 23), "wing_ll": (28, 32), "wing_ur": (36, 23), "wing_lr": (36, 32),
               "body": (32, 36), "head": (32, 20)},
        sgn={"wing_ul": 1, "wing_ll": 1, "wing_ur": -1, "wing_lr": -1},
        limb_box=(24, 36, 40, 48), body_x=32, fwd=(0.9, -0.45), back=True),
    "N": dict(
        poly={"wing_ul": [(0, 0), (30, 0), (30, 31), (14, 36), (0, 36)],
              "wing_ll": [(10, 31), (30, 29), (30, 49), (10, 49)],
              "wing_ur": [(39, 0), (69, 0), (69, 36), (54, 36), (39, 31)],
              "wing_lr": [(39, 29), (58, 31), (58, 49), (39, 49)],
              "body": [(29, 16), (40, 16), (40, 49), (29, 49)],
              "head": [(28, 0), (42, 0), (40, 20), (30, 20)]},
        label=["wing_ul", "wing_ll", "wing_ur", "wing_lr", "body", "head"],
        draw=["wing_ll", "wing_lr", "wing_ul", "wing_ur", "body", "head"],
        pivot={"wing_ul": (30, 24), "wing_ll": (30, 33), "wing_ur": (39, 24), "wing_lr": (39, 33),
               "body": (35, 36), "head": (35, 20)},
        sgn={"wing_ul": 1, "wing_ll": 1, "wing_ur": -1, "wing_lr": -1},
        limb_box=(27, 37, 42, 49), body_x=35, fwd=(0.0, -1.0), back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    mn = a.min(axis=2)
    spread = np.ptp(a, axis=2)
    white = (mn >= 246) & (spread < 8)          # pure backdrop white; the wings are lavender (b > r)
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg |= ndi.binary_dilation(bg, iterations=2) & (mn > 238) & (spread < 8)
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def make_stills(dst: Path):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    scale = {v: OFFICIAL_H / GRID_H_S for v in src}
    scale["SW"] = OFFICIAL_H / SW_H
    dst.mkdir(parents=True, exist_ok=True)
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        im = im.crop(im.getbbox())
        sm = im.resize((round(im.width * scale[v]), round(im.height * scale[v])), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        Image.fromarray(a, "RGBA").save(dst / f"{v}.png")


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


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    lab = np.zeros(op.shape, int)
    names = R["label"]
    for i, n in enumerate(names, start=1):
        lab[poly_mask(R["poly"][n], op.shape) & op] = i
    # the teal hands and blue feet (g clearly above r; the lavender wings have g ~ r) belong to the body
    r0, g0, b0 = [still[..., c].astype(int) for c in range(3)]
    x0, y0, x1, y1 = R["limb_box"]
    yy, xx = np.mgrid[0:op.shape[0], 0:op.shape[1]]
    limb = op & (g0 - r0 > 25) & (b0 > 120) & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    limb = ndi.binary_dilation(limb, iterations=1) & op & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1) \
        & ((g0 - r0 > 25) | (np.maximum(np.maximum(r0, g0), b0) < 70))
    lab[limb] = names.index("body") + 1
    un = op & (lab == 0)
    if un.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[un] = lab[iy[un], ix[un]]
    # every part must be one piece: stray islands go to the nearest other part (antennae may be 2 islands)
    for i, n in enumerate(names, start=1):
        if n == "head":
            continue
        m = lab == i
        cl, cn = ndi.label(m)
        if cn > 1:
            sz = ndi.sum(np.ones_like(cl), cl, range(1, cn + 1))
            big = int(np.argmax(sz)) + 1
            stray = m & (cl != big)
            other = op & (lab != i)
            if other.any():
                _, (iy, ix) = ndi.distance_transform_edt(~other, return_indices=True)
                lab[stray] = lab[iy[stray], ix[stray]]
    masks = {n: lab == (i + 1) for i, n in enumerate(names)}
    layers = {}
    order = R["draw"]
    bright = np.maximum(np.maximum(r0, g0), b0)
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 4 if n.startswith("wing") else 2
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        if seam.any() and m.any():
            fillable = m & (bright > 70)
            if not fillable.any():
                fillable = m
            _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
            sy, sx = np.nonzero(seam)
            lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
            lyr[sy, sx, 3] = 255
        layers[n] = lyr
    return layers, masks


def place_layer(layer, view):
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - RIG[view]["body_x"]
    oy = FOOT_Y - (h - 1)
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(h, CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
def wave(f, amp, cycles=1, phase=0.0):
    """0 at frame 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def hump(f, amp, cycles=1):
    """0 -> amp -> 0, cycles times (0 at frame 0 and at the loop end)."""
    return amp * 0.5 * (1 - math.cos(2 * math.pi * cycles * f / N))


def pose(action, f):
    """wing_up / wing_lo: [rot deg (+ raises the tip), x-scale toward the root]; body / head: [fwd, up];
    g: [forward px, up px, lean deg]."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        flap = wave(f, 9.0, 2, 0.0)                         # two slow beats
        squash = 1.0 - hump(f, 0.14, 2)
        P["wing_up"] = [flap, squash]
        P["wing_lo"] = [wave(f, 6.0, 2, -0.5), 1.0 - hump(f, 0.10, 2)]
        P["body"] = [0, 0]
        P["head"] = [0, round(wave(f, 0.7, 1, -1.0))]
        P["g"] = [0, round(hump(f, 2.0, 1)), 0]           # hover bob (never below the rest height)
    elif action == "walk":
        flap = wave(f, 16.0, 2, 0.0)                        # flight: deeper beats
        P["wing_up"] = [flap, 1.0 - hump(f, 0.28, 2)]
        P["wing_lo"] = [wave(f, 11.0, 2, -0.5), 1.0 - hump(f, 0.2, 2)]
        P["body"] = [0, 0]
        P["head"] = [0, 1 if math.sin(2 * 2 * math.pi * f / N - 0.8) > 0.4 else 0]
        lift = hump(f, 3.0, 2)                              # rises on each down-stroke
        P["g"] = [round(wave(f, 1.0, 1, 0.0)), round(lift), wave(f, 3.0, 1, 0.0)]
        if f in (4, 12):
            fx.append(("dust", f))
    elif action == "attack":
        g_f = [0, -2, -3, -4, 2, 6, 9, 10, 10, 9, 7, 5, 3, 2, 1, 0][f]
        g_u = [0, 2, 3, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -3, -5, -6, 2, 6, 8, 8, 7, 6, 4, 3, 2, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [[0, -1, -1, -1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0][f], 0]
        P["body"] = [0, 0]
        # gust: both wing pairs rise high (wind-up), then one hard down-stroke (foreshortened), then settle
        P["wing_up"] = [[0, 12, 22, 28, 10, -14, -22, -24, -20, -14, -9, -5, -3, -1, 0, 0][f],
                        [1, 1, 1, 1, 0.85, 0.66, 0.6, 0.62, 0.7, 0.8, 0.88, 0.94, 0.97, 1, 1, 1][f]]
        P["wing_lo"] = [[0, 8, 14, 18, 6, -10, -16, -16, -13, -9, -6, -3, -2, -1, 0, 0][f],
                        [1, 1, 1, 1, 0.9, 0.75, 0.7, 0.72, 0.8, 0.86, 0.92, 0.96, 1, 1, 1, 1][f]]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 5 <= f <= 10:
            fx.append(("gust", f))
        if 6 <= f <= 11:
            fx.append(("powder", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-6.0 * env + jit, 2.0 * env, -8.0 * env]
        P["head"] = [-1.0 * env, 1.0 * env]
        P["body"] = [0, 0]
        P["wing_up"] = [18.0 * env, 1.0 - 0.1 * env]
        P["wing_lo"] = [-8.0 * env, 1.0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 8:
            fx.append(("scales", f))
    return P, fx


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
    if abs(ang) < 0.3:
        return img4
    im = Image.fromarray(img4, "RGBA").rotate(-ang, resample=Image.NEAREST, center=(piv[0] * UP + UP / 2, piv[1] * UP + UP / 2))
    return np.array(im)


def sx4(img4, s, piv):
    """foreshorten horizontally toward piv x (nearest)."""
    if abs(s - 1.0) < 0.01:
        return img4
    px = piv[0] * UP + UP / 2
    im = Image.fromarray(img4, "RGBA").transform((img4.shape[1], img4.shape[0]), Image.AFFINE,
                                                  (1.0 / s, 0, px - px / s, 0, 1, 0), resample=Image.NEAREST)
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
    zero = [0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    side_view = abs(fwx) > 0.95
    for n in R["draw"]:
        l4 = up4(cells[n])
        if n.startswith("wing"):
            kind = "wing_lo" if n in ("wing_ll", "wing_lr") else "wing_up"
            rot, s = P.get(kind, [0.0, 1.0])
            if side_view:
                # profile: the wings beat up / down about the root; foreshortening is vertical-ish -> keep x
                s = 1.0 - (1.0 - s) * 0.4
            if n == "wing_ul" and side_view:
                rot *= 0.8                                # far wing lags a little
            l4 = sx4(l4, s, piv[n])
            l4 = rot4(l4, rot * R["sgn"][n], piv[n])
        else:
            df, du = P.get(n, zero)
            l4 = shift4(l4, df * fwx, df * fwy - du)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4
    centre = (CELL_W // 2, FOOT_Y - 25)
    if abs(lean) > 0.3 and abs(fwx) > 0.3:
        canvas = rot4(canvas, lean * side, centre)
    canvas = shift4(canvas, gf * fwx, gf * fwy - gu)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, (gf * fwx, gf * fwy - gu)


# ---------------------------------------------------------------- effects (wind + wing-scale dust, reserved colours)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def sparkle(a, x, y, big):
    put(a, x, y, FX_HI)
    if big:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            put(a, x + dx, y + dy, FX_LO)


def arc(a, cx, cy, ux, uy, r, half, col, col2=None):
    """arc of radius r around (cx, cy), centred on direction (ux, uy), +-half radians (squashed for depth)."""
    base = math.atan2(uy, ux)
    k = max(6, int(r * half * 2))
    for j in range(k + 1):
        t = base - half + 2 * half * j / k
        x, y = cx + math.cos(t) * r, cy + math.sin(t) * r * 0.75
        put(a, x, y, col)
        if col2 is not None:
            put(a, x - math.cos(t), y - math.sin(t) * 0.75, col2)


def draw_fx(a, fx, fwd, centre):
    fwx, fwy = fwd
    depth = abs(fwx) < 0.5                              # toward / away from the camera
    for kind, k in fx:
        if kind == "dust":
            for j in range(3):
                put(a, centre[0] - 18 - 3 * j, centre[1] + 6 + 2 * j, FX_LO)
                put(a, centre[0] + 18 + 3 * j, centre[1] + 6 + 2 * j, FX_LO)
        elif kind == "charge":
            # air gathers under the raised wings: small arcs closing in on the body
            for sgn_ in (-1, 1):
                r = 22 - 4 * k
                arc(a, centre[0], centre[1] - 6, sgn_, -0.3, r, 0.45, FX_LO)
        elif kind == "gust":
            # three wind arcs blowing forward from the down-stroke
            for j in range(3):
                d = (k - 5) * 5 + 8 + j * 7
                if depth:
                    # toward / away from the camera: rings widening around the body
                    r = 13 + (k - 5) * 4 + j * 6
                    cy = centre[1] + (9 if fwy > 0 else -6)
                    arc(a, centre[0], cy, 0, 1 if fwy > 0 else -1, r, 1.1 - 0.15 * j, FX_HI if j == 0 else FX_LO,
                        FX_LO if j == 0 else None)
                else:
                    cx, cy = centre[0] + fwx * d, centre[1] + fwy * d
                    arc(a, cx - fwx * 10, cy - fwy * 10, fwx, fwy, 10 + j * 2, 0.9, FX_HI if j == 0 else FX_LO,
                        FX_LO if j == 0 else None)
        elif kind == "powder":
            rng = np.random.default_rng(100 + k)
            for j in range(9):
                t = rng.uniform(-0.9, 0.9)
                if depth:
                    dist = 8 + (k - 6) * 4 + rng.uniform(0, 8)
                    ang = rng.uniform(0, 2 * math.pi)
                    x, y = centre[0] + math.cos(ang) * dist * 1.4, centre[1] + math.sin(ang) * dist * 0.7
                else:
                    dist = 14 + (k - 6) * 5 + rng.uniform(0, 10)
                    x = centre[0] + fwx * dist - fwy * t * 12
                    y = centre[1] + fwy * dist + fwx * t * 12 + rng.uniform(-3, 3)
                sparkle(a, x, y, j % 3 == 0 and k < 10)
        elif kind == "scales":
            for j in range(5):
                ang = j * 1.3 + k * 0.7
                cx = centre[0] + math.cos(ang) * 26
                cy = centre[1] - 8 + math.sin(ang) * 16 + (k - 2) * 1.5
                sparkle(a, cx, cy, j % 2 == 0 and k < 6)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FX_HI
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    work = OUT_DIR / "_work"
    h = state_hash()
    st_p = work / "state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    if args.force or st.get("hash") != h:
        if work.exists():
            shutil.rmtree(work)
        st = {"hash": h, "done": []}
    work.mkdir(parents=True, exist_ok=True)
    save_st = lambda: st_p.write_text(json.dumps(st))

    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw)
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % FX_HI, "#%02x%02x%02x" % FX_LO,
                  "--balance", "--name", "butterfree", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    rig = {}
    for v in RIG:
        layers, masks = build_rig(v, still[v])
        cells, piv = {}, {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v)
        for n, p in RIG[v]["pivot"].items():
            piv[n] = (p[0] + ox, p[1] + oy)
        rig[v] = (cells, piv, (ox, oy))

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, (ox, oy) = rig[v]
        R = RIG[v]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, (gx, gy) = compose(v, cells, piv, act, f)
                centre = (CELL_W / 2 + gx, FOOT_Y - 22 + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, R["fwd"], centre)
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (3, 6, 8, 10)):
            src = sheets[act][:, fc * CELL_W:(fc + 1) * CELL_W]
            mm = src[..., 3] > 0
            pv[:, (ai * 4 + si) * CELL_W:(ai * 4 + si + 1) * CELL_W][mm] = src[mm]
    Image.fromarray(pv, "RGBA").save(OUT_DIR / "preview.png")

    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(FORM_DIR / "8dir_stills.png")

    # 5. QC
    qa = work / "qa"; qa.mkdir(exist_ok=True)
    pj = json.loads((lock_dir / "palette.json").read_text())
    cols = set()
    for c in pj.get("colors", []):
        hx = (c["hex"] if isinstance(c, dict) else c).lstrip("#")
        cols.add((int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)))
    aa = master[..., 3] > 0
    used = {tuple(int(v) for v in c) for c in np.unique(master[aa][:, :3], axis=0)}
    off = sorted(used - cols)
    alpha_bad = int(((master[..., 3] > 0) & (master[..., 3] < 255)).sum())
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
        for act in ACTIONS:
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            d = int((np.abs(a0.astype(int) - ref.astype(int)).sum(axis=2) > 0).sum())
            if d:
                still_diff[f"{v}/{act}"] = d
    s_h = int(np.ptp(np.nonzero(still["S"][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off), "alpha_partial": alpha_bad,
              "edge_touching_frames": clip, "still_diff": still_diff, "S_still_height": s_h, "official_height": OFFICIAL_H}
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
        c = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W, 3] > 0
        ys, xs = np.nonzero(c)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    meta = {"generator": "forge_gen1_0012_butterfree_actions", "version": VERSION, "key": KEY, "dex": 12,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(bounds[3]), "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

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
