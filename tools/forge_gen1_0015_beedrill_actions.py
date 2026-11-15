#!/usr/bin/env python3
"""Gen-1 lance wasp, 3rd stage (0015_beedrill): 8-direction x 4-action sheet (gen1 forge batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py (structure only; rig, colours and attack are this form's own):
  1. The official front.png / back.png (x8 nearest -> art/ref_front.png, art/ref_back.png) were given to the
     Antigravity CLI (agy, image generation) with art/agy_prompt.md (no proper nouns, exact part counts):
       art/out/keys_2x2.png  : top-left S, top-right E (faces right), bottom-left NE, bottom-right N
       art/out/sw_single.png : SW (3/4 front, faces down-left)
     The other three are mirrors: SE = SW, W = E, NW = NE.
     (try 1 got another job's images through a shared /tmp file -> art/try1; try 2 is the one used.)
  2. Key the white backdrop: border-connected white, plus enclosed white that is NOT ringed by wing veins
     (the wing cells stay white = membrane; the gaps inside the antenna curls go). Area-downscale: S height = the
     official front.png opaque height (59 px); E / NE / N share the grid scale; SW is matched on height.
  3. agent-sprite-forge palette_tool: build (30 colours + 2 reserved poison-FX colours) -> lock -> apply.
  4. Anatomical rig per view: wings (only wing-coloured pixels; pivot at the wing root), head + 2 antennae,
     thorax + 4 hanging legs, striped abdomen + stinger (pivot at the waist), two arm lances (pivot at the shoulder).
     It flies: no steps, it hovers. Posed at 4x (nearest), back to 1x.
     Actions (16 frames): idle (hover bob, wing buzz, abdomen sway), walk (forward flight: dip, surge, fast wing
     buzz, lances swing in turn, abdomen pumps), attack (twin lance jab: rear back with lances drawn -> lunge, jab
     with one lance then the other, poison sparks at the tips -> stinger curls forward and stabs -> recover),
     hurt (white flash, knocked back and tilted, wings flare, recover).
  5. QC: off-palette, edge-touch, still diff (frame 0 of every action = the rest pose), gait_loop.py idle / walk.

The official front / back png / gif are never touched. Writes 8dir_stills.png and agent-sprite-forge/ in the form
folder and syncs agent-sprite-forge/ (files only, not _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0015_beedrill_actions.py [--force]
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
KEY = "0015_beedrill"
REL = "monsterSprite/gen1/0015_beedrill/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0015_beedrill_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
OFFICIAL_H = 59                      # opaque height of the official front.png
GRID_H_S = 345                       # S height in the generated grid (px)
SW_H = 678                           # SW height in sw_single.png (px)
CELL_W, CELL_H, FOOT_Y = 160, 96, 84
UP = 4
FX_HI, FX_LO = (246, 214, 255), (156, 64, 206)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("E", False), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("E", True), "SW": ("SW", False)}
# white gaps inside the antenna curls (background) that the keying keeps: cleared after the downscale
CLEAR = {"S": [(29, 8, 35, 15), (42, 8, 48, 15)], "N": [(29, 4, 37, 13), (42, 4, 50, 13)]}


def mirror_poly(p, w):
    return [(w - 1 - x, y) for x, y in reversed(p)]


# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label: polygons in order, later wins; wings only take wing-coloured pixels (white / pale lavender) and win over
# body / head / abdomen, lances win over everything. draw: back to front.
# wsgn: rotation sign per wing (flap), csgn: abdomen rotation that curls the stinger forward / under.
# lance_a jabs first, lance_b second. fwd = screen direction of forward.
_S_LL = [(0, 44), (14, 28), (20, 23), (26, 22), (29, 24), (27, 27), (24, 30), (22, 34), (10, 46), (2, 51), (0, 51)]
_S_WL = [(0, 0), (38, 0), (38, 10), (31, 16), (31, 44), (16, 44), (0, 30)]
_N_LL = [(0, 47), (9, 32), (14, 29), (20, 33), (19, 42), (7, 53), (0, 53)]
_N_WL = [(0, 0), (39, 0), (39, 48), (16, 48), (0, 34)]
RIG = {
    "S": dict(
        poly={"wing_l": _S_WL, "wing_r": mirror_poly(_S_WL, 77),
              "body": [(27, 21), (49, 21), (52, 30), (59, 59), (17, 59), (24, 30)],
              "abdomen": [(29, 33), (47, 33), (47, 50), (42, 59), (35, 59), (29, 50)],
              "head": [(28, 5), (49, 5), (49, 22), (44, 27), (33, 27), (28, 22)],
              "lance_l": _S_LL, "lance_r": mirror_poly(_S_LL, 77)},
        wings=["wing_l", "wing_r"], lances=["lance_l", "lance_r"],
        draw=["wing_l", "wing_r", "body", "abdomen", "head", "lance_l", "lance_r"],
        pivot={"wing_l": (34, 22), "wing_r": (42, 22), "head": (38, 26), "abdomen": (38, 34), "body": (38, 40),
               "lance_l": (27, 25), "lance_r": (49, 25)},
        wsgn={"wing_l": 1, "wing_r": -1}, csgn=0, body_x=38, fwd=(0.0, 1.0), back=False),
    "SW": dict(
        poly={"wing_l": [(8, 6), (33, 6), (33, 27), (18, 31), (8, 27)],
              "wing_r": [(35, 0), (62, 0), (62, 49), (45, 49), (39, 36), (39, 18)],
              "body": [(30, 29), (46, 27), (50, 35), (59, 47), (57, 53), (48, 49), (46, 41), (40, 41), (27, 45), (25, 59),
                       (14, 59), (17, 43), (26, 37)],
              "abdomen": [(30, 39), (46, 37), (49, 46), (43, 56), (37, 59), (32, 56), (30, 47)],
              "head": [(9, 15), (30, 15), (38, 21), (40, 30), (36, 38), (28, 40), (18, 38), (9, 31)],
              "lance_up": [(29, 0), (36, 1), (46, 16), (49, 23), (46, 30), (41, 28), (40, 22), (31, 8)],
              "lance_low": [(0, 45), (0, 50), (5, 50), (16, 45), (28, 40), (31, 35), (25, 33), (18, 35), (11, 36), (3, 42)]},
        wings=["wing_l", "wing_r"], lances=["lance_low", "lance_up"],
        draw=["wing_r", "wing_l", "body", "abdomen", "head", "lance_up", "lance_low"],
        pivot={"wing_l": (30, 24), "wing_r": (42, 26), "head": (30, 34), "abdomen": (40, 39), "body": (40, 36),
               "lance_up": (44, 28), "lance_low": (29, 37)},
        wsgn={"wing_l": 1, "wing_r": -1}, csgn=1, body_x=38, fwd=(-0.9, 0.45), back=False),
    "E": dict(
        poly={"wing": [(0, 0), (33, 0), (35, 20), (31, 30), (23, 44), (0, 44)],
              "body": [(24, 24), (41, 25), (43, 40), (39, 61), (25, 61), (24, 40)],
              "abdomen": [(7, 34), (22, 27), (28, 25), (31, 32), (28, 38), (26, 44), (22, 53), (14, 58), (8, 58)],
              "head": [(29, 7), (53, 5), (53, 17), (46, 28), (34, 28), (29, 22)],
              "lance_far": [(35, 27), (48, 23), (60, 25), (74, 28), (74, 33), (60, 35), (48, 35), (37, 33)],
              "lance_near": [(35, 33), (48, 30), (55, 34), (63, 47), (63, 53), (56, 51), (44, 43), (36, 38)]},
        wings=["wing"], lances=["lance_far", "lance_near"],
        draw=["wing", "body", "abdomen", "head", "lance_far", "lance_near"],
        pivot={"wing": (30, 26), "head": (36, 26), "abdomen": (28, 32), "body": (32, 40),
               "lance_far": (38, 30), "lance_near": (38, 35)},
        wsgn={"wing": -1}, csgn=-1, body_x=32, fwd=(1.0, 0.0), back=False),
    "NE": dict(
        poly={"wing_l": [(0, 0), (27, 0), (31, 20), (31, 37), (21, 44), (3, 44), (0, 30)],
              "wing_r": [(31, 0), (61, 0), (61, 35), (45, 35), (41, 49), (29, 49), (29, 31), (35, 21)],
              "body": [(21, 13), (39, 13), (41, 33), (49, 39), (49, 62), (35, 62), (31, 45), (21, 45), (19, 57), (7, 57),
                       (9, 39), (19, 31)],
              "abdomen": [(17, 35), (37, 35), (37, 51), (31, 59), (27, 62), (21, 59), (17, 49)],
              "head": [(23, 3), (45, 3), (43, 14), (37, 22), (26, 22), (23, 14)],
              "lance": [(42, 33), (53, 33), (70, 47), (70, 53), (63, 53), (46, 43), (41, 39)]},
        wings=["wing_l", "wing_r"], lances=["lance"],
        draw=["lance", "body", "abdomen", "head", "wing_l", "wing_r"],
        pivot={"wing_l": (28, 20), "wing_r": (34, 20), "head": (31, 20), "abdomen": (27, 36), "body": (30, 34),
               "lance": (44, 38)},
        wsgn={"wing_l": 1, "wing_r": -1}, csgn=-1, body_x=30, fwd=(0.9, -0.45), back=True),
    "N": dict(
        poly={"wing_l": _N_WL, "wing_r": mirror_poly(_N_WL, 80),
              "body": [(32, 17), (48, 17), (50, 31), (63, 39), (63, 59), (52, 59), (50, 41), (30, 41), (28, 59), (16, 59),
                       (16, 39), (30, 31)],
              "abdomen": [(29, 32), (51, 32), (51, 48), (44, 60), (36, 60), (29, 48)],
              "head": [(28, 3), (52, 3), (50, 18), (44, 22), (36, 22), (30, 18)],
              "lance_l": _N_LL, "lance_r": mirror_poly(_N_LL, 80)},
        wings=["wing_l", "wing_r"], lances=["lance_r", "lance_l"],
        draw=["lance_l", "lance_r", "body", "abdomen", "head", "wing_l", "wing_r"],
        pivot={"wing_l": (38, 24), "wing_r": (41, 24), "head": (40, 20), "abdomen": (40, 33), "body": (40, 30),
               "lance_l": (17, 34), "lance_r": (62, 34)},
        wsgn={"wing_l": 1, "wing_r": -1}, csgn=0, body_x=40, fwd=(0.0, -1.0), back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    mn, mx = a.min(axis=2), a.max(axis=2)
    white = mn > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    vein = (~white) & (b >= r - 8) & (b > g + 5) & (mx > 110)       # wing veins / outline (pale lavender)
    objs = ndi.find_objects(lab)
    holes = []
    for i in range(n):
        if (i + 1) in border or objs[i] is None:
            continue
        sl = objs[i]
        y0, y1, x0, x1 = max(0, sl[0].start - 4), sl[0].stop + 4, max(0, sl[1].start - 4), sl[1].stop + 4
        m = lab[y0:y1, x0:x1] == i + 1
        ring = ndi.binary_dilation(m, iterations=3) & ~m & ~white[y0:y1, x0:x1]
        if not ring.any() or vein[y0:y1, x0:x1][ring].mean() < 0.08:
            holes.append(i + 1)                                      # not a wing cell -> background gap
    bg |= np.isin(lab, holes)
    bg |= ndi.binary_dilation(bg, iterations=2) & (mn > 200)
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def make_stills(dst: Path):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "E": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
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
        for x0, y0, x1, y1 in CLEAR.get(v, []):
            sub = a[y0:y1 + 1, x0:x1 + 1]
            sub[(sub[..., :3].min(axis=2) > 215)] = 0
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
    r0, g0, b0 = [still[..., c].astype(int) for c in range(3)]
    mx = np.maximum(np.maximum(r0, g0), b0)
    mn = np.minimum(np.minimum(r0, g0), b0)
    wingcol = op & (mx >= 135) & (b0 >= r0 - 10) & (b0 >= g0) & (mn >= 95)    # white / pale lavender
    order_lab = ["body", "abdomen", "head"]
    names = list(R["draw"])
    idx = {n: i + 1 for i, n in enumerate(names)}
    lab = np.zeros(op.shape, int)
    for n in order_lab:
        lab[poly_mask(R["poly"][n], op.shape) & op] = idx[n]
    for n in R["wings"]:
        lab[poly_mask(R["poly"][n], op.shape) & wingcol] = idx[n]
    # wing outline pixels (darker lavender rim) next to wing pixels, outside the body parts' cores, also go to the wing
    for n in R["lances"]:
        lab[poly_mask(R["poly"][n], op.shape) & op] = idx[n]
    un = op & (lab == 0)
    if un.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[un] = lab[iy[un], ix[un]]
    # every part one piece: stray islands go to the nearest other part (wings may be split into cells: keep them)
    for n in names:
        if n in R["wings"] or n == "body":       # wings: cells; body: the legs are separate islands (body is never moved alone)
            continue
        i = idx[n]
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
    masks = {n: lab == idx[n] for n in names}
    layers = {}
    order = R["draw"]
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 3 if n in ("body", "abdomen") else 2
        if n in R["wings"] or n in R["lances"]:
            cushion = 0
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion) if cushion else np.zeros_like(m)
        src_m = m
        if n in R["wings"] and m.any():
            # membrane under the lances / head / legs that overlap the wing: filled with the nearest wing colour, so no
            # hole opens when a lance jabs away
            hull = ndi.binary_fill_holes(ndi.binary_closing(np.pad(m, 4), iterations=3))[4:-4, 4:-4]
            seam = above & ~m & hull & ~masks["body"] & ~masks["abdomen"]
        if n == "body" and m.any():
            # underlay inside the body + head + abdomen outline for the parts drawn above it (lances, head)
            bh = m | masks["head"] | masks["abdomen"]
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 4), iterations=4))[4:-4, 4:-4]
            seam |= above & ~m & inside & ~masks["head"] & ~masks["abdomen"]
            src_m = m
        if seam.any() and m.any():
            fillable = src_m & (mx > 70)
            if not fillable.any():
                fillable = src_m
            _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
            sy, sx = np.nonzero(seam)
            lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
            lyr[sy, sx, 3] = 255
        layers[n] = lyr
    return layers, masks


def place_layer(layer, view, h=None):
    h = h or layer.shape[0]
    w = layer.shape[1]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - RIG[view]["body_x"]
    oy = FOOT_Y - (h - 1)
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(layer.shape[0], CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
def wave(f, amp, cycles=1, phase=0.0):
    """0 at frame 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def hump(f, amp):
    """0 -> amp -> 0 over the loop."""
    return amp * (1 - math.cos(2 * math.pi * f / N)) / 2


def pose(action, f):
    """Per-part [rot deg, fwd px, up px]; global [fwd px, up px, lean deg]."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        P["g"] = [0.0, round(wave(f, 1.6, 1)), 0.0]                  # hover bob
        P["wings"] = [wave(f, 9.0, 4), 0, 0]                         # wing buzz
        P["head"] = [0, 0, round(wave(f, 0.8, 1, -0.9))]
        P["lance_a"] = [0, 0, round(wave(f, 1.0, 1, 0.6))]
        P["lance_b"] = [0, 0, round(wave(f, 1.0, 1, 1.4))]
        P["abdomen"] = [wave(f, 5.0, 1, 0.4), 0, 0]
    elif action == "walk":
        P["g"] = [round(wave(f, 1.5, 2)), round(wave(f, 2.0, 2, 0.8)), hump(f, 6.0)]
        P["wings"] = [wave(f, 13.0, 4), 0, 0]
        P["head"] = [0, 0, round(wave(f, 0.8, 2, -0.6))]
        P["lance_a"] = [wave(f, 6.0, 1), round(wave(f, 2.0, 1)), 0]
        P["lance_b"] = [-wave(f, 6.0, 1), round(-wave(f, 2.0, 1)), 0]
        P["abdomen"] = [wave(f, 8.0, 2, 0.5), 0, 0]
        if f in (3, 11):
            fx.append(("trail", f))
    elif action == "attack":
        g_f = [0, -2, -4, -5, 1, 6, 9, 9, 8, 9, 10, 9, 6, 3, 1, 0][f]
        g_u = [0, 1, 3, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -3, -6, -8, -2, 5, 8, 7, 6, 7, 9, 8, 5, 3, 1, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["wings"] = [[0, 8, 14, 16, -6, -12, 10, -10, 12, -10, -14, 8, 6, 3, 1, 0][f], 0, 0]
        P["head"] = [0, [0, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0][f], 0]
        # twin jab: lance a, then lance b (drawn back first)
        P["lance_a"] = [[0, -8, -14, -16, -6, 4, 6, 2, -4, -4, -2, 0, 0, 0, 0, 0][f],
                        [0, -2, -4, -5, 2, 11, 14, 6, 0, -1, -1, 0, 0, 0, 0, 0][f], 0]
        P["lance_b"] = [[0, -8, -14, -16, -10, -8, -6, -2, 4, 6, 2, -2, -1, 0, 0, 0][f],
                        [0, -2, -4, -5, -4, -3, -2, 4, 11, 14, 6, 1, 0, 0, 0, 0][f], 0]
        P["abdomen"] = [[0, -6, -10, -12, -4, 4, 6, 6, 8, 14, 24, 26, 16, 8, 3, 0][f], 0, 0]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if f in (5, 6):
            fx.append(("jab_a", f))
        if f in (8, 9):
            fx.append(("jab_b", f))
        if f in (6, 9, 11):
            fx.append(("burst", f))
        if f in (10, 11, 12):
            fx.append(("sting", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-6.0 * env + jit, 2.0 * env, -12.0 * env]
        P["wings"] = [18.0 * env * (1 if f % 2 == 0 else 0.6), 0, 0]
        P["head"] = [0, -1.5 * env, 1.0 * env]
        P["lance_a"] = [-14.0 * env, -2.0 * env, 1.0 * env]
        P["lance_b"] = [-14.0 * env, -2.0 * env, 1.0 * env]
        P["abdomen"] = [-12.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 7:
            fx.append(("stars", f))
    return P, fx


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
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


def rot_pt(p, ang, piv):
    t = math.radians(ang)
    dx, dy = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + dx * math.cos(t) - dy * math.sin(t), piv[1] + dx * math.sin(t) + dy * math.cos(t))


def far_point(cell, piv):
    ys, xs = np.nonzero(cell[..., 3] > 0)
    i = int(np.argmax((xs - piv[0]) ** 2 + (ys - piv[1]) ** 2))
    return (float(xs[i]), float(ys[i]))


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    front = abs(fwx) < 0.5
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    tips = {}
    la, lb = R["lances"][0], R["lances"][-1]
    for n in R["draw"]:
        rot, dx, dy = 0.0, 0.0, 0.0
        if n == "head":
            h = P.get("head", zero)
            dx, dy = h[1] * fwx, h[1] * fwy - h[2]
        elif n in R["wings"]:
            rot = P.get("wings", zero)[0] * R["wsgn"][n]
        elif n == "abdomen":
            a = P.get("abdomen", zero)[0]
            if R["csgn"]:
                rot = a * R["csgn"]
            else:
                # front / back: the stinger swings sideways a little, the curl reads as a lift
                rot = a * 0.35
                dy = -abs(a) * 0.12 if action == "attack" else 0.0
        elif n in R["lances"]:
            key = "lance_a" if n == la else "lance_b"
            if n != la and n != lb:
                key = "lance_a"
            a = P.get(key, zero)
            # rotation that lifts the lance tip backward: front / back views mirror it per side
            if front:
                sx = 1.0 if piv[n][0] < CELL_W // 2 else -1.0
                rot = a[0] * sx * 0.6
                dx = -sx * max(0.0, a[1]) * 0.25                    # jab toward the middle line
                dy = a[1] * fwy - a[2]
            else:
                rot = -a[0] * side
                dx, dy = a[1] * fwx, a[1] * fwy - a[2]
            tp = rot_pt(far_point(cells[n], piv[n]), rot, piv[n])
            tips[key] = (tp[0] + dx, tp[1] + dy)
        if n == "abdomen":
            tp = rot_pt(far_point(cells[n], piv[n]), rot, piv[n])
            tips["sting"] = (tp[0] + dx, tp[1] + dy)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if front:
        gf *= 0.4
    centre = (CELL_W // 2, FOOT_Y - 30)
    if abs(lean) > 0.3 and not front:
        canvas = rot4(canvas, lean * side, centre)
        tips = {k: rot_pt(p, lean * side, centre) for k, p in tips.items()}
    elif abs(lean) > 0.3 and action == "hurt":
        canvas = rot4(canvas, lean * 0.5, centre)
        tips = {k: rot_pt(p, lean * 0.5, centre) for k, p in tips.items()}
    gx, gy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gx, gy)
    tips = {k: (p[0] + gx, p[1] + gy) for k, p in tips.items()}
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, (gx, gy), tips


# ---------------------------------------------------------------- effects (poison, reserved colours only)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def stroke(a, x0, y0, x1, y1, col):
    k = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for j in range(k + 1):
        u = j / k
        put(a, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, col)


def star(a, x, y, r, col_c, col_r):
    put(a, x, y, col_c)
    for d in range(1, r + 1):
        for ex, ey in ((d, 0), (-d, 0), (0, d), (0, -d)):
            put(a, x + ex, y + ey, col_r if d == r else col_c)


def draw_fx(a, fx, tips, fwd, centre):
    fwx, fwy = fwd
    px, py = -fwy, fwx
    for kind, k in fx:
        if kind == "trail":
            for j in range(3):
                bx, by = centre[0] - fwx * (22 + 4 * j), centre[1] - fwy * (16 + 3 * j) + (j - 1) * 6
                stroke(a, bx, by, bx - fwx * 4 - (0 if abs(fwx) > 0.3 else 0), by - fwy * 3, FX_HI)
        elif kind == "charge":
            for key in ("lance_a", "lance_b"):
                if key not in tips:
                    continue
                tx, ty = tips[key]
                for j in range(2 + k):
                    ang = j * 2.4 + k * 0.7
                    r = 7 - k
                    put(a, tx + math.cos(ang) * r, ty + math.sin(ang) * r, FX_LO if j % 2 else FX_HI)
        elif kind in ("jab_a", "jab_b"):
            key = "lance_a" if kind == "jab_a" else "lance_b"
            if key not in tips:
                continue
            tx, ty = tips[key]
            L = 8 if k in (5, 8) else 12
            for o in (-3, 0, 3):
                bx, by = tx + px * o - fwx * L, ty + py * o - fwy * L
                stroke(a, bx, by, bx + fwx * (L - 3), by + fwy * (L - 3), FX_HI if o == 0 else FX_LO)
        elif kind == "burst":
            key = {6: "lance_a", 9: "lance_b", 11: "sting"}[k]
            if key not in tips:
                continue
            cx, cy = tips[key][0] + fwx * 3, tips[key][1] + fwy * 3
            for j in range(8):
                ang = j * math.pi / 4 + k * 0.3
                stroke(a, cx + math.cos(ang) * 3, cy + math.sin(ang) * 3, cx + math.cos(ang) * 7, cy + math.sin(ang) * 7,
                       FX_HI if j % 2 == 0 else FX_LO)
            star(a, cx, cy, 2, FX_HI, FX_LO)
        elif kind == "sting" and "sting" in tips:
            tx, ty = tips["sting"]
            drops = {10: [(2, 2)], 11: [(3, 4), (-2, 6)], 12: [(1, 8), (-3, 10), (4, 9)]}[k]
            for ddx, ddy in drops:
                put(a, tx + ddx, ty + ddy, FX_LO)
                put(a, tx + ddx, ty + ddy - 1, FX_HI)
        elif kind == "stars":
            for j in range(3):
                ang = j * 2.1 + k * 0.8
                cx, cy = centre[0] + math.cos(ang) * 34, centre[1] - 30 + math.sin(ang) * 5
                star(a, cx, cy, 1 if k % 2 else 2, FX_HI, FX_LO)


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
                  "--balance", "--name", "beedrill", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}
    s_h = int(np.ptp(np.nonzero(still["S"][..., 3].any(axis=1))[0]) + 1)

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
                a, fx, (gx, gy), tips = compose(v, cells, piv, act, f)
                centre = (CELL_W / 2 + gx, FOOT_Y - 30 + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, tips, R["fwd"], centre)
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (2, 6, 9, 11)):
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
    meta = {"generator": "forge_gen1_0015_beedrill_actions", "version": VERSION, "key": KEY, "dex": 15,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": s_h, "frames": N, "frame_time": AD,
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
