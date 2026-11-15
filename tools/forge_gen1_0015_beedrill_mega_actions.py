#!/usr/bin/env python3
"""Mega Beedrill (0015_beedrill-mega): 8-direction x 4-action sheet (gen1 forge batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py (structure only; rig, colours and attack are this form's own):
  1. The official front.png / back.png (x8 nearest -> art/ref_front.png, art/ref_back.png) were given to the
     Antigravity CLI (agy, image generation) with art/agy_prompt.md (no proper nouns, exact part counts):
       art/out/keys_2x2.png  : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
       art/out/sw_single.png : SW (3/4 front, faces down-left)
     The other three are mirrors: SE = SW, E = W, NW = NE.
  2. Key the white backdrop (border-connected white only: the near-white wing panes are enclosed by their veins and
     stay), area-downscale. S is scaled so its height = the official front.png opaque height (74 px); W / NE / N
     share the grid scale; SW is matched to the same height (the official front is itself a SW-like 3/4 view).
  3. agent-sprite-forge palette_tool: build (30 colours + 2 reserved needle-FX colours) -> lock -> apply.
  4. Anatomical rig per view: four wings (one part per side, pivot at the wing root), head with two antennae,
     thorax, two lance forearms (pivot at the shoulder), the striped abdomen with the stinger (pivot at the waist),
     two dangling spiked legs. It flies: nothing stands on the ground; the stinger tip is the baseline.
     Actions (16 frames): idle (hover bob, fast wing buzz, abdomen / leg sway), walk (forward glide: lean, wing beat,
     legs paddle alternately, abdomen swings), attack (twin-needle lance thrust: lances draw back -> lunge, first
     lance stabs -> second lance stabs, abdomen curls the stinger forward, needle streaks + impact stars -> recover),
     hurt (white flash, knocked back, wings flare, recover).
  5. QC: off-palette, edge-touch, still diff (frame 0 of every action = the rest pose), gait_loop.py idle / walk.

The official front / back png / gif are never touched. Writes 8dir_stills.png and agent-sprite-forge/ in the form
folder and syncs agent-sprite-forge/ (files only, not _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0015_beedrill_mega_actions.py [--force]
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
KEY = "0015_beedrill-mega"
REL = "monsterSprite/gen1/0015_beedrill/mega/beedrill-mega"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0015_beedrill_mega_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
OFFICIAL_H = 74                      # opaque height of the official front.png
GRID_H_S = 316                       # S height in the generated grid (px)
SW_H = 628                           # SW height in sw_single.png (px)
CELL_W, CELL_H, FOOT_Y = 168, 112, 100
UP = 4
FX_HI, FX_LO = (255, 250, 190), (190, 80, 230)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}


def mx(poly, w):
    return [(w - 1 - x, y) for x, y in poly]


# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label: later polygons win. draw: back to front. sgn: rotation sign that RAISES the tip of a wing / lance
# (+ = clockwise on screen). abd: sign that curls the stinger forward. lanceA stabs first, lanceB second.
_S_L = {"wing_l": [(8, 0), (42, 0), (40, 14), (34, 22), (30, 28), (22, 32), (12, 32), (8, 16)],
        "lance_l": [(0, 54), (18, 36), (26, 28), (34, 26), (37, 30), (37, 40), (12, 58), (0, 62)],
        "leg_l": [(26, 40), (36, 40), (36, 56), (31, 72), (22, 72), (25, 56)]}
_N_L = {"lance_l": [(0, 52), (14, 36), (22, 28), (33, 28), (32, 40), (14, 54), (0, 60)],
        "wing_l": [(8, 0), (43, 0), (44, 40), (32, 44), (22, 42), (16, 30), (10, 18)],
        "leg_l": [(20, 42), (37, 42), (37, 56), (30, 70), (20, 70), (21, 56)]}
RIG = {
    "S": dict(
        w=94,
        poly={**_S_L, **{k.replace("_l", "_r"): mx(v, 94) for k, v in _S_L.items()},
              "abdomen": [(35, 47), (59, 47), (56, 64), (50, 74), (43, 74), (38, 64)],
              "body": [(32, 35), (62, 35), (60, 48), (34, 48)],
              "head": [(36, 4), (58, 4), (60, 18), (62, 30), (57, 36), (37, 36), (32, 30), (34, 18)]},
        label=["wing_l", "wing_r", "lance_l", "lance_r", "abdomen", "body", "leg_l", "leg_r", "head"],
        draw=["wing_l", "wing_r", "lance_l", "lance_r", "abdomen", "leg_l", "leg_r", "body", "head"],
        pivot={"wing_l": (36, 26), "wing_r": (57, 26), "lance_l": (34, 30), "lance_r": (59, 30), "abdomen": (47, 48),
               "body": (47, 48), "head": (47, 36), "leg_l": (32, 42), "leg_r": (61, 42)},
        sgn={"wing_l": 1, "wing_r": -1, "lance_l": 1, "lance_r": -1, "leg_l": 1, "leg_r": -1},
        abd=0, lanceA="lance_r", lanceB="lance_l", lite=[("head", 22, 99)], body_x=47, fwd=(0.0, 1.0), back=False),
    "SW": dict(
        w=67,
        poly={"wing": [(35, 0), (67, 0), (67, 12), (56, 12), (48, 24), (42, 20), (40, 12)],
              "lance_r": [(48, 22), (55, 11), (67, 12), (67, 34), (63, 68), (55, 68), (51, 48), (47, 34)],
              "lance_l": [(0, 50), (14, 26), (22, 17), (32, 21), (33, 32), (10, 50), (2, 56)],
              "abdomen": [(27, 42), (50, 42), (48, 56), (38, 64), (32, 74), (23, 74), (27, 58)],
              "body": [(25, 28), (50, 28), (52, 43), (27, 43)],
              "leg_l": [(17, 46), (29, 46), (27, 56), (22, 64), (14, 64)],
              "leg_r": [(48, 36), (58, 36), (57, 52), (55, 68), (48, 68), (48, 52)],
              "head": [(20, 2), (50, 2), (48, 20), (46, 32), (36, 38), (24, 36), (20, 24)]},
        label=["wing", "lance_r", "lance_l", "abdomen", "body", "leg_l", "leg_r", "head"],
        draw=["wing", "lance_r", "leg_r", "abdomen", "leg_l", "body", "lance_l", "head"],
        pivot={"wing": (46, 22), "lance_r": (52, 26), "lance_l": (28, 28), "abdomen": (38, 44), "body": (38, 44),
               "head": (36, 36), "leg_l": (26, 47), "leg_r": (52, 38)},
        sgn={"wing": -1, "lance_l": 1, "lance_r": -1, "leg_l": 1, "leg_r": -1},
        abd=1, lanceA="lance_l", lanceB="lance_r", lite=[], body_x=38, fwd=(-0.9, 0.45), back=False),
    "W": dict(
        w=72,
        poly={"wing": [(38, 0), (72, 0), (72, 44), (54, 44), (46, 36), (40, 24)],
              "body": [(36, 30), (54, 30), (56, 44), (42, 48), (34, 40)],
              "abdomen": [(24, 46), (42, 42), (58, 42), (58, 56), (42, 66), (22, 76), (10, 76), (14, 64)],
              "leg": [(42, 54), (51, 52), (51, 75), (42, 75)],
              "lance": [(0, 56), (28, 36), (40, 26), (54, 26), (57, 36), (22, 58), (8, 68), (0, 64)],
              "head": [(14, 4), (38, 4), (41, 20), (40, 34), (30, 41), (16, 39), (14, 24)]},
        label=["wing", "body", "abdomen", "leg", "lance", "head"],
        draw=["wing", "leg", "abdomen", "body", "lance", "head"],
        pivot={"wing": (44, 30), "body": (46, 44), "abdomen": (46, 46), "leg": (46, 53), "lance": (46, 32),
               "head": (34, 36)},
        sgn={"wing": -1, "lance": 1, "leg": 1},
        abd=1, lanceA="lance", lanceB="lance", lite=[], body_x=44, fwd=(-1.0, 0.0), back=False),
    "NE": dict(
        w=86,
        poly={"wing": [(0, 0), (38, 0), (38, 22), (34, 30), (30, 44), (24, 60), (14, 60), (0, 30)],
              "lance": [(46, 20), (56, 22), (70, 40), (86, 62), (86, 70), (80, 70), (58, 50), (47, 40)],
              "abdomen": [(22, 40), (50, 40), (57, 50), (60, 64), (56, 77), (46, 77), (26, 64), (20, 50)],
              "body": [(28, 30), (50, 30), (52, 44), (30, 46)],
              "head": [(39, 12), (54, 6), (56, 22), (50, 34), (40, 36), (34, 30), (34, 20)],
              "leg_l": [(23, 58), (34, 58), (32, 73), (25, 73)],
              "leg_r": [(53, 58), (61, 58), (63, 75), (57, 75), (53, 64)]},
        label=["wing", "lance", "abdomen", "body", "head", "leg_l", "leg_r"],
        draw=["leg_l", "leg_r", "abdomen", "body", "head", "lance", "wing"],
        pivot={"wing": (34, 30), "lance": (50, 28), "abdomen": (40, 40), "body": (40, 40), "head": (42, 34),
               "leg_l": (29, 58), "leg_r": (55, 60)},
        sgn={"wing": 1, "lance": -1, "leg_l": 1, "leg_r": -1},
        abd=-1, lanceA="lance", lanceB="lance", lite=[("abdomen", 62, 33)], body_x=40, fwd=(0.9, -0.45), back=True),
    "N": dict(
        w=98,
        poly={**_N_L, **{k.replace("_l", "_r"): mx(v, 98) for k, v in _N_L.items()},
              "abdomen": [(36, 38), (62, 38), (60, 56), (52, 73), (46, 73), (38, 56)],
              "head": [(41, 2), (57, 2), (56, 38), (42, 38)]},
        label=["lance_l", "lance_r", "wing_l", "wing_r", "abdomen", "leg_l", "leg_r", "head"],
        draw=["lance_l", "lance_r", "leg_l", "leg_r", "abdomen", "head", "wing_l", "wing_r"],
        pivot={"lance_l": (32, 32), "lance_r": (65, 32), "wing_l": (42, 26), "wing_r": (55, 26), "abdomen": (49, 40),
               "head": (49, 38), "leg_l": (33, 44), "leg_r": (64, 44)},
        sgn={"wing_l": 1, "wing_r": -1, "lance_l": 1, "lance_r": -1, "leg_l": 1, "leg_r": -1},
        abd=0, lanceA="lance_l", lanceB="lance_r", lite=[], body_x=49, fwd=(0.0, -1.0), back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    mn = a.min(axis=2)
    lab, n = ndi.label(mn > 225)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))            # enclosed near-white = wing panes (kept)
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
    un = op & (lab == 0)
    if un.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[un] = lab[iy[un], ix[un]]
    r0, g0, b0 = [still[..., c].astype(int) for c in range(3)]
    lite_px = op & (np.minimum(np.minimum(r0, g0), b0) > 140) & (b0 >= r0 - 4)        # wing / lance lavender-white
    yy, xx = np.mgrid[0:op.shape[0], 0:op.shape[1]]
    wid = [i + 1 for i, n in enumerate(names) if n.startswith("wing")]
    for part, ymax, xmax in R["lite"]:
        # light wing-pane pixels caught in a body-part polygon belong to the nearest wing
        hit = (lab == names.index(part) + 1) & lite_px & (yy < ymax) & (xx <= xmax)
        if hit.any() and wid:
            wm = np.isin(lab, wid)
            _, (iy, ix) = ndi.distance_transform_edt(~wm, return_indices=True)
            lab[hit] = lab[iy[hit], ix[hit]]
    # every part must be one piece: stray islands go to the nearest other part
    for i, n in enumerate(names, start=1):
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
        cushion = 4 if n in ("body", "abdomen", "head") else 2
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        if n.startswith("wing") or n.startswith("lance") or n.startswith("leg"):
            seam &= False                      # thin translucent parts: no cushion (would thicken them)
        src_m = m
        if n == "body" and m.any():
            bh = m | masks.get("head", np.zeros_like(m)) | masks.get("abdomen", np.zeros_like(m))
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 6), iterations=6))[6:-6, 6:-6]
            seam |= above & ~m & inside & ~masks.get("head", np.zeros_like(m))
            src_m = bh
        if seam.any() and m.any():
            fillable = src_m & (bright > 70)
            if not fillable.any():
                fillable = src_m
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


def pose(action, f):
    """Per-part [rot deg, fwd px, up px]; global [fwd px, up px, lean deg].
    wings rot: + raises; lanceA / lanceB rot: + raises the tip; abd rot: + curls the stinger forward;
    leg_a / leg_b rot: + swings the spike back."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        P["g"] = [0.0, round(wave(f, 1.6, 1, 0.0)), 0.0]                       # hover bob
        P["wings"] = [wave(f, 7.0, 4, 0.0), 0, 0]                                # fast buzz
        P["lanceA"] = [wave(f, 2.5, 1, -0.6), 0, 0]
        P["lanceB"] = [wave(f, 2.5, 1, -0.6), 0, 0]
        P["abd"] = [wave(f, 4.0, 1, -1.0), 0, 0]
        P["leg_a"] = [wave(f, 5.0, 1, -0.4), 0, 0]
        P["leg_b"] = [wave(f, 5.0, 1, -2.0), 0, 0]
        P["head"] = [0, 0, round(wave(f, 0.8, 1, -0.8))]
    elif action == "walk":
        P["g"] = [0.0, round(wave(f, 2.0, 2, 0.0)), wave(f, 3.0, 1, 0.0)]      # bob + rocking glide
        P["wings"] = [wave(f, 10.0, 4, 0.0), 0, 0]
        P["lanceA"] = [wave(f, -4.0, 1, 0.0), 0, 0]
        P["lanceB"] = [wave(f, 4.0, 1, 0.0), 0, 0]
        P["abd"] = [wave(f, 8.0, 1, 0.0), 0, 0]                                # stinger swings back and forth
        P["leg_a"] = [wave(f, 12.0, 1, 0.0), 0, 0]                             # legs paddle alternately
        P["leg_b"] = [wave(f, -12.0, 1, 0.0), 0, 0]
        P["head"] = [0, round(wave(f, 1.0, 2, 0.0)), 0]
    elif action == "attack":
        g_f = [0, -2, -4, -5, 3, 8, 10, 10, 11, 12, 11, 8, 5, 3, 1, 0][f]
        g_u = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -3, -6, -7, 2, 6, 8, 8, 8, 8, 7, 5, 3, 2, 1, 0][f]
        P["g"] = [g_f, g_u, lean]
        # lance A: drawn back (tip raised) -> stab (tip down/forward, pushed out) -> held -> back
        P["lanceA"] = [[0, 14, 24, 28, 6, -10, -14, -12, -6, -2, 0, 0, 0, 0, 0, 0][f],
                       [0, -2, -3, -4, 2, 7, 9, 7, 3, 1, 0, 0, 0, 0, 0, 0][f], 0]
        # lance B: follows a beat later (twin needle)
        P["lanceB"] = [[0, 10, 18, 22, 20, 14, 4, -6, -12, -14, -10, -6, -3, -1, 0, 0][f],
                       [0, -1, -2, -3, -3, -2, 0, 4, 8, 9, 6, 3, 1, 0, 0, 0][f], 0]
        P["abd"] = [[0, -4, -8, -10, -4, 6, 12, 14, 14, 12, 9, 6, 3, 1, 0, 0][f], 0, 0]
        P["wings"] = [[0, 10, 16, 18, 6, -8, -12, -10, -12, -10, -6, -3, -1, 0, 0, 0][f], 0, 0]
        P["leg_a"] = [[0, 4, 8, 8, 2, -6, -8, -8, -8, -6, -4, -2, -1, 0, 0, 0][f], 0, 0]
        P["leg_b"] = list(P["leg_a"])
        P["head"] = [0, [0, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0][f], 0]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 5 <= f <= 7:
            fx.append(("stabA", f))
        if 8 <= f <= 10:
            fx.append(("stabB", f))
        if f in (6, 7):
            fx.append(("impactA", f))
        if f in (9, 10):
            fx.append(("impactB", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-7.0 * env + jit, 2.0 * env, -10.0 * env]
        P["head"] = [0, -1.5 * env, 1.0 * env]
        P["wings"] = [16.0 * env, 0, 0]
        P["lanceA"] = [10.0 * env, 0, 0]
        P["lanceB"] = [10.0 * env, 0, 0]
        P["abd"] = [-10.0 * env, 0, 0]
        P["leg_a"] = [8.0 * env, 0, 0]
        P["leg_b"] = [8.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 6:
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
    """rotate a point (+ = clockwise on screen, y down) about piv."""
    t = math.radians(ang)
    dx, dy = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + dx * math.cos(t) - dy * math.sin(t), piv[1] + dx * math.sin(t) + dy * math.cos(t))


def tip_of(cell, piv):
    ys, xs = np.nonzero(cell[..., 3] > 0)
    i = int(np.argmax((xs - piv[0]) ** 2 + (ys - piv[1]) ** 2))
    return (float(xs[i]), float(ys[i]))


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    frontback = abs(fwx) < 0.5
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    legs = [n for n in R["draw"] if n.startswith("leg")]
    tips = {}
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "head":
            h = P.get("head", zero); df, du = h[1], h[2]
        elif n.startswith("wing"):
            rot = P.get("wings", zero)[0] * R["sgn"][n]
        elif n.startswith("lance"):
            roles = [r for r in ("lanceA", "lanceB") if R[r] == n]
            if len(roles) == 2 and action == "attack":
                # one lance part (side views): stab A then stab B with the same lance
                a, b = P["lanceA"], P["lanceB"]
                pa = a if f <= 6 else b
                rot, df = pa[0], pa[1]
            else:
                pa = P.get(roles[0], zero) if roles else zero
                rot, df = pa[0], pa[1]
            rot *= R["sgn"][n]
            if frontback:
                df *= 0.5
        elif n.startswith("leg"):
            key = "leg_a" if n == legs[0] else "leg_b"
            rot = P.get(key, zero)[0] * R["sgn"][n]
        elif n == "abdomen":
            a = P.get("abd", zero)
            rot = a[0] * (R["abd"] if R["abd"] else 0.35)
        dx, dy = df * fwx, df * fwy - du
        if n.startswith("lance"):
            t = rot_pt(tip_of(cells[n], piv[n]), rot, piv[n])
            tips[n] = (t[0] + dx, t[1] + dy)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if frontback:
        gf *= 0.4
    centre = (CELL_W // 2, FOOT_Y - 36)
    if abs(lean) > 0.3 and not frontback:
        canvas = rot4(canvas, lean * side, centre)
        tips = {k: rot_pt(v, lean * side, centre) for k, v in tips.items()}
    elif abs(lean) > 0.3 and action == "hurt":
        canvas = rot4(canvas, lean * 0.4, centre)       # front / back: a small sideways tilt
        tips = {k: rot_pt(v, lean * 0.4, centre) for k, v in tips.items()}
    gx, gy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gx, gy)
    tips = {k: (v[0] + gx, v[1] + gy) for k, v in tips.items()}
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, tips, (gx, gy)


# ---------------------------------------------------------------- effects (reserved colours only)
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


def star(a, cx, cy, r, rot):
    for j in range(8):
        ang = j * math.pi / 4 + rot
        rr = r if j % 2 == 0 else r * 0.55
        stroke(a, cx + math.cos(ang) * 2, cy + math.sin(ang) * 2, cx + math.cos(ang) * rr, cy + math.sin(ang) * rr,
               FX_HI if j % 2 == 0 else FX_LO)
    put(a, cx, cy, FX_HI)


def draw_fx(a, fx, tips, view, centre):
    R = RIG[view]
    fwx, fwy = R["fwd"]
    tA, tB = tips.get(R["lanceA"]), tips.get(R["lanceB"])
    for kind, k in fx:
        if kind == "charge":
            for t in {tA, tB} - {None}:
                for j in range(1 + k):
                    ang = j * 2.4 + k * 0.7
                    r = 8 - 2 * (k - 1)
                    stroke(a, t[0] + math.cos(ang) * (r + 3), t[1] + math.sin(ang) * (r + 3),
                           t[0] + math.cos(ang) * r, t[1] + math.sin(ang) * r, FX_LO)
                put(a, t[0], t[1], FX_HI)
        elif kind in ("stabA", "stabB"):
            t = tA if kind == "stabA" else tB
            if not t:
                continue
            # needle streaks trailing behind the tip along the thrust (tip -> pivot direction)
            L = {5: 9, 6: 14, 7: 10, 8: 9, 9: 14, 10: 10}[k]
            ux, uy = t[0] - centre[0], t[1] - centre[1]
            Ln = math.hypot(ux, uy) or 1.0
            ux, uy = ux / Ln, uy / Ln
            px, py = -uy, ux
            for o in (-3, 0, 3):
                stroke(a, t[0] - ux * (L + 4) + px * o, t[1] - uy * (L + 4) + py * o,
                       t[0] - ux * 4 + px * o, t[1] - uy * 4 + py * o, FX_HI if o == 0 else FX_LO)
        elif kind in ("impactA", "impactB"):
            t = tA if kind == "impactA" else tB
            if not t:
                continue
            ux, uy = t[0] - centre[0], t[1] - centre[1]
            Ln = math.hypot(ux, uy) or 1.0
            cx, cy = t[0] + ux / Ln * 3, t[1] + uy / Ln * 3
            star(a, cx, cy, 7 if k in (6, 9) else 10, 0.3 * k)
        elif kind == "stars":
            for j in range(3):
                ang = j * 2.1 + k * 0.8
                cx, cy = centre[0] + math.cos(ang) * 22, centre[1] - 8 + math.sin(ang) * 14
                put(a, cx, cy, FX_HI)
                put(a, cx + 1, cy, FX_LO); put(a, cx - 1, cy, FX_LO)
                put(a, cx, cy + 1, FX_LO); put(a, cx, cy - 1, FX_LO)


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
    ap.add_argument("--debug-rig", action="store_true", help="write _work/rig_<view>.png (parts coloured) and stop")
    args = ap.parse_args()

    work = OUT_DIR / "_work"
    h = state_hash()
    st_p = work / "state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    art_h = hashlib.sha1((ART / "out/keys_2x2.png").read_bytes() + (ART / "out/sw_single.png").read_bytes()).hexdigest()[:12]
    if args.force or st.get("hash") != h:
        keep_stills = (not args.force) and st.get("art") == art_h and "stills" in st.get("done", [])
        if work.exists():
            for p in work.iterdir():
                if keep_stills and p.name in ("stills_raw", "stills", "palette", "palette_lock"):
                    continue
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        st = {"hash": h, "art": art_h, "done": ["stills"] if keep_stills else []}
    work.mkdir(parents=True, exist_ok=True)
    save_st = lambda: st_p.write_text(json.dumps(st))

    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw)
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % FX_HI, "#%02x%02x%02x" % FX_LO,
                  "--balance", "--name", "beedrill-mega", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    save_st()
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    if args.debug_rig:
        cols = [(255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 200, 0), (200, 0, 255), (0, 220, 220),
                (255, 128, 0), (255, 255, 255), (128, 128, 128), (120, 60, 0)]
        for v in RIG:
            _, masks = build_rig(v, still[v])
            hh, ww = still[v].shape[:2]
            img = np.zeros((hh, ww * 2, 3), np.uint8)
            img[:, :ww] = still[v][..., :3]
            for i, n in enumerate(RIG[v]["label"]):
                img[:, ww:][masks[n]] = cols[i % len(cols)]
            Image.fromarray(img).resize((ww * 16, hh * 8), Image.NEAREST).save(work / f"rig_{v}.png")
            print(v, {n: cols[i % len(cols)] for i, n in enumerate(RIG[v]["label"])})
        return

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
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, tips, (gx, gy) = compose(v, cells, piv, act, f)
                centre = (CELL_W / 2 + gx, FOOT_Y - 36 + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, tips, v, centre)
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (2, 6, 9, 12)):
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
    meta = {"generator": "forge_gen1_0015_beedrill_mega_actions", "version": VERSION, "key": KEY, "dex": 15,
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
