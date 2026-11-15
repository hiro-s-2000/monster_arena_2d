#!/usr/bin/env python3
"""Mega Raichu X (0026_raichu-mega-x): 8-direction stills + 8-direction x 4-action sheet.

Source (2026-11-03, user instruction "Antigravity CLI でできる内容で"):
  The user supplied a front and a back render (art/raimX_front.webp, art/raimX_back.webp). Following the Antigravity
  skill monster-8dir-sheet-architect, the Antigravity CLI (`agy`, image-generator) drew the key views from them with
  a morphological description prompt (art/agy_prompt.md):
    art/agy_keys_2x2.png  : top-left S (front), top-right W (side, faces left), bottom-left NE (back 3/4), bottom-right N
    art/agy_sw_single.png : SW (3/4 front, faces down-left)
  The other three are mirrors: SE = SW, E = W, NW = NE.

Pipeline (md/AGENTS.md locked spec 2 -> monster-action-forge-architect):
  1. Key the white backdrop (border-connected only, so the white belly stays; small enclosed white = the ear
     swirls' holes, cleared), split, area-downscale every view to one game scale (S/W/NE/N share the grid's scale,
     SW is matched on body height).
  2. agent-sprite-forge palette_tool: build (30 colours + reserved spark colours) -> lock -> apply on the 5 stills.
  3. Anatomical part rig per view (polygons in sprite px): two lightning tails (root pivots at the rump), head with
     ears, two fists, torso, two feet. Biped boxer + two huge tails. Hidden seams get cushions (nearest own colour).
  4. Poses at 4x (nearest), sampled back to 1x (RotSprite-like: no new colours, crisp outline).
     Actions (16 frames each): idle (breath, tails / ears sway, cheek sparks), walk (alternating steps, fists and
     tails swing), attack (thunder punch: charge -> lunge punch -> electric burst -> recover), hurt (flash, recoil,
     static).
  5. QC: off-palette colours, frames touching the cell edge, agent-sprite-forge gait_loop.py on idle / walk.

Writes monsterSprite/gen1/0026_raichu/mega/raichu-mega-x/: front.png (SW), back.png (N) + -reverse / .gif,
8dir_stills.png (S SE E NE N NW W SW), agent-sprite-forge/ (sheet.png = 4 actions x 8 rows, preview.png, meta.json),
then syncs agent-sprite-forge/ to pokemon_arena_2d. Resumable: _work/state.json (hash of this script + art);
--force rebuilds.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_raichu_mega_x_actions.py
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
KEY = "0026_raichu-mega-x"
MEGA_DIR = ROOT / "monsterSprite/gen1/0026_raichu/mega/raichu-mega-x"
ART = MEGA_DIR / "art"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen1/0026_raichu/mega/raichu-mega-x/agent-sprite-forge")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "raichu_mega_x_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
GRID_SCALE = 0.258                 # sprite px per generated px (2x2 key sheet)
CELL_W, CELL_H, FOOT_Y = 232, 132, 122
UP = 4
SPARK_HI, SPARK_LO = (255, 252, 214), (255, 214, 40)
# mirrored rows
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label order: later polygons win. draw order: back to front. body_x = body centre (placed on the cell centre).
# fwd = screen direction of "forward" for the view (lunges, punches); cheeks = spark points (front views).
RIG = {
    "S": dict(
        poly={"tail_l": [(0, 0), (14, 0), (44, 36), (56, 48), (62, 60), (62, 76), (50, 76), (36, 72), (0, 62)],
              "tail_r": [(159, 0), (145, 0), (115, 36), (103, 48), (97, 60), (97, 76), (109, 76), (123, 72), (159, 62)],
              "body": [(62, 54), (98, 54), (102, 84), (58, 84)],
              "head": [(46, 8), (114, 8), (112, 40), (98, 56), (62, 56), (48, 40)],
              "foot_l": [(56, 83), (78, 83), (78, 92), (56, 92)],
              "foot_r": [(82, 83), (104, 83), (104, 92), (82, 92)],
              "fist_l": [(53, 43), (69, 43), (70, 61), (55, 62)],
              "fist_r": [(91, 43), (107, 43), (105, 62), (90, 61)]},
        label=["tail_l", "tail_r", "body", "head", "foot_l", "foot_r", "fist_l", "fist_r"],
        draw=["tail_l", "tail_r", "body", "foot_l", "foot_r", "head", "fist_l", "fist_r"],
        pivot={"tail_l": (62, 70), "tail_r": (97, 70), "head": (80, 56), "fist_l": (66, 54), "fist_r": (94, 54),
               "body": (80, 84), "foot_l": (67, 90), "foot_r": (93, 90)},
        punch="fist_r", body_x=80, fwd=(0.0, 1.0), cheeks=[(68, 50), (92, 50)], back=False),
    "SW": dict(
        poly={"tail_l": [(0, 0), (14, 0), (50, 36), (56, 52), (60, 66), (60, 80), (48, 80), (36, 74), (0, 62)],
              "tail_r": [(169, 0), (155, 0), (120, 36), (110, 52), (104, 66), (104, 80), (116, 80), (130, 74), (169, 62)],
              "body": [(66, 56), (100, 56), (104, 88), (64, 88)],
              "head": [(60, 14), (112, 14), (108, 44), (96, 56), (68, 56), (62, 40)],
              "foot_l": [(56, 86), (80, 86), (80, 96), (56, 96)],
              "foot_r": [(86, 86), (106, 86), (106, 96), (86, 96)],
              "fist_l": [(54, 50), (70, 50), (70, 67), (54, 67)],
              "fist_r": [(74, 52), (93, 52), (93, 71), (74, 71)]},
        label=["tail_l", "tail_r", "body", "head", "foot_l", "foot_r", "fist_l", "fist_r"],
        draw=["tail_l", "tail_r", "body", "foot_l", "foot_r", "head", "fist_l", "fist_r"],
        pivot={"tail_l": (60, 74), "tail_r": (104, 74), "head": (84, 56), "fist_l": (70, 58), "fist_r": (86, 60),
               "body": (84, 88), "foot_l": (68, 94), "foot_r": (96, 94)},
        punch="fist_r", body_x=84, fwd=(-0.9, 0.45), cheeks=[(90, 48)], back=False),
    "W": dict(
        poly={"tail_r": [(50, 46), (64, 38), (84, 0), (112, 0), (112, 70), (92, 76), (72, 82), (58, 84), (48, 70)],
              "body": [(14, 54), (50, 54), (52, 84), (16, 84)],
              "head": [(8, 12), (56, 12), (54, 42), (48, 54), (30, 58), (10, 58), (8, 40)],
              "foot_l": [(12, 82), (30, 82), (30, 93), (12, 93)],
              "foot_r": [(30, 82), (52, 82), (52, 93), (30, 93)],
              "fist_l": [(0, 50), (26, 50), (26, 68), (0, 68)]},
        label=["tail_r", "body", "head", "foot_l", "foot_r", "fist_l"],
        draw=["tail_r", "foot_r", "body", "foot_l", "head", "fist_l"],
        pivot={"tail_r": (50, 72), "head": (30, 58), "fist_l": (26, 60), "body": (32, 84), "foot_l": (21, 91), "foot_r": (41, 91)},
        punch="fist_l", body_x=32, fwd=(-1.0, 0.0), cheeks=[(24, 50)], back=False),
    "NE": dict(
        poly={"tail_l": [(0, 0), (14, 0), (44, 40), (56, 52), (64, 64), (68, 80), (54, 80), (36, 74), (0, 62)],
              "tail_r": [(156, 0), (142, 0), (112, 40), (104, 52), (96, 64), (92, 80), (106, 80), (122, 74), (156, 62)],
              "body": [(62, 48), (102, 48), (106, 86), (58, 86)],
              "head": [(54, 12), (110, 12), (106, 44), (94, 52), (68, 52), (56, 42)],
              "foot_l": [(58, 84), (80, 84), (80, 93), (58, 93)],
              "foot_r": [(82, 84), (110, 84), (110, 93), (82, 93)]},
        label=["tail_l", "tail_r", "body", "head", "foot_l", "foot_r"],
        draw=["foot_l", "foot_r", "body", "head", "tail_l", "tail_r"],
        pivot={"tail_l": (72, 72), "tail_r": (90, 72), "head": (82, 52), "body": (82, 86), "foot_l": (69, 91), "foot_r": (96, 91)},
        punch=None, body_x=82, fwd=(0.9, -0.45), cheeks=[], back=True),
    "N": dict(
        poly={"tail_l": [(0, 0), (14, 0), (44, 40), (54, 52), (62, 64), (66, 80), (52, 80), (36, 74), (0, 62)],
              "tail_r": [(156, 0), (142, 0), (112, 40), (102, 52), (94, 64), (90, 80), (104, 80), (120, 74), (156, 62)],
              "body": [(58, 54), (98, 54), (100, 86), (56, 86)],
              "head": [(44, 12), (112, 12), (108, 42), (94, 56), (62, 56), (48, 42)],
              "foot_l": [(56, 84), (78, 84), (78, 92), (56, 92)],
              "foot_r": [(80, 84), (102, 84), (102, 92), (80, 92)]},
        label=["tail_l", "tail_r", "body", "head", "foot_l", "foot_r"],
        draw=["foot_l", "foot_r", "body", "head", "tail_l", "tail_r"],
        pivot={"tail_l": (70, 72), "tail_r": (86, 72), "head": (78, 56), "body": (78, 86), "foot_l": (67, 90), "foot_r": (91, 90)},
        punch=None, body_x=78, fwd=(0.0, -1.0), cheeks=[], back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    # small enclosed white areas = holes of the ear swirls (eye highlights are smaller, the belly is larger)
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    holes = [i + 1 for i, s in enumerate(sizes) if 150 <= s <= 3000 and (i + 1) not in border]
    bg |= np.isin(lab, holes)
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 200)        # light fringe
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def make_stills(dst: Path):
    k = np.array(Image.open(ART / "agy_keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "agy_sw_single.png").convert("RGB"))}
    scale = {v: GRID_SCALE for v in src}
    scale["SW"] = GRID_SCALE * 305.0 / 615.0             # body height (ear tip -> feet): 305 px in the grid, 615 px in SW
    dst.mkdir(parents=True, exist_ok=True)
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        im = im.crop(im.getbbox())
        sm = im.resize((round(im.width * scale[v]), round(im.height * scale[v])), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        if v in ("N", "NE"):
            # back views have no white parts: the light swirl ornaments beside the ears read as eyes -> dark grey
            lt = (a[..., 3] > 0) & (a[..., :3].min(axis=2) > 170) & (np.ptp(a[..., :3].astype(int), axis=2) < 40)
            a[lt, :3] = (64, 60, 68)
        Image.fromarray(a, "RGBA").save(dst / f"{v}.png")


# ---------------------------------------------------------------- palette (agent-sprite-forge palette_tool)
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
    polys = {n: poly_mask(R["poly"][n], op.shape) for n in names}
    for i, n in enumerate(names, start=1):
        lab[polys[n] & op] = i
    if R["back"]:      # the tails' dark roots run over the lower back: dark grey pixels there stay tail
        r, g, b = [still[..., c].astype(int) for c in range(3)]
        mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
        dark = op & (mx < 120) & (mx - mn < 30)
        yy, xx = np.mgrid[0:op.shape[0], 0:op.shape[1]]
        low = yy > 54
        lab[dark & low & (xx < R["body_x"])] = names.index("tail_l") + 1
        lab[dark & low & (xx >= R["body_x"])] = names.index("tail_r") + 1
    un = op & (lab == 0)
    if un.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[un] = lab[iy[un], ix[un]]
    # a fist is only its brown glove (+1px outline); chest / cheek pixels inside the fist polygon go to the
    # nearest other part, so they do not travel with the punch
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    brown = op & (r > 80) & (r < 215) & (g > 30) & (g < 140) & (b < 100) & (r - g > 30) & (r - b > 50)
    fist_ids = [names.index(n) + 1 for n in names if n.startswith("fist")]
    if fist_ids:
        isfist = np.isin(lab, fist_ids)
        keep = isfist & ndi.binary_dilation(brown & isfist, iterations=1)
        drop = isfist & ~keep
        if drop.any():
            other = op & ~isfist
            _, (iy, ix) = ndi.distance_transform_edt(~other, return_indices=True)
            lab[drop] = lab[iy[drop], ix[drop]]
    masks = {n: lab == (i + 1) for i, n in enumerate(names)}
    layers = {}
    order = R["draw"]
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 5 if (n.startswith("tail") or n == "body") else 3
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n == "body" and m.any():
            # the body also carries an underlay for everything drawn above it inside the closed body + head outline
            # (fists over the belly or the chin): filled from body / head colours, so no hole opens when a fist moves
            bh = m | masks.get("head", np.zeros_like(m))
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 8), iterations=8))[8:-8, 8:-8]
            seam |= above & ~m & inside & ~masks.get("head", np.zeros_like(m))
            src_m = bh
            # dark shading right next to the fists (the shadow under a raised fist) would read as a hole once the
            # fist moves away: repaint it with the body colour too
            fists = np.zeros_like(m)
            for fn in masks:
                if fn.startswith("fist"):
                    fists |= masks[fn]
            if fists.any():
                r0, g0, b0 = [still[..., c].astype(int) for c in range(3)]
                darkb = m & (np.maximum(np.maximum(r0, g0), b0) < 70) & (ndi.distance_transform_edt(~fists) <= 4) & inside
                darkb &= ndi.binary_erosion(bh, iterations=1)       # keep the outer outline
                seam |= darkb
        if seam.any() and m.any():
            # fill with the nearest non-outline colour of the part (body: body or head)
            r, g, b = [still[..., c].astype(int) for c in range(3)]
            fillable = src_m & (np.maximum(np.maximum(r, g), b) > 70)
            if not fillable.any():
                fillable = src_m
            _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
            sy, sx = np.nonzero(seam)
            lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
            lyr[sy, sx, 3] = 255
        layers[n] = lyr
    return layers, masks


def place_layer(layer, view):
    """Put a view-sized layer on the cell: body centre on the cell centre, feet on FOOT_Y."""
    h, w = layer.shape[:2]
    ys = np.nonzero(layer[..., 3].any(axis=1))[0]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - RIG[view]["body_x"]
    oy = FOOT_Y - (h - 1)
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(h, CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
def pose(action, f):
    """Per-part [rot deg, fwd px, up px] (+ rot raises a tail / tilts the head forward) and global
    [forward px, up px, lean deg]. Effects list."""
    t = f / N
    ph = 2 * math.pi * t
    s1, c1 = math.sin(ph), math.cos(ph)
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        bob = 1.0 * s1
        P["body"] = [0, 0, -max(0.0, bob)]
        P["head"] = [0, 0, -round(1.0 * math.sin(ph - 0.5) + 0.4)]
        P["fists"] = [0, 0, -round(1.0 * math.sin(ph - 0.3))]
        P["tail_l"] = [3.0 * math.sin(ph - 0.4), 0, 0]
        P["tail_r"] = [3.0 * math.sin(ph - 1.2), 0, 0]
        if f in (5, 6, 13, 14):
            fx.append(("cheek", 1 if f in (5, 13) else 2))
    elif action == "walk":
        up = 2.0 * abs(math.sin(ph))
        P["body"] = [0, 0, up]
        P["head"] = [0, 0, up + (1 if math.sin(2 * ph - 0.6) > 0.3 else 0)]
        P["foot_a"] = [0, 3.0 * c1, 3.0 * max(0.0, s1)]
        P["foot_b"] = [0, -3.0 * c1, 3.0 * max(0.0, -s1)]
        P["fist_a"] = [0, -2.0 * c1, up]
        P["fist_b"] = [0, 2.0 * c1, up]
        P["tail_l"] = [5.0 * math.sin(2 * ph - 0.8), 0, 0]
        P["tail_r"] = [-5.0 * math.sin(2 * ph - 1.4), 0, 0]
        P["g"] = [0, 0, 2.0]
    elif action == "attack":
        g_f = [0, -2, -3, -4, 2, 6, 9, 10, 10, 9, 7, 5, 3, 2, 1, 0][f]
        g_u = [0, -1, -2, -2, 1, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -2, -4, -5, 3, 6, 8, 8, 7, 6, 4, 3, 2, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [0, 0, [0, -1, -1, -1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0][f]]
        P["punch"] = [0, [0, -3, -5, -6, 4, 10, 14, 15, 15, 14, 10, 6, 3, 1, 0, 0][f], [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0][f]]
        P["guard"] = [0, [0, 1, 2, 2, -1, -2, -3, -3, -3, -2, -2, -1, -1, 0, 0, 0][f], 0]
        P["foot_a"] = [0, [0, 0, 0, 0, 2, 4, 5, 5, 5, 5, 4, 3, 2, 1, 0, 0][f], 0]
        fl = [0, 4, 8, 10, -2, -5, -7, -7, -6, -5, -3, -2, -1, -1, 0, 0][f]
        P["tail_l"] = [fl, 0, 0]
        P["tail_r"] = [fl, 0, 0]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 5 <= f <= 9:
            fx.append(("bolt", f))
        if f in (6, 7, 8):
            fx.append(("burst", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-6.0 * env + jit, 1.0 * env, -6.0 * env]
        P["head"] = [0, -1.0 * env, 1.5 * env]
        P["fists"] = [0, -1.0 * env, 2.0 * env]
        P["tail_l"] = [-7.0 * env, 0, 0]
        P["tail_r"] = [-7.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 7:
            fx.append(("static", f))
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
    """cells: part -> layer placed on the cell (1x). Returns 1x RGBA cell, effects, part offsets."""
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0               # screen side the creature faces (front/back views: +1)
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    feet_names = [n for n in R["draw"] if n.startswith("foot")]
    fist_names = [n for n in R["draw"] if n.startswith("fist")]
    offs = {}
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        b = P.get("body", zero)
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero); df, du = h[1], h[2]
        elif n.startswith("fist"):
            key = None
            if action == "attack":
                key = "punch" if n == R["punch"] else "guard"
            elif action == "walk":
                key = "fist_a" if n == fist_names[0] else "fist_b"
            a = P.get(key, P.get("fists", zero)) if key else P.get("fists", zero)
            df, du = a[1], a[2]
        elif n.startswith("foot"):
            key = "foot_a" if n == feet_names[0] else "foot_b"
            ft = P.get(key, zero); df, du = ft[1], ft[2]
        elif n.startswith("tail"):
            tt = P.get(n, zero)
            rot = tt[0] * (1.0 if n == "tail_l" else -1.0)    # + raises the tail (left tail turns clockwise)
            du = b[2] * 0.6
        dx, dy = df * fwx, df * fwy - du
        if action == "walk" and n.startswith("foot") and abs(fwx) < 0.5:
            dx = 0.0                                          # front / back: steps are vertical only
        offs[n] = (dx, dy)
        lyr = cells[n]
        p = piv[n]
        l4 = rot4(up4(lyr), rot, p)
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4                                    # toward / away from the camera: a short step, not a slide
    if abs(lean) > 0.3 and abs(fwx) > 0.3:
        feet = (CELL_W // 2, FOOT_Y)
        canvas = rot4(canvas, lean * side, feet)
    canvas = shift4(canvas, gf * fwx, gf * fwy - gu)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, offs, (gf * fwx, gf * fwy - gu)


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def zigzag(a, x0, y0, x1, y1, seed):
    rng = np.random.default_rng(seed)
    n = max(3, int(math.hypot(x1 - x0, y1 - y0) / 4))
    pts = [(x0, y0)]
    for i in range(1, n):
        u = i / n
        nx, ny = -(y1 - y0), (x1 - x0)
        L = math.hypot(nx, ny) or 1
        o = rng.uniform(-2.5, 2.5)
        pts.append((x0 + (x1 - x0) * u + nx / L * o, y0 + (y1 - y0) * u + ny / L * o))
    pts.append((x1, y1))
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        k = int(max(abs(bx - ax), abs(by - ay))) + 1
        for j in range(k + 1):
            u = j / k
            x, y = ax + (bx - ax) * u, ay + (by - ay) * u
            put(a, x, y, SPARK_HI)
            put(a, x + 1, y, SPARK_LO)


def draw_fx(a, fx, fist, fwd, centre, cheeks):
    fwx, fwy = fwd
    for kind, k in fx:
        if kind == "cheek":
            for cx, cy in cheeks:
                for j in range(k):
                    zigzag(a, cx, cy, cx + (5 + 3 * j) * (1 if cx >= centre[0] else -1), cy - 3 - 2 * j, 11 + k + j)
        elif kind == "charge" and fist:
            for j in range(2 + k):
                ang = j * 2.1 + k
                r = 6 + 2 * k
                zigzag(a, fist[0] + math.cos(ang) * r, fist[1] + math.sin(ang) * r,
                       fist[0] + math.cos(ang + 0.6) * (r - 5), fist[1] + math.sin(ang + 0.6) * (r - 5), 20 + j + k)
        elif kind == "bolt" and fist:
            L = 8 + (k - 5) * 3
            px, py = -fwy, fwx
            for j, o in enumerate((-5, 0, 5)):
                zigzag(a, fist[0] - fwx * 3 + px * o * 0.3, fist[1] - fwy * 3 + py * o * 0.3,
                       fist[0] + fwx * L + px * o, fist[1] + fwy * L + py * o, 30 + j + k)
        elif kind == "burst" and fist:
            r = {6: 7, 7: 11, 8: 9}[k]
            for j in range(8):
                ang = j * math.pi / 4 + k * 0.3
                zigzag(a, fist[0] + math.cos(ang) * 2, fist[1] + math.sin(ang) * 2,
                       fist[0] + math.cos(ang) * r, fist[1] + math.sin(ang) * r, 40 + j + k)
        elif kind == "static":
            for j in range(3):
                ang = j * 2.2 + k * 0.9
                cx, cy = centre[0] + math.cos(ang) * 16, centre[1] + math.sin(ang) * 12
                zigzag(a, cx, cy, cx + math.cos(ang + 1.3) * 6, cy + math.sin(ang + 1.3) * 6, 60 + j + k)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = SPARK_HI
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("agy_keys_2x2.png", "agy_sw_single.png"):
        h.update((ART / n).read_bytes())
    return h.hexdigest()[:12]


def save_gif(im, path):
    p = im.convert("RGBA")
    al = np.array(p.getchannel("A"))
    q = p.convert("RGB").quantize(colors=255, method=Image.MEDIANCUT, dither=Image.Dither.NONE)
    pal = q.getpalette()[:255 * 3] + [0, 0, 0]
    px = np.array(q)
    px[al < 128] = 255
    g = Image.fromarray(px.astype(np.uint8), "P")
    g.putpalette(pal)
    g.save(path, transparency=255, disposal=2, loop=0, duration=100)


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

    # 1-2. stills + palette
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw)
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % SPARK_HI, "#%02x%02x%02x" % SPARK_LO,
                  "--balance", "--name", "raichu-mega-x", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    # 3. rigs, placed on the cell
    rig = {}
    for v in RIG:
        layers, masks = build_rig(v, still[v])
        cells, piv = {}, {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v)
        for n, p in RIG[v]["pivot"].items():
            piv[n] = (p[0] + ox, p[1] + oy)
        rig[v] = (cells, piv, (ox, oy))

    # 4. frames (cached per view / action)
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
                a, fx, offs, (gx, gy) = compose(v, cells, piv, act, f)
                fist = None
                if R["punch"]:
                    m = cells[R["punch"]][..., 3] > 0
                    ys, xs = np.nonzero(m)
                    dx, dy = offs[R["punch"]]
                    fist = (xs.mean() + dx + gx + R["fwd"][0] * 4, ys.mean() + dy + gy + R["fwd"][1] * 4)
                elif act == "attack":
                    fist = (CELL_W / 2 + gx + R["fwd"][0] * 18, FOOT_Y - 40 + gy + R["fwd"][1] * 10)
                centre = (CELL_W / 2 + gx, FOOT_Y - 30 + gy)
                cheeks = [(c[0] + ox + gx, c[1] + oy + gy + offs.get("head", (0, 0))[1]) for c in R["cheeks"]]
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, fist, R["fwd"], centre, cheeks)
                Image.fromarray(a, "RGBA").save(od / f"{f:02d}.png")
            st["done"].append(key); save_st()
            print("frames", key, flush=True)

    # assemble
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
        for si, fc in enumerate((0, 4, 8, 12)):
            src = sheets[act][:, fc * CELL_W:(fc + 1) * CELL_W]
            mm = src[..., 3] > 0
            pv[:, (ai * 4 + si) * CELL_W:(ai * 4 + si + 1) * CELL_W][mm] = src[mm]
    Image.fromarray(pv, "RGBA").save(OUT_DIR / "preview.png")

    # stills: 8-direction strip + front / back
    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(MEGA_DIR / "8dir_stills.png")
    def crop_sq(a):
        ys, xs = np.nonzero(a[..., 3] > 0)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        S = max(x1 - x0, y1 - y0) + 7
        c = np.zeros((S, S, 4), np.uint8)
        ox, oy = (S - (x1 - x0 + 1)) // 2, S - (y1 - y0 + 1) - 3
        c[oy:oy + y1 - y0 + 1, ox:ox + x1 - x0 + 1] = a[y0:y1 + 1, x0:x1 + 1]
        return Image.fromarray(c, "RGBA")
    fr, bk = crop_sq(idle0[DIRS.index("SW")]), crop_sq(idle0[DIRS.index("N")])
    fr.save(MEGA_DIR / "front.png"); bk.save(MEGA_DIR / "back.png")
    save_gif(fr, MEGA_DIR / "front.gif"); save_gif(fr.transpose(Image.FLIP_LEFT_RIGHT), MEGA_DIR / "front-reverse.gif")
    save_gif(bk, MEGA_DIR / "back.gif"); save_gif(bk.transpose(Image.FLIP_LEFT_RIGHT), MEGA_DIR / "back-reverse.gif")

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
    clip = {}
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(N):
                c = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] > 0
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off), "edge_touching_frames": clip}
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
    meta = {"generator": "forge_raichu_mega_x_actions", "version": VERSION, "key": KEY, "dex": 26,
            "source_art": ["art/agy_keys_2x2.png", "art/agy_sw_single.png"], "made_from": ["art/raimX_front.webp", "art/raimX_back.webp"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(bounds[3]), "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else (v.get("policy") if "policy" in v else len(v))) for k, v in report.items()}, ensure_ascii=False))

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
