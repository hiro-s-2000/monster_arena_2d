#!/usr/bin/env python3
"""0017_pidgeotto (normal): 8-direction x 4-action sheet (gen1 forge batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py (structure only; rig, colours and attack are this form's own):
  1. The official front.png / back.png (x8 nearest -> art/ref_front.png, art/ref_back.png) were given to the
     Antigravity CLI (agy, image generation) with art/agy_prompt.md (no proper nouns, exact part counts):
       art/out/keys_2x2.png  : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
       art/out/sw_single.png : SW (3/4 front, faces down-left)
     The other three are mirrors: SE = SW, E = W, NW = NE. (1st try was correct.)
  2. Key the white backdrop (border-connected white + big / pure-white enclosed gaps between feathers), area
     downscale. S is scaled so its height = the official front.png opaque height (80 px); W / NE / N share that
     scale; SW (same pose as the official front) is also matched to 80 px.
  3. agent-sprite-forge palette_tool: build (30 colours + 2 reserved wind-FX colours) -> lock -> apply.
  4. Anatomical rig per view (counted on the official sprite): a hovering bird of prey with TWO feathered wings
     (pivot at the shoulder), ONE head with the red crest and hooked beak, the body, TWO pink taloned legs hanging
     below (one hidden in NE), ONE red / cream striped tail fan. Posed at 4x (nearest), back to 1x.
     It flies: no standing pose, the whole bird hovers (bobs) with the wing beat; legs never grow / vanish.
     Actions (16 frames):
       idle   : hover - two wing beats per loop, the body rises on each down-stroke, tail and talons sway.
       walk   : forward flight - faster, deeper wing beats, the body dips forward, legs trail back, tail steers.
       attack : wing gust + talon strike - rises and raises both wings high (wind gathers) -> dives forward, wings
                sweep down hard and both talon feet thrust forward -> three claw slash marks + gust arcs at the
                talons -> recover to the hover.
       hurt   : white flash, knocked back and up, wings flare, loose feathers (own colours), recover.
  5. QC: off-palette, edge-touch, still diff (frame 0 of every action = the rest pose), gait_loop.py idle / walk.

The official front / back png / gif are never touched. Writes 8dir_stills.png and agent-sprite-forge/ in the form
folder and syncs agent-sprite-forge/ (files only, not _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0017_pidgeotto_actions.py [--force]
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
KEY = "0017_pidgeotto"
REL = "monsterSprite/gen1/0017_pidgeotto/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0017_pidgeotto_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
OFFICIAL_H = 80                      # opaque height of the official front.png
GRID_H_S = 362                       # S full height (wing tips -> tail fan) in the generated grid (px)
SW_H = 660                           # SW full height in sw_single.png (px)
CELL_W, CELL_H, FOOT_Y = 184, 128, 114
UP = 4
FX_HI, FX_LO = (246, 252, 255), (156, 206, 238)      # reserved: wind / slash light
FEATHER = None                       # light cream palette colour (set in main) for loose feathers
FEATHER_D = None                     # darker ochre palette colour
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label: later polygons win. draw: back to front. sgn: rotation sign that RAISES a wing tip / the tail fan.
# fwd = screen direction of forward (dives, talon thrusts), talon = point between the talons (slash FX).
RIG = {
    "S": dict(
        poly={"wing_l": [(0, 0), (40, 0), (43, 30), (42, 40), (38, 54), (0, 54)],
              "wing_r": [(107, 0), (67, 0), (64, 30), (65, 40), (69, 54), (107, 54)],
              "body": [(38, 38), (70, 38), (70, 66), (38, 66)],
              "tail": [(40, 64), (68, 64), (68, 80), (40, 80)],
              "head": [(41, 18), (66, 18), (66, 44), (41, 44)],
              "foot_l": [(25, 55), (44, 55), (44, 75), (25, 75)],
              "foot_r": [(63, 55), (82, 55), (82, 75), (63, 75)]},
        label=["wing_l", "wing_r", "body", "tail", "head", "foot_l", "foot_r"],
        draw=["wing_l", "wing_r", "tail", "body", "foot_l", "foot_r", "head"],
        pivot={"wing_l": (42, 40), "wing_r": (65, 40), "head": (53, 44), "tail": (53, 64), "body": (53, 66),
               "foot_l": (43, 58), "foot_r": (64, 58)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": 1},
        body_x=53, fwd=(0.0, 1.0), back=False),
    "SW": dict(
        poly={"wing_l": [(0, 0), (26, 0), (28, 24), (22, 34), (20, 54), (0, 54)],
              "wing_r": [(40, 0), (68, 0), (68, 52), (50, 52), (44, 46), (40, 30)],
              "body": [(24, 42), (46, 40), (50, 66), (36, 72), (26, 62)],
              "tail": [(30, 66), (54, 60), (54, 80), (30, 80)],
              "head": [(19, 23), (42, 23), (40, 44), (28, 50), (18, 48)],
              "foot_l": [(6, 54), (30, 54), (30, 71), (6, 71)],
              "foot_r": [(31, 53), (44, 53), (44, 71), (31, 71)]},
        label=["wing_l", "wing_r", "body", "tail", "head", "foot_l", "foot_r"],
        draw=["wing_r", "wing_l", "tail", "body", "head", "foot_r", "foot_l"],
        pivot={"wing_l": (25, 38), "wing_r": (44, 40), "head": (32, 46), "body": (36, 72), "tail": (40, 64),
               "foot_l": (29, 58), "foot_r": (37, 57)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": 1},
        body_x=34, fwd=(-0.9, 0.45), back=False),
    "W": dict(
        poly={"wing_far": [(0, 0), (28, 0), (30, 20), (26, 31), (14, 31), (0, 20)],
              "wing_near": [(36, 0), (75, 0), (75, 57), (52, 57), (44, 49), (38, 38)],
              "body": [(16, 36), (50, 36), (56, 60), (46, 70), (20, 70), (14, 52)],
              "tail": [(48, 58), (75, 58), (75, 78), (50, 78), (46, 66)],
              "head": [(4, 25), (36, 25), (38, 34), (34, 48), (20, 52), (4, 48)],
              "foot_a": [(6, 60), (24, 60), (26, 79), (6, 79)],
              "foot_b": [(24, 62), (44, 62), (44, 82), (24, 82)]},
        label=["wing_far", "wing_near", "body", "tail", "head", "foot_a", "foot_b"],
        draw=["wing_far", "tail", "foot_b", "body", "foot_a", "wing_near", "head"],
        pivot={"wing_far": (27, 31), "wing_near": (42, 44), "head": (24, 48), "tail": (50, 66), "body": (34, 70),
               "foot_a": (24, 62), "foot_b": (34, 64)},
        sgn={"wing_far": 1, "wing_near": -1, "tail": -1},
        body_x=32, fwd=(-1.0, 0.0), back=False),
    "NE": dict(
        poly={"wing_l": [(0, 0), (36, 0), (40, 22), (40, 40), (34, 54), (0, 54)],
              "wing_r": [(94, 0), (70, 0), (62, 28), (58, 40), (62, 52), (94, 52)],
              "body": [(30, 34), (64, 34), (64, 72), (40, 72), (30, 60)],
              "tail": [(44, 64), (72, 64), (72, 83), (44, 83)],
              "head": [(36, 14), (60, 14), (60, 37), (36, 37)],
              "foot_l": [(23, 56), (39, 56), (39, 72), (23, 72)]},
        label=["wing_l", "wing_r", "body", "tail", "head", "foot_l"],
        draw=["foot_l", "body", "head", "wing_l", "wing_r", "tail"],
        pivot={"wing_l": (38, 40), "wing_r": (60, 38), "head": (48, 36), "tail": (56, 66), "foot_l": (38, 58),
               "body": (48, 72)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": -1},
        body_x=48, fwd=(0.9, -0.45), back=True),
    "N": dict(
        poly={"wing_l": [(0, 0), (42, 0), (44, 34), (42, 46), (34, 54), (0, 54)],
              "wing_r": [(107, 0), (65, 0), (63, 34), (65, 46), (73, 54), (107, 54)],
              "body": [(38, 34), (70, 34), (70, 68), (38, 68)],
              "tail": [(40, 64), (68, 64), (68, 83), (40, 83)],
              "head": [(42, 14), (65, 14), (65, 37), (42, 37)],
              "foot_l": [(31, 53), (44, 53), (44, 70), (31, 70)],
              "foot_r": [(63, 53), (77, 53), (77, 70), (63, 70)]},
        label=["wing_l", "wing_r", "body", "tail", "head", "foot_l", "foot_r"],
        draw=["foot_l", "foot_r", "body", "head", "wing_l", "wing_r", "tail"],
        pivot={"wing_l": (44, 40), "wing_r": (63, 40), "head": (53, 37), "body": (53, 68), "tail": (53, 64),
               "foot_l": (42, 55), "foot_r": (65, 55)},
        sgn={"wing_l": 1, "wing_r": -1, "tail": -1},
        body_x=53, fwd=(0.0, -1.0), back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    mn = a.min(axis=2)
    white = mn > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    means = ndi.mean(mn, lab, range(1, n + 1))
    # enclosed gaps (between feathers / legs) are big or pure white; talon tips / eye highlights are small
    holes = [i + 1 for i in range(n) if (i + 1) not in border and (sizes[i] >= 1500 or (sizes[i] >= 150 and means[i] > 252))]
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
    for q in (p if isinstance(p[0][0], (list, tuple)) else [p]):
        ImageDraw.Draw(m).polygon([tuple(v) for v in q], fill=255)
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
    # a pink leg is only its pink / grey-talon pixels (+1 px outline); cream belly pixels inside a foot polygon go
    # to the nearest other part so they do not travel with the leg
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    pinkish = op & (((r > 150) & (b > 140) & (g < 190)) | ((np.abs(r - b) < 25) & (np.abs(r - g) < 25) & (r > 100)))
    foot_ids = [names.index(n) + 1 for n in names if n.startswith("foot")]
    isfoot = np.isin(lab, foot_ids)
    keep = isfoot & ndi.binary_dilation(pinkish & isfoot, iterations=1)
    drop = isfoot & ~keep
    if drop.any():
        other = op & ~isfoot
        _, (iy, ix) = ndi.distance_transform_edt(~other, return_indices=True)
        lab[drop] = lab[iy[drop], ix[drop]]
    # every part one piece: stray islands go to the nearest other part
    for i, n in enumerate(names, start=1):
        m = lab == i
        cl, cn = ndi.label(m)
        if cn > 1:
            sz = ndi.sum(np.ones_like(cl), cl, range(1, cn + 1))
            keepc = {int(np.argmax(sz)) + 1}
            stray = m & ~np.isin(cl, list(keepc))
            other = op & (lab != i)
            if other.any() and stray.any():
                _, (iy, ix) = ndi.distance_transform_edt(~other, return_indices=True)
                lab[stray] = lab[iy[stray], ix[stray]]
    masks = {n: lab == (i + 1) for i, n in enumerate(names)}
    layers = {}
    order = R["draw"]
    bright = np.maximum(np.maximum(r, g), b)
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 5 if (n.startswith("wing") or n in ("body", "tail")) else 3
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n == "body" and m.any():
            # underlay inside the body + head outline, so no hole opens when a leg / the head moves
            bh = m | masks.get("head", np.zeros_like(m))
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 8), iterations=8))[8:-8, 8:-8]
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
    """Per-part [rot deg (+ raises a wing tip / the tail fan), fwd px, up px]; global [fwd px, up px, lean deg]."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        # two wing beats; + = up-stroke. Body rises while the wings push down (lags the beat).
        P["wings"] = [wave(f, 12.0, 2, 0.0), 0, 0]
        P["g"] = [0.0, round(wave(f, -1.6, 2, -0.9)), 0.0]
        P["head"] = [0, 0, round(wave(f, 0.8, 2, -1.8))]
        P["tail"] = [wave(f, 4.0, 1, 0.0), 0, 0]
        P["foot_a"] = [0, round(wave(f, 1.0, 1, 0.0)), round(wave(f, 1.0, 2, -0.5))]
        P["foot_b"] = [0, round(wave(f, -1.0, 1, 0.0)), round(wave(f, 1.0, 2, -0.5))]
    elif action == "walk":
        # forward flight: deep fast beats, surge up on each down-stroke, dip forward, legs trail back
        P["wings"] = [wave(f, 22.0, 2, 0.0), 0, 0]
        P["g"] = [round(wave(f, 1.5, 2, -1.0)), round(wave(f, -2.5, 2, -0.9)), wave(f, 3.0, 2, -math.pi / 2)]
        P["head"] = [0, round(wave(f, 1.0, 2, -1.4)), 0]
        P["tail"] = [wave(f, 6.0, 1, 0.0), 0, 0]
        P["foot_a"] = [0, round(wave(f, -2.0, 2, -math.pi / 2)), round(wave(f, 1.0, 2, -math.pi / 2))]
        P["foot_b"] = [0, round(wave(f, -2.0, 2, -math.pi / 2 + 0.8)), round(wave(f, 1.0, 2, -math.pi / 2 + 0.8))]
        if f in (5, 13):
            fx.append(("trail", f))
    elif action == "attack":
        # rise + wings high (1-3) -> dive, wings sweep down, talons thrust (4-6) -> slash + gust (6-9) -> recover
        g_f = [0, -1, -2, -3, 2, 7, 10, 11, 11, 10, 8, 6, 4, 2, 1, 0][f]
        g_u = [0, 2, 4, 5, 3, 0, -1, -1, -1, 0, 0, 1, 1, 1, 0, 0][f]
        lean = [0, -3, -6, -7, 2, 8, 11, 11, 10, 8, 6, 4, 2, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["wings"] = [[0, 10, 20, 26, 10, -14, -24, -24, -20, -14, -8, -4, -1, 0, 0, 0][f], 0, 0]
        P["head"] = [0, [0, -1, -1, -1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0][f], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        thr = [0, -1, -2, -2, 2, 6, 8, 8, 7, 6, 4, 3, 2, 1, 0, 0][f]
        dn = [0, 1, 2, 2, 0, -1, -2, -2, -2, -1, -1, 0, 0, 0, 0, 0][f]
        P["foot_a"] = [0, thr, dn]
        P["foot_b"] = [0, thr, dn]
        P["tail"] = [[0, -4, -8, -10, -4, 6, 10, 10, 8, 6, 4, 2, 1, 0, 0, 0][f], 0, 0]
        if 1 <= f <= 3:
            fx.append(("gather", f))
        if 6 <= f <= 9:
            fx.append(("slash", f))
        if 5 <= f <= 9:
            fx.append(("gust", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-6.0 * env + jit, 3.0 * env, -9.0 * env]
        P["wings"] = [18.0 * env, 0, 0]
        P["head"] = [0, -1.5 * env, 1.0 * env]
        P["foot_a"] = [0, 2.0 * env, -1.0 * env]
        P["foot_b"] = [0, 2.0 * env, -1.0 * env]
        P["tail"] = [-8.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 9:
            fx.append(("feathers", f))
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


def compose(view, cells, piv, action, f):
    """Returns the 1x frame, effects, part offsets and the global shift."""
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    feet_names = [n for n in R["label"] if n.startswith("foot")]
    offs = {}
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "head":
            h = P.get("head", zero); df, du = h[1], h[2]
        elif n.startswith("foot"):
            key = "foot_a" if n == feet_names[0] else "foot_b"
            ft = P.get(key, zero); df, du = ft[1], ft[2]
        elif n.startswith("wing"):
            w = P.get("wings", zero)
            rot = w[0] * R["sgn"][n]
        elif n == "tail":
            t = P.get("tail", zero)
            rot = t[0] * R["sgn"]["tail"]
        dx, dy = df * fwx, df * fwy - du
        if n.startswith("foot"):
            if abs(fwx) < 0.5:
                dx, dy = 0.0, df * fwy * 0.5 - du         # toward / away: talons reach down / up a little
            else:
                # legs swing about the hip (never detach): +df = talons forward
                rot = -side * df * 4.0
                dx, dy = df * fwx * 0.25, df * fwy * 0.25 - du
        offs[n] = (dx, dy)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4
    if abs(lean) > 0.3 and abs(fwx) > 0.3:
        canvas = rot4(canvas, lean * side, (CELL_W // 2, FOOT_Y - 30))
    gx, gy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, offs, (gx, gy)


# ---------------------------------------------------------------- effects (reserved wind colours + own feathers)
def put(a, x, y, col, behind=False):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        if behind and a[y, x, 3] > 0 and tuple(a[y, x, :3]) not in (FX_HI, FX_LO):
            return
        a[y, x, :3] = col
        a[y, x, 3] = 255


def arc(a, cx, cy, r, a0, a1, col, behind=False):
    steps = max(4, int(abs(a1 - a0) * r))
    for j in range(steps + 1):
        t = a0 + (a1 - a0) * j / steps
        put(a, cx + math.cos(t) * r, cy + math.sin(t) * r, col, behind)


def draw_fx(a, fx, talon, fwd, centre, back):
    fwx, fwy = fwd
    base = math.atan2(fwy, fwx)
    for kind, k in fx:
        if kind == "gather":
            # wind curls drawn in toward the raised wings
            for j in range(2 + k):
                ang = j * 2.4 + k * 0.7
                r = 30 - 4 * k
                cx, cy = centre[0] + math.cos(ang) * r, centre[1] - 14 + math.sin(ang) * r * 0.6
                arc(a, cx, cy, 3, ang, ang + 2.2, FX_LO if j % 2 else FX_HI, True)
        elif kind == "trail":
            # speed streaks behind the flying bird
            for j in range(3):
                d0 = 26 + 3 * j
                px, py = -fwy, fwx
                o = (j - 1) * 9
                for s in range(6):
                    put(a, centre[0] - fwx * (d0 + s) + px * o, centre[1] - fwy * (d0 + s) + py * o, FX_LO, True)
        elif kind == "gust":
            # crescent gust arcs pushed forward by the wing down-stroke
            d = {5: 0, 6: 5, 7: 10, 8: 15, 9: 20}[k]
            if abs(fwx) >= 0.5:
                # side views: crescents rolling forward ahead of the beak (never over the body)
                for j, (rr, col) in enumerate(((12, FX_HI), (8, FX_LO), (16, FX_LO))):
                    if k == 9 and j == 2:
                        continue
                    o = 30 + d + 3 * j
                    cx, cy = centre[0] + fwx * (o - rr), centre[1] - 4 + fwy * (o - rr)
                    arc(a, cx, cy, rr, base - 0.7, base + 0.7, col, True)
            else:
                # toward / away from the camera: the down-stroke throws gust crescents out past both wing tips
                for sd in (-1, 1):
                    for j, (rr, col) in enumerate(((10, FX_HI), (14, FX_LO))):
                        if k == 9 and j == 0:
                            continue
                        cx = centre[0] + sd * (34 + d)
                        cy = centre[1] - 4 + d * 0.4
                        b0 = 0.0 if sd > 0 else math.pi
                        arc(a, cx - sd * rr, cy, rr, b0 - 0.8, b0 + 0.8, col, True)
        elif kind == "slash" and talon:
            # three parallel claw marks raked across the target in front of the talons
            L = {6: 8, 7: 16, 8: 18, 9: 14}[k]
            # three diagonal rake marks (top-front to bottom-back) in front of the talons
            if abs(fwx) >= 0.5:
                dx, dy = -0.55 * fwx / abs(fwx), 0.85
            else:
                dx, dy = 0.6, -0.8
            px, py = dy, -dx
            if px < 0:
                px, py = -px, -py
            for j in (-1, 0, 1):
                sx, sy = talon[0] + px * j * 5, talon[1] + py * j * 5
                for s in range(L):
                    u = s - L / 2
                    col = FX_HI if (k < 9 and abs(u) < L / 3) else FX_LO
                    put(a, sx + dx * u, sy + dy * u, col, True)
        elif kind == "feathers" and FEATHER is not None:
            for j in range(4):
                ang = j * 1.6 + 0.4
                rr = 14 + (k - 2) * 2.5
                cx = centre[0] + math.cos(ang) * rr * 1.3
                cy = centre[1] - 10 + math.sin(ang) * rr * 0.7 + (k - 2) * 0.8
                tilt = 1 if (k + j) % 2 else -1
                put(a, cx, cy, FEATHER)
                put(a, cx + tilt, cy + 1, FEATHER)
                put(a, cx + 2 * tilt, cy + 2, FEATHER_D)


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
                  "--balance", "--name", "pidgeotto", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}
    # crisp outline: light pixels on the outer silhouette edge (downscale fringe) -> the darkest palette colour
    allpal = set()
    for v in RIG:
        s = still[v]
        allpal |= {tuple(int(c) for c in p) for p in np.unique(s[s[..., 3] > 0][:, :3], axis=0)}
    ink = min(allpal - {FX_HI, FX_LO}, key=lambda c: sum(c))
    for v in RIG:
        s = still[v]
        op = s[..., 3] > 0
        edge = op & ~ndi.binary_erosion(op)
        lum = s[..., :3].astype(int).sum(axis=2)
        s[edge & (lum > 330), :3] = ink
    # loose-feather colours from the sprite's own palette: the lightest cream and a mid ochre
    global FEATHER, FEATHER_D
    pal = set()
    for v in RIG:
        s = still[v]
        pal |= {tuple(int(c) for c in p) for p in np.unique(s[s[..., 3] > 0][:, :3], axis=0)}
    pal -= {FX_HI, FX_LO}
    creams = [c for c in pal if c[0] > 200 and c[1] > 160 and c[2] < 170]
    FEATHER = max(creams, key=lambda c: sum(c)) if creams else None
    browns = [c for c in pal if 100 < c[0] < 200 and c[0] - c[2] > 50 and c[1] < c[0]]
    FEATHER_D = max(browns, key=lambda c: sum(c)) if browns else FEATHER

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
        feet = [n for n in R["label"] if n.startswith("foot")]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, offs, (gx, gy) = compose(v, cells, piv, act, f)
                centre = (CELL_W / 2 + gx, FOOT_Y - 34 + gy)
                pts = []
                for n in feet:
                    ys, xs = np.nonzero(cells[n][..., 3] > 0)
                    pts.append((xs.mean() + offs[n][0], ys.max() + offs[n][1]))
                talon = (sum(p[0] for p in pts) / len(pts) + gx + R["fwd"][0] * 14,
                         sum(p[1] for p in pts) / len(pts) + gy + R["fwd"][1] * 4)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, talon, R["fwd"], centre, R["back"])
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (3, 5, 7, 10)):
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
    meta = {"generator": "forge_gen1_0017_pidgeotto_actions", "version": VERSION, "key": KEY, "dex": 17,
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
