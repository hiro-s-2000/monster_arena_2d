#!/usr/bin/env python3
"""Ivysaur (0002_ivysaur, normal form): 8-direction x 4-action sheet (gen1 batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py (structure only; rig, colours and attack come from this form's art):
  art/ref_front.png / ref_back.png  : the official front.png / back.png, x8 nearest, on white
  art/agy_prompt.md                 : morphological prompt for the Antigravity CLI (agy)
  art/out/keys_2x2.png              : top-left S, top-right W (side, faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png             : SW (3/4 front, faces down-left)
  SE = SW mirrored, E = W mirrored, NW = NE mirrored.

Pipeline:
  1. key the border-connected white backdrop (white claws / eye highlights / fangs inside the outline stay),
     area-downscale every view so the body height = opaque height of the official front.png (51 px).
  2. agent-sprite-forge palette_tool: build (official 14 colours reserved + k-means fill) -> lock -> apply.
  3. part rig per view (polygons in sprite px): bud on the back, the broad leaves around it, head, body, legs.
     Quadruped: 4 short legs (2-3 visible per view). Hidden seams get cushions (nearest own colour).
  4. poses at 4x (nearest), sampled back to 1x. Actions (16 frames):
     idle  : breathing, leaves rise and fall, bud sways (feet stay on the ground)
     walk  : diagonal-pair quadruped steps, body bob, leaves flap, bud sways
     attack: leaf tackle - crouch with leaves raised and bud pulled back -> lunge, leaves sweep forward,
             mouth opens (tools/mouth_open.py, max 2 px), razor-leaf shards fly ahead -> impact -> recover
     hurt  : white flash, recoil back (lean <= 8 deg in side views), leaves droop, shake, recover
  5. QC: off-palette 0, edge-touch 0, still diff 0, gait_loop.py on idle / walk.

Never writes the official front/back png/gif. Syncs agent-sprite-forge/ (without _work) to pokemon_arena_2d.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0002_ivysaur_actions.py [--force] [--no-sync]
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import mouth_open  # noqa: E402

KEY = "0002_ivysaur"
SLUG = "0002_ivysaur"
REL = "monsterSprite/gen1/0002_ivysaur/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0002_ivysaur_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AN = [16, 16, 16, 16]
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 128, 88, 70
UP = 4
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# official colours (front.png + back.png): reserved so the sprite keeps the game's palette
INK = (16, 16, 16)
WHITE = (255, 255, 255)
LEAF_HI, LEAF_MID, LEAF_DK, LEAF_INK = (123, 213, 115), (98, 172, 90), (49, 123, 82), (16, 74, 57)
SKIN_HI = (131, 230, 213)
BUD_RED = (213, 65, 90)


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))[..., 3] > 0
    ys = np.nonzero(a.any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


def official_colours():
    cols = set()
    for n in ("front.png", "back.png"):
        p = FORM_DIR / n
        if p.exists():
            a = np.array(Image.open(p).convert("RGBA"))
            for c in np.unique(a[a[..., 3] > 0][:, :3], axis=0):
                cols.add("#%02x%02x%02x" % tuple(int(v) for v in c))
    return sorted(cols)


# ---------------------------------------------------------------- rig (sprite px of each 51 px tall view)
# label order: later polygons win. draw order: back to front. body_x = body centre (placed on the cell centre).
# legs: name -> walk phase (0 / 1 = diagonal pairs). mouth: closed mouth line (hinge -> lip / corner -> corner).
RIG = {
    "S": dict(
        poly={"body": [(8, 26), (54, 26), (54, 47), (8, 47)],
              "leg_l": [(8, 36), (23, 36), (23, 51), (8, 51)],
              "leg_r": [(39, 36), (54, 36), (54, 51), (39, 51)],
              "head": [(12, 19), (50, 19), (50, 32), (44, 37), (42, 43), (20, 43), (18, 37), (12, 32)],
              "leaf_l": [(0, 11), (21, 11), (21, 19), (14, 22), (14, 31), (0, 31)],
              "leaf_r": [(41, 11), (62, 11), (62, 31), (48, 31), (48, 22), (41, 19)],
              "bud": [(19, 0), (43, 0), (43, 21), (19, 21)]},
        label=["body", "leg_l", "leg_r", "head", "leaf_l", "leaf_r", "bud"],
        draw=["bud", "leaf_l", "leaf_r", "body", "leg_l", "leg_r", "head"],
        pivot={"bud": (31, 20), "leaf_l": (20, 22), "leaf_r": (42, 22), "head": (31, 30), "body": (31, 47),
               "leg_l": (16, 40), "leg_r": (46, 40)},
        legs={"leg_l": 0, "leg_r": 1}, body_x=31, fwd=(0.0, 1.0), back=False, mouth=None),
    "SW": dict(
        poly={"body": [(4, 24), (54, 24), (54, 46), (4, 46)],
              "leg_h": [(42, 35), (55, 35), (55, 49), (42, 49)],
              "head": [(6, 24), (40, 24), (42, 30), (40, 40), (32, 45), (18, 45), (6, 40)],
              "leg_fl": [(3, 41), (18, 41), (18, 50), (3, 50)],
              "leg_fr": [(32, 42), (41, 42), (41, 51), (32, 51)],
              "leaf_l": [(0, 12), (18, 12), (18, 27), (0, 27)],
              "leaf_r": [(37, 7), (59, 7), (59, 31), (42, 31), (37, 26)],
              "bud": [(20, 0), (42, 0), (42, 22), (20, 22)]},
        label=["body", "leg_h", "head", "leg_fl", "leg_fr", "leaf_l", "leaf_r", "bud"],
        draw=["leaf_r", "bud", "leaf_l", "body", "leg_h", "head", "leg_fr", "leg_fl"],
        pivot={"bud": (31, 21), "leaf_l": (18, 22), "leaf_r": (38, 22), "head": (32, 34), "body": (30, 46),
               "leg_fl": (14, 43), "leg_fr": (36, 44), "leg_h": (47, 38)},
        legs={"leg_fl": 0, "leg_h": 0, "leg_fr": 1}, body_x=30, fwd=(-0.9, 0.45), back=False, mouth=None),
    "W": dict(
        poly={"body": [(18, 18), (53, 18), (53, 46), (18, 46)],
              "leg_f": [(11, 38), (30, 38), (30, 51), (11, 51)],
              "leg_h": [(33, 39), (53, 39), (53, 51), (33, 51)],
              "leaf_b": [(44, 15), (61, 15), (61, 29), (46, 29)],
              "leaf_f": [(10, 9), (27, 9), (28, 20), (20, 20), (10, 18)],
              "head": [(0, 15), (25, 13), (28, 22), (28, 37), (18, 40), (0, 38)],
              "bud": [(24, 0), (47, 0), (47, 18), (24, 18)]},
        label=["body", "leg_f", "leg_h", "leaf_b", "leaf_f", "head", "bud"],
        draw=["leaf_f", "bud", "leaf_b", "body", "leg_h", "leg_f", "head"],
        pivot={"bud": (36, 18), "leaf_f": (26, 18), "leaf_b": (45, 22), "head": (25, 32), "body": (32, 46),
               "leg_f": (22, 40), "leg_h": (43, 41)},
        legs={"leg_f": 0, "leg_h": 1}, body_x=32, fwd=(-1.0, 0.0), back=False, mouth=None),
    "NE": dict(
        poly={"body": [(8, 24), (52, 24), (52, 48), (8, 48)],
              "leg_a": [(9, 37), (21, 37), (21, 51), (9, 51)],
              "leg_b": [(28, 40), (42, 40), (42, 51), (28, 51)],
              "leg_c": [(43, 36), (57, 36), (57, 49), (43, 49)],
              "head": [(43, 10), (57, 10), (57, 34), (47, 34), (43, 26)],
              "leaf_l": [(0, 14), (24, 14), (26, 31), (12, 31), (0, 28)],
              "leaf_r": [(33, 7), (47, 7), (50, 18), (53, 26), (34, 28)],
              "bud": [(14, 0), (36, 0), (36, 23), (14, 23)]},
        label=["body", "leg_a", "leg_b", "leg_c", "head", "leaf_l", "leaf_r", "bud"],
        draw=["head", "leg_c", "body", "leg_a", "leg_b", "leaf_r", "bud", "leaf_l"],
        pivot={"bud": (25, 22), "leaf_l": (24, 24), "leaf_r": (35, 22), "head": (46, 28), "body": (30, 48),
               "leg_a": (15, 40), "leg_b": (35, 43), "leg_c": (49, 39)},
        legs={"leg_a": 0, "leg_c": 0, "leg_b": 1}, body_x=30, fwd=(0.9, -0.45), back=True, mouth=None),
    "N": dict(
        poly={"body": [(8, 9), (52, 9), (52, 48), (8, 48)],
              "leg_l": [(8, 39), (25, 39), (25, 51), (8, 51)],
              "leg_r": [(34, 39), (51, 39), (51, 51), (34, 51)],
              "leaf_l": [(0, 17), (24, 17), (24, 30), (0, 30)],
              "leaf_r": [(35, 17), (59, 17), (59, 30), (35, 30)],
              "leaf_c": [(21, 23), (38, 23), (38, 38), (21, 38)],
              "bud": [(19, 0), (40, 0), (40, 22), (19, 22)]},
        label=["body", "leg_l", "leg_r", "leaf_l", "leaf_r", "bud", "leaf_c"],
        draw=["body", "leg_l", "leg_r", "leaf_l", "leaf_r", "bud", "leaf_c"],
        pivot={"bud": (29, 21), "leaf_l": (23, 23), "leaf_r": (36, 23), "leaf_c": (29, 23), "body": (29, 48),
               "leg_l": (16, 41), "leg_r": (42, 41)},
        legs={"leg_l": 0, "leg_r": 1}, body_x=29, fwd=(0.0, -1.0), back=True, mouth=None),
}
# closed mouth lines (measured on the stills, see find_mouth): set at runtime from the dark line in the face
MOUTH_SEARCH = {"S": (16, 46, 35, 43), "SW": (14, 36, 38, 45), "W": (1, 20, 31, 38)}   # x0, x1, y0, y1


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 200)        # light fringe
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def make_stills(dst: Path, height):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    dst.mkdir(parents=True, exist_ok=True)
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        im = im.crop(im.getbbox())
        sc = height / im.height                      # every view: body height = official front.png height
        sm = im.resize((round(im.width * sc), height), Image.BOX)
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


def find_mouth(view, still):
    """Darkest horizontal run inside the face box = closed mouth line (corner -> corner / hinge -> lip)."""
    if view not in MOUTH_SEARCH:
        return None
    x0, x1, y0, y1 = MOUTH_SEARCH[view]
    lum = still[..., :3].astype(int).sum(axis=2)
    dark = (still[..., 3] > 0) & (lum < 200)
    best = None
    for y in range(y0, y1):
        xs = [x for x in range(x0, x1) if dark[y, x]]
        if len(xs) >= 4 and (best is None or len(xs) > best[0]):
            best = (len(xs), y, min(xs), max(xs))
    if best is None:
        return None
    _, y, xa, xb = best
    if view == "W":            # faces left: hinge (back of the mouth) on the right
        return [(xb, y), (xa + 1, y)]
    return [(xa, y), (xb, y)]


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    lab = np.zeros(op.shape, int)
    names = R["label"]
    for i, n in enumerate(names, start=1):
        lab[poly_mask(R["poly"][n], op.shape) & op] = i
    # green pixels belong to a leaf / red ones to the bud even where the boxes overlap the body or head
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    green = op & (g > r + 25) & (g > b + 5)
    red = op & (r > g + 60) & (r > b + 40)
    leaf_ids = [names.index(n) + 1 for n in names if n.startswith("leaf")]
    body_like = np.isin(lab, [names.index(n) + 1 for n in names if n in ("body", "head")])
    if leaf_ids:
        stray = green & body_like
        if stray.any():
            leafm = np.isin(lab, leaf_ids)
            _, (iy, ix) = ndi.distance_transform_edt(~leafm, return_indices=True)
            near = stray & (ndi.distance_transform_edt(~leafm) <= 6)
            lab[near] = lab[iy[near], ix[near]]
    budm = lab == names.index("bud") + 1          # red eyes / mouth stay with the face: only red next to the bud
    stray = red & ~budm & (ndi.distance_transform_edt(~budm) <= 3)
    lab[stray] = names.index("bud") + 1
    un = op & (lab == 0)
    if un.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[un] = lab[iy[un], ix[un]]
    masks = {n: lab == (i + 1) for i, n in enumerate(names)}
    layers = {}
    order = R["draw"]
    lum = still[..., :3].astype(int).max(axis=2)
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 6 if n in ("body", "bud") else 3
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n == "body" and m.any():
            # underlay for everything drawn above the body inside the closed body + head outline
            bh = m | masks.get("head", np.zeros_like(m))
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 8), iterations=8))[8:-8, 8:-8]
            seam |= above & ~m & inside & ~masks.get("head", np.zeros_like(m))
            src_m = bh
        if seam.any() and m.any():
            fillable = src_m & (lum > 70)
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
def wave(amp, cycles, f, phase=0.0):
    """Loop-safe wave: 0 at frame 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """P[part-kind] = [rot deg (+ = raise / tip forward), fwd px, up px]; P['g'] = [fwd, up, lean]. fx list."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        breath = wave(0.75, 1, f, -math.pi / 2)              # 0 -> 1.5 -> 0, smooth
        P["body"] = [0, 0, round(breath * 0.0)]
        P["head"] = [0, 0, -round(wave(0.6, 1, f, -math.pi / 2 + 0.6))]
        P["leaf"] = [wave(3.0, 1, f, -math.pi / 2), 0, round(breath * 0.6)]
        P["bud"] = [wave(2.0, 1, f, 0.0), 0, round(breath)]
        P["leaf_c"] = [wave(2.0, 1, f, 0.0), 0, round(breath * 0.6)]
    elif action == "walk":
        bob = round(abs(math.sin(2 * math.pi * f / N)) * 1.6)
        P["body"] = [0, 0, bob]
        P["head"] = [0, 0, bob + (1 if math.sin(4 * math.pi * f / N - 0.6) > 0.4 else 0)]
        P["leaf"] = [wave(5.0, 2, f, 0.3), 0, bob]
        P["leaf_c"] = [wave(3.0, 2, f, 0.3), 0, bob]
        P["bud"] = [wave(3.0, 1, f, 0.0), 0, bob]
        ph = 2 * math.pi * f / N
        P["legA"] = [0, 2.5 * math.sin(ph), 2.4 * max(0.0, math.sin(ph))]
        P["legB"] = [0, -2.5 * math.sin(ph), 2.4 * max(0.0, -math.sin(ph))]
    elif action == "attack":
        g_f = [0, -2, -3, -4, -4, 3, 8, 10, 10, 9, 7, 5, 3, 2, 1, 0][f]
        g_u = [0, 0, -1, -1, -1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -1, -3, -4, -4, 3, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [0, [0, -1, -1, -1, -1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0][f], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        lf = [0, 6, 12, 16, 18, 4, -10, -16, -16, -14, -10, -7, -4, -2, -1, 0][f]
        P["leaf"] = [lf, 0, 0]
        P["leaf_c"] = [lf * 0.4, 0, 0]
        P["bud"] = [[0, -2, -4, -6, -6, 2, 6, 8, 7, 6, 4, 3, 2, 1, 0, 0][f], 0, 0]
        P["mouth"] = [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0][f]
        if 2 <= f <= 4:
            fx.append(("gather", f))
        if 5 <= f <= 11:
            fx.append(("shards", f))
        if f in (8, 9, 10):
            fx.append(("impact", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-5.0 * env + jit, 0.0, -7.0 * env]
        P["head"] = [0, -1.0 * env, 1.0 * env]
        P["leaf"] = [-10.0 * env, 0, 0]
        P["leaf_c"] = [-4.0 * env, 0, 0]
        P["bud"] = [-7.0 * env, 0, 0]
        P["mouth"] = 1 if 2 <= f <= 6 else 0
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
    dx, dy = int(round(dx)) * UP, int(round(dy)) * UP          # whole pixels only: the face never splits
    if not dx and not dy:
        return img4
    out = np.zeros_like(img4)
    h, w = img4.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = img4[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def compose(view, cells, piv, side_sign, mouth_line, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    frontish = abs(fwx) < 0.5
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    offs = {}
    b = P.get("body", zero)
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero)
            df, du = h[1], h[2] + b[2]
        elif n.startswith("leg"):
            k = "legA" if R["legs"][n] == 0 else "legB"
            lg = P.get(k, zero)
            df, du = lg[1], lg[2]
            if frontish:
                df = 0.0                                    # toward / away from camera: steps are lifts only
        elif n == "bud":
            bd = P.get("bud", zero)
            # + tips the bud top forward (side views) / sways right (front, back)
            rot = bd[0] * (side if not frontish else 1.0)
            du = bd[2] + b[2]
        elif n == "leaf_c":
            lc = P.get("leaf_c", zero)
            rot = lc[0]
            du = lc[2] + b[2]
        elif n.startswith("leaf"):
            lf = P.get("leaf", zero)
            rot = lf[0] * side_sign[n]                      # + raises the leaf tip
            du = lf[2] + b[2]
        dx, dy = df * fwx, df * fwy - du
        offs[n] = (dx, dy)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if frontish:
        gf *= 0.45
    if abs(lean) > 0.3 and not frontish:
        canvas = rot4(canvas, lean * side, (CELL_W // 2, FOOT_Y))
    gx, gy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()
    mo = P.get("mouth", 0)
    if mo and mouth_line:
        hd = offs.get("head", (0, 0))
        ln = mouth_open.transform_points(mouth_line, dx=round(hd[0]) + round(gx), dy=round(hd[1]) + round(gy))
        if abs(lean) > 0.3 and not frontish:
            ln = mouth_open.transform_points(mouth_line, angle=lean * side, pivot=(CELL_W // 2, FOOT_Y))
            ln = [(x + round(hd[0]) + round(gx), y + round(hd[1]) + round(gy)) for x, y in ln]
        out = mouth_open.open_mouth(out, ln, amount=mo, deep=INK, inner=BUD_RED, peak=0.5 if view == "S" else 0.7)
    return out, fx, offs, (gx, gy)


# ---------------------------------------------------------------- effects (official palette colours only)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def leaf_shard(a, x, y, dx, dy, big):
    """razor-leaf blade (outlined, light top / mid underside / dark vein) pointing along (dx, dy)."""
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    n = 4.5 if big else 3.5
    wmax = 1.9 if big else 1.4
    inside = {}
    R = int(n) + 3
    xi, yi = int(round(x)), int(round(y))
    for qy in range(yi - R, yi + R + 1):
        for qx in range(xi - R, xi + R + 1):
            u = (qx - x) * ux + (qy - y) * uy
            v = -(qx - x) * uy + (qy - y) * ux
            if abs(u) <= n and abs(v) <= wmax * (1 - (u / n) ** 2) + 0.35:
                inside[(qx, qy)] = (u, v)
    for (qx, qy) in inside:
        for ex, ey in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (qx + ex, qy + ey) not in inside:
                put(a, qx + ex, qy + ey, LEAF_INK)
    for (qx, qy), (u, v) in inside.items():
        put(a, qx, qy, LEAF_DK if abs(v) < 0.5 and abs(u) < n - 1 else (LEAF_HI if v < 0 else LEAF_MID))


def star(a, x, y, r):
    for k in range(8):
        ang = k * math.pi / 4
        rr = r if k % 2 == 0 else r * 0.55
        for s in range(1, int(rr) + 1):
            put(a, x + math.cos(ang) * s, y + math.sin(ang) * s, WHITE if s < rr - 1 else SKIN_HI)
    put(a, x, y, WHITE)


def ease(u):
    return 1 - (1 - u) ** 2


def draw_fx(a, fx, target, fwd, centre, leaf_roots):
    fwx, fwy = fwd
    px, py = -fwy, fwx
    for kind, k in fx:
        if kind == "gather":
            for j, (lx, ly) in enumerate(leaf_roots):
                for q in range(2):
                    ang = k * 1.3 + j * 2.0 + q * 3.1
                    put(a, lx + math.cos(ang) * (9 - k), ly - 3 + math.sin(ang) * 3, WHITE if q else SKIN_HI)
        elif kind == "shards":
            # each leaf throws two blades; they arc out and converge on the target in front of the snout
            for j, (lx, ly) in enumerate(leaf_roots[:3]):
                for q in range(2):
                    launch = 5 + q + (j % 2)
                    u = (k - launch + 1) / 4.0
                    if not (0 < u <= 1.0):
                        continue
                    e = ease(u)
                    arc = (1 if (j + q) % 2 == 0 else -1) * 7 * math.sin(math.pi * u)
                    sx = lx + (target[0] - lx) * e + px * arc
                    sy = ly - 3 + (target[1] - ly + 3) * e + py * arc
                    spin = k * 1.9 + j + q * 2.3
                    leaf_shard(a, sx, sy, math.cos(spin) + fwx, math.sin(spin) + fwy, big=(q == 0))
        elif kind == "impact":
            r = {8: 4, 9: 7, 10: 5}[k]
            star(a, target[0], target[1], r)
            if k == 9:
                for q in range(4):
                    ang = q * math.pi / 2 + 0.6
                    leaf_shard(a, target[0] + math.cos(ang) * 8, target[1] + math.sin(ang) * 6, math.cos(ang), math.sin(ang), False)
        elif kind == "stars":
            for j in range(3):
                ang = j * 2.1 + k * 0.8
                cx, cy = centre[0] + math.cos(ang) * 14, centre[1] - 16 + math.sin(ang) * 5
                put(a, cx, cy, WHITE)
                put(a, cx + 1, cy, SKIN_HI); put(a, cx - 1, cy, SKIN_HI)
                put(a, cx, cy + 1, SKIN_HI); put(a, cx, cy - 1, SKIN_HI)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = WHITE
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    h.update((ROOT / "tools/mouth_open.py").read_bytes())
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
    height = official_height()

    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw, height)
        run_tool(["build", "--input", str(raw), "--colors", "26", "--reserve", *official_colours(),
                  "--balance", "--name", SLUG, "--output-dir", str(pal_dir)])
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
        side_sign = {}
        for n in cells:
            if n.startswith("leaf") and n != "leaf_c":
                ys, xs = np.nonzero(cells[n][..., 3] > 0)
                side_sign[n] = 1.0 if (xs.mean() if len(xs) else piv[n][0]) < piv[n][0] else -1.0
        ml = find_mouth(v, still[v])
        if ml:
            ml = [(x + ox, y + oy) for x, y in ml]
        rig[v] = (cells, piv, (ox, oy), side_sign, ml)

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, (ox, oy), side_sign, ml = rig[v]
        R = RIG[v]
        leaf_cells = [n for n in cells if n.startswith("leaf")]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, offs, (gx, gy) = compose(v, cells, piv, side_sign, ml, act, f)
                # front of the creature (snout / leading edge) for the attack effects
                op = a[..., 3] > 0
                ys, xs = np.nonzero(op)
                fwx, fwy = R["fwd"]
                cx, cy = xs.mean(), ys.mean()
                proj = (xs - cx) * fwx + (ys - cy) * fwy
                i = int(np.argmax(proj))
                frontish = abs(fwx) < 0.5
                dist = (26 if fwy > 0 else 36) if frontish else 46
                target = (CELL_W / 2 + fwx * dist, FOOT_Y - 18 + fwy * dist * (1.0 if frontish else 0.7))
                centre = (CELL_W / 2 + gx, FOOT_Y - 20 + gy)
                roots = [(piv[n][0] + gx, piv[n][1] + gy) for n in leaf_cells]
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, target, R["fwd"], centre, roots)
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
    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(FORM_DIR / "8dir_stills.png")

    # QC
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
    # still diff: idle frame 0 and every action's frame 0 vs the palette-applied still on the cell
    still_diff = 0
    for v in RIG:
        ref, _ = place_layer(still[v], v)
        for act in ACTIONS:
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            still_diff += int((a0 != ref).any(axis=2).sum())
        aN = np.array(Image.open(frames_dir / v / "idle" / f"{N - 1:02d}.png").convert("RGBA"))
    s_h = np.nonzero((sheets["idle"][:CELL_H, :CELL_W, 3] > 0).any(axis=1))[0]; s_h = int(s_h.max() - s_h.min() + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "alpha_not_binary": alpha_bad, "edge_touching_frames": clip, "still_diff_px": still_diff,
              "S_height": s_h, "official_height": height,
              "mouth_lines": {v: rig[v][4] for v in RIG}}
    if not args.skip_qa and GAIT_LOOP.exists():
        for act, kind in (("idle", "idle"), ("walk", "gait")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FPY), "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / "S" / act), "--fps", "13",
                                "--output-dir", str(od), "--kind", kind, "--no-aids"], capture_output=True, text=True)
            rep = od / "loop-report.json"
            report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr.strip().splitlines() or ["no report"])[-1]
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    bounds = []
    for di in range(8):
        c = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W, 3] > 0
        ys, xs = np.nonzero(c)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    meta = {"generator": "forge_gen1_0002_ivysaur_actions", "version": VERSION, "key": KEY, "dex": 2,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": s_h, "frames": N, "an": AN, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else (v.get("policy") if "policy" in v else v)) for k, v in report.items()}, ensure_ascii=False, default=str))

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
