#!/usr/bin/env python3
"""Mega Venusaur (0003_venusaur-mega): 8-direction x 4-action sheet (gen1 batch, md/gen1-forge.md).

Same flow as tools/forge_raichu_mega_x_actions.py, rig / colours / attack built from this form's own art.

Source (2026-11-04):
  art/ref_front.png / art/ref_back.png = official front.png / back.png x8 (nearest). The Antigravity CLI (`agy -p`,
  prompt art/agy_prompt.md, no proper nouns) drew from them:
    art/out/keys_2x2.png : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
    art/out/sw_single.png: SW. The other three are mirrors: SE = SW, E = W, NW = NE.
  (first agy try drew a different creature -> art/try1/, retried once with a stronger prompt.)

Creature: heavy low quadruped (4 legs, white claws), one huge pink flower on the back (petals + brown trunk with a
yellow crown), two big jagged fern leaves under it, a small pink flower on the forehead. No tail, no arms.

Pipeline:
  1. key the white backdrop (border-connected only), drop the small caption text agy wrote under the grid,
     area-downscale so the S body height = opaque height of the official front.png (74 px); SW matched on height.
  2. agent-sprite-forge palette_tool build (30 colours + 2 reserved effect colours) -> lock -> apply.
  3. part rig: colour first (leaf greens -> leaves, flower region -> flower), polygons for head and legs, the rest is
     the body; outline pixels go to the nearest coloured part. Hidden seams get cushions (nearest own colour).
  4. poses at 4x (nearest rotate), sampled back to 1x. Actions (16 frames):
     idle  = slow breath (body sinks 1 px), flower sways on its trunk, leaves breathe; feet stay planted
     walk  = quadruped diagonal-pair steps (legs alternate), body bob, leaves / flower bounce
     attack= crouch + leaves raised + sun light gathers at the flower crown -> lunge with a front-leg stomp, both
             leaves whip forward (leaf slash arcs) and the flower tips forward -> stomp shockwave + flying leaf bits
             -> recover
     hurt  = white flash, recoil back with lean, leaves flare, recover
  5. QC: off-palette 0, edge-touch 0, still diff (idle frame 0 == still) 0, gait_loop.py on idle / walk.

Never writes the official front/back png/gif. Syncs agent-sprite-forge/ (without _work) to pokemon_arena_2d.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0003_venusaur_mega_actions.py [--force]
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
KEY = "0003_venusaur-mega"
REL = "monsterSprite/gen1/0003_venusaur/mega/venusaur-mega"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "venusaur_mega_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 184, 124, 112
UP = 4
FX_HI, FX_LO = (246, 255, 206), (156, 232, 84)       # reserved: sun light / leaf-slash colours
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# flower: region where non-green pixels belong to the back flower (petals + trunk); green pixels anywhere are leaves
# (split left / right at leaf_split), except inside `leaf_body` (a leaf lying on the back that stays with the body).
# head / legs: polygons. draw = back to front. legs_a / legs_b = diagonal pairs for the walk.
RIG = {
    "S": dict(
        flower=[(10, 0), (88, 0), (88, 24), (66, 34), (56, 40), (40, 40), (28, 34), (10, 24)],
        head=[(16, 38), (72, 38), (74, 56), (70, 68), (20, 68), (14, 56)],
        legs={"leg_l": [(0, 52), (20, 52), (24, 60), (24, 74), (0, 74)],
              "leg_r": [(70, 54), (92, 52), (97, 60), (97, 74), (66, 74), (66, 64)]},
        leaf_split=48, leaf_body=None,
        draw=["body", "leaf_l", "leaf_r", "flower", "head", "leg_l", "leg_r"],
        pivot={"flower": (48, 38), "leaf_l": (32, 32), "leaf_r": (62, 34), "head": (44, 44)},
        legs_a=["leg_l"], legs_b=["leg_r"], body_x=48, fwd=(0.0, 1.0), crown=(48, 6), back=False),
    "SW": dict(
        flower=[(14, 0), (90, 0), (90, 22), (62, 34), (52, 38), (36, 36), (14, 22)],
        head=[(6, 30), (44, 30), (54, 38), (58, 50), (54, 62), (40, 66), (6, 64)],
        legs={"leg_fl": [(0, 58), (22, 58), (24, 74), (0, 74)],
              "leg_fr": [(44, 58), (64, 54), (68, 64), (66, 74), (44, 74)],
              "leg_br": [(70, 54), (98, 50), (98, 74), (72, 74)]},
        leaf_split=48, leaf_body=None,
        draw=["body", "leg_br", "leaf_l", "leaf_r", "flower", "head", "leg_fl", "leg_fr"],
        pivot={"flower": (50, 36), "leaf_l": (36, 30), "leaf_r": (62, 32), "head": (44, 42)},
        legs_a=["leg_fl", "leg_br"], legs_b=["leg_fr"], body_x=50, fwd=(-0.9, 0.45), crown=(50, 6), back=False),
    "W": dict(
        flower=[(14, 0), (90, 0), (90, 16), (62, 24), (40, 24), (14, 16)],
        head=[(0, 24), (36, 24), (40, 36), (40, 56), (0, 58)],
        legs={"leg_f": [(28, 54), (54, 54), (54, 70), (28, 70)],
              "leg_b": [(64, 50), (100, 48), (100, 70), (64, 70)]},
        leaf_split=40, leaf_body=None,
        draw=["leaf_l", "body", "leg_b", "leg_f", "leaf_r", "flower", "head"],
        pivot={"flower": (52, 22), "leaf_l": (44, 26), "leaf_r": (50, 26), "head": (34, 40)},
        legs_a=["leg_f"], legs_b=["leg_b"], body_x=52, fwd=(-1.0, 0.0), crown=(50, 4), back=False),
    "NE": dict(
        flower=[(8, 0), (80, 0), (80, 26), (60, 28), (24, 28), (8, 24)],
        head=[(64, 20), (89, 18), (89, 46), (66, 46)],
        legs={"leg_a": [(0, 50), (26, 50), (26, 71), (0, 71)],
              "leg_b": [(44, 56), (64, 56), (64, 71), (44, 71)],
              "leg_c": [(68, 50), (89, 48), (89, 71), (68, 71)]},
        leaf_split=40, leaf_body=None,
        draw=["body", "leg_a", "leg_b", "leg_c", "head", "leaf_l", "leaf_r", "flower"],
        pivot={"flower": (38, 28), "leaf_l": (34, 30), "leaf_r": (44, 30), "head": (68, 34)},
        legs_a=["leg_a", "leg_c"], legs_b=["leg_b"], body_x=44, fwd=(0.9, -0.45), crown=(40, 6), back=True),
    "N": dict(
        flower=[(14, 0), (86, 0), (86, 22), (64, 32), (38, 32), (14, 22)],
        head=None,
        legs={"leg_l": [(4, 52), (32, 52), (32, 71), (4, 71)],
              "leg_r": [(66, 52), (97, 50), (97, 71), (66, 71)]},
        leaf_split=50, leaf_body=[(24, 34), (52, 34), (52, 60), (24, 60)],
        draw=["body", "leg_l", "leg_r", "leaf_l", "leaf_r", "flower"],
        pivot={"flower": (50, 32), "leaf_l": (40, 30), "leaf_r": (60, 30)},
        legs_a=["leg_l"], legs_b=["leg_r"], body_x=50, fwd=(0.0, -1.0), crown=(50, 6), back=True),
}
CUSHION = {"body": 6, "flower": 5, "leaf_l": 4, "leaf_r": 4, "head": 3}


# ---------------------------------------------------------------- 1. stills
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, _ = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 200)
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 3000])] = 0     # also drops the caption text
    return out


def make_stills(dst: Path, s_height: int):
    k = key_white(np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB")))
    lab, _ = ndi.label(k[..., 3] > 0)
    src = {}
    H, W = k.shape[:2]
    for i, o in enumerate(ndi.find_objects(lab)):
        cy, cx = (o[0].start + o[0].stop) / 2, (o[1].start + o[1].stop) / 2
        v = {(0, 0): "S", (0, 1): "W", (1, 0): "NE", (1, 1): "N"}[(int(cy > H / 2), int(cx > W / 2))]
        sub = k[o].copy()
        sub[lab[o] != i + 1] = 0
        src[v] = sub
    src["SW"] = key_white(np.array(Image.open(ART / "out/sw_single.png").convert("RGB")))
    crops = {v: Image.fromarray(a, "RGBA").crop(Image.fromarray(a, "RGBA").getbbox()) for v, a in src.items()}
    grid = s_height / crops["S"].height
    scale = {v: grid for v in crops}
    scale["SW"] = s_height / crops["SW"].height
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in crops.items():
        sm = im.resize((round(im.width * scale[v]), round(im.height * scale[v])), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        Image.fromarray(a, "RGBA").save(dst / f"{v}.png")


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))[..., 3] > 0
    ys = np.nonzero(a.any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


# ---------------------------------------------------------------- 3. rig
def poly_mask(p, shape):
    m = Image.new("L", (shape[1], shape[0]), 0)
    if p:
        ImageDraw.Draw(m).polygon(p, fill=255)
    return np.array(m) > 0


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    mx = np.maximum(np.maximum(r, g), b)
    core = op & (mx > 60)                                # outline pixels are assigned afterwards
    green = core & (g > r + 25) & (g > b + 25)
    yy, xx = np.mgrid[0:op.shape[0], 0:op.shape[1]]
    names = ["body", "leaf_l", "leaf_r", "flower"] + (["head"] if R["head"] else []) + list(R["legs"])
    idx = {n: i + 1 for i, n in enumerate(names)}
    lab = np.zeros(op.shape, int)
    lab[core] = idx["body"]
    if R["head"]:
        lab[core & poly_mask(R["head"], op.shape)] = idx["head"]
    for ln, p in R["legs"].items():
        lab[core & poly_mask(p, op.shape)] = idx[ln]
    fl = poly_mask(R["flower"], op.shape)
    lab[core & fl & ~green] = idx["flower"]
    lb = poly_mask(R["leaf_body"], op.shape) if R["leaf_body"] else np.zeros_like(op)
    # leaves = big green blobs (tiny green specks in the body stay body)
    gl, gn = ndi.label(green & ~lb)
    if gn:
        gs = ndi.sum(np.ones_like(gl), gl, range(1, gn + 1))
        big = np.isin(gl, [i + 1 for i, s in enumerate(gs) if s >= 12])
        lab[big & (xx < R["leaf_split"])] = idx["leaf_l"]
        lab[big & (xx >= R["leaf_split"])] = idx["leaf_r"]
    un = op & ~core
    if un.any():
        _, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
        lab[un] = lab[iy[un], ix[un]]
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
        cush = CUSHION.get(n, 0)
        if n.startswith("leg"):
            cush = 0
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cush) if cush and m.any() else np.zeros_like(m)
        if n == "body" and m.any():
            # legs only get a short body underlay at their top (a lifted leg must not leave a body-coloured stump)
            for ln in R["legs"]:
                lm = masks[ln]
                if lm.any():
                    top = np.nonzero(lm.any(axis=1))[0].min()
                    seam &= ~(lm & (yy > top + 5))
        if seam.any() and m.any():
            fillable = m & (mx > 70)
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
def wave(amp, cyc, f, phase=0.0):
    """loop-safe wave: 0 at frame 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cyc * f / N + phase) - math.sin(phase))


def pose(action, f):
    """Per part: [rot deg, fwd px, up px]. rot: leaves + = raise, flower + = tip forward. g = [fwd, up, lean]."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    ph = 2 * math.pi * f / N
    if action == "idle":
        sink = -0.5 * (1 - math.cos(ph))                       # 0 -> -1 -> 0
        P["body"] = [0, 0, sink]
        P["head"] = [0, 0, -0.5 * (1 - math.cos(ph - 0.0))]
        P["flower"] = [wave(2.5, 1, f, 0.0), 0, sink]
        P["leaf_l"] = [wave(4.0, 1, f, 0.0), 0, sink]
        P["leaf_r"] = [wave(4.0, 1, f, 0.9), 0, sink]
    elif action == "walk":
        s1, c1 = math.sin(ph), math.cos(ph)
        up = 1.2 * abs(s1)
        P["body"] = [0, 0, up]
        P["head"] = [0, 0, up + (1 if math.sin(2 * ph - 0.6) > 0.5 else 0)]
        P["legs_a"] = [0, 3.0 * s1, 3.0 * max(0.0, s1)]
        P["legs_b"] = [0, -3.0 * s1, 3.0 * max(0.0, -s1)]
        P["flower"] = [wave(3.0, 2, f, 0.0), 0, up]
        P["leaf_l"] = [wave(6.0, 2, f, 0.0), 0, up]
        P["leaf_r"] = [wave(6.0, 2, f, 0.6), 0, up]
    elif action == "attack":
        g_f = [0, -2, -3, -4, -3, 4, 8, 9, 9, 8, 6, 4, 3, 2, 1, 0][f]
        g_u = [0, -1, -2, -2, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -2, -4, -6, -5, 3, 6, 6, 5, 4, 3, 2, 1, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [0, [0, 0, -1, -1, -1, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0][f], [0, 0, 1, 1, 1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        # front stomp: lift during the wind-up, slam on the lunge
        P["legs_a"] = [0, [0, 0, 1, 2, 3, 3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0][f], [0, 2, 4, 6, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        lf = [0, 6, 12, 16, 18, -8, -18, -20, -18, -14, -10, -6, -3, -1, 0, 0][f]
        P["leaf_l"] = [lf, 0, 0]
        P["leaf_r"] = [lf, 0, 0]
        P["flower"] = [[0, -2, -4, -5, -5, 3, 6, 6, 5, 4, 3, 2, 1, 0, 0, 0][f], 0, 0]
        if 1 <= f <= 4:
            fx.append(("charge", f))
        if 5 <= f <= 8:
            fx.append(("slash", f))
        if 6 <= f <= 9:
            fx.append(("quake", f))
        if 6 <= f <= 11:
            fx.append(("bits", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [-6.0 * env + jit, 1.0 * env, -7.0 * env]
        P["head"] = [0, -1.0 * env, 1.0 * env]
        P["leaf_l"] = [14.0 * env, 0, 0]
        P["leaf_r"] = [14.0 * env, 0, 0]
        P["flower"] = [-5.0 * env, 0, 0]
        if f == 1:
            fx.append(("flash", 0))
        if 1 <= f <= 4:
            fx.append(("hit", f))
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
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    offs = {}
    b = P.get("body", zero)
    for n in R["draw"]:
        rot, df, du = 0.0, 0.0, 0.0
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero)
            df, du = h[1], h[2]
        elif n.startswith("leg"):
            grp = "legs_a" if n in R["legs_a"] else "legs_b"
            lg = P.get(grp, zero)
            df, du = lg[1], lg[2]
        elif n.startswith("leaf"):
            lf = P.get(n, zero)
            rot = lf[0] * (1.0 if n == "leaf_l" else -1.0)
            du = lf[2]
        elif n == "flower":
            fl = P.get("flower", zero)
            rot = fl[0] * side if abs(fwx) > 0.3 else fl[0] * (0.5 if action == "idle" or action == "walk" else 0.0)
            if abs(fwx) <= 0.3 and action not in ("idle", "walk"):
                du = -abs(fl[0]) * 0.2 * (1 if fl[0] < 0 else -1)   # front / back: dip instead of a flat turn
            else:
                du = fl[2]
        dx, dy = df * fwx, df * fwy - du
        if n.startswith("leg") and abs(fwx) < 0.5:
            dx, dy = 0.0, -du                             # front / back: steps are vertical only
        offs[n] = (dx, dy)
        l4 = rot4(up4(cells[n]), rot, piv[n]) if n in piv else up4(cells[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4
    if abs(lean) > 0.3 and abs(fwx) > 0.3:
        canvas = rot4(canvas, lean * side, (CELL_W // 2, FOOT_Y))
    canvas = shift4(canvas, gf * fwx, gf * fwy - gu)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, offs, (gf * fwx, gf * fwy - gu)


# ---------------------------------------------------------------- effects (reserved colours only)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def line(a, x0, y0, x1, y1, col):
    k = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for j in range(k + 1):
        u = j / k
        put(a, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, col)


def arc(a, cx, cy, rx, ry, a0, a1, col, col2=None):
    steps = int(abs(a1 - a0) * max(rx, ry)) + 4
    for j in range(steps + 1):
        t = a0 + (a1 - a0) * j / steps
        put(a, cx + math.cos(t) * rx, cy + math.sin(t) * ry, col)
        if col2:
            put(a, cx + math.cos(t) * (rx - 1), cy + math.sin(t) * (ry - 1), col2)


def draw_fx(a, fx, R, crown, front, feet):
    fwx, fwy = R["fwd"]
    side = -1.0 if fwx < 0 else 1.0
    for kind, k in fx:
        if kind == "charge":
            # sun light gathers at the flower crown: converging rays + a growing core
            r = 14 - 2 * k
            for j in range(8):
                t = j * math.pi / 4 + k * 0.4
                line(a, crown[0] + math.cos(t) * r, crown[1] + math.sin(t) * r * 0.7,
                     crown[0] + math.cos(t) * (r - 3), crown[1] + math.sin(t) * (r - 3) * 0.7, FX_LO if j % 2 else FX_HI)
            rc = 1 + k                                          # growing sun orb above the crown
            for dy in range(-rc, rc + 1):
                for dx in range(-rc, rc + 1):
                    d = math.hypot(dx, dy)
                    if d <= rc:
                        put(a, crown[0] + dx, crown[1] - 4 - rc + dy, FX_HI if d < rc - 0.8 else FX_LO)
        elif kind == "slash":
            # two crescent leaf-slash arcs sweeping in front of the body
            rr = 10 + (k - 5) * 3
            if abs(fwx) > 0.3:
                c = (front[0] - side * 4, front[1] - 6)
                base = 0.0 if side > 0 else math.pi
                arc(a, c[0], c[1], rr, rr * 0.8, base - 1.1, base + 1.1, FX_HI, FX_LO)
                arc(a, c[0] - side * 4, c[1] + 6, rr - 3, rr * 0.6, base - 0.9, base + 0.9, FX_LO)
            elif fwy > 0:
                c = (front[0], front[1] + 4)
                base = math.pi / 2
                arc(a, c[0] - 10, c[1], rr, rr * 0.5, base - 1.2, base + 1.2, FX_HI, FX_LO)
                arc(a, c[0] + 10, c[1], rr, rr * 0.5, base - 1.2, base + 1.2, FX_HI, FX_LO)
            else:
                # back view: the leaves whip outward -> crescents just outside both leaf tips
                hw, cy = front
                for sd in (-1, 1):
                    cx = CELL_W / 2 + sd * (hw - rr + 4)
                    b0 = 0.0 if sd > 0 else math.pi
                    arc(a, cx, cy, rr, rr * 0.9, b0 - 1.0, b0 + 1.0, FX_HI, FX_LO)
        elif kind == "quake":
            # stomp shockwave: flat rings on the ground in front of the front feet
            rx = 8 + (k - 6) * 7
            ry = max(2, rx // 4)
            cx, cy = feet
            col = FX_HI if k < 8 else FX_LO
            arc(a, cx, cy, rx, ry, math.pi * 0.95, math.pi * 2.05, col)
            if k >= 7:
                arc(a, cx, cy, rx - 6, max(2, ry - 2), math.pi, math.pi * 2, FX_LO)
        elif kind == "bits":
            # leaf bits flying out from the impact
            for j in range(6):
                t = -math.pi / 2 + (j - 2.5) * 0.45 + (0 if abs(fwx) < 0.3 else side * 0.6)
                d = 6 + (k - 6) * 4 + (j % 3)
                x, y = feet[0] + math.cos(t) * d, feet[1] - 4 + math.sin(t) * d * 0.8 + (k - 6) ** 2 * 0.35
                put(a, x, y, FX_LO)
                put(a, x + 1, y, FX_LO)
                put(a, x, y - 1, FX_HI)
        elif kind == "hit":
            cx, cy = front
            r = 4 + k * 2
            for j in range(6):
                t = j * math.pi / 3 + k * 0.5
                line(a, cx + math.cos(t) * (r - 3), cy + math.sin(t) * (r - 3), cx + math.cos(t) * r, cy + math.sin(t) * r, FX_HI)


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

    s_height = official_height()
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw, s_height)
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % FX_HI, "#%02x%02x%02x" % FX_LO,
                  "--balance", "--name", "venusaur-mega", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    rig, placed = {}, {}
    for v in RIG:
        layers, masks = build_rig(v, still[v])
        cells, piv = {}, {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v)
        for n, p in RIG[v]["pivot"].items():
            piv[n] = (p[0] + ox, p[1] + oy)
        rig[v] = (cells, piv, (ox, oy))
        placed[v] = place_layer(still[v], v)[0]
    for v in RIG:
        _, masks = build_rig(v, still[v])
        dbg = np.zeros(still[v].shape[:2] + (3,), np.uint8)
        cols = [(90, 90, 90), (60, 200, 60), (30, 120, 30), (240, 120, 160), (220, 200, 40), (60, 120, 240), (40, 220, 220), (240, 140, 40), (180, 80, 240)]
        for i, (n, m) in enumerate(masks.items()):
            dbg[m] = cols[i % len(cols)]
        (work / "qa").mkdir(exist_ok=True)
        Image.fromarray(dbg).resize((dbg.shape[1] * 4, dbg.shape[0] * 4), Image.NEAREST).save(work / "qa" / f"parts_{v}.png")

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
                fo = offs.get("flower", (0, 0))
                crown = (R["crown"][0] + ox + gx + fo[0], R["crown"][1] + oy + gy + fo[1])
                fwx, fwy = R["fwd"]
                ys, xs = np.nonzero(a[..., 3] > 0)
                if abs(fwx) > 0.3:
                    fx_x = xs.min() - 2 if fwx < 0 else xs.max() + 2
                    front = (fx_x, FOOT_Y - 22 + gy)
                    feet = (fx_x + (6 if fwx < 0 else -6), FOOT_Y)
                else:
                    if fwy > 0:
                        front = (CELL_W / 2 + gx, ys.max() - 6)
                    else:
                        front = ((xs.max() - xs.min()) / 2, ys.min() + 34)     # (half width, leaf height)
                    feet = (CELL_W / 2 + gx, FOOT_Y)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, R, crown, front, feet)
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (3, 6, 7, 10)):
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
    alpha_bad = int(np.count_nonzero((master[..., 3] > 0) & (master[..., 3] < 255)))
    clip = {}
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(N):
                c = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] > 0
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    still_diff = {}
    for v in RIG:
        f0 = np.array(Image.open(frames_dir / v / "idle" / "00.png").convert("RGBA"))
        still_diff[v] = int(np.count_nonzero(np.any(f0 != placed[v], axis=2)))
    s_h = int(np.ptp(np.nonzero(placed["S"][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off), "alpha_partial": alpha_bad,
              "edge_touching_frames": clip, "still_diff": still_diff, "s_height": s_h, "official_height": s_height}
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
    meta = {"generator": "forge_gen1_0003_venusaur_mega_actions", "version": VERSION, "key": KEY, "dex": 3,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["art/ref_front.png", "art/ref_back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(bounds[3]), "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else (v.get("policy") if "policy" in v else v)) for k, v in report.items()}, ensure_ascii=False))

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
