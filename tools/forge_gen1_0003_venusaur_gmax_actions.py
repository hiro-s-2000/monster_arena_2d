#!/usr/bin/env python3
"""Gigantamax Venusaur (0003_venusaur-gmax): 8-direction stills + 8-direction x 4-action sheet.

Batch md/gen1-forge.md (2026-11-04), same flow as tools/forge_raichu_mega_x_actions.py:
  art/ref_front.png / ref_back.png  : official front.png / back.png x8 nearest (references for the Antigravity CLI)
  art/agy_prompt.md                 : morphological prompt (no proper nouns, exact part counts and colours)
  art/out/keys_2x2.png              : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png             : SW (3/4 front, faces down-left)
  Mirrors: SE = SW, E = W, NW = NE.

Pipeline: key the white backdrop (border-connected only: claws, teeth and eye whites stay) -> one game scale
(S still height = opaque height of the official front.png = 89 px; SW is matched to the same height) ->
agent-sprite-forge palette_tool build / lock / apply -> part rig (flower dome, crown, head, body with fern fronds,
the visible stubby legs) -> poses at 4x nearest, sampled back to 1x -> QC -> agent-sprite-forge/ -> sync.

Actions (16 frames each):
  idle   : heavy breathing; the flower dome rises a beat after the body, the crown sways.
  walk   : diagonal quadruped gait (front-left + rear-right, then front-right + rear-left), body bob, flower sway.
  attack : rears up on the hind legs while the crown gathers light -> slams the front legs down, the head thrusts,
           the flower dome swings forward and throws a storm of its own petals forward, shock ring at the feet
           -> recovers.
  hurt   : white flash, recoil backwards (lean <= 7 deg), the flower jolts back and sheds two petals, recover.

The official front / back png / gif are never written.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0003_venusaur_gmax_actions.py [--force] [--no-sync]
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
KEY = "0003_venusaur-gmax"
REL = "monsterSprite/gen1/0003_venusaur/gigantamax/venusaur-gmax"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0003_venusaur_gmax_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 224, 140, 128
UP = 4
SOL_HI, SOL_LO = (255, 253, 226), (255, 240, 140)      # reserved: the crown's light / shock ring only
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))
    ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# flower : only petal-coloured pixels (red / pink / brown + their outline) inside the polygon belong to it.
# legs   : pair "a" = front-left + rear-right, pair "b" = front-right + rear-left (diagonal gait).
# fwd    : screen direction of "forward"; rear = pivot of the rearing lean (hind feet).
RIG = {
    "S": dict(
        poly={"crown": [(38, 0), (86, 0), (84, 21), (70, 24), (56, 23), (40, 21)],
              "flower": [(0, 14), (137, 14), (137, 62), (0, 62)],
              "head": [(28, 54), (110, 54), (112, 70), (100, 84), (38, 84), (26, 70)],
              "leg_fl": [(10, 68), (32, 68), (40, 80), (42, 89), (8, 89)],
              "leg_fr": [(106, 68), (130, 68), (130, 89), (96, 89), (98, 80)]},
        draw=["body", "leg_fl", "leg_fr", "head", "flower", "crown"],
        pivot={"crown": (62, 21), "flower": (69, 44), "head": (69, 60), "body": (69, 88),
               "leg_fl": (26, 70), "leg_fr": (112, 70)},
        pair={"leg_fl": "a", "leg_fr": "b"}, front_legs=["leg_fl", "leg_fr"],
        body_x=69, fwd=(0.0, 1.0), rear=None),
    "SW": dict(
        poly={"crown": [(32, 0), (76, 0), (74, 19), (62, 22), (40, 21), (32, 19)],
              "flower": [(0, 12), (100, 12), (100, 66), (88, 60), (70, 52), (56, 52), (50, 54), (0, 54)],
              "head": [(0, 50), (14, 45), (34, 43), (52, 46), (60, 56), (58, 68), (46, 76), (20, 78), (6, 72), (0, 62)],
              "leg_fl": [(8, 76), (34, 76), (34, 89), (8, 89)],
              "leg_fr": [(52, 74), (72, 72), (72, 89), (52, 89)],
              "leg_rr": [(74, 72), (96, 72), (96, 89), (74, 89)]},
        draw=["leg_rr", "body", "leg_fr", "head", "leg_fl", "flower", "crown"],
        pivot={"crown": (54, 20), "flower": (54, 46), "head": (52, 60), "body": (52, 88),
               "leg_fl": (21, 77), "leg_fr": (62, 74), "leg_rr": (84, 73)},
        pair={"leg_fl": "a", "leg_rr": "a", "leg_fr": "b"}, front_legs=["leg_fl", "leg_fr"],
        body_x=52, fwd=(-0.9, 0.45), rear=(84, 88)),
    "W": dict(
        poly={"crown": [(52, 0), (92, 0), (90, 17), (54, 17)],
              "flower": [(4, 8), (137, 8), (137, 50), (50, 50), (44, 41), (4, 41)],
              "head": [(0, 44), (14, 38), (40, 38), (48, 46), (48, 64), (38, 72), (30, 80), (4, 80), (0, 70)],
              "leg_f": [(32, 70), (56, 64), (64, 72), (64, 89), (30, 89)],
              "leg_r": [(94, 68), (124, 66), (128, 89), (94, 89)]},
        draw=["body", "leg_r", "leg_f", "head", "flower", "crown"],
        pivot={"crown": (72, 16), "flower": (70, 44), "head": (46, 56), "body": (70, 89),
               "leg_f": (46, 68), "leg_r": (108, 70)},
        pair={"leg_f": "a", "leg_r": "b"}, front_legs=["leg_f"],
        body_x=70, fwd=(-1.0, 0.0), rear=(110, 89)),
    "NE": dict(
        poly={"crown": [(38, 0), (72, 0), (70, 16), (40, 16)],
              "flower": [(0, 6), (112, 6), (112, 50), (0, 50)],
              "leg_rl": [(0, 70), (32, 70), (32, 92), (0, 92)],
              "leg_rr": [(52, 78), (86, 76), (88, 92), (52, 92)],
              "leg_fr": [(92, 64), (112, 64), (112, 92), (90, 92)]},
        draw=["leg_fr", "body", "leg_rl", "leg_rr", "flower", "crown"],
        pivot={"crown": (55, 15), "flower": (56, 42), "body": (56, 92),
               "leg_rl": (16, 72), "leg_rr": (70, 78), "leg_fr": (102, 66)},
        pair={"leg_rr": "a", "leg_rl": "b", "leg_fr": "b"}, front_legs=["leg_fr"],
        body_x=56, fwd=(0.9, -0.45), rear=(40, 92)),
    "N": dict(
        poly={"crown": [(38, 0), (78, 0), (76, 20), (42, 20)],
              "flower": [(0, 12), (116, 12), (116, 54), (0, 54)],
              "leg_rl": [(4, 78), (36, 78), (36, 93), (4, 93)],
              "leg_rr": [(76, 78), (114, 78), (114, 93), (76, 93)]},
        draw=["body", "leg_rl", "leg_rr", "flower", "crown"],
        pivot={"crown": (58, 19), "flower": (58, 46), "body": (58, 92), "leg_rl": (20, 80), "leg_rr": (92, 80)},
        pair={"leg_rr": "a", "leg_rl": "b"}, front_legs=[],
        body_x=58, fwd=(0.0, -1.0), rear=None),
}


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


def colour_classes(still):
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    op = still[..., 3] > 0
    mx = np.maximum(np.maximum(r, g), b)
    red = op & (r > 150) & (r - g > 40)                              # petals, pink speckles
    brown = op & (r > 90) & (r - b > 60) & (g < 130) & (r - g > 30)  # seams, petal undersides
    yellow = op & (r > 170) & (g > 140) & (b < 140) & (r - b > 60)   # crown
    dark = op & (mx < 70)
    return red, brown, yellow, dark


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    shape = op.shape
    names = R["draw"]
    red, brown, yellow, dark = colour_classes(still)
    lab = np.zeros(shape, int)
    idx = {n: i + 1 for i, n in enumerate(names)}
    lab[op] = idx["body"]
    if "head" in R["poly"]:
        lab[poly_mask(R["poly"]["head"], shape) & op] = idx["head"]
    for n in names:
        if n.startswith("leg"):
            lab[poly_mask(R["poly"][n], shape) & op] = idx[n]
    fp = poly_mask(R["poly"]["flower"], shape)
    petal = fp & (red | brown | yellow)
    petal |= fp & dark & ndi.binary_dilation(petal, iterations=1)
    # small specks of green trapped between petals stay with the body; dark outline next to petals is flower
    lab[petal] = idx["flower"]
    # crown: everything in its polygon except the pinkish petal reds (petals: g ~ b; crown orange: g >> b)
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    petal_red = red & (np.abs(g - b) < 30)
    cp = poly_mask(R["poly"]["crown"], shape) & op
    lab[cp & ~petal_red] = idx["crown"]
    masks = {n: lab == idx[n] for n in names}

    layers = {}
    allop = op
    for i, n in enumerate(names):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in names[i + 1:]:
            above |= masks[n2]
        cushion = {"body": 6, "head": 5, "flower": 4}.get(n, 2)
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion) & allop
        if seam.any() and m.any():
            r, g, b = [still[..., c].astype(int) for c in range(3)]
            fillable = m & (np.maximum(np.maximum(r, g), b) > 70)
            if n == "flower":
                fillable &= ~(still[..., :3].min(axis=2) > 200)          # not the pale speckles
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
def wave(amp, cycles, f, phase=0.0):
    """0 at frame 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """Per part: (tilt deg toward forward, fwd px, up px). Global: (fwd px, up px, rear deg). Effects list."""
    P = {}
    G = [0.0, 0.0, 0.0]
    fx = []
    ph = 2 * math.pi * f / N
    if action == "idle":
        P["body"] = (0, 0, wave(1.0, 1, f))
        P["head"] = (0, 0, wave(1.0, 1, f, -0.7))
        P["flower"] = (wave(1.2, 1, f, -0.5), 0, wave(1.4, 1, f, -1.0))
        P["crown"] = (wave(3.0, 2, f), 0, wave(1.4, 1, f, -1.0))
    elif action == "walk":
        bob = 1.2 * abs(math.sin(ph))
        P["body"] = (0, 0, bob)
        P["head"] = (0, 0.8 * math.sin(2 * ph), bob)
        P["flower"] = (1.5 * math.sin(2 * ph - 0.6), 0, 1.2 * abs(math.sin(ph - 0.5)))
        P["crown"] = (4.0 * math.sin(2 * ph - 1.0), 0, 1.2 * abs(math.sin(ph - 0.5)))
        P["leg_a"] = (0, 3.2 * math.sin(ph), 2.6 * max(0.0, math.cos(ph)))
        P["leg_b"] = (0, -3.2 * math.sin(ph), 2.6 * max(0.0, -math.cos(ph)))
    elif action == "attack":
        rear = [0, 3, 6, 8, 9, 4, -2, -3, -2, -1, 0, 0, 0, 0, 0, 0][f]
        g_up = [0, 1, 2, 3, 3, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0][f]
        g_f = [0, -1, -2, -2, -2, 2, 4, 5, 5, 4, 3, 2, 1, 1, 0, 0][f]
        G = [g_f, g_up, rear]
        lift = [0, 1, 2, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        P["front_leg"] = (0, [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f], lift)
        P["head"] = (0, [0, 0, -1, -1, -1, 1, 2, 3, 3, 2, 1, 1, 0, 0, 0, 0][f], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f])
        tilt = [0, -1, -2, -3, -3, 2, 6, 7, 6, 4, 3, 2, 1, 1, 0, 0][f]
        P["flower"] = (tilt, 0, [0, 0, 1, 1, 1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0][f])
        P["crown"] = (tilt * 1.5, 0, P["flower"][2])
        P["body"] = (0, 0, 0)
        if 1 <= f <= 5:
            fx.append(("charge", f))
        if 6 <= f <= 11:
            fx.append(("petals", f))
        if 6 <= f <= 9:
            fx.append(("shock", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        G = [-5.0 * env + jit, 1.0 * env, 5.0 * env]
        P["head"] = (0, -1.0 * env, 1.0 * env)
        lag = math.sin(math.pi * min(1.0, max(0.0, (f - 1) / 12.0))) if f < 13 else 0.0
        P["flower"] = (-5.0 * lag, 0, 1.0 * lag)
        P["crown"] = (-8.0 * lag, 0, 1.0 * lag)
        P["front_leg"] = (0, 0, 1.5 * env)
        if f == 1:
            fx.append(("flash", 0))
        if 3 <= f <= 11:
            fx.append(("shed", f))
    return P, G, fx


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
    """img4 at 4x; rotate (+ = clockwise on screen) about piv (1x coords)."""
    if abs(ang) < 0.3:
        return img4
    im = Image.fromarray(img4, "RGBA").rotate(-ang, resample=Image.NEAREST,
                                              center=(piv[0] * UP + UP / 2, piv[1] * UP + UP / 2))
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
    P, G, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    side = 0.0 if abs(fwx) < 0.5 else (-1.0 if fwx < 0 else 1.0)
    diag = 0.6 if (abs(fwx) > 0.5 and abs(fwy) > 0.3) else 1.0
    zero = (0.0, 0.0, 0.0)
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    offs = {}
    for n in R["draw"]:
        if n.startswith("leg"):
            if action == "walk":
                t = P["leg_" + R["pair"][n]]
            elif n in R["front_legs"]:
                t = P.get("front_leg", zero)
            else:
                t = zero
            if action != "walk" and n not in R["front_legs"]:
                t = zero
        else:
            t = P.get(n, zero)
        tilt, df, du = t
        if side == 0.0:
            dx, dy = 0.0, -du + (df * fwy * 0.3 if not n.startswith("leg") else 0.0)
        else:
            dx, dy = df * fwx, df * fwy * 0.4 - du
        ang = tilt * side * diag if side != 0.0 else 0.0
        if n == "crown" and side == 0.0:
            ang = tilt * (0.6 if action != "attack" else 0.25)       # crown wobble in front / back views
        offs[n] = (dx, dy)
        l4 = rot4(up4(cells[n]), ang, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, rear = G
    if side == 0.0:
        gf *= 0.4
    if abs(rear) > 0.3 and side != 0.0 and R["rear"] is not None:
        pv = piv["__rear"]
        canvas = rot4(canvas, -side * rear * (0.75 if diag < 1 else 1.0), pv)
    gdx, gdy = gf * fwx, gf * fwy - gu
    canvas = shift4(canvas, gdx, gdy)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, offs, (gdx, gdy)


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 2 <= x < a.shape[1] - 2 and 2 <= y < a.shape[0] - 2:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def petal(a, x, y, ang, cols):
    """An own-colour petal (about 5x3 px): pale tip, coral body, crimson base, dark outline at the base."""
    lt, md, dk, ink = cols
    c, s = math.cos(ang), math.sin(ang)
    pts = ((-2, 0, ink), (-1, 0, dk), (-1, 1, dk), (-1, -1, dk), (0, 0, md), (0, 1, md), (0, -1, md),
           (1, 0, md), (1, -1, lt), (1, 1, md), (2, 0, lt))
    for u, v, col in pts:
        put(a, x + u * c - v * s, y + u * s + v * c, col)


def ring(a, cx, cy, rx, ry, cols, gap_seed):
    rng = np.random.default_rng(gap_seed)
    for j in range(48):
        t = 2 * math.pi * j / 48
        if rng.random() < 0.18:
            continue
        put(a, cx + rx * math.cos(t), cy + ry * math.sin(t), cols[j % 2])


def draw_fx(a, fx, ctx, pcols):
    fwx, fwy = ctx["fwd"]
    for kind, k in fx:
        if kind == "charge":
            cx, cy = ctx["crown"]
            r = 18 - 2.6 * k                                  # light gathers in toward the crown
            n = 10
            for j in range(n):
                t = j * 2 * math.pi / n + k * 0.45
                x0, y0 = cx + math.cos(t) * r, cy + math.sin(t) * r * 0.75
                x1, y1 = cx + math.cos(t) * (r - 3), cy + math.sin(t) * (r - 3) * 0.75
                for q in range(4):
                    put(a, x0 + (x1 - x0) * q / 3, y0 + (y1 - y0) * q / 3, SOL_HI if q >= 2 else SOL_LO)
            if k >= 3:
                L = 2 + 2 * (k - 2)
                for j in range(-L, L + 1):
                    put(a, cx + j, cy - 2, SOL_HI if abs(j) < L - 1 else SOL_LO)
                    put(a, cx, cy - 2 + j, SOL_HI if abs(j) < L - 1 else SOL_LO)
                for j in (-1, 1):
                    for q in range(1, L // 2 + 1):
                        put(a, cx + j * q, cy - 2 - q, SOL_LO)
                        put(a, cx + j * q, cy - 2 + q, SOL_LO)
        elif kind == "petals":
            rng = np.random.default_rng(100)
            u = (k - 5) / 6.0
            for j in range(18):
                ox, oy, dx, dy = ctx["emit"][j % len(ctx["emit"])]
                L = math.hypot(dx, dy) or 1
                dist = (6 + 46 * u) * (0.55 + 0.6 * rng.random())
                spread = rng.uniform(-0.45, 0.45) * (1 + u)
                sw = math.sin(k * 0.9 + j) * 3
                nx, ny = -dy / L, dx / L
                x = ox + dx / L * dist + nx * (dist * spread + sw)
                y = oy + dy / L * dist + ny * (dist * spread + sw) + 2.0 * u * u * 6
                petal(a, x, y, k * 0.8 + j * 1.3, pcols)
        elif kind == "shock":
            cx, cy = ctx["front_feet"]
            r = 5 + (k - 6) * 7
            ring(a, cx, cy, r, max(2, r * 0.3), (SOL_HI, SOL_LO), 300 + k)
            if k <= 8:
                ring(a, cx, cy, r * 0.6, max(1.5, r * 0.18), (SOL_LO, SOL_HI), 310 + k)
            if k == 6:
                for j in range(-2, 3):
                    put(a, cx + j * 2, cy - 3 - abs(j), SOL_HI)
        elif kind == "shed":
            for j, (sx, sy) in enumerate(ctx["shed"]):
                u = (k - 3)
                petal(a, sx + math.sin(u * 0.9 + j * 2) * 3, sy + u * 2.2, u * 0.7 + j, pcols)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = SOL_HI
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    return h.hexdigest()[:12]


def nearest(cols, target):
    arr = np.array(sorted(cols), float)
    return tuple(int(v) for v in arr[np.argmin(((arr - np.array(target, float)) ** 2).sum(1))])


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
        if work.exists():
            shutil.rmtree(work)
        st = {"hash": h, "done": []}
    work.mkdir(parents=True, exist_ok=True)
    save_st = lambda: st_p.write_text(json.dumps(st))
    target_h = official_height()

    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw, target_h)
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % SOL_HI, "#%02x%02x%02x" % SOL_LO,
                  "--balance", "--name", "venusaur-gmax", "--output-dir", str(pal_dir)])
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
    body_cols = cols - {SOL_HI, SOL_LO}
    pcols = (nearest(body_cols, (255, 189, 189)), nearest(body_cols, (255, 123, 115)),
             nearest(body_cols, (222, 65, 65)), nearest(body_cols, (16, 16, 16)))

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
        if RIG[v]["rear"]:
            piv["__rear"] = (RIG[v]["rear"][0] + ox, RIG[v]["rear"][1] + oy)
        rig[v] = (cells, piv, (ox, oy), masks)
        # debug: part labels
        pal = [(200, 60, 60), (240, 200, 40), (60, 120, 220), (40, 180, 90), (180, 80, 200), (240, 140, 40), (90, 220, 220), (120, 120, 120)]
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
        R = RIG[v]
        fm = cells["flower"][..., 3] > 0
        fys, fxs = np.nonzero(fm)
        flower_c = (float(fxs.mean()), float(fys.mean()))
        fwx, fwy = R["fwd"]
        x0f, x1f, ymid = float(fxs.min()), float(fxs.max()), flower_c[1] + 4
        if abs(fwx) < 0.5:          # front / back views: petals burst out of both rims, away from the face
            emit = [(x0f + 3, ymid, -1.0, 0.25 * fwy), (x1f - 3, ymid, 1.0, 0.25 * fwy),
                    (x0f + 6, ymid - 6, -0.8, -0.5), (x1f - 6, ymid - 6, 0.8, -0.5)]
        else:                       # side / diagonal views: from the front rim, forward
            fr = x0f + 3 if fwx < 0 else x1f - 3
            emit = [(fr, ymid, fwx, fwy - 0.35), (fr, ymid - 5, fwx, fwy - 0.7), (fr, ymid + 4, fwx, fwy + 0.1)]
        crown_m = cells["crown"][..., 3] > 0
        cys, cxs = np.nonzero(crown_m)
        crown_c = (float(cxs.mean()), float(cys.min()) + 4)
        fl = [n for n in R["front_legs"]] or [n for n in R["draw"] if n.startswith("leg")]
        lx = np.mean([np.nonzero(cells[n][..., 3] > 0)[1].mean() for n in fl])
        front_feet = (float(lx) + R["fwd"][0] * 6, FOOT_Y + 1 + (2 if R["fwd"][1] > 0.5 else 0))
        shed = [(fxs.min() + 6, float(fys.max()) - 2), (fxs.max() - 6, float(fys.max()) - 4)]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, offs, (gx, gy) = compose(v, cells, piv, act, f)
                fo = offs.get("flower", (0, 0))
                ctx = {"fwd": R["fwd"],
                       "crown": (crown_c[0] + gx + offs["crown"][0], crown_c[1] + gy + offs["crown"][1]),
                       "emit": [(ex + gx + fo[0], ey + gy + fo[1], dx, dy) for ex, ey, dx, dy in emit],
                       "front_feet": (front_feet[0] + gx, front_feet[1]),
                       "shed": [(x + gx, y + gy) for x, y in shed]}
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, ctx, pcols)
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
    meta = {"generator": "forge_gen1_0003_venusaur_gmax_actions", "version": VERSION, "key": KEY, "dex": 3,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": sprite_h_S, "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) or k == "still_diff_px" else (v.get("policy") if "policy" in v else len(v))) for k, v in report.items()}, ensure_ascii=False))

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
