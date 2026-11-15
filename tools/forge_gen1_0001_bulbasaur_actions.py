#!/usr/bin/env python3
"""0001_bulbasaur (normal): 8-direction x 4-action sheet, same flow as forge_raichu_mega_x_actions.py.

Source (2026-11-04, md/gen1-forge.md):
  art/ref_front.png / art/ref_back.png = official front.png / back.png x8 nearest.
  The Antigravity CLI (`agy -p`, prompt art/agy_prompt.md, no proper nouns) drew the key views:
    art/out/keys_2x2.png  : top-left S, top-right W (side, faces left), bottom-left NE, bottom-right N
    art/out/sw_single.png : SW (3/4 front, faces down-left)
  SE = SW, E = W, NW = NE mirrored.

Creature (counted on the official sprite): squat four-legged teal body, big wide head with two small ears, two red
eyes, wide mouth, ONE yellow-green bulb on the back, no tail.

Pipeline: key white (border-connected) -> area downscale (S body height = official front.png opaque height, 33 px;
all grid views share that scale, SW matched on height) -> palette_tool build / lock / apply -> outer light edge
pixels -> darkest outline colour -> part rig (bulb by colour, head / legs by polygons, body = rest; cushions under
seams) -> poses at 4x nearest -> 1x -> QC -> sheets -> sync agent-sprite-forge/ to pokemon_arena_2d.

Actions (16 frames each):
  idle   : breathing, the bulb swells / sways a little, head nods. Feet stay on the ground.
  walk   : four legs step in diagonal pairs (near-front + far-hind, then the other pair), body bob, bulb sway.
  attack : the bulb gathers light and lifts (crouch back) -> head-first tackle while sharp leaves shoot out of the
           bulb toward the target -> impact star -> recover.
  hurt   : white flash, recoil back, head thrown up, bulb lags, dizzy stars over the head, return.

Never touches the official front/back png/gif.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0001_bulbasaur_actions.py [--force] [--no-sync]
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
KEY = "0001_bulbasaur"
REL = "monsterSprite/gen1/0001_bulbasaur/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0001_bulbasaur_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 104, 72, 62
UP = 4
FLASH, GLOW = (255, 255, 255), (236, 255, 170)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# poly: part -> list of polygons. Bulb = bulb-coloured pixels (+ their dark rim). body = everything else.
# legs: (name, pair) pair "a" = near-front + far-hind, "b" = the other diagonal.
RIG = {
    "S": dict(
        poly={"head": [[(1, 6), (28, 6), (28, 23), (1, 23)]],
              "leg_fl": [[(4, 27), (11, 27), (11, 33), (4, 33)]],
              "leg_fr": [[(18, 27), (25, 27), (25, 33), (18, 33)]],
              "leg_bl": [[(0, 24), (4, 24), (4, 32), (0, 32)]],
              "leg_br": [[(25, 24), (29, 24), (29, 32), (25, 32)]]},
        legs={"leg_fl": "a", "leg_br": "a", "leg_fr": "b", "leg_bl": "b"},
        draw=["leg_bl", "leg_br", "bulb", "body", "head", "leg_fl", "leg_fr"],
        head_piv=(14, 22), body_x=14, fwd=(0.0, 1.0), back=False),
    "SW": dict(
        poly={"head": [[(0, 4), (23, 4), (24, 12), (24, 24), (0, 24)]],
              "leg_fl": [[(5, 26), (15, 26), (15, 33), (5, 33)]],
              "leg_fr": [[(15, 27), (25, 27), (25, 33), (15, 33)]],
              "leg_b": [[(26, 25), (36, 25), (36, 32), (26, 32)]]},
        legs={"leg_fl": "a", "leg_b": "a", "leg_fr": "b"},
        draw=["leg_b", "bulb", "body", "leg_fr", "head", "leg_fl"],
        head_piv=(18, 22), body_x=19, fwd=(-0.9, 0.45), back=False),
    "W": dict(
        poly={"head": [[(0, 3), (15, 3), (16, 12), (16, 22), (0, 22)]],
              "leg_f": [[(6, 24), (17, 24), (17, 32), (6, 32)]],
              "leg_b": [[(21, 24), (34, 24), (34, 32), (21, 32)]]},
        legs={"leg_f": "a", "leg_b": "b"},
        draw=["bulb", "body", "leg_b", "leg_f", "head"],
        head_piv=(14, 18), body_x=18, fwd=(-1.0, 0.0), back=False),
    "NE": dict(
        poly={"head": [[(15, 0), (33, 0), (33, 20), (15, 20)]],
              "leg_bl": [[(0, 25), (12, 25), (12, 32), (0, 32)]],
              "leg_br": [[(13, 26), (22, 26), (22, 32), (13, 32)]],
              "leg_f": [[(23, 25), (33, 25), (33, 32), (23, 32)]]},
        legs={"leg_bl": "a", "leg_f": "a", "leg_br": "b"},
        draw=["leg_f", "head", "body", "leg_br", "leg_bl", "bulb"],
        head_piv=(22, 18), body_x=16, fwd=(0.9, -0.45), back=True),
    "N": dict(
        poly={"head": [[(0, 2), (9, 2), (9, 13), (0, 13)], [(20, 2), (29, 2), (29, 13), (20, 13)]],
              "leg_bl": [[(0, 27), (11, 27), (11, 33), (0, 33)]],
              "leg_br": [[(18, 27), (29, 27), (29, 33), (18, 33)]]},
        legs={"leg_bl": "a", "leg_br": "b"},
        draw=["head", "body", "leg_bl", "leg_br", "bulb"],
        head_piv=(14, 14), body_x=14, fwd=(0.0, -1.0), back=True),
}
KEYS = {"S": "keys", "W": "keys", "NE": "keys", "N": "keys", "SW": "sw"}
OFFICIAL_H = None      # measured from front.png


# ---------------------------------------------------------------- 1. stills
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 225
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 200)
    out = np.dstack([a, np.where(bg, 0, 255)]).astype(np.uint8)
    lab2, n2 = ndi.label(out[..., 3] > 0)
    if n2:
        sz = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
        out[~np.isin(lab2, [i + 1 for i, s in enumerate(sz) if s > 200])] = 0
    return out


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))
    ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


def make_stills(dst: Path, target_h):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    crops = {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        crops[v] = im.crop(im.getbbox())
    grid = target_h / crops["S"].height          # S body height = official front height
    scale = {v: grid for v in src}
    scale["SW"] = target_h / crops["SW"].height
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in crops.items():
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


def palette_colors(pj_path):
    pj = json.loads(Path(pj_path).read_text())
    cols = []
    for c in pj.get("colors", []):
        hx = (c["hex"] if isinstance(c, dict) else c).lstrip("#")
        cols.append((int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)))
    return cols


def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def crisp_outline(a, ink):
    """Light pixels on the outer silhouette edge -> outline colour (the official sprites have a dark outline)."""
    op = a[..., 3] > 0
    edge = op & ~ndi.binary_erosion(op, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], border_value=0)
    L = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    m = edge & (L > 75)
    a[m, :3] = ink
    return a


# ---------------------------------------------------------------- 2. rig
def poly_mask(polys, shape):
    m = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        d.polygon(p, fill=255)
    return np.array(m) > 0


def bulb_colour(still):
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    return (still[..., 3] > 0) & (g > 60) & (b < g * 0.62) & (r > b + 10)


def build_rig(view, still):
    R = RIG[view]
    op = still[..., 3] > 0
    names = ["body", "head"] + list(R["legs"]) + ["bulb"]
    lab = np.zeros(op.shape, int)
    lab[op] = 1
    for i, n in enumerate(names):
        if n in R["poly"]:
            lab[poly_mask(R["poly"][n], op.shape) & op] = i + 1
    bc = bulb_colour(still)
    # the bulb also owns its dark rim: dark pixels touching bulb colour that are not next to skin
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    dark = op & (np.maximum(np.maximum(r, g), b) < 70)
    skin = op & ~bc & ~dark
    rim = dark & ndi.binary_dilation(bc, iterations=1) & ~ndi.binary_dilation(skin, iterations=1)
    bulb = bc | rim
    # legs keep their pixels (feet never belong to the bulb)
    legm = np.isin(lab, [names.index(n) + 1 for n in R["legs"]])
    lab[bulb & ~legm] = names.index("bulb") + 1
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
        cushion = 3 if n in ("body", "bulb") else 2
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        if n == "body":
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(op, 4), iterations=4))[4:-4, 4:-4]
            inside = ndi.binary_erosion(inside, iterations=1)
            seam |= above & inside & ~masks["bulb"]
        if seam.any() and m.any():
            fillable = m & (np.maximum(np.maximum(r, g), b) > 70)
            if not fillable.any():
                fillable = m
            _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
            sy, sx = np.nonzero(seam)
            lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
            lyr[sy, sx, 3] = 255
        layers[n] = lyr
    piv = {"head": R["head_piv"]}
    for n in masks:
        ys, xs = np.nonzero(masks[n])
        if not len(ys):
            continue
        if n == "bulb":
            piv[n] = (float(xs.mean()), float(ys.max()))
        elif n.startswith("leg"):
            piv[n] = (float(xs.mean()), float(ys.min()))
        elif n == "body":
            piv[n] = (float(xs.mean()), float(ys.max()))
    return layers, masks, piv


def place_layer(layer, view):
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - RIG[view]["body_x"]
    oy = FOOT_Y - (h - 1)
    cell[oy:oy + h, ox:ox + w] = layer
    return cell, (ox, oy)


# ---------------------------------------------------------------- 3. poses
def wave(f, amp, cycles=1.0, phase=0.0):
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """Translations only (no part rotation: a 30 px sprite turns jaggy when rotated).
    part -> [unused, fwd px, up px]; 'bulb_x' = screen-x sway of the bulb; 'g' = [fwd, up, lean]."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        P["body"] = [0, 0, round(wave(f, 0.5, 1, -math.pi / 2))]
        P["head"] = [0, 0, round(wave(f, 0.9, 1, -math.pi / 2 - 0.6))]
        P["bulb"] = [0, 0, round(wave(f, 1.0, 1, -math.pi / 2 - 0.3))]
        P["bulb_x"] = round(wave(f, 1.0, 1, 0.0))
    elif action == "walk":
        P["body"] = [0, 0, round(abs(wave(f, 1.0, 2, 0.0)))]
        P["head"] = [0, 0, round(abs(wave(f, 1.0, 2, -0.5)))]
        P["bulb"] = [0, 0, round(abs(wave(f, 1.0, 2, -0.9)))]
        P["bulb_x"] = round(wave(f, 1.0, 1, 0.0))
        s = math.sin(2 * math.pi * f / N)
        # pair a swings forward while lifted (s > 0), pair b on the other half of the cycle
        P["leg_a"] = [0, round(2.0 * s), round(2.0 * max(0.0, s))]
        P["leg_b"] = [0, round(-2.0 * s), round(2.0 * max(0.0, -s))]
    elif action == "attack":
        # charge (bulb gathers light, body crouches back) -> head-first tackle + leaves -> impact -> recover
        g_f = [0, -1, -2, -3, -3, 2, 7, 10, 10, 9, 7, 5, 3, 2, 1, 0][f]
        g_u = [0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0][f]
        P["g"] = [g_f, g_u, 0]
        P["body"] = [0, 0, [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["head"] = [0, [0, 0, -1, -1, -1, 1, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0][f], [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["bulb"] = [0, [0, 0, -1, -1, -1, -1, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0][f], [0, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["leg_a"] = [0, [0, 0, 0, 0, 0, 2, 3, 3, 3, 2, 2, 1, 1, 0, 0, 0][f], [0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["leg_b"] = [0, [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0][f], 0]
        if 1 <= f <= 4:
            fx.append(("charge", f))
        if 5 <= f <= 11:
            fx.append(("leaves", f))
        if f in (7, 8, 9):
            fx.append(("impact", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [round(-5.0 * env) + jit, round(1.0 * env), 0]
        P["head"] = [0, round(-1.0 * env), round(1.0 * env)]
        P["bulb"] = [0, round(-1.4 * env), round(-1.0 * env)]
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 9:
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


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    side = -1.0 if fwx < 0 else 1.0
    sideways = abs(fwx) > 0.3
    zero = [0.0, 0.0, 0.0]
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    offs = {}
    for n in R["draw"]:
        rot, df, du, bx = 0.0, 0.0, 0.0, 0
        b = P.get("body", zero)
        if n == "body":
            df, du = b[1], b[2]
        elif n == "head":
            h = P.get("head", zero); df, du = h[1], h[2]
        elif n == "bulb":
            bb = P.get("bulb", zero)
            df, du = bb[1], bb[2]
            bx = P.get("bulb_x", 0)
        elif n.startswith("leg"):
            lg = P.get("leg_" + R["legs"][n], zero)
            df, du = lg[1], lg[2]
            if not sideways:
                df = 0.0
        dx, dy = df * fwx, df * fwy - du
        if not sideways and n != "body":
            dy = -du + (df * fwy if n == "head" else 0.0)
        if n == "bulb":
            dx += bx
        dx, dy = round(dx), round(dy)
        offs[n] = (dx, dy)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if not sideways:
        gf *= 0.4
    if abs(lean) > 0.3 and sideways:
        canvas = rot4(canvas, lean * side, (CELL_W // 2, FOOT_Y))
    gx, gy = round(gf * fwx), round(gf * fwy - gu)
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()
    return out, fx, offs, (gx, gy)


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


LEAF_SHAPE = ["..o....",
              ".omMo..",
              "omMMmo.",
              ".omMMmo",
              "..omMo.",
              "...oo.."]          # diagonal leaf, tip at the lower right (o rim, m body, M midrib light)


def leaf(a, x, y, ang, cols):
    """A small sharp leaf stamped in one of 4 orientations (no rotation blur): dark rim, body, light midrib."""
    dk, md, lt = cols
    g = [list(r) for r in LEAF_SHAPE]
    q = int(round(ang / (math.pi / 2))) % 4
    arr = np.array(g)
    arr = np.rot90(arr, -q)
    h, w = arr.shape
    for yy in range(h):
        for xx in range(w):
            ch = arr[yy, xx]
            if ch == ".":
                continue
            put(a, x - w // 2 + xx, y - h // 2 + yy, {"o": dk, "m": md, "M": lt}[ch])


def star(a, x, y, r, c1, c2):
    for t in range(-r, r + 1):
        put(a, x + t, y, c1 if abs(t) < r else c2)
        put(a, x, y + t, c1 if abs(t) < r else c2)
    if r >= 3:
        for t in (-1, 1):
            put(a, x + t, y + t, c2)
            put(a, x + t, y - t, c2)


def draw_fx(a, fx, fwd, bulb_top, head_front, centre, leafc, head_top=0):
    fwx, fwy = fwd
    for kind, k in fx:
        if kind == "charge":
            # light gathers on the bulb tip: twinkles that close in
            for j in range(3):
                ang = j * 2.1 + k * 0.7
                r = 9 - 2 * k
                put(a, bulb_top[0] + math.cos(ang) * r, bulb_top[1] - 1 + math.sin(ang) * r * 0.7, GLOW)
            if k >= 3:
                star(a, bulb_top[0], bulb_top[1] - 1, 1 if k == 3 else 2, FLASH, GLOW)
        elif kind == "leaves":
            # three sharp leaves pop out of the bulb tip, arc up and fly at the target (above the face, never
            # across it); toward the camera they fan out to both sides
            t = k - 5
            for j, spread in enumerate((-1, 0, 1)):
                d = 3 + 6 * t - 3 * abs(spread) - (j == 1) * 2
                if d < 2:
                    continue
                if abs(fwx) > 0.3:
                    d = 3 + 7 * t - 6 * j                                   # a stream of leaves, one behind the other
                    if d < 2:
                        continue
                    x = bulb_top[0] + fwx * d
                    y = bulb_top[1] - 3 - 3 * math.sin(min(1.0, d / 16) * math.pi) + fwy * d * 0.6 + spread
                elif fwy > 0:                                            # toward the camera: fan out sideways
                    x = bulb_top[0] + spread * (5 + d * 0.9) + (0 if spread else 0)
                    y = bulb_top[1] - 3 + (d * 0.35 if spread else -d * 0.5)
                else:                                                    # away: up the screen
                    x = bulb_top[0] + spread * (2 + d * 0.3)
                    y = bulb_top[1] - 3 - d
                leaf(a, x, y, (t + j) * math.pi / 2, leafc)
        elif kind == "impact":
            r = {7: 3, 8: 5, 9: 3}[k]
            hx, hy = head_front
            star(a, hx, hy, r, FLASH, GLOW)
        elif kind == "stars":
            # dizzy stars circling over the head
            for j in range(3):
                ang = j * 2.1 + k * 0.8
                cx, cy = centre[0] + math.cos(ang) * 11, head_top + math.sin(ang) * 3
                star(a, cx, cy, 1, FLASH, GLOW)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FLASH
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
        h.update((ART / n).read_bytes())
    h.update((FORM_DIR / "front.png").read_bytes())
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

    target_h = official_height()
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw, target_h)
        run_tool(["build", "--input", str(raw), "--colors", "24", "--reserve", "#%02x%02x%02x" % FLASH, "#%02x%02x%02x" % GLOW,
                  "--balance", "--name", "bulb-quad", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        pal = palette_colors(lock_dir / "palette.json")
        ink = min(pal, key=lum)
        for v in RIG:
            p = stills / f"{v}.png"
            a = crisp_outline(np.array(Image.open(p).convert("RGBA")), ink)
            Image.fromarray(a, "RGBA").save(p)
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    pal = palette_colors(lock_dir / "palette.json")
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}

    # leaf colours from the palette (the bulb's own greens)
    def nearest(c):
        return min(pal, key=lambda p: sum((int(p[i]) - c[i]) ** 2 for i in range(3)))
    leafc = (nearest((82, 98, 41)), nearest((140, 200, 50)), nearest((189, 255, 115)))

    rig = {}
    for v in RIG:
        layers, masks, piv0 = build_rig(v, still[v])
        cells = {}
        for n, lyr in layers.items():
            cells[n], (ox, oy) = place_layer(lyr, v)
        piv = {n: (p[0] + ox, p[1] + oy) for n, p in piv0.items()}
        rig[v] = (cells, piv, (ox, oy), masks)

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        cells, piv, (ox, oy), masks = rig[v]
        R = RIG[v]
        bm = cells["bulb"][..., 3] > 0
        bys, bxs = np.nonzero(bm)
        top_y = bys.min()
        bulb_top0 = (float(bxs[bys <= top_y + 2].mean()), float(top_y))
        allm = np.zeros(bm.shape, bool)
        for c in cells.values():
            allm |= c[..., 3] > 0
        top0 = int(np.nonzero(allm.any(axis=1))[0].min())
        hm = cells["head"][..., 3] > 0
        hys, hxs = np.nonzero(hm)
        fwx, fwy = R["fwd"]
        if abs(fwx) > 0.3:
            hx = hxs.min() if fwx < 0 else hxs.max()
            head_front0 = (float(hx + fwx * 2), float(hys.mean() + 2))
        else:
            head_front0 = (float(hxs.mean()), float(hys.max() + 2 if fwy > 0 else hys.min() - 2))
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, offs, (gx, gy) = compose(v, cells, piv, act, f)
                bo = offs.get("bulb", (0, 0))
                ho = offs.get("head", (0, 0))
                bulb_top = (bulb_top0[0] + bo[0] + gx, bulb_top0[1] + bo[1] + gy)
                head_front = (head_front0[0] + ho[0] + gx, head_front0[1] + ho[1] + gy)
                centre = (CELL_W / 2 + gx, FOOT_Y - 12 + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, R["fwd"], bulb_top, head_front, centre, leafc, top0 + gy - 4)
                Image.fromarray(a, "RGBA").save(od / f"{f:02d}.png")
            st["done"].append(key); save_st()
        print("frames", v, flush=True)

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
    cols = set(pal)
    aa = master[..., 3] > 0
    used = {tuple(int(x) for x in c) for c in np.unique(master[aa][:, :3], axis=0)}
    off = sorted(used - cols)
    alpha_bad = int(((master[..., 3] > 0) & (master[..., 3] < 255)).sum())
    clip = {}
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(N):
                c = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] > 0
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    # still diff: idle frame 0 (and every action's frame 0) == the still placed on the cell
    still_diff = 0
    for v in RIG:
        ref, _ = place_layer(still[v], v)
        for act in ACTIONS:
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            still_diff += int((a0 != ref).any(axis=2).sum())
    loop_diff = 0
    for v in RIG:
        for act in ("idle", "walk"):
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            a15 = np.array(Image.open(frames_dir / v / act / "15.png").convert("RGBA"))
            loop_diff += int((a0 != a15).any(axis=2).sum())
    S_h = int(np.ptp(np.nonzero((still["S"][..., 3] > 0).any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "semi_alpha_px": alpha_bad, "edge_touching_frames": clip, "still_diff_px": still_diff,
              "loop_end_diff_px(info)": loop_diff, "S_height": S_h, "official_front_height": target_h}
    if not args.skip_qa and GAIT_LOOP.exists():
        for act, kind in (("idle", "idle"), ("walk", "gait")):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FPY), "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / "S" / act), "--fps", "13",
                                "--output-dir", str(od), "--kind", kind, "--no-aids"], capture_output=True, text=True)
            rep = od / "loop-report.json"
            rec = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr.strip().splitlines() or ["no report"])[-1]
            report[f"gait_{act}"] = rec
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    bounds = []
    for di in range(8):
        c = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W, 3] > 0
        ys, xs = np.nonzero(c)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    meta = {"generator": "forge_gen1_0001_bulbasaur_actions", "version": VERSION, "key": KEY, "dex": 1,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["front.png", "back.png"],
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
