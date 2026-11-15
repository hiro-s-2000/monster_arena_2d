#!/usr/bin/env python3
"""Mega Latias (0380_latias-mega) v3: 8-direction stills + 8-direction x 4-action sheet from Antigravity CLI key art.

2026-11-04, user instruction: rebuild Mega Latias from the official illustration (art/mega-latias.webp) and a second
illustration (art/ref_gemini.jpeg) with agent-sprite-forge, using only what the Antigravity CLI can do, replacing the
existing implementation INCLUDING front/back png/gif (the user explicitly lifted locked spec 3 for this).

  art/agy_prompt.md -> Antigravity CLI (`agy -p`, image-generator):
    art/out/keys_2x2.png  : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
    art/out/sw_single.png : SW.   SE / E / NW are mirrors of SW / W / NE.
  White key (background = white connected to the border only; the white-grey chest is darker than the key) ->
  area downscale -> agent-sprite-forge palette_tool (build / lock / apply) -> per-view part rig
  (head, body + tail, two wings with their arm connectors and claw hands) -> poses at 4x nearest -> 1x.

  idle   : hover bob, slow wing beats, head trails by a pixel.
  walk   : glide: lean into the flight direction, fast wing beats, speed lines.
  attack : roar (wings rise, psychic rings from the mouth, 0-3) -> 360 degree barrel turn through the real
           8 direction views (4-9) -> dash with streaks and a psychic burst ahead (10-11) -> recover (12-15).
  hurt   : white flash, recoil and lean back, wings fold, sparkles.
  QC: off-palette colours, edge-touching frames, gait_loop.py.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_latias_mega_agy_actions.py
"""
import argparse
import colorsys
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
KEY = "0380_latias-mega"
MEGA_DIR = ROOT / "monsterSprite/gen3/0380_latias/mega/latias-mega"
ART = MEGA_DIR / "art"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"
SYNC_MEGA = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d/monsterSprite/gen3/0380_latias/mega/latias-mega")
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "latias_mega_agy_v3"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
GRID_SCALE = 0.26                  # sprite px per generated px (2x2 key sheet): body height 365 -> 95
SW_SCALE = GRID_SCALE * 365.0 / 740.0
CELL_W, CELL_H, FOOT_Y = 232, 172, 150
UP = 4
PSY_HI, PSY_LO = (255, 236, 250), (226, 110, 196)
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each downscaled view)
# label order: later polygons win. draw order: back to front. wings: the arm connector and claw hand go with them.
# mouth = roar origin; fwd = screen direction the creature faces.
RIG = {
    "S": dict(
        poly={"body": [(58, 22), (117, 22), (117, 95), (58, 95)],
              "wing_a": [(0, 22), (66, 22), (66, 37), (74, 37), (74, 48), (66, 48), (66, 80), (0, 80)],
              "wing_b": [(108, 22), (175, 22), (175, 80), (108, 80), (108, 48), (100, 48), (100, 37), (108, 37)],
              "head": [(70, 0), (104, 0), (102, 27), (95, 30), (79, 30), (72, 27)]},
        label=["body", "wing_a", "wing_b", "head"],
        draw=["wing_a", "wing_b", "body", "head"],
        pivot={"wing_a": (72, 43), "wing_b": (102, 43), "head": (87, 29), "body": (87, 60)},
        body_x=87, fwd=(0.0, 1.0), mouth=(87, 24)),
    "SW": dict(
        poly={"body": [(36, 20), (106, 20), (106, 95), (36, 95)],
              "wing_b": [(68, 6), (130, 6), (130, 60), (100, 60), (92, 46), (82, 32), (68, 24)],
              "wing_a": [(0, 30), (46, 30), (55, 43), (55, 56), (30, 60), (0, 60)],
              "head": [(34, 0), (64, 0), (64, 22), (57, 31), (40, 30), (34, 24)]},
        label=["body", "wing_b", "wing_a", "head"],
        draw=["wing_b", "body", "wing_a", "head"],
        pivot={"wing_a": (52, 50), "wing_b": (88, 42), "head": (50, 30), "body": (70, 60)},
        body_x=70, fwd=(-0.9, 0.45), mouth=(38, 24)),
    "W": dict(
        poly={"body": [(24, 18), (84, 18), (84, 88), (24, 88)],
              "wing_a": [(0, 36), (31, 36), (31, 58), (0, 58)],
              "head": [(0, 0), (37, 0), (37, 28), (27, 34), (0, 30)],
              "wing_b": [(51, 0), (122, 0), (122, 62), (98, 62), (66, 42), (51, 31)]},
        label=["body", "wing_a", "head", "wing_b"],
        draw=["wing_a", "body", "head", "wing_b"],
        pivot={"wing_a": (28, 46), "wing_b": (63, 38), "head": (26, 30), "body": (50, 60)},
        body_x=50, fwd=(-1.0, 0.0), mouth=(4, 20)),
    "NE": dict(
        poly={"body": [(55, 22), (110, 22), (110, 95), (55, 95)],
              "wing_a": [(0, 15), (62, 15), (62, 31), (75, 31), (75, 45), (62, 45), (62, 72), (0, 72)],
              "wing_b": [(95, 28), (163, 28), (163, 74), (97, 74), (92, 52)],
              "head": [(80, 0), (117, 0), (117, 26), (101, 37), (86, 37), (82, 20)]},
        label=["body", "wing_a", "wing_b", "head"],
        draw=["body", "head", "wing_a", "wing_b"],
        pivot={"wing_a": (73, 38), "wing_b": (97, 46), "head": (92, 36), "body": (82, 60)},
        body_x=82, fwd=(0.9, -0.45), mouth=(112, 16)),
    "N": dict(
        poly={"body": [(60, 18), (116, 18), (116, 95), (60, 95)],
              "wing_a": [(0, 18), (64, 18), (64, 32), (75, 32), (75, 47), (64, 47), (64, 80), (0, 80)],
              "wing_b": [(112, 18), (176, 18), (176, 80), (112, 80), (112, 47), (101, 47), (101, 32), (112, 32)],
              "head": [(72, 0), (104, 0), (101, 26), (75, 26)]},
        label=["body", "wing_a", "wing_b", "head"],
        draw=["body", "head", "wing_a", "wing_b"],
        pivot={"wing_a": (73, 40), "wing_b": (103, 40), "head": (88, 26), "body": (88, 60)},
        body_x=88, fwd=(0.0, -1.0), mouth=(88, 12)),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
def key_white(rgb):
    a = rgb.astype(int)
    white = a.min(axis=2) > 238                       # the white-grey chest is <= ~230, the key is 255
    lab, n = ndi.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg |= ndi.binary_dilation(bg, iterations=2) & (a.min(axis=2) > 215)       # light fringe
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
    dst.mkdir(parents=True, exist_ok=True)
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        im = im.crop(im.getbbox())
        sc = SW_SCALE if v == "SW" else GRID_SCALE
        sm = im.resize((round(im.width * sc), round(im.height * sc)), Image.BOX)
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
    masks = {n: lab == (i + 1) for i, n in enumerate(names)}
    layers = {}
    order = R["draw"]
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    bright = np.maximum(np.maximum(r, g), b) > 70
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        seam = np.zeros_like(m)
        if n == "body" and m.any():
            # underlay: the closed body + head outline is filled with body colour, so a moving wing root or the
            # head never opens a hole into the torso
            bh = m | masks["head"]
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 8), iterations=6))[8:-8, 8:-8]
            seam = above & ~m & inside & ~masks["head"]
            seam |= above & ~m & (ndi.distance_transform_edt(~m) <= 3)
        elif m.any():
            seam = above & ~m & (ndi.distance_transform_edt(~m) <= 2)
        if seam.any() and m.any():
            src_m = (m | masks["head"]) if n == "body" else m
            fillable = src_m & bright
            if not fillable.any():
                fillable = src_m
            _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
            sy, sx = np.nonzero(seam)
            lyr[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
            lyr[sy, sx, 3] = 255
        layers[n] = lyr
    return layers, masks


def place_layer(layer, view):
    """Put a view-sized layer on the cell: body centre on the cell centre, tail tip on FOOT_Y."""
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - RIG[view]["body_x"]
    oy = FOOT_Y - (h - 1)
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(h, CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
SPIN_STEPS = [1, 2, 3, 5, 6, 7]          # attack frames 4-9: view offset (in the 8 directions) of the barrel turn


def pose(action, f):
    """flap (deg, + raises both wing tips), head (dx_fwd, dy_px), global [forward px, up px, lean deg], fx."""
    ph = 2 * math.pi * f / N
    s1 = math.sin(ph)
    P = {"flap": 0.0, "flap_b": None, "head": (0, 0), "g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        P["g"] = [0.0, 2.0 * s1, 0.0]
        P["flap"] = 7.0 * math.sin(ph - 0.6) + 7.0 * math.sin(0.6)
        P["head"] = (0, -round(1.0 * (math.sin(ph - 0.7) + math.sin(0.7))))
    elif action == "walk":
        P["g"] = [2.0, 2.0 * math.sin(2 * ph), 7.0]
        P["flap"] = 12.0 * math.sin(2 * ph)
        P["head"] = (1, 0)
        fx.append(("lines", f))
    elif action == "attack":
        P["flap"] = [6, 14, 20, 22, 0, 0, 0, 0, 0, 0, -16, -18, -10, -4, -1, 0][f]
        P["g"] = [[0, -1, -2, -2, 0, 0, 0, 0, 0, 0, 10, 13, 9, 5, 2, 0][f],
                  [0, 1, 3, 4, 7, 10, 12, 12, 10, 7, 3, 2, 2, 1, 0, 0][f],
                  [0, -3, -6, -7, 0, 0, 0, 0, 0, 0, 10, 12, 8, 4, 1, 0][f]]
        P["head"] = (0, [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f])
        if 1 <= f <= 3:
            fx.append(("roar", f))
        if 4 <= f <= 9:
            fx.append(("spin", f))
        if f in (10, 11):
            fx.append(("dash", f))
            fx.append(("burst", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0
        P["g"] = [-7.0 * env + jit, 2.0 * env, -9.0 * env]
        P["flap"] = -12.0 * env
        P["head"] = (-1 if env > 0.5 else 0, 1 if env > 0.5 else 0)
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 7:
            fx.append(("sparkle", f))
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
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    for n in R["draw"]:
        rot, dx, dy = 0.0, 0.0, 0.0
        if n.startswith("wing"):
            cx = np.nonzero(cells[n][..., 3].any(axis=0))[0].mean()
            rot = P["flap"] * (1.0 if cx < piv[n][0] else -1.0)     # + raises the wing tip on either side
        elif n == "head":
            hx, hy = P["head"]
            dx, dy = hx * (fwx if abs(fwx) > 0.3 else 0.0), hy
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.4                                    # toward / away from the camera: short, not a slide
    if abs(lean) > 0.3:
        c = (CELL_W // 2, FOOT_Y - 45)
        canvas = rot4(canvas, lean * side if abs(fwx) > 0.3 else lean * 0.3, c)
    canvas = shift4(canvas, gf * fwx, gf * fwy - gu)
    return canvas[1::UP, 1::UP].copy(), fx, (gf * fwx, gf * fwy - gu)


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def draw_fx(a, fx, fwd, mouth, centre):
    fwx, fwy = fwd
    side_on = abs(fwx) > 0.3
    for kind, k in fx:
        if kind == "roar":
            # psychic rings opening from the mouth, toward where the creature faces
            for j in range(k + 1):
                r = 6 + 7 * j + 3 * k
                for deg in range(-60, 61, 6):
                    t = math.radians(deg)
                    if side_on:
                        x = mouth[0] + fwx * r * math.cos(t)
                        y = mouth[1] + r * 0.8 * math.sin(t) + fwy * r * 0.5
                    else:
                        x = mouth[0] + r * math.sin(t) * 1.2
                        y = mouth[1] - r * 0.55 * math.cos(t) * (1 if fwy < 0 else -1) + fwy * 4
                    put(a, x, y, PSY_HI if j == 0 else PSY_LO)
        elif kind == "spin":
            # swirl of light around the body while it turns
            for j in range(10):
                t = j * 0.62 + k * 0.9
                put(a, centre[0] + math.cos(t) * 46, centre[1] + math.sin(t) * 14, PSY_HI if j % 2 else PSY_LO)
        elif kind == "lines" or kind == "dash":
            n_l = 4 if kind == "lines" else 7
            L = 10 if kind == "lines" else 18
            for j in range(n_l):
                if side_on:
                    oy = -24 + j * (48 / max(1, n_l - 1)) + (k % 2) * 2
                    bx = centre[0] - fwx * (40 + (j * 7) % 12)
                    by = centre[1] + oy - fwy * 20
                    for i in range(L):
                        put(a, bx - fwx * i, by - fwy * i * 0.5, PSY_HI if i < 3 else PSY_LO)
                else:
                    # toward / away from the camera: streaks beside the wings, not through the body
                    sx = -1 if j % 2 else 1
                    bx = centre[0] + sx * (92 + (j * 5) % 10)
                    by = centre[1] - 10 + (j // 2) * 9 + (k % 2) * 2
                    for i in range(L // 2):
                        put(a, bx, by - fwy * i, PSY_HI if i < 2 else PSY_LO)
        elif kind == "burst":
            cx, cy = centre[0] + fwx * 62, centre[1] + fwy * (30 if side_on else 62)
            r = 8 if k == 10 else 13
            for j in range(12):
                t = j * math.pi / 6 + k * 0.2
                for q in range(r // 2, r):
                    put(a, cx + math.cos(t) * q, cy + math.sin(t) * q * 0.8, PSY_HI if q < r - 2 else PSY_LO)
        elif kind == "sparkle":
            for j in range(4):
                t = j * 1.7 + k * 0.8
                x, y = centre[0] + math.cos(t) * 30, centre[1] + math.sin(t) * 22
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    put(a, x + dx, y + dy, PSY_HI if (dx, dy) == (0, 0) else PSY_LO)


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = PSY_HI
    return out


# ---------------------------------------------------------------- shiny (purple -> green hue swap, like v1/v2)
def to_shiny(a):
    out = a.copy()
    m = a[..., 3] > 0
    for y, x in zip(*np.nonzero(m)):
        r, g, b = a[y, x, :3] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s > 0.12 and 0.62 <= h <= 0.92:
            h = (h - 0.75 + 0.33) % 1.0
            out[y, x, :3] = [int(c * 255 + 0.5) for c in colorsys.hsv_to_rgb(h, s, v)]
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for n in ("out/keys_2x2.png", "out/sw_single.png"):
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


def crop_sq(a):
    ys, xs = np.nonzero(a[..., 3] > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    S = max(x1 - x0, y1 - y0) + 7
    c = np.zeros((S, S, 4), np.uint8)
    ox, oy = (S - (x1 - x0 + 1)) // 2, (S - (y1 - y0 + 1)) // 2
    c[oy:oy + y1 - y0 + 1, ox:ox + x1 - x0 + 1] = a[y0:y1 + 1, x0:x1 + 1]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--samples", action="store_true", help="only write _work/samples.png (no sheets, no front/back)")
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
        run_tool(["build", "--input", str(raw), "--colors", "30", "--reserve", "#%02x%02x%02x" % PSY_HI, "#%02x%02x%02x" % PSY_LO,
                  "--balance", "--name", "latias-mega", "--output-dir", str(pal_dir)])
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
        if args.samples:
            col = {"body": (180, 180, 190), "head": (255, 220, 40), "wing_a": (60, 200, 90), "wing_b": (60, 140, 255)}
            pm = np.zeros((*still[v].shape[:2], 4), np.uint8)
            for n, m in masks.items():
                pm[m, :3] = col[n]; pm[m, 3] = 255
            Image.fromarray(np.repeat(np.repeat(pm, 4, 0), 4, 1), "RGBA").save(work / f"parts_{v}.png")

    def render(v, act, f):
        cells, piv, (ox, oy) = rig[v]
        R = RIG[v]
        a, fx, (gx, gy) = compose(v, cells, piv, act, f)
        centre = (CELL_W / 2 + gx, FOOT_Y - 48 + gy)
        mouth = (R["mouth"][0] + ox + gx, R["mouth"][1] + oy + gy)
        if any(k == "flash" for k, _ in fx):
            a = flash(a)
        draw_fx(a, fx, R["fwd"], mouth, centre)
        return a

    if args.samples:
        tiles = []
        for v in ("S", "W", "N"):
            for act, fs in (("idle", (0, 4, 8)), ("walk", (2,)), ("attack", (2, 6, 10)), ("hurt", (3,))):
                for f in fs:
                    tiles.append(render(v, act, f))
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sh = np.full((rows * CELL_H, cols * CELL_W, 4), (15, 20, 31, 255), np.uint8)
        for i, t in enumerate(tiles):
            r_, c_ = divmod(i, cols)
            m = t[..., 3] > 0
            sh[r_ * CELL_H:(r_ + 1) * CELL_H, c_ * CELL_W:(c_ + 1) * CELL_W][m] = t[m]
        Image.fromarray(sh, "RGBA").save(work / "samples.png")
        print("samples", work / "samples.png")
        return

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                Image.fromarray(render(v, act, f), "RGBA").save(od / f"{f:02d}.png")
            st["done"].append(key); save_st()
            print("frames", key, flush=True)

    def frame_of(d, act, f):
        v, mirror = DIR_SRC[d]
        a = np.array(Image.open(frames_dir / v / act / f"{f:02d}.png").convert("RGBA"))
        return a[:, ::-1] if mirror else a

    sheets = {}
    for act in ACTIONS:
        sh = np.zeros((8 * CELL_H, N * CELL_W, 4), np.uint8)
        for di, d in enumerate(DIRS):
            for f in range(N):
                if act == "attack" and 4 <= f <= 9:
                    # barrel turn: show the real view of the direction the body is turning through
                    a = frame_of(DIRS[(di + SPIN_STEPS[f - 4]) % 8], act, f)
                else:
                    a = frame_of(d, act, f)
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
    Image.fromarray(pv, "RGBA").save(OUT_DIR / "preview_motion.png")

    # stills: 8-direction strip + front (SW) / back (N) png/gif incl. shiny (user lifted locked spec 3 on 2026-11-04)
    idle0 = [sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W] for di in range(8)]
    Image.fromarray(np.concatenate(idle0, axis=1), "RGBA").save(MEGA_DIR / "8dir_stills.png")
    fr, bk = crop_sq(idle0[DIRS.index("SW")]), crop_sq(idle0[DIRS.index("N")])
    for name, arr in (("front", fr), ("back", bk), ("front-shiny", to_shiny(fr)), ("back-shiny", to_shiny(bk))):
        im = Image.fromarray(arr, "RGBA")
        im.save(MEGA_DIR / f"{name}.png")
        base, _, shiny = name.partition("-")
        gif = MEGA_DIR / (f"{base}-shiny.gif" if shiny else f"{base}.gif")
        rev = MEGA_DIR / (f"{base}-shiny-reverse.gif" if shiny else f"{base}-reverse.gif")
        save_gif(im, gif)
        save_gif(im.transpose(Image.FLIP_LEFT_RIGHT), rev)

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
    clip = {}
    for act in ACTIONS:
        for di, d in enumerate(DIRS):
            for f in range(N):
                c = sheets[act][di * CELL_H:(di + 1) * CELL_H, f * CELL_W:(f + 1) * CELL_W, 3] > 0
                if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
                    clip.setdefault(act, []).append(f"{d}:{f}")
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off), "edge_touching_frames": clip}
    if not args.skip_qa and GAIT_LOOP.exists():
        for act in ("idle", "walk"):
            od = qa / f"gait_{act}_S"
            if od.exists():
                shutil.rmtree(od)
            r = subprocess.run([str(FPY), "-I", str(GAIT_LOOP), "select", "--frames-dir", str(frames_dir / "S" / act), "--fps", "12",
                                "--output-dir", str(od), "--kind", "hover", "--no-aids"], capture_output=True, text=True)
            rep = od / "loop-report.json"
            report[f"gait_{act}"] = json.loads(rep.read_text()).get("recommendation") if rep.exists() else (r.stderr.strip().splitlines() or ["no report"])[-1]
    (qa / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    bounds = []
    for di in range(8):
        c = sheets["idle"][di * CELL_H:(di + 1) * CELL_H, :CELL_W, 3] > 0
        ys, xs = np.nonzero(c)
        bounds += [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    meta = {"generator": "forge_latias_mega_agy_actions", "version": VERSION, "key": KEY, "dex": 380,
            "source_art": ["art/out/keys_2x2.png", "art/out/sw_single.png"], "made_from": ["art/mega-latias.webp", "art/ref_gemini.jpeg"],
            "cell_w": CELL_W, "cell_h": CELL_H, "foot_y": FOOT_Y, "sprite_h": int(bounds[3]), "frames": N, "frame_time": AD,
            "directions": DIRS, "direction_sources": DIR_SRC, "actions": ACTIONS, "palette_colors": len(cols),
            "off_palette_colors": len(off), "idle_bounds_S_to_SW": bounds}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

    if not args.no_sync and SYNC_MEGA.exists():
        sd = SYNC_MEGA / "agent-sprite-forge"
        if sd.exists():
            shutil.rmtree(sd)
        sd.mkdir(parents=True)
        for fp in OUT_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, sd / fp.name)
        for fp in MEGA_DIR.iterdir():
            if fp.is_file() and (fp.suffix in (".png", ".gif")) and (fp.name.startswith(("front", "back")) or fp.name == "8dir_stills.png"):
                shutil.copy2(fp, SYNC_MEGA / fp.name)
        print("synced ->", SYNC_MEGA)


if __name__ == "__main__":
    main()
