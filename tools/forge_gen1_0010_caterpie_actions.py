#!/usr/bin/env python3
"""0010_caterpie (gen1 normal): 8-direction x 4-action sheet, Antigravity key art + agent-sprite-forge.

Batch procedure md/gen1-forge.md (2026-11-04); structure copied from tools/forge_raichu_mega_x_actions.py, rig /
colours / attack built from this form's own art (a legless green larva: raised front half, head with one forked
red antenna, segments lying on the ground, tail ending in a cream bulb).

Source:
  art/ref_front.png, art/ref_back.png : official front.png / back.png x8 nearest (references only)
  art/agy_prompt.md                   : morphological prompt (no proper nouns) for the Antigravity CLI (`agy -p`)
  art/out/keys_2x2.png                : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png               : SW
  Mirrors: SE = SW, E = W, NW = NE.

Pipeline:
  1. key the white backdrop (border-connected only), crop, "mode" downscale + 1 px outline. S height = opaque
     height of the official front.png; W / NE / N share the grid scale, SW is matched on height.
  2. agent-sprite-forge palette_tool build (incl. reserved silk / flash colours) -> lock -> apply.
  3. part rig per view: tail (pivot at the joint to the lying segments), rear (lying segments), body (raised
     segments), head (+ antenna). No legs: nothing is added.
  4. poses at 4x nearest, sampled back to 1x.
     idle   : breathing of the raised body, head nods / tilts, tail bulb sways.
     walk   : legless crawl: a peristaltic wave (rear segments bunch and stretch, body humps, head leads, tail
              follows); no feet are drawn.
     attack : silk shot: rear up (head raised, lean back) -> lunge, silk strands shoot from the mouth -> the silk
              spreads into a web net -> recover.
     hurt   : white flash, recoil back (lean <= 10 deg), recover.
  5. QC: off-palette, edge touch, still diff (idle f0 == still), gait_loop.py on idle / walk.

Writes <form>/8dir_stills.png and <form>/agent-sprite-forge/ (sheet.png = 4 actions x 8 rows, preview.png,
meta.json, idle.png ...). The official front/back png/gif are never touched. Syncs agent-sprite-forge/ (without
_work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0010_caterpie_actions.py [--force] [--no-sync] [--stills-only]
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
KEY = "0010_caterpie"
REL = "monsterSprite/gen1/0010_caterpie/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0010_caterpie_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 112, 80, 70
UP = 4
SILK = (246, 246, 236)           # reserved: silk strands / web
FLASH = (255, 255, 255)          # reserved: hit flash
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each still)
# label: later polygons win. draw: back to front. body_x: placed on the cell centre. fwd: screen "forward".
# mouth: point where the silk leaves (None = hidden, effect starts in front of the head).
def _v(head, body, rear, tail, piv, body_x, fwd, mouth, tail_dir, back):
    return dict(poly={"head": head, "body": body, "rear": rear, "tail": tail},
                label=["rear", "body", "tail", "head"],
                draw=(["tail", "rear", "body", "head"] if not back else ["rear", "tail", "body", "head"]),
                pivot=piv, body_x=body_x, fwd=fwd, mouth=mouth, tail_dir=tail_dir, back=back)


RIG = {
    "S": _v(head=[(0, 0), (23, 0), (23, 15), (16, 17), (5, 17), (0, 15)],
            body=[(0, 15), (23, 15), (23, 26), (0, 26)],
            rear=[(3, 26), (30, 26), (30, 35), (3, 35)],
            tail=[(23, 13), (30, 13), (30, 26), (23, 26)],
            piv={"head": (11, 17), "body": (12, 26), "rear": (16, 34), "tail": (25, 26)},
            body_x=14, fwd=(-0.45, 0.9), mouth=(5, 15), tail_dir=1, back=False),   # the front art looks down-left
    "SW": _v(head=[(0, 0), (24, 0), (24, 15), (17, 18), (4, 18), (0, 15)],
             body=[(0, 15), (25, 15), (25, 26), (0, 26)],
             rear=[(3, 26), (32, 26), (32, 35), (3, 35)],
             tail=[(25, 13), (32, 13), (32, 26), (25, 26)],
             piv={"head": (12, 18), "body": (13, 26), "rear": (17, 34), "tail": (27, 25)},
             body_x=14, fwd=(-0.9, 0.45), mouth=(6, 16), tail_dir=1, back=False),
    "W": _v(head=[(0, 0), (23, 0), (23, 14), (10, 17), (0, 16)],
            body=[(0, 13), (24, 13), (24, 24), (0, 24)],
            rear=[(5, 24), (41, 24), (41, 35), (5, 35)],
            tail=[(32, 13), (41, 13), (41, 26), (32, 26)],
            piv={"head": (12, 16), "body": (12, 24), "rear": (24, 33), "tail": (37, 25)},
            body_x=18, fwd=(-1.0, 0.0), mouth=(3, 15), tail_dir=1, back=False),
    "NE": _v(head=[(8, 0), (30, 0), (30, 14), (20, 17), (10, 15)],
             body=[(8, 14), (30, 14), (30, 27), (8, 27)],
             rear=[(0, 27), (30, 27), (30, 37), (0, 37)],
             tail=[(0, 17), (8, 17), (8, 30), (0, 30)],
             piv={"head": (19, 16), "body": (19, 27), "rear": (14, 36), "tail": (4, 30)},
             body_x=18, fwd=(0.9, -0.45), mouth=(28, 13), tail_dir=-1, back=True),
    "N": _v(head=[(0, 0), (21, 0), (21, 15), (15, 17), (5, 17), (0, 15)],
            body=[(0, 15), (21, 15), (21, 29), (0, 29)],
            rear=[(0, 29), (29, 29), (29, 38), (0, 38)],
            tail=[(22, 20), (29, 20), (29, 29), (22, 29)],
            piv={"head": (10, 16), "body": (10, 29), "rear": (14, 37), "tail": (25, 29)},
            body_x=12, fwd=(0.0, -1.0), mouth=(10, 6), tail_dir=1, back=True),
}


# ---------------------------------------------------------------- 1. stills from the Antigravity key art
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


def mode_down(a, s):
    """Each output pixel takes the dominant (4-bit quantised) colour of its source block; dark lines win at 30 %."""
    H, W = a.shape[:2]
    oh, ow = round(H * s), round(W * s)
    out = np.zeros((oh, ow, 4), np.uint8)
    q = (a[..., 0].astype(int) >> 4) * 256 + (a[..., 1].astype(int) >> 4) * 16 + (a[..., 2].astype(int) >> 4)
    for i in range(oh):
        y0 = int(i / s); y1 = max(y0 + 1, int((i + 1) / s))
        for j in range(ow):
            x0 = int(j / s); x1 = max(x0 + 1, int((j + 1) / s))
            blk = a[y0:y1, x0:x1].reshape(-1, 4)
            qq = q[y0:y1, x0:x1].reshape(-1)
            op = blk[:, 3] > 0
            if op.mean() < 0.5:
                continue
            px, qk = blk[op], qq[op]
            dark = px[:, :3].max(axis=1) < 60
            if dark.mean() >= 0.3:
                sel = px[dark]
            else:
                ks, cs = np.unique(qk, return_counts=True)
                sel = px[qk == ks[cs.argmax()]]
            out[i, j, :3] = sel[:, :3].mean(axis=0)
            out[i, j, 3] = 255
    return out


def outline(a, col=(8, 8, 8)):
    op = a[..., 3] > 0
    ring = ndi.binary_dilation(op) & ~op
    b = a.copy()
    b[ring, :3] = col
    b[ring, 3] = 255
    return b


def official_height():
    a = np.array(Image.open(FORM_DIR / "front.png").convert("RGBA"))
    ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
    return int(ys.max() - ys.min() + 1)


def make_stills(dst: Path):
    k = np.array(Image.open(ART / "out/keys_2x2.png").convert("RGB"))
    h2, w2 = k.shape[0] // 2, k.shape[1] // 2
    src = {"S": k[:h2, :w2], "W": k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    crops = {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        crops[v] = np.array(im.crop(im.getbbox()))
    target = official_height()

    def render(v, s):
        a = outline(mode_down(crops[v], s))
        ys, xs = np.nonzero(a[..., 3])
        return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def fit(v):            # scale whose rendered height equals the official height
        base = target / crops[v].shape[0]
        for d in np.linspace(-0.06, 0.06, 25):
            s = base * (1 + d) if d else base
            if render(v, s).shape[0] == target:
                return s
        raise SystemExit(f"cannot fit {v} to {target}px")
    g = fit("S")
    scale = {v: g for v in src}
    scale["SW"] = fit("SW")
    dst.mkdir(parents=True, exist_ok=True)
    for v in src:
        Image.fromarray(render(v, scale[v]), "RGBA").save(dst / f"{v}.png")
    return {v: float(s) for v, s in scale.items()}


# ---------------------------------------------------------------- palette
def run_tool(args):
    r = subprocess.run([str(FPY), "-I", str(PALETTE_TOOL)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"palette_tool {args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def load_palette(lock_dir):
    pj = json.loads((lock_dir / "palette.json").read_text())
    cols = []
    for c in pj.get("colors", []):
        hx = (c["hex"] if isinstance(c, dict) else c).lstrip("#")
        cols.append((int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)))
    return cols


def nearest(cols, rgb):
    a = np.array(cols, float)
    return tuple(int(v) for v in cols[int(((a - np.array(rgb, float)) ** 2).sum(1).argmin())])


# ---------------------------------------------------------------- 3. rig
def poly_mask(p, shape):
    m = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(m).polygon([tuple(q) for q in p], fill=255)
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
    bright = np.maximum(np.maximum(r, g), b) > 60
    for i, n in enumerate(order):
        m = masks[n]
        lyr = np.zeros_like(still)
        lyr[m] = still[m]
        above = np.zeros_like(m)
        for n2 in order[i + 1:]:
            above |= masks[n2]
        cushion = 3 if n in ("rear", "body") else 2
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n == "body" and m.any():
            # underlay inside the closed body + head outline so the head can nod without holes
            bh = m | masks.get("head", np.zeros_like(m))
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 4), iterations=4))[4:-4, 4:-4]
            seam |= above & ~m & inside & ~masks.get("head", np.zeros_like(m))
            src_m = bh
        if seam.any() and m.any():
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
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - RIG[view]["body_x"]
    oy = FOOT_Y - (h - 1)
    cell[oy:oy + h, ox:ox + w] = layer
    return cell, (ox, oy)


# ---------------------------------------------------------------- 4. posing
def wave(amp, cycles, f, phase=0.0):
    """0 at f = 0 and at the loop end."""
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """part -> [rot deg (+ = raise / tip toward forward), fwd px, up px]; 'g' = [forward px, up px, lean deg]."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        br = wave(0.6, 1, f, -math.pi / 2)              # 0 .. 1.2 (1 - cos)
        P["body"] = [0, 0, round(br * 0.9)]
        P["head"] = [wave(3.0, 1, f, 0.0), 0, round(br)]
        P["tail"] = [wave(8.0, 1, f, 0.0), 0, 0]
    elif action == "walk":
        # peristaltic crawl: the rear bunches forward and up while the head reaches, then the head pulls the
        # body along and the rear stretches back. Two waves per loop.
        ph = 2 * math.pi * 2 * f / N
        hump = 0.5 - 0.5 * math.cos(ph)              # 0 .. 1 .. 0
        P["rear"] = [0, round(1.5 * math.sin(ph)), round(1.0 * hump)]
        P["body"] = [0, round(1.0 * math.sin(ph - 0.8)), round(1.0 * (0.5 - 0.5 * math.cos(ph - 0.8)))]
        P["head"] = [wave(3.0, 2, f, 0.0), round(1.5 * math.sin(ph - 1.6)), round(1.0 * (0.5 - 0.5 * math.cos(ph - 1.6)))]
        P["tail"] = [wave(10.0, 2, f, 0.6), round(1.5 * math.sin(ph)), round(1.0 * hump)]
    elif action == "attack":
        g_f = [0, -1, -2, -2, 1, 3, 4, 4, 4, 3, 3, 2, 1, 1, 0, 0][f]
        g_u = [0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -4, -8, -10, -2, 6, 9, 9, 8, 7, 5, 4, 3, 2, 1, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["head"] = [[0, -4, -8, -10, 0, 6, 8, 8, 7, 6, 4, 3, 2, 1, 0, 0][f],
                     [0, -1, -1, -1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0][f], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]]
        P["rear"] = [0, [0, 1, 2, 2, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0][f], 0]
        P["tail"] = [[0, 10, 18, 22, 8, -6, -10, -10, -8, -6, -4, -2, -1, 0, 0, 0][f], 0, 0]
        if 1 <= f <= 3:
            fx.append(("charge", f))
        if 4 <= f <= 11:
            fx.append(("silk", f))
        if 8 <= f <= 12:
            fx.append(("web", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [-4.0 * env + jit, 1.0 * env, -10.0 * env]
        P["head"] = [-6.0 * env, -1.0 * env, 1.0 * env]
        P["tail"] = [-14.0 * env, 0, 0]
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


def drop_crumbs(a, max_px=2):
    op = a[..., 3] > 0
    lab, n = ndi.label(op, structure=np.ones((3, 3)))
    if n <= 1:
        return a
    sz = ndi.sum(op, lab, range(1, n + 1))
    small = np.isin(lab, [i + 1 for i, s in enumerate(sz) if s <= max_px])
    if small.any():
        a = a.copy()
        a[small] = 0
    return a


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    offs = {}
    for n in R["draw"]:
        p = P.get(n, zero)
        rot, df, du = p[0], p[1], p[2]
        if n == "head":
            # + tips the head toward forward: side views turn about the neck, front / back views tilt sideways
            rot = rot * (side if abs(fwx) > 0.3 else 0.4)
        elif n == "tail":
            tail_right = R["tail_dir"] > 0
            rot = rot * (-1.0 if tail_right else 1.0)     # + raises the bulb
        if abs(fwx) < 0.5:                                 # toward / away from the camera: forward = mostly up/down
            dx, dy = 0.0, df * fwy * 0.5 - du
        else:
            dx, dy = df * fwx, df * fwy * 0.5 - du
        offs[n] = (dx, dy, rot)
        l4 = rot4(up4(cells[n]), rot, piv[n])
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.5
    feetpt = (CELL_W // 2, FOOT_Y)
    ang = 0.0
    if abs(lean) > 0.3:
        ang = lean * side if abs(fwx) > 0.3 else lean * 0.4 * (1 if fwy < 0 else -1)
        canvas = rot4(canvas, ang, feetpt)
    gx, gy = gf * fwx, gf * fwy * 0.6 - gu
    canvas = shift4(canvas, gx, gy)
    out = canvas[1::UP, 1::UP].copy()
    out = drop_crumbs(out)
    return out, fx, offs, (gx, gy), ang


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def line(a, x0, y0, x1, y1, col):
    k = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for j in range(k + 1):
        u = j / k
        put(a, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, col)


def web(a, cx, cy, r, C, sy=1.0):
    """Small silk net: 8 spokes + 2 rings (ellipse when seen from the front)."""
    for j in range(8):
        ang = j * math.pi / 4 + 0.2
        line(a, cx, cy, cx + math.cos(ang) * r, cy + math.sin(ang) * r * sy, SILK)
    for rr in (r * 0.5, r):
        if rr < 1.5:
            continue
        steps = int(2 * math.pi * rr) + 6
        for s in range(steps):
            ang = 2 * math.pi * s / steps
            put(a, cx + math.cos(ang) * rr, cy + math.sin(ang) * rr * sy, C["cream"] if rr < r else SILK)


def draw_fx(a, fx, mouth_pt, fwd, C, front_hit):
    fwx, fwy = fwd
    L = math.hypot(fwx, fwy) or 1
    ux, uy = fwx / L, fwy / L
    px, py = -uy, ux
    toward = fwy > 0.5 and abs(fwx) < 0.3
    away = fwy < -0.5 and abs(fwx) < 0.3
    sc = 0.55 if (toward or away) else 1.0
    for kind, k in fx:
        if kind == "charge" and mouth_pt:
            for j in range(k):
                put(a, mouth_pt[0] + ux * 2 + px * (j - 1) * 2, mouth_pt[1] + uy * 2 + py * (j - 1) * 2, SILK)
        elif kind == "silk" and mouth_pt:
            t = k - 4
            head = [5, 12, 19, 26, 30, 30, 30, 30][t] * sc
            tail = [0, 0, 0, 0, 4, 10, 18, 26][t] * sc
            if toward:
                # silk shot at the camera: a growing tangle in front of the mouth
                r = [1, 2, 3, 4, 5, 5, 4, 3][t]
                web(a, mouth_pt[0], mouth_pt[1] + 3, r, C, sy=0.8)
                continue
            for s, (o, col) in enumerate(((-1.0, SILK), (1.0, C["cream"]))):
                d = tail
                prev = None
                while d <= head:
                    w = o * (1.0 + 0.8 * math.sin(d * 0.5 - k * 1.1 + s))
                    pt = (mouth_pt[0] + ux * d + px * w, mouth_pt[1] + uy * d + py * w)
                    if prev:
                        line(a, prev[0], prev[1], pt[0], pt[1], col)
                    prev = pt
                    d += 2.0
        elif kind == "web" and mouth_pt:
            r = {8: 3, 9: 5, 10: 6, 11: 6, 12: 5}[k]
            if toward:
                bx, by, sy = mouth_pt[0], mouth_pt[1] + 4, 0.8
            else:
                d = 31 * sc
                bx, by, sy = mouth_pt[0] + ux * d, mouth_pt[1] + uy * d, (0.6 if away else 1.0)
            web(a, bx, by, r, C, sy=sy)
        elif kind == "hit" and front_hit:
            hx, hy = front_hit
            rr = 2 + k
            for j in range(6):
                ang = j * math.pi / 3 + 0.3
                put(a, hx + math.cos(ang) * rr, hy + math.sin(ang) * rr, FLASH if k < 3 else C["yel"])
                put(a, hx + math.cos(ang) * (rr - 1), hy + math.sin(ang) * (rr - 1), C["cream"])


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FLASH
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for p in (ART / "out/keys_2x2.png", ART / "out/sw_single.png"):
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--stills-only", action="store_true")
    args = ap.parse_args()

    work = OUT_DIR / "_work"
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    # the stills only depend on the key art: keep them across script edits
    art_h = hashlib.sha1((ART / "out/keys_2x2.png").read_bytes() + (ART / "out/sw_single.png").read_bytes()).hexdigest()[:12]
    st_p = work / "state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    h = state_hash()
    if args.force or st.get("art") != art_h:
        if work.exists():
            shutil.rmtree(work)
        st = {"art": art_h, "hash": h, "done": []}
    elif st.get("hash") != h:
        st = {"art": art_h, "hash": h, "done": [d for d in st.get("done", []) if d == "stills"], "scale": st.get("scale")}
        if (work / "frames").exists():
            shutil.rmtree(work / "frames")
    work.mkdir(parents=True, exist_ok=True)
    save_st = lambda: st_p.write_text(json.dumps(st))

    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        st["scale"] = make_stills(raw)
        run_tool(["build", "--input", str(raw), "--colors", "22", "--reserve", "#%02x%02x%02x" % SILK,
                  "#%02x%02x%02x" % FLASH, "--balance", "--name", KEY, "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1",
                  "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", st["scale"], flush=True)
    if args.stills_only:
        return
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}
    pal = load_palette(lock_dir)
    C = {"cream": nearest(pal, (230, 205, 148)), "yel": nearest(pal, (246, 238, 139))}

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
                a, fx, offs, (gx, gy), ang = compose(v, cells, piv, act, f)
                hdx, hdy, hrot = offs.get("head", (0, 0, 0))
                feetpt = (CELL_W // 2, FOOT_Y)
                mp = R["mouth"]
                pt = [(mp[0] + ox, mp[1] + oy)]
                pt = [(x + hdx, y + hdy) for x, y in pt]
                c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                pt = [(feetpt[0] + (x - feetpt[0]) * c - (y - feetpt[1]) * s,
                       feetpt[1] + (x - feetpt[0]) * s + (y - feetpt[1]) * c) for x, y in pt]
                mouth_pt = (pt[0][0] + gx, pt[0][1] + gy)
                front_hit = (CELL_W / 2 + gx + R["fwd"][0] * 8, FOOT_Y - 20 + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, mouth_pt, R["fwd"], C, front_hit)
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (3, 6, 9, 11)):
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
    still_diff = 0
    loop_diff = 0
    for v in RIG:
        cells, piv, (ox, oy) = rig[v]
        ref = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        ref[oy:oy + still[v].shape[0], ox:ox + still[v].shape[1]] = still[v]
        f0 = np.array(Image.open(frames_dir / v / "idle" / "00.png").convert("RGBA"))
        still_diff += int((np.abs(f0.astype(int) - ref.astype(int)).sum(axis=2) > 0).sum())
    s_h = int(np.ptp(np.nonzero(idle0[0][..., 3].any(axis=1))[0]) + 1)
    report = {"version": VERSION, "hash": h, "palette_colors": len(cols), "off_palette_colors": len(off),
              "alpha_not_binary": alpha_bad, "edge_touching_frames": clip, "still_diff_px": still_diff,
              "S_height": s_h, "official_height": official_height()}
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
    meta = {"generator": "forge_gen1_0010_caterpie_actions", "version": VERSION, "key": KEY, "dex": 10,
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
