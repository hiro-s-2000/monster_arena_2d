#!/usr/bin/env python3
"""0050_diglett-alola (regional): 8-direction x 4-action sheet, same flow as forge_raichu_mega_x_actions.py.

Source (2026-11-04, md/gen1-forge.md):
  art/ref_front.png / art/ref_back.png = official front.png / back.png x8 nearest.
  The Antigravity CLI (`agy -p`, prompt art/agy_prompt.md, no proper nouns) drew the key views:
    art/out/keys_2x2.png  : top-left S, top-right side view, bottom-left NE, bottom-right N
    art/out/sw_single.png : SW (3/4 front, faces down-left)
  SE = SW mirrored, NW = NE mirrored, the other side view = the drawn one mirrored.

Creature (counted on the official sprite): ONE capsule-shaped reddish-brown body poking out of the ground, two tiny
black eyes, ONE big pink-red nose, exactly THREE stiff golden hair strands on the dome top, ONE low ring of grey
stones around the base. No arms, no legs, no mouth, no tail.

Pipeline: key white (border-connected) -> area downscale (S height = official front.png opaque height; grid views
share that scale, SW matched on height) -> palette_tool build / lock / apply -> light outer edge -> outline colour ->
part rig by colour (hair = golden strands + their dark edge above the dome, stones = grey low-chroma pixels at the
base, body = rest; outline pixels follow their nearest part) -> poses (integer shifts, a row shear for leaning,
hair rotated at 4x nearest about its root) -> QC -> sheets -> sync agent-sprite-forge/ to pokemon_arena_2d.

The body lives inside the ground: when it sinks, the rows below its original ground line are hidden; when it rises,
its lowest visible rows are repeated downward (own colours, behind the stones) so no gap opens above the stones.

Actions (16 frames each):
  idle   : the body slowly rises and settles in its hole (breath), the three hairs sway, stones stay still.
  walk   : no legs -> burrowing glide: the body bobs twice per loop and sways, the stone ring rumbles (left / right
           halves bounce in turn), the hair trails, a little grit flies behind.
  attack : dig + headbutt: the body ducks into the stones (only the hair shows) -> bursts out high leaning forward,
           the hair whipping -> slams the dome forward, stones fly and an impact star flashes -> settles back.
  hurt   : white flash, the body leans back and drops into the hole, the stones jolt, hair droops, recovers.

Never touches the official front/back png/gif.
run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0050_diglett_alola_actions.py [--force] [--no-sync]
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
KEY = "0050_diglett-alola"
REL = "monsterSprite/gen1/0050_diglett/regional/diglett-alola"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0050_diglett_alola_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 88, 68, 58
UP = 4
FLASH, GLOW = (255, 255, 255), (255, 240, 170)

# the side view drawn top-right in keys_2x2 (checked by eye): "E" (faces right) or "W" (faces left)
SIDE = "W"
OTHER = "W" if SIDE == "E" else "E"
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), SIDE: (SIDE, False), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), OTHER: (SIDE, True), "SW": ("SW", False)}
RIG = {
    "S": dict(fwd=(0.0, 1.0)),
    "SW": dict(fwd=(-0.9, 0.45)),
    SIDE: dict(fwd=((1.0, 0.0) if SIDE == "E" else (-1.0, 0.0))),
    "NE": dict(fwd=(0.9, -0.45)),
    "N": dict(fwd=(0.0, -1.0)),
}


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
    src = {"S": k[:h2, :w2], SIDE: k[:h2, w2:], "NE": k[h2:, :w2], "N": k[h2:, w2:],
           "SW": np.array(Image.open(ART / "out/sw_single.png").convert("RGB"))}
    crops = {}
    for v, rgb in src.items():
        im = Image.fromarray(key_white(rgb), "RGBA")
        crops[v] = im.crop(im.getbbox())
    grid = target_h / crops["S"].height
    scale = {v: grid for v in src}
    scale["SW"] = target_h / crops["SW"].height
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in crops.items():
        sm = im.resize((round(im.width * scale[v]), round(im.height * scale[v])), Image.BOX)
        a = np.array(sm)
        a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        a[a[..., 3] == 0] = 0
        lab, n = ndi.label(a[..., 3] > 0)
        if n > 1:
            sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
            keep = [i + 1 for i, s in enumerate(sz) if s >= 6]
            a[~np.isin(lab, keep)] = 0
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
    op = a[..., 3] > 0
    edge = op & ~ndi.binary_erosion(op, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], border_value=0)
    L = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    a[edge & (L > 90), :3] = ink
    return a


# ---------------------------------------------------------------- 2. rig (by colour)
def classify(still):
    """-> masks hair / body / stones."""
    op = still[..., 3] > 0
    h, w = op.shape
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    yy, xx = np.mgrid[0:h, 0:w]
    gold = op & (r > 180) & (g > 140) & (b < 110) & (r - b > 110)
    ys = np.nonzero(gold.any(axis=1))[0]
    # hair = golden strands that sit in the top part of the sprite
    gold &= yy < h * 0.45
    grey = op & (chroma <= 34) & (mx >= 40) & (yy > h * 0.45)
    # keep only grey pieces that touch the lower part (stones), not eye whites etc.
    lab, n = ndi.label(grey)
    keep = [i for i in range(1, n + 1) if (yy[lab == i].max() > h * 0.7)]
    grey = np.isin(lab, keep)
    seed = np.zeros(op.shape, int)                          # 1 body, 2 hair, 3 stones
    seed[op & (chroma > 34) & (mx >= 40)] = 1
    seed[gold] = 2
    seed[grey] = 3
    dark = op & (seed == 0)
    if dark.any():
        _, (iy, ix) = ndi.distance_transform_edt(seed == 0, return_indices=True)
        seed[dark] = seed[iy[dark], ix[dark]]
    hair = seed == 2
    # hair outline pixels may have been assigned to body: dark body pixels above the hair's lowest gold row and
    # touching the hair stay hair
    if ys.size:
        hy = int(np.nonzero(hair.any(axis=1))[0].max()) if hair.any() else 0
        dk = op & (seed == 1) & (mx < 70) & (yy < hy) & ndi.binary_dilation(hair, iterations=1)
        seed[dk] = 2
    return {"body": seed == 1, "hair": seed == 2, "stones": seed == 3}


def fill_from(still, src_m, target):
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    fillable = src_m & (np.maximum(np.maximum(r, g), b) > 70)
    if not fillable.any():
        fillable = src_m
    _, (iy, ix) = ndi.distance_transform_edt(~fillable, return_indices=True)
    out = np.zeros_like(still)
    sy, sx = np.nonzero(target)
    out[sy, sx] = still[iy[sy, sx], ix[sy, sx]]
    out[sy, sx, 3] = 255
    return out


def build_rig(still):
    masks = classify(still)
    body, hair, stones = masks["body"], masks["hair"], masks["stones"]
    h, w = body.shape
    layers = {}
    # body layer + cushion under the hair root (the dome continues under it)
    lb = np.zeros_like(still)
    lb[body] = still[body]
    bh = ndi.binary_fill_holes(ndi.binary_closing(np.pad(body | hair, 4), iterations=3))[4:-4, 4:-4]
    under = hair & bh & (ndi.distance_transform_edt(~body) <= 2)
    if under.any():
        f = fill_from(still, body, under)
        lb[under] = f[under]
    # ground line: lowest body pixel per column; extension = the row 2 above it repeated downward (behind stones)
    bottom = np.full(w, -1)
    ext = np.zeros((h + 16, w, 4), np.uint8)
    for x in range(w):
        col = np.nonzero(body[:, x])[0]
        if col.size:
            bottom[x] = col.max()
            yref = max(col.min(), bottom[x] - 2)
            ext[bottom[x] + 1:bottom[x] + 17, x] = still[yref, x]
            ext[bottom[x] + 1:bottom[x] + 17, x, 3] = 255
    layers["body"] = lb
    lh = np.zeros_like(still)
    lh[hair] = still[hair]
    layers["hair"] = lh
    ls = np.zeros_like(still)
    ls[stones] = still[stones]
    layers["stones"] = ls
    hy = np.nonzero(hair.any(axis=1))[0]
    hx = np.nonzero(hair.any(axis=0))[0]
    root = (float(hx.mean()) if hx.size else w / 2, float(hy.max()) if hy.size else 0.0)
    return layers, masks, bottom, ext, root


# ---------------------------------------------------------------- 3. poses
def wave(f, amp, cycles=1.0, phase=0.0):
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """rise: body up px (neg = sinks into the hole); lean: px the dome top moves (screen side of forward);
    hair: deg; g: whole sprite [fwd, up]; st: stone rumble (left up, right up); fx list."""
    P = {"rise": 0, "lean": 0.0, "hair": 0.0, "g": [0, 0], "st": (0, 0), "sway": 0.0}
    fx = []
    if action == "idle":
        P["rise"] = round(wave(f, 1.0, 1, 0.0))
        P["hair"] = wave(f, 8.0, 1, -0.6)
        P["sway"] = wave(f, 0.8, 1, 0.0)
    elif action == "walk":
        t = (f % 8) / 8.0
        P["rise"] = round(2.0 * math.sin(math.pi * t))
        P["sway"] = wave(f, 1.6, 1, 0.0)
        P["lean"] = round(wave(f, 1.0, 2, 0.0))
        P["hair"] = -10.0 * math.sin(math.pi * t) + wave(f, 6.0, 1, 0.0)
        P["st"] = (1 if (f % 8) in (1, 2, 3) else 0, 1 if (f % 8) in (5, 6, 7) else 0)
        if f % 4 in (1, 2):
            fx.append(("grit", f))
    elif action == "attack":
        P["rise"] = [0, -3, -7, -10, -10, -2, 5, 6, 4, 2, 1, 0, 0, 0, 0, 0][f]
        P["lean"] = [0, 0, 0, 0, 0, 2, 5, 7, 6, 4, 3, 2, 1, 1, 0, 0][f]
        P["g"] = [[0, 0, 0, 0, 0, 1, 3, 4, 4, 3, 2, 2, 1, 1, 0, 0][f], 0]
        P["hair"] = [0, -6, -10, -12, -12, 18, 26, 14, -8, -14, -8, -4, 2, 2, 0, 0][f]
        P["st"] = [(0, 0), (1, 0), (0, 1), (1, 0), (0, 1), (2, 2), (1, 1), (0, 0), (0, 0), (0, 0),
                   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)][f]
        if 1 <= f <= 4:
            fx.append(("dust", f))
        if 5 <= f <= 8:
            fx.append(("rocks", f))
        if f in (7, 8, 9):
            fx.append(("impact", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["rise"] = round(-3.0 * env)
        P["lean"] = round(-3.0 * env)
        P["g"] = [round(-2.0 * env) + jit, 0]
        P["hair"] = 16.0 * env
        P["st"] = ((1, 0) if f in (2, 4) else (0, 1) if f in (3, 5) else (0, 0))
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 9:
            fx.append(("stars", f))
    return P, fx


def shift(img, dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if not dx and not dy:
        return img
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def shear(img, lean_x, base_y, top_y):
    """Lean: row y moves lean_x * (base_y - y) / (base_y - top_y) px (integer per row, pixels only moved)."""
    if abs(lean_x) < 0.25:
        return img
    out = np.zeros_like(img)
    H = max(1.0, float(base_y - top_y))
    for y in range(img.shape[0]):
        d = int(round(lean_x * max(0.0, base_y - y) / H))
        out[y] = shift(img[y:y + 1], d, 0)[0]
    return out


def rot4(img, ang, piv):
    """rotate (+ = clockwise) about piv at 4x nearest, then back to 1x."""
    if abs(ang) < 0.5:
        return img
    up = np.repeat(np.repeat(img, UP, 0), UP, 1)
    im = Image.fromarray(up, "RGBA").rotate(-ang, resample=Image.NEAREST, center=(piv[0] * UP + UP / 2, piv[1] * UP + UP / 2))
    return np.array(im)[1::UP, 1::UP].copy()


def compose(R, C, action, f):
    """C: placed cells + ground line. Returns 1x RGBA cell, fx, (dome top x, y), grid offset."""
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    sideways = abs(fwx) > 0.3
    side = 1.0 if fwx > 0 else -1.0
    rise = P["rise"]
    # body (+ extension when risen, clipped at the ground line when sunk)
    body = C["body"]
    if rise > 0:
        ext = C["ext"]
        body = body.copy()
        m = (ext[..., 3] > 0) & (body[..., 3] == 0)
        body[m] = ext[m]
    body = shift(body, 0, -rise)
    hair = shift(C["hair"], 0, -rise)
    if rise != 0:
        gl = C["ground"]                                    # per column: rows > gl are inside the ground
        yy = np.arange(CELL_H)[:, None]
        hide = (yy > gl[None, :]) | (gl[None, :] < 0)
        body[hide] = 0
    # hair on the dome
    piv = (C["root"][0], C["root"][1] - rise)
    hair = rot4(hair, P["hair"] * (side if sideways else 1.0), piv)
    up = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    for lyr in (body, hair):
        m = lyr[..., 3] > 0
        up[m] = lyr[m]
    # lean (row shear about the ground line) + sideways sway
    lean = P["lean"]
    sx = (lean * side if sideways else lean * 0.5) + P["sway"]
    base = C["base_y"]
    top = C["top_y"] - rise
    up = shear(up, sx, base, top)
    # stones: left / right halves rumble
    st = C["stones"]
    sl, sr = P["st"]
    if sl or sr:
        L = st.copy(); L[:, int(C["cx"]):] = 0
        Rr = st.copy(); Rr[:, :int(C["cx"])] = 0
        st = np.zeros_like(st)
        for lyr, d in ((L, sl), (Rr, sr)):
            lyr = shift(lyr, 0, -d)
            m = lyr[..., 3] > 0
            st[m] = lyr[m]
    canvas = up
    m = st[..., 3] > 0
    canvas[m] = st[m]
    gf, gu = P["g"]
    if not sideways:
        gf *= 0.6
    gx, gy = round(gf * fwx), round(gf * fwy - gu)
    canvas = shift(canvas, gx, gy)
    return canvas, fx, (C["top_x"] + sx + gx, top + gy), (gx, gy)


# ---------------------------------------------------------------- effects
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def star(a, x, y, r, c1, c2):
    for t in range(-r, r + 1):
        put(a, x + t, y, c1 if abs(t) < r else c2)
        put(a, x, y + t, c1 if abs(t) < r else c2)
    if r >= 3:
        for t in (-1, 1):
            put(a, x + t, y + t, c2)
            put(a, x + t, y - t, c2)


def pebble(a, x, y, cols, big=False):
    lt, dk = cols
    put(a, x, y, lt)
    put(a, x + 1, y, dk)
    if big:
        put(a, x, y + 1, dk)
        put(a, x + 1, y + 1, dk)


def draw_fx(a, fx, fwd, top, centre, ground_y, rock_cols, half_w):
    fwx, fwy = fwd
    sideways = abs(fwx) > 0.3
    for kind, k in fx:
        if kind == "dust":
            for j in range(2 + k // 2):
                ang = j * 1.9 + k * 0.7
                pebble(a, centre[0] + math.cos(ang) * (half_w + 1 + k), ground_y - 2 - abs(math.sin(ang)) * (k + 1), rock_cols)
        elif kind == "rocks":
            t = k - 4
            for j, (ox, vy) in enumerate(((-1.0, 3.0), (1.0, 3.5), (-0.6, 5.0), (0.7, 4.5))):
                x = centre[0] + ox * (half_w - 2 + 3 * t)
                y = ground_y - 6 - vy * t + 0.9 * t * t
                pebble(a, x, y, rock_cols, big=(j < 2))
        elif kind == "impact":
            r = {7: 3, 8: 5, 9: 3}[k]
            if sideways:
                hx, hy = top[0] + fwx * (half_w + 2), top[1] + 5
            else:
                hx, hy = (centre[0] - half_w - 3, top[1] + 4) if fwy > 0 else (centre[0] + half_w + 3, top[1] + 4)
            star(a, hx, hy, r, FLASH, GLOW)
        elif kind == "grit":
            bx = centre[0] - (fwx if sideways else 0.0) * (half_w + 2)
            for j in range(2):
                pebble(a, bx + (j * 2 - 1) * (2 if not sideways else 1) - (fwx * j * 2 if sideways else 0), ground_y - 1 - (k % 2) - j, rock_cols)
        elif kind == "stars":
            for j in range(3):
                ang = j * 2.1 + k * 0.8
                cx, cy = centre[0] + math.cos(ang) * 8, top[1] - 4 + math.sin(ang) * 2.0
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


def place(layer):
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    ox = CELL_W // 2 - w // 2
    oy = FOOT_Y - (h - 1)
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(h, CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell, (ox, oy)


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
        run_tool(["build", "--input", str(raw), "--colors", "22", "--reserve", "#%02x%02x%02x" % FLASH, "#%02x%02x%02x" % GLOW,
                  "--balance", "--name", "mole", "--output-dir", str(pal_dir)])
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

    # rock colours for debris = the stone pixels' lightest and a darker grey (palette colours)
    rig = {}
    for v in RIG:
        layers, masks, bottom, ext, root = build_rig(still[v])
        C = {}
        for n, lyr in layers.items():
            C[n], (ox, oy) = place(lyr)
        e = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        eh = min(ext.shape[0], CELL_H - oy)
        x0, x1 = max(0, ox), min(CELL_W, ox + ext.shape[1])
        e[oy:oy + eh, x0:x1] = ext[:eh, x0 - ox:x1 - ox]
        C["ext"] = e
        gl = np.full(CELL_W, -1)
        for x in range(ext.shape[1]):
            if 0 <= x + ox < CELL_W and bottom[x] >= 0:
                gl[x + ox] = bottom[x] + oy
        C["ground"] = gl
        C["root"] = (root[0] + ox, root[1] + oy)
        bm = np.zeros((CELL_H, CELL_W), bool)
        bm[oy:oy + masks["body"].shape[0], ox:ox + masks["body"].shape[1]] = masks["body"]
        ys, xs = np.nonzero(bm)
        allm = (C["body"][..., 3] > 0) | (C["hair"][..., 3] > 0)
        ays, axs = np.nonzero(allm)
        C["top_y"] = int(ays.min())
        C["top_x"] = float(xs[ys <= ys.min() + 1].mean())
        C["base_y"] = int(ys.max())
        C["cx"] = float(xs.mean())
        C["half_w"] = float((xs.max() - xs.min()) / 2)
        sm = masks["stones"]
        sp = still[v][sm][:, :3]
        if len(sp):
            Ls = sp @ np.array([0.299, 0.587, 0.114])
            lt = tuple(int(c) for c in sp[int(np.argmax(Ls))])
            mid = sorted(set(map(tuple, sp.tolist())), key=lum)
            dk = tuple(int(c) for c in mid[len(mid) // 3])
        else:
            lt, dk = pal[-1], pal[0]
        C["rock"] = (lt, dk)
        rig[v] = C
        print(v, "parts px:", {k: int(m.sum()) for k, m in masks.items()}, "rock", C["rock"], flush=True)

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in RIG:
        C = rig[v]
        R = RIG[v]
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx, top, (gx, gy) = compose(R, C, act, f)
                centre = (C["cx"] + gx, FOOT_Y - 12 + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, R["fwd"], top, centre, C["base_y"] + gy, C["rock"], C["half_w"])
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
        for si, fc in enumerate((0, 4, 8, 12) if act != "attack" else (0, 3, 6, 8)):
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
    still_diff = 0
    for v in RIG:
        ref, _ = place(still[v])
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
    meta = {"generator": "forge_gen1_0050_diglett_alola_actions", "version": VERSION, "key": KEY, "dex": 50,
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
