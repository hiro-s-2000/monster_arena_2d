#!/usr/bin/env python3
"""0024_arbok (gen1 normal): 8-direction x 4-action sheet, Antigravity key art + agent-sprite-forge.

Batch procedure md/gen1-forge.md (2026-11-04); structure copied from tools/forge_raichu_mega_x_actions.py, rig /
colours / attack built from this form's own art (a legless purple cobra: one thick ground coil, a raised neck that
spreads into one wide hood with a face-like pattern, one small head with open jaws and two fangs, one thin tail).

Source:
  art/ref_front.png, art/ref_back.png : official front.png / back.png x8 nearest (references only)
  art/agy_prompt.md                   : morphological prompt (no proper nouns) for the Antigravity CLI (`agy -p`)
  art/out/keys_2x2.png                : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png               : SW
  Mirrors: SE = SW, E = W, NW = NE.

Pipeline:
  1. key the white backdrop (border-connected only), crop, "mode" downscale + 1 px outline. S height = opaque
     height of the official front.png; W / NE / N share the grid scale, SW is matched on height.
  2. agent-sprite-forge palette_tool build (incl. reserved venom / flash colours) -> lock -> apply.
  3. part rig per view: coil (ground, the base), tail (pivot where it leaves the coil), hood (raised neck + hood,
     pivot at the neck base on the coil), head (pivot at the top of the hood). No legs: nothing is added.
  4. poses at 4x nearest, sampled back to 1x.
     idle   : the raised hood sways like a cobra (small tilt about the neck base), the head bobs, the tail flicks.
     walk   : legless slither: the coil shifts side to side, the hood sways the other way and bobs, tail whips.
     attack : fang strike: rear back (hood leans back, head lifts) -> the head and hood lunge forward and bite
              -> two fang slash arcs + venom drops at the bite point -> recover. The tail coils up and lashes.
     hurt   : white flash, recoil back (lean <= 14 deg), recover.
  5. QC: off-palette, edge touch, still diff (idle f0 == still), gait_loop.py on idle / walk.

Writes <form>/8dir_stills.png and <form>/agent-sprite-forge/ (sheet.png = 4 actions x 8 rows, preview.png,
meta.json, idle.png ...). The official front/back png/gif are never touched. Syncs agent-sprite-forge/ (without
_work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0024_arbok_actions.py [--force] [--no-sync] [--stills-only]
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
KEY = "0024_arbok"
REL = "monsterSprite/gen1/0024_arbok/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0024_arbok_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 160, 100, 88
UP = 4
VENOM = (214, 92, 255)           # reserved: venom drops
VENOM_HI = (246, 206, 255)       # reserved: venom highlight / fang slash
FLASH = (255, 255, 255)          # reserved: hit flash
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each still)
# label: later polygons win. draw: back to front. body_x: placed on the cell centre. fwd: screen "forward".
# mouth: bite point on the head (None = hidden: the bite lands in front of the head). tail_dir: +1 tail on the right.
RIG_FILE = Path(__file__).with_suffix(".rig.json")


def _v(head, hood, coil, tail, piv, body_x, fwd, mouth, tail_dir, back, tail_front=False):
    draw = ["coil", "tail", "hood", "head"] if tail_front is False else ["coil", "hood", "head", "tail"]
    if back:
        draw = ["hood", "head", "coil", "tail"] if tail_front else ["tail", "hood", "head", "coil"]
    return dict(poly={"head": head, "hood": hood, "coil": coil, "tail": tail},
                label=["coil", "hood", "tail", "head"], draw=draw,
                pivot=piv, body_x=body_x, fwd=fwd, mouth=mouth, tail_dir=tail_dir, back=back)


RIG = {}


def load_rig():
    RIG.clear()
    if RIG_FILE.exists():
        for v, d in json.loads(RIG_FILE.read_text()).items():
            d = dict(d)
            piv = {k: tuple(p) for k, p in d.pop("piv").items()}
            RIG[v] = _v(piv=piv, **{k: (tuple(x) if k in ("fwd", "mouth") and x else x) for k, x in d.items()})


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


def outline(a, col=(16, 16, 16)):
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
        cushion = 4 if n in ("coil", "hood") else 2
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n == "hood" and m.any():
            # underlay inside the closed hood + head outline so the head can bob without holes
            bh = m | masks["head"]
            inside = ndi.binary_fill_holes(ndi.binary_closing(np.pad(bh, 4), iterations=4))[4:-4, 4:-4]
            seam |= above & ~m & inside & ~masks["head"]
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
    """part -> [rot deg, fwd px, up px, side px]; hood rot + = lean toward forward (side views) / tilt (front).
    'g' = [forward px, up px, lean deg]."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        br = wave(0.6, 1, f, -math.pi / 2)              # 0 .. 1.2
        P["hood"] = [wave(2.5, 1, f, 0.0), 0, round(br * 0.8), 0]
        P["head"] = [wave(4.0, 1, f, -0.6), 0, round(br), 0]
        P["tail"] = [wave(9.0, 2, f, 0.0), 0, 0, 0]
    elif action == "walk":
        # slither: the coil slides side to side, the hood counter-sways, the head leads, the tail whips (2 waves)
        ph = 2 * math.pi * 2 * f / N
        P["coil"] = [0, 0, 0, round(1.5 * math.sin(ph))]
        P["hood"] = [wave(4.0, 2, f, 0.0) * -1, 0, round(1.0 * (0.5 - 0.5 * math.cos(ph))), 0]
        P["head"] = [wave(5.0, 2, f, -0.8), round(1.0 * math.sin(ph - 0.8)), round(1.0 * (0.5 - 0.5 * math.cos(ph))), 0]
        P["tail"] = [wave(14.0, 2, f, 0.5), 0, 0, round(1.5 * math.sin(ph))]
    elif action == "attack":
        g_f = [0, -1, -2, -2, 2, 5, 6, 6, 5, 4, 3, 2, 1, 1, 0, 0][f]
        lean = [0, -4, -8, -10, 0, 8, 12, 12, 10, 8, 6, 4, 3, 1, 0, 0][f]
        P["g"] = [g_f, 0, lean]
        P["hood"] = [[0, -3, -6, -7, 2, 7, 9, 9, 8, 6, 4, 3, 2, 1, 0, 0][f], 0,
                     [0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f], 0]
        P["head"] = [[0, -6, -10, -12, 4, 12, 16, 16, 14, 10, 7, 5, 3, 1, 0, 0][f],
                     [0, -1, -2, -2, 3, 6, 8, 8, 7, 5, 4, 3, 2, 1, 0, 0][f],
                     [0, 1, 2, 2, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0][f], 0]
        P["tail"] = [[0, 12, 22, 26, 10, -8, -14, -14, -12, -8, -5, -3, -2, -1, 0, 0][f], 0, 0, 0]
        if 1 <= f <= 3:
            fx.append(("hiss", f))
        if 5 <= f <= 9:
            fx.append(("fang", f))
        if 6 <= f <= 11:
            fx.append(("venom", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [-4.0 * env + jit, 1.0 * env, -8.0 * env]
        P["hood"] = [-6.0 * env, 0, 0, 0]
        P["head"] = [-10.0 * env, -1.0 * env, 1.0 * env, 0]
        P["tail"] = [-16.0 * env, 0, 0, 0]
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


def rot_pt(pt, ang, piv):
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    x, y = pt[0] - piv[0], pt[1] - piv[1]
    return (piv[0] + x * c - y * s, piv[1] + x * s + y * c)


def compose(view, cells, piv, action, f):
    R = RIG[view]
    P, fx = pose(action, f)
    fwx, fwy = R["fwd"]
    zero = [0.0, 0.0, 0.0, 0.0]
    side = -1.0 if fwx < 0 else 1.0
    sideview = abs(fwx) > 0.3
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    offs = {}
    # the hood carries the head: the head's pivot / offset follow the hood's tilt
    hp = P.get("hood", zero)
    hood_rot = hp[0] * (side if sideview else 0.5 * (1 if fwy > 0 else -1))
    for n in R["draw"]:
        p = P.get(n, zero)
        rot, df, du, ds = p[0], p[1], p[2], p[3]
        extra = (0.0, 0.0)
        pv = piv[n]
        if n == "hood":
            rot = hood_rot
        elif n == "head":
            rot = rot * (side if sideview else 0.4 * (1 if fwy > 0 else -1))
            # ride on the hood: move with the hood's tilt about its pivot, plus its own lift
            npv = rot_pt(pv, hood_rot, piv["hood"])
            extra = (npv[0] - pv[0], npv[1] - pv[1] - hp[2])
            rot += hood_rot
        elif n == "tail":
            rot = rot * (-1.0 if R["tail_dir"] > 0 else 1.0)     # + raises the tip
        if not sideview:                                      # toward / away from the camera
            dx, dy = ds, df * fwy * 0.5 - du
        else:
            dx, dy = df * fwx + ds * 0.5, df * fwy * 0.5 - du
        dx += extra[0]
        dy += extra[1]
        offs[n] = (dx, dy, rot)
        l4 = rot4(up4(cells[n]), rot, pv)
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if not sideview:
        gf *= 0.5
    feetpt = (CELL_W // 2, FOOT_Y)
    ang = 0.0
    if abs(lean) > 0.3 and sideview:
        ang = lean * side
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


def draw_fx(a, fx, bite, fwd, C, front_hit):
    fwx, fwy = fwd
    L = math.hypot(fwx, fwy) or 1
    ux, uy = fwx / L, fwy / L
    px, py = -uy, ux
    for kind, k in fx:
        if kind == "hiss" and bite:
            # short hiss lines in front of the jaws
            for j in range(k):
                d = 3 + 2 * j
                o = (j - (k - 1) / 2) * 2.5
                line(a, bite[0] + ux * d + px * o, bite[1] + uy * d + py * o,
                     bite[0] + ux * (d + 2) + px * o * 1.3, bite[1] + uy * (d + 2) + py * o * 1.3, VENOM_HI)
        elif kind == "fang" and bite:
            # two curved fang slash arcs across the bite point
            t = k - 5
            r = [4, 6, 8, 8, 7][t]
            cx, cy = bite[0] + ux * 4, bite[1] + uy * 4
            for o in (-2.5, 2.5):
                for s in range(-6, 7):
                    u = s / 6.0
                    x = cx + px * (u * r) + ux * (o + 1.5 * (1 - u * u))
                    y = cy + py * (u * r) + uy * (o + 1.5 * (1 - u * u))
                    put(a, x, y, FLASH if t < 3 else VENOM_HI)
                    put(a, x - ux, y - uy, VENOM if t >= 1 else VENOM_HI)
        elif kind == "venom" and bite:
            t = k - 6
            cx, cy = bite[0] + ux * 5, bite[1] + uy * 5
            for j in range(6):
                ang = j * math.pi / 3 + 0.4
                rr = 3 + 2.2 * t
                x = cx + math.cos(ang) * rr
                y = cy + math.sin(ang) * rr * 0.8 + 0.6 * t * t * 0.5
                put(a, x, y, VENOM)
                if t < 4:
                    put(a, x, y - 1, VENOM_HI)
                    put(a, x + 1, y, VENOM)
        elif kind == "hit" and front_hit:
            hx, hy = front_hit
            rr = 2 + k
            for j in range(6):
                ang = j * math.pi / 3 + 0.3
                put(a, hx + math.cos(ang) * rr, hy + math.sin(ang) * rr, FLASH if k < 3 else C["yel"])
                put(a, hx + math.cos(ang) * (rr - 1), hy + math.sin(ang) * (rr - 1), C["org"])


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FLASH
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for p in (ART / "out/keys_2x2.png", ART / "out/sw_single.png", RIG_FILE):
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
    load_rig()

    work = OUT_DIR / "_work"
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
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
        run_tool(["build", "--input", str(raw), "--colors", "24", "--reserve", "#%02x%02x%02x" % VENOM,
                  "#%02x%02x%02x" % VENOM_HI, "#%02x%02x%02x" % FLASH, "--balance", "--name", KEY,
                  "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1",
                  "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", st["scale"], flush=True)
    if args.stills_only:
        return
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}
    pal = load_palette(lock_dir)
    C = {"yel": nearest(pal, (238, 230, 82)), "org": nearest(pal, (246, 115, 74))}

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
                if mp:
                    p0 = (mp[0] + ox, mp[1] + oy)
                    p0 = rot_pt(p0, hrot, piv["head"])
                    p0 = (p0[0] + hdx, p0[1] + hdy)
                else:
                    hc = piv["head"]
                    p0 = (hc[0] + hdx + R["fwd"][0] * 6, hc[1] + hdy - 4)
                p0 = rot_pt(p0, ang, feetpt)
                bite = (p0[0] + gx, p0[1] + gy)
                front_hit = (CELL_W / 2 + gx + R["fwd"][0] * 8, FOOT_Y - 36 + gy)
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, bite, R["fwd"], C, front_hit)
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
    for v in RIG:
        cells, piv, (ox, oy) = rig[v]
        ref = np.zeros((CELL_H, CELL_W, 4), np.uint8)
        ref[oy:oy + still[v].shape[0], ox:ox + still[v].shape[1]] = still[v]
        for act in ACTIONS:
            f0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
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
    meta = {"generator": "forge_gen1_0024_arbok_actions", "version": VERSION, "key": KEY, "dex": 24,
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
