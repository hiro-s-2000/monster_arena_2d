#!/usr/bin/env python3
"""0023_ekans (gen1 normal): 8-direction x 4-action sheet, Antigravity key art + agent-sprite-forge.

Batch procedure md/gen1-forge.md (2026-11-04); structure copied from tools/forge_raichu_mega_x_actions.py, rig /
colours / attack built from this form's own art (a legless pink-purple snake: one raised neck with a yellow collar,
one rounded head with open jaws, one spiral coil on the ground, one thin tail with a four-ring rattle).

Source:
  art/ref_front.png, art/ref_back.png : official front.png / back.png x8 nearest (references only)
  art/agy_prompt.md                   : morphological prompt (no proper nouns) for the Antigravity CLI (`agy -p`)
  art/out/keys_2x2.png                : top-left S, top-right W (faces left), bottom-left NE, bottom-right N
  art/out/sw_single.png               : SW
  Mirrors: SE = SW, E = W, NW = NE.

Pipeline:
  1. key the white backdrop (border-connected only), crop, "mode" downscale + 1 px outline. S height = opaque
     height of the official front.png; W / NE / N share the grid scale, SW is matched on height.
  2. agent-sprite-forge palette_tool build (incl. reserved flash / spark colours) -> lock -> apply.
  3. part rig per view: coil (on the ground), neck (pivot where it leaves the coil), head (pivot at the collar,
     carried by the neck's rotation), tail + rattle (pivot at its root on the coil). No legs: nothing is added.
  4. poses at 4x nearest, sampled back to 1x.
     idle   : neck sways, head bobs, the rattle shakes (fast small tail wiggle).
     walk   : legless slither: the coil surges and humps, the neck sways side to side, head leads, tail follows.
     attack : bite: coil back (neck and head pulled back, rattle shaking) -> snap strike forward with the head
              -> bite impact + venom drops -> recover.
     hurt   : white flash, recoil back (lean <= 12 deg), recover.
  5. QC: off-palette, edge touch, still diff (idle f0 == still), gait_loop.py on idle / walk.

Writes <form>/8dir_stills.png and <form>/agent-sprite-forge/ (sheet.png = 4 actions x 8 rows, preview.png,
meta.json, idle.png ...). The official front/back png/gif are never touched. Syncs agent-sprite-forge/ (without
_work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0023_ekans_actions.py [--force] [--no-sync] [--stills-only]
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
KEY = "0023_ekans"
REL = "monsterSprite/gen1/0023_ekans/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0023_ekans_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 120, 84, 74
UP = 4
FLASH = (255, 255, 255)          # reserved: hit flash / bite spark core
SPARK = (255, 230, 140)          # reserved: bite spark rim
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), "E": ("W", True), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), "W": ("W", False), "SW": ("SW", False)}

# ---------------------------------------------------------------- rig (sprite px of each still)
# label: later polygons win. draw: back to front. body_x: placed on the cell centre. fwd: screen "forward".
# mouth: point in front of the jaws where the bite lands. tail_dir: +1 = tail on the right of the coil.
RIG = json.loads((Path(__file__).with_suffix(".rig.json")).read_text()) if Path(__file__).with_suffix(".rig.json").exists() else {}


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
        cushion = 3 if n in ("coil", "neck") else 2
        seam = above & ~m & (ndi.distance_transform_edt(~m) <= cushion)
        src_m = m
        if n == "neck" and m.any():
            # underlay inside the closed neck + head outline so the head can bob without holes
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
    """part -> [rot deg (+ = tip toward forward), fwd px, up px]; 'g' = [forward px, up px, lean deg].
    The head is carried by the neck's rotation (its pivot follows the neck top)."""
    P = {"g": [0.0, 0.0, 0.0]}
    fx = []
    if action == "idle":
        br = wave(0.6, 1, f, -math.pi / 2)              # 0 .. 1.2 (1 - cos)
        P["neck"] = [wave(2.5, 1, f, 0.0), 0, 0]
        P["head"] = [wave(-2.0, 1, f, 0.0), 0, round(br)]
        P["coil"] = [0, 0, 0]
        P["tail"] = [wave(4.0, 4, f, 0.0), 0, 0]       # rattle shake
    elif action == "walk":
        ph = 2 * math.pi * 2 * f / N
        hump = 0.5 - 0.5 * math.cos(ph)
        P["coil"] = [0, round(1.5 * math.sin(ph)), round(1.0 * hump)]
        P["neck"] = [wave(6.0, 1, f, 0.0), round(1.0 * math.sin(ph - 0.8)), round(1.0 * (0.5 - 0.5 * math.cos(ph - 0.8)))]
        P["head"] = [wave(-5.0, 1, f, 0.0), round(1.0 * math.sin(ph - 1.4)), 0]
        P["tail"] = [wave(12.0, 1, f, 1.2), round(1.5 * math.sin(ph)), round(1.0 * hump)]
    elif action == "attack":
        g_f = [0, -1, -2, -2, 2, 5, 6, 6, 5, 4, 3, 2, 1, 1, 0, 0][f]
        g_u = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        lean = [0, -3, -6, -7, 2, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0, 0][f]
        P["g"] = [g_f, g_u, lean]
        P["neck"] = [[0, -6, -11, -13, 4, 13, 16, 16, 14, 11, 8, 6, 4, 2, 1, 0][f], 0, 0]
        P["head"] = [[0, -3, -6, -7, 3, 7, 8, 8, 7, 5, 4, 3, 2, 1, 0, 0][f],
                     [0, -1, -1, -1, 2, 4, 5, 5, 4, 3, 2, 1, 1, 0, 0, 0][f], 0]
        P["coil"] = [0, [0, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0][f], 0]
        P["tail"] = [[0, 6, -4, 8, -6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f] + [0, 4, 8, 8, 4, 0, -2, -2, -2, -1, -1, 0, 0, 0, 0, 0][f], 0, 0]
        if 1 <= f <= 3:
            fx.append(("rattle", f))
        if 5 <= f <= 8:
            fx.append(("bite", f))
        if 7 <= f <= 11:
            fx.append(("venom", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        jit = (1 if f % 2 == 0 else -1) if 1 <= f <= 5 else 0
        P["g"] = [-4.0 * env + jit, 1.0 * env, -12.0 * env]
        P["neck"] = [-8.0 * env, 0, 0]
        P["head"] = [-6.0 * env, -1.0 * env, 0]
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


def rot_pt(p, piv, ang):
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    x, y = p[0] - piv[0], p[1] - piv[1]
    return (piv[0] + x * c - y * s, piv[1] + x * s + y * c)


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
    sidev = abs(fwx) > 0.3
    canvas = np.zeros((CELL_H * UP, CELL_W * UP, 4), np.uint8)
    offs = {}
    neck_rot, neck_d = 0.0, (0.0, 0.0)
    for n in R["draw"]:
        p = P.get(n, zero)
        rot, df, du = p[0], p[1], p[2]
        if n in ("neck", "head"):
            # + tips toward forward: side views turn about the joint, front / back views sway sideways (smaller)
            rot = rot * (side if sidev else 0.5)
        elif n == "tail":
            rot = rot * (-1.0 if R["tail_dir"] > 0 else 1.0)      # + swings the rattle outward / up
        if abs(fwx) < 0.5:                                         # toward / away from the camera
            dx, dy = 0.0, df * fwy * 0.5 - du
        else:
            dx, dy = df * fwx, df * fwy * 0.5 - du
        if n == "neck":
            neck_rot, neck_d = rot, (dx, dy)
        l4 = up4(cells[n])
        if n == "head":
            # carried by the neck: the head pivot follows the neck top
            hp = rot_pt(piv["head"], piv["neck"], neck_rot)
            cdx, cdy = hp[0] - piv["head"][0] + neck_d[0], hp[1] - piv["head"][1] + neck_d[1]
            l4 = rot4(l4, rot + neck_rot, piv["head"])
            dx, dy = dx + cdx, dy + cdy
        else:
            l4 = rot4(l4, rot, piv[n])
        offs[n] = (dx, dy, rot)
        l4 = shift4(l4, dx, dy)
        m = l4[..., 3] > 0
        canvas[m] = l4[m]
    gf, gu, lean = P["g"]
    if abs(fwx) < 0.5:
        gf *= 0.5
    feetpt = (CELL_W // 2, FOOT_Y)
    ang = 0.0
    if abs(lean) > 0.3:
        ang = lean * side if sidev else lean * 0.4 * (1 if fwy < 0 else -1)
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


def draw_fx(a, fx, mouth_pt, fwd, C, front_hit, rattle_pt):
    fwx, fwy = fwd
    L = math.hypot(fwx, fwy) or 1
    ux, uy = fwx / L, fwy / L
    for kind, k in fx:
        if kind == "rattle" and rattle_pt:
            # short shake ticks beside the rattle
            rx, ry = rattle_pt
            for s in (-1, 1):
                line(a, rx + s * 3, ry - 1 + k % 2, rx + s * 5, ry - 2 + k % 2, C["ochre"])
        elif kind == "bite" and mouth_pt:
            # snapping fang marks: two short white arcs closing on the target in front of the jaws
            bx, by = mouth_pt[0] + ux * 3, mouth_pt[1] + uy * 3
            r = {5: 5, 6: 4, 7: 3, 8: 4}[k]
            for j in range(7):
                t = math.pi * (0.15 + 0.7 * j / 6)
                put(a, bx + math.cos(t) * r, by - math.sin(t) * r * 0.8, FLASH)
                put(a, bx + math.cos(t) * r, by + math.sin(t) * r * 0.8, FLASH)
            if k in (6, 7):
                for j in range(8):
                    ang = j * math.pi / 4 + 0.4
                    rr = 6 if k == 6 else 8
                    put(a, bx + math.cos(ang) * rr, by + math.sin(ang) * rr, SPARK)
                    put(a, bx + math.cos(ang) * (rr - 1), by + math.sin(ang) * (rr - 1), FLASH)
        elif kind == "venom" and mouth_pt:
            # a few purple venom drops spattering from the bite point
            bx, by = mouth_pt[0] + ux * 3, mouth_pt[1] + uy * 3
            t = k - 7
            for j, (ox, oy) in enumerate(((1.0, -0.6), (1.4, 0.2), (0.8, 0.9), (0.3, -1.1))):
                d = 2 + t * 2.0
                x = bx + (ox * ux - oy * uy * 0.6) * d + (oy if abs(ux) < 0.5 else 0) * d
                y = by + (ox * uy + oy * 0.6) * d + 0.5 * t * t * 0.4
                put(a, x, y, C["venom"])
                put(a, x + 1, y, C["venom_dk"])
                if t < 2:
                    put(a, x, y - 1, C["venom_lt"])
        elif kind == "hit" and front_hit:
            hx, hy = front_hit
            rr = 2 + k
            for j in range(6):
                ang = j * math.pi / 3 + 0.3
                put(a, hx + math.cos(ang) * rr, hy + math.sin(ang) * rr, FLASH if k < 3 else SPARK)
                put(a, hx + math.cos(ang) * (rr - 1), hy + math.sin(ang) * (rr - 1), C["yel"])


def flash(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FLASH
    return out


# ---------------------------------------------------------------- main
def state_hash():
    h = hashlib.sha1(Path(__file__).read_bytes())
    for p in (ART / "out/keys_2x2.png", ART / "out/sw_single.png", Path(__file__).with_suffix(".rig.json")):
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def to_cell(pt, view, rig, offs, gxy, ang, part="head"):
    """A still point (sprite px of the view) -> cell point following the part, the lean and the global shift."""
    cells, piv, (ox, oy) = rig
    x, y = pt[0] + ox, pt[1] + oy
    dx, dy, _ = offs.get(part, (0, 0, 0))
    x, y = x + dx, y + dy
    feetpt = (CELL_W // 2, FOOT_Y)
    x, y = rot_pt((x, y), feetpt, ang)
    return (x + gxy[0], y + gxy[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--stills-only", action="store_true")
    args = ap.parse_args()

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
        run_tool(["build", "--input", str(raw), "--colors", "24", "--reserve", "#%02x%02x%02x" % FLASH,
                  "#%02x%02x%02x" % SPARK, "--balance", "--name", KEY, "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1",
                  "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", st["scale"], flush=True)
    if args.stills_only:
        return
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in RIG}
    pal = load_palette(lock_dir)
    C = {"yel": nearest(pal, (255, 213, 106)), "ochre": nearest(pal, (230, 172, 90)),
         "venom": nearest(pal, (164, 74, 139)), "venom_dk": nearest(pal, (123, 49, 106)), "venom_lt": nearest(pal, (205, 98, 180))}

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
                a, fx, offs, gxy, ang = compose(v, cells, piv, act, f)
                mouth_pt = to_cell(R["mouth"], v, rig[v], offs, gxy, ang, "head") if R.get("mouth") else None
                rattle_pt = to_cell(R["rattle"], v, rig[v], offs, gxy, ang, "tail") if R.get("rattle") else None
                front_hit = (CELL_W / 2 + gxy[0] + R["fwd"][0] * 8, FOOT_Y - 26 + gxy[1])
                if any(k == "flash" for k, _ in fx):
                    a = flash(a)
                draw_fx(a, fx, mouth_pt, R["fwd"], C, front_hit, rattle_pt)
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
    meta = {"generator": "forge_gen1_0023_ekans_actions", "version": VERSION, "key": KEY, "dex": 23,
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
