#!/usr/bin/env python3
"""gen1 0050_diglett (normal): 8-direction stills + 8-direction x 4-action sheet (md/gen1-forge.md).

Source:
  art/ref_front.png / art/ref_back.png = official front.png / back.png x8 nearest (white backdrop).
  The Antigravity CLI (`agy`, image-generator) drew the key views from them with a morphological prompt
  (art/agy_prompt.md, no proper nouns):
    art/out/keys_2x2.png  : top-left S, top-right SIDE (side profile), bottom-left NE, bottom-right N
    art/out/sw_single.png : SW (3/4 front)
  The other three are mirrors.

Creature (counted on the official sprite): NO limbs. ONE upright brown capsule body poking out of a hole, 2 tiny
black eyes, 1 big pink-red nose, no mouth; 1 low ring of round khaki pebbles around the base + dirt specks.

Rig (by colour, per view): "body" (brown / mauve / tan / pink nose / eyes + its outline) and "ground" (khaki /
olive pebbles + specks + their outline), plus a "hole" underlay (dark brown) inside the body's base, hidden at rest.
The body never gets legs: it telescopes out of / into the hole (rows above a reference row move, the reference row is
repeated or swallowed, the bottom contour stays), leans about its base, and the pebble ring is drawn over it.

Actions (16 frames each, waves are 0 at frame 0 and the loop end):
  idle   = breathe in the hole (rise / sink 1 px) + tiny sway, pebbles still
  walk   = burrow glide: bob in and out of the hole twice per cycle, forward lean, the two halves of the pebble ring
           rumble in turn, dirt specks kicked behind
  attack = dig in (sink into the hole, pebbles jitter) -> burst out tall and head-butt forward with the dome
           (lean), pebble clods (copied from its own ring) fly forward, impact star -> settle back
  hurt   = white flash, recoil: lean back + sink into the hole, pebbles shake, recover (max ~12 deg)

QC: off-palette 0, edge-touch 0, still diff 0 (frame 0 of every action = the placed still), gait_loop.py on idle / walk.
Never touches the official front/back png/gif. Syncs agent-sprite-forge/ (without _work) to pokemon_arena_2d.

run: <agent-sprite-forge>/.venv/bin/python -I tools/forge_gen1_0050_diglett_actions.py [--force] [--no-sync]
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
KEY = "0050_diglett"
DEX = 50
REL = "monsterSprite/gen1/0050_diglett/normal"
FORM_DIR = ROOT / REL
ART = FORM_DIR / "art"
OUT_DIR = FORM_DIR / "agent-sprite-forge"
SYNC_DIR = Path("/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_2d") / REL / "agent-sprite-forge"
FORGE = Path("/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge")
FPY = FORGE / ".venv/bin/python"
PALETTE_TOOL = FORGE / "skills/generate2dsprite/scripts/palette_tool.py"
GAIT_LOOP = FORGE / "skills/video2dsprite/scripts/gait_loop.py"

VERSION = "0050_diglett_agy_v1"
DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N = 16
AD = [0.08, 0.062, 0.05, 0.055]
CELL_W, CELL_H, FOOT_Y = 88, 72, 62
UP = 4
FLASH = (255, 255, 255)           # hurt flash / impact star core (reserved)
STAR = (255, 236, 150)            # impact star rim (reserved)
SIDE = "W"
OTHER = "W" if SIDE == "E" else "E"
DIR_SRC = {"S": ("S", False), "SE": ("SW", True), SIDE: (SIDE, False), "NE": ("NE", False), "N": ("N", False),
           "NW": ("NE", True), OTHER: (SIDE, True), "SW": ("SW", False)}
VIEWS = ["S", "SW", SIDE, "NE", "N"]
FWD = {"S": (0.0, 1.0), "SW": (-0.9, 0.45), SIDE: ((-1.0, 0.0) if SIDE == "W" else (1.0, 0.0)),
       "NE": (0.9, -0.45), "N": (0.0, -1.0)}


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
        big = max(sz)
        # keep the creature and its ground specks (small blobs near the main body), drop stray noise
        keep = [i + 1 for i, s in enumerate(sz) if s > max(60, big * 0.002)]
        out[~np.isin(lab2, keep)] = 0
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
    dst.mkdir(parents=True, exist_ok=True)
    for v, im in crops.items():
        # every view gets the official height (same creature, same ground ring): S exactly, others the same scale
        sc = target_h / im.height
        if v != "S":
            sc = target_h / crops["S"].height * (crops["S"].height / im.height) ** 0.0 * (im.height / im.height)
            sc = target_h / im.height
        for tries in range(6):
            sm = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.BOX)
            a = np.array(sm)
            a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
            a[a[..., 3] == 0] = 0
            ys = np.nonzero((a[..., 3] > 0).any(axis=1))[0]
            hh = ys.max() - ys.min() + 1
            if hh == target_h:
                break
            sc *= target_h / hh
        ys, xs = np.nonzero(a[..., 3] > 0)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
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


# ---------------------------------------------------------------- 2. rig by colour
def classify(still):
    """-> body mask, ground mask (pebbles / specks), each with its own outline."""
    op = still[..., 3] > 0
    r, g, b = [still[..., c].astype(int) for c in range(3)]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    # khaki / olive pebbles: green ~ red, blue clearly lower; body: red clearly above green (brown, mauve, pink)
    pebble = op & (mx >= 70) & (np.abs(r - g) <= 28) & (r - b >= 18)
    bodyc = op & (mx >= 70) & ((r - g) > 28)
    # pale highlights (eye glints, very light khaki): decide by neighbours below
    dark = op & ~pebble & ~bodyc
    lab = np.zeros(op.shape, int)
    lab[bodyc] = 1
    lab[pebble] = 2
    # body = the large connected body-colour blob (+ enclosed details: eyes, glints)
    bl, bn = ndi.label(bodyc)
    if bn:
        sz = ndi.sum(np.ones_like(bl), bl, range(1, bn + 1))
        main = bl == (int(np.argmax(sz)) + 1)
        filled = ndi.binary_fill_holes(main)
        lab[bodyc & ~main] = 2 if True else 1
        lab[filled & op & ~pebble] = 1
    # outline / unknown pixels: nearest labelled part
    un = op & (lab == 0)
    if un.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[un] = lab[iy[un], ix[un]]
    return lab == 1, lab == 2


def build_rig(view, still):
    body, ground = classify(still)
    ys, xs = np.nonzero(body)
    top = int(ys.min())
    bot = int(ys.max())
    # ref row: lowest row where the body is still (nearly) its full width and not covered by pebbles
    widths = body.sum(axis=1)
    full = int(widths.max())
    ref = top + int((bot - top) * 0.55)
    for y in range(bot, top, -1):
        if widths[y] >= full * 0.9 and not (ground[y] & ndi.binary_dilation(body[y][None], iterations=1)[0]).any():
            ref = y
            break
    ref = max(top + 6, min(ref, bot - 2))
    lyr_b = np.zeros_like(still); lyr_b[body] = still[body]
    lyr_g = np.zeros_like(still); lyr_g[ground] = still[ground]
    # hole underlay: the body's lower band (below ref) in the darkest body colours; hidden at rest
    pal_dark = sorted({tuple(int(c) for c in still[y, x, :3]) for y, x in zip(*np.nonzero(body))}, key=lum)
    rim, inner = pal_dark[0], pal_dark[min(1, len(pal_dark) - 1)]
    hole = np.zeros_like(still)
    band = body.copy(); band[:ref - 2] = False
    hole[band, :3] = inner; hole[band, 3] = 255
    edge = band & ~ndi.binary_erosion(band, iterations=1)
    hole[edge, :3] = rim
    return dict(body=lyr_b, ground=lyr_g, hole=hole, top=top, bot=bot, ref=ref,
                cx=float(xs.mean()), body_mask=body, ground_mask=ground)


def place(layer, ox, oy):
    h, w = layer.shape[:2]
    cell = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    y0, x0 = max(0, -oy), max(0, -ox)
    y1, x1 = min(h, CELL_H - oy), min(w, CELL_W - ox)
    cell[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = layer[y0:y1, x0:x1]
    return cell


# ---------------------------------------------------------------- 3. posing
def wave(f, amp, cycles=1.0, phase=0.0):
    return amp * (math.sin(2 * math.pi * cycles * f / N + phase) - math.sin(phase))


def pose(action, f):
    """rise (px, + = out of the hole), lean (deg, + = toward fwd), fwd shift of the body (px),
    pebble jitter (left half dy, right half dy), global x jitter, effects."""
    P = dict(rise=0.0, lean=0.0, push=0.0, gl=0.0, gr=0.0, jx=0.0)
    fx = []
    if action == "idle":
        P["rise"] = wave(f, 1.0)
        P["lean"] = wave(f, 2.0, 1.0, 0.0) * 0.0 + wave(f, 2.0, 1.0, math.pi / 2)
    elif action == "walk":
        P["rise"] = wave(f, 2.0, 2.0)
        P["lean"] = wave(f, 3.0, 2.0, math.pi / 2) + (0 if f in (0,) else 0)
        P["gl"] = -round(max(0.0, math.sin(2 * math.pi * f / N)))
        P["gr"] = -round(max(0.0, -math.sin(2 * math.pi * f / N)))
        if f % 4 in (1, 2):
            fx.append(("grit", f))
    elif action == "attack":
        P["rise"] = [0, -2, -4, -5, -3, 3, 6, 7, 6, 4, 2, 1, 0, 0, 0, 0][f]
        P["lean"] = [0, -2, -4, -5, -3, 4, 10, 14, 14, 10, 6, 3, 1, 0, 0, 0][f]
        P["push"] = [0, 0, 0, 0, 0, 1, 2, 3, 3, 2, 1, 1, 0, 0, 0, 0][f]
        jit = [0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][f]
        P["gl"], P["gr"] = -abs(jit) * (f % 2), -abs(jit) * ((f + 1) % 2)
        if 1 <= f <= 3:
            fx.append(("dust", f))
        if 5 <= f <= 10:
            fx.append(("clods", f))
        if f in (7, 8, 9):
            fx.append(("impact", f))
    elif action == "hurt":
        env = math.sin(math.pi * min(1.0, f / 12.0)) if f < 12 else 0.0
        P["rise"] = round(-3.0 * env)
        P["lean"] = -12.0 * env
        P["jx"] = ((1 if f % 2 == 0 else -1) if 1 <= f <= 6 else 0)
        P["gl"] = -1 if f in (2, 4) else 0
        P["gr"] = -1 if f in (3, 5) else 0
        if f == 1:
            fx.append(("flash", 0))
        if 2 <= f <= 7:
            fx.append(("stars", f))
    return P, fx


def telescope(layer, ref, rise):
    """rows above `ref` move up by rise (rise < 0: down, swallowed at ref); the ref row repeats to fill."""
    rise = int(round(rise))
    if rise == 0:
        return layer
    out = layer.copy()
    out[:ref + 1] = 0
    if rise > 0:
        if ref - rise + 1 > 0:
            out[:ref + 1 - rise] = layer[rise:ref + 1]
        for y in range(max(0, ref + 1 - rise), ref + 1):
            out[y] = layer[ref]
    else:
        d = -rise
        if ref + 1 - d > 0:
            out[d:ref + 1] = layer[:ref + 1 - d]
    return out


def up4(a):
    return np.repeat(np.repeat(a, UP, 0), UP, 1)


def rot4(img4, ang, piv):
    if abs(ang) < 0.3:
        return img4
    im = Image.fromarray(img4, "RGBA").rotate(-ang, resample=Image.NEAREST,
                                             center=(piv[0] * UP + UP / 2, piv[1] * UP + UP / 2))
    return np.array(im)


def shift(img, dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if not dx and not dy:
        return img
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def over(dst, src):
    m = src[..., 3] > 0
    dst[m] = src[m]


def compose(view, R, action, f):
    P, fx = pose(action, f)
    fwx, fwy = FWD[view]
    side = -1.0 if fwx < -0.3 else (1.0 if fwx > 0.3 else 0.0)
    canvas = np.zeros((CELL_H, CELL_W, 4), np.uint8)
    rise, lean = P["rise"], P["lean"]
    moving = abs(rise) >= 0.5 or abs(lean) >= 0.3 or P["push"]
    if moving:
        over(canvas, R["hole"])
    body = telescope(R["body"], R["ref"], rise)
    if abs(lean) >= 0.3 and side != 0.0:
        piv = (R["cx"], R["bot"])
        b4 = rot4(up4(body), lean * side, piv)
        body = b4[1::UP, 1::UP].copy()
    elif abs(lean) >= 0.3:
        # toward / away from the camera: lean reads as a 1 px nod (no flat spin)
        body = shift(telescope(body, R["ref"] - 4, -round(abs(lean) / 7.0)), 0, 0)
    body = shift(body, P["push"] * fwx + P["jx"], P["push"] * fwy * 0.5)
    over(canvas, body)
    g = R["ground"]
    if P["gl"] or P["gr"]:
        cx = int(round(R["cx"]))
        left = g.copy(); left[:, cx:] = 0
        right = g.copy(); right[:, :cx] = 0
        over(canvas, shift(left, 0, P["gl"]))
        over(canvas, shift(right, 0, P["gr"]))
    else:
        over(canvas, g)
    return canvas, fx


# ---------------------------------------------------------------- effects (palette colours only; flash / star reserved)
def put(a, x, y, col):
    x, y = int(round(x)), int(round(y))
    if 1 <= x < a.shape[1] - 1 and 1 <= y < a.shape[0] - 1:
        a[y, x, :3] = col
        a[y, x, 3] = 255


def star(a, x, y, r):
    for k in range(-r, r + 1):
        put(a, x + k, y, STAR if abs(k) > 1 else FLASH)
        put(a, x, y + k, STAR if abs(k) > 1 else FLASH)
    for k in range(-(r // 2), r // 2 + 1):
        put(a, x + k, y + k, STAR)
        put(a, x + k, y - k, STAR)


def paste(a, clod, x, y):
    h, w = clod.shape[:2]
    x, y = int(round(x - w / 2)), int(round(y - h / 2))
    for yy in range(h):
        for xx in range(w):
            if clod[yy, xx, 3] and 1 <= x + xx < a.shape[1] - 1 and 1 <= y + yy < a.shape[0] - 1:
                a[y + yy, x + xx] = clod[yy, xx]


def draw_fx(a, fx, view, R, clods, specks):
    fwx, fwy = FWD[view]
    cx, bot, top = R["cx"], R["bot"], R["top"]
    for kind, k in fx:
        if kind == "grit":
            # dirt specks kicked out behind the glide
            for j in range(2):
                x = cx - fwx * (10 + 3 * j + (k % 4)) + (j * 3 - 1) * (1 if fwx == 0 else 0)
                y = bot - 1 - fwy * (4 + j) - (k % 4) - j
                put(a, x, y, specks[j % len(specks)])
        elif kind == "dust":
            for j in range(2 + k):
                s = -1 if j % 2 else 1
                put(a, cx + s * (9 + j * 2), bot - 2 - k - (j // 2), specks[j % len(specks)])
        elif kind == "clods":
            t = (k - 5) / 5.0
            for j, cl in enumerate(clods[:2]):
                sp = 1 if j == 0 else -1
                d = 6 + 18 * t + 3 * j
                lift = 10 * math.sin(math.pi * min(1.0, t * 1.1)) + 2
                if abs(fwx) > 0.3:
                    x = cx + fwx * d
                    y = bot - 6 - lift + fwy * d * 0.5 + 2 * j
                else:
                    x = cx + sp * (6 + 12 * t)
                    y = bot - 4 - lift + fwy * 4 * t
                paste(a, cl, x, y)
        elif kind == "impact":
            r = {7: 3, 8: 4, 9: 2}[k]
            if abs(fwx) > 0.3:
                x = cx + fwx * 13
                y = top + 6 + fwy * 4
            else:
                x = cx + (5 if fwy > 0 else -5)
                y = top - 1 if fwy < 0 else top + 3
            star(a, x, y, r)
        elif kind == "stars":
            for j in range(2):
                ang = j * math.pi + k * 0.8
                put(a, cx + math.cos(ang) * 9, top + 2 + math.sin(ang) * 2, FLASH)
                put(a, cx + math.cos(ang) * 9 + 1, top + 2 + math.sin(ang) * 2, STAR)


def flash_img(a):
    out = a.copy()
    out[a[..., 3] > 0, :3] = FLASH
    return out


def pick_clods(R):
    """two small pebble clods cut (copied) from the creature's own ground ring."""
    g = R["ground"]
    m = g[..., 3] > 0
    r, gg, b = [g[..., c].astype(int) for c in range(3)]
    light = m & (np.maximum(np.maximum(r, gg), b) > 110)
    lab, n = ndi.label(light)
    out = []
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        order = np.argsort(-sz)
        for i in order[:4]:
            ys, xs = np.nonzero(lab == i + 1)
            cy, cx = int(ys.mean()), int(xs.mean())
            y0, y1, x0, x1 = max(0, cy - 2), cy + 3, max(0, cx - 2), cx + 3
            c = g[y0:y1, x0:x1].copy()
            if (c[..., 3] > 0).sum() >= 8:
                out.append(c)
    if not out:
        out = [g[-5:, :5].copy()]
    while len(out) < 2:
        out.append(out[0])
    return out


def speck_cols(R):
    g = R["ground"]
    m = g[..., 3] > 0
    cols = [tuple(int(v) for v in c) for c in np.unique(g[m][:, :3], axis=0)]
    cols = sorted(cols, key=lum)
    return [cols[len(cols) // 2], cols[-1]] if cols else [(128, 120, 80)]


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

    target_h = official_height()
    raw, stills, pal_dir, lock_dir = work / "stills_raw", work / "stills", work / "palette", work / "palette_lock"
    if "stills" not in st["done"]:
        for d in (raw, stills, pal_dir, lock_dir):
            if d.exists():
                shutil.rmtree(d)
        make_stills(raw, target_h)
        run_tool(["build", "--input", str(raw), "--colors", "18", "--reserve", "#%02x%02x%02x" % FLASH, "#%02x%02x%02x" % STAR,
                  "--balance", "--name", "burrower", "--output-dir", str(pal_dir)])
        run_tool(["lock", "--palette", str(pal_dir / "palette.json"), "--source", str(raw), "--output-dir", str(lock_dir)])
        run_tool(["apply", "--palette", str(lock_dir / "palette.json"), "--input", str(raw), "--orphans", "1", "--output-dir", str(stills)])
        st["done"].append("stills"); save_st()
        print("stills + palette done", flush=True)
    pal = palette_colors(lock_dir / "palette.json")
    still = {v: np.array(Image.open(stills / f"{v}.png").convert("RGBA")) for v in VIEWS}

    rigs = {}
    for v in VIEWS:
        s = still[v]
        hh, ww = s.shape[:2]
        R0 = build_rig(v, s)
        ox = int(round(CELL_W / 2 - R0["cx"]))
        oy = FOOT_Y - (hh - 1)
        R = dict(R0)
        for n in ("body", "ground", "hole"):
            R[n] = place(R0[n], ox, oy)
        R["cx"] += ox; R["top"] += oy; R["bot"] += oy; R["ref"] += oy
        R["ox"], R["oy"] = ox, oy
        R["still"] = place(s, ox, oy)
        rigs[v] = R

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    for v in VIEWS:
        R = rigs[v]
        clods, specks = pick_clods(R), speck_cols(R)
        for act in ACTIONS:
            key = f"{v}/{act}"
            if key in st["done"]:
                continue
            od = frames_dir / v / act
            od.mkdir(parents=True, exist_ok=True)
            for f in range(N):
                a, fx = compose(v, R, act, f)
                if any(k == "flash" for k, _ in fx):
                    a = flash_img(a)
                draw_fx(a, fx, v, R, clods, specks)
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
    still_diff = 0
    for v in VIEWS:
        ref = rigs[v]["still"]
        for act in ACTIONS:
            a0 = np.array(Image.open(frames_dir / v / act / "00.png").convert("RGBA"))
            still_diff += int((a0 != ref).any(axis=2).sum())
    loop_diff = 0
    for v in VIEWS:
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
    meta = {"generator": "forge_gen1_0050_diglett_actions", "version": VERSION, "key": KEY, "dex": DEX,
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
