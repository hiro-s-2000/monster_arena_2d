"""Mega Gengar idle taunt (2026-11-04, user request:
「メガゲンガーの歯が見える時は、時折、口を開いて、長いベロを出して相手を挑発するアクションに」).

Input : ../_backup_gengar_mega_taunt_20261104/0094_gengar-mega.png  (original hand-made idle sheet,
        8 direction columns S,SE,E,NE,N,NW,W,SW x 8 idle frames, cell 229x196)
Output: monsterSprite/gen1/0094_gengar/mega/gengar-mega/0094_gengar-mega.png + the "n"/"d"/"v" of its data/dir8.js entry.

Views where the grin is visible (S, SE, E, W, SW) get a longer idle: the original 8-frame breath loop twice
(2.4 s), then a taunt (~1.5 s): the grin parts along its own crescent (upper teeth stay, the lower teeth row
drops a few px, lens-shaped: 0 at both mouth corners, so no square mouth and no jutting chin), a long pink
tongue slides out and hangs over the lower teeth / belly, wags left-right, then goes back and the mouth closes.
Back views (NE, N, NW) keep their 8 frames.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "_backup_gengar_mega_taunt_20261104" / "0094_gengar-mega.png"
DST = ROOT / "monsterSprite/gen1/0094_gengar/mega/gengar-mega/0094_gengar-mega.png"
DIR8 = ROOT / "data/dir8.js"
KEY = "0094_gengar-mega"
VER = "gengar_mega_taunt3"
CW, CH = 229, 196
FRONT = {0: 0, 1: 1, 2: 1, 6: -1, 7: -1}      # column -> facing side (0 = toward camera, +1 right, -1 left)
BASE_D = 0.15

# tongue palette (pink; reads apart from the crimson belly)
T_OUT = (28, 18, 36)      # same near-black as the sprite's outline
T_DARK = (214, 84, 132)
T_MID = (248, 140, 178)
T_LIT = (255, 178, 206)
T_SHINE = (255, 232, 240)
MOUTH = (34, 10, 40)
THROAT = (92, 22, 60)

# taunt timeline: (mouth open 0..1, tongue length 0..1, wag phase, duration)
TAUNT = [
    (0.35, 0.0, 0.0, 0.07),
    (0.7, 0.0, 0.0, 0.07),
    (1.0, 0.25, 0.0, 0.06),
    (1.0, 0.55, 0.0, 0.06),
    (1.0, 0.85, 0.0, 0.06),
    (1.0, 1.0, 0.0, 0.10),
    (1.0, 1.0, 0.9, 0.10),
    (1.0, 1.0, 0.0, 0.10),
    (1.0, 1.0, -0.9, 0.10),
    (1.0, 1.0, 0.0, 0.10),
    (1.0, 1.0, 0.9, 0.10),
    (1.0, 1.0, 0.0, 0.12),
    (1.0, 0.6, 0.0, 0.06),
    (1.0, 0.25, 0.0, 0.06),
    (0.6, 0.0, 0.0, 0.07),
    (0.25, 0.0, 0.0, 0.08),
]


def teeth_mask(c):
    a = c.astype(int)
    w = (a[..., 3] > 200) & (a[..., :3].min(-1) > 185) & ((a[..., :3].max(-1) - a[..., :3].min(-1)) < 40)
    w = ndi.binary_closing(w, iterations=2)
    lab, n = ndi.label(w)
    if n == 0:
        return None
    s = ndi.sum(w, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(s)))


def open_grin(c, T, amt, peak=0.5):
    """Part the grin along its crescent. The gap is widest at `peak` (0 = left end, 1 = right end of the teeth;
    toward the snout in side views) and 0 at both corners. Returns frame, mouth gap per column, widest column."""
    out = c.copy()
    xs = np.nonzero(T.any(0))[0]
    xl, xr = xs.min(), xs.max()
    span = max(1, xr - xl)
    G = max(4, round(span * 0.15)) * amt
    gaps = {}
    for x in range(xl, xr + 1):
        ys = np.nonzero(T[:, x])[0]
        if len(ys) == 0:
            continue
        t0, t1 = ys.min(), ys.max()
        u = (x - xl) / span
        v = u / peak if u <= peak else (1 - u) / (1 - peak)
        g = int(round(G * math.sin(0.5 * math.pi * max(0.0, min(1.0, v))) ** 0.8))
        if g <= 0:
            continue
        m = (t0 + t1) // 2
        lo = min(CH - 1, t1 + 3)                      # lower teeth + their dark lip line
        col = c[m:lo + 1, x].copy()
        end = min(CH, m + g + len(col))
        out[m + g:end, x] = col[:end - m - g]
        for k in range(g):                            # mouth interior, darker at the throat
            out[m + k, x, :3] = THROAT if (0 < k < g - 1 and g >= 4) else MOUTH
            out[m + k, x, 3] = 255
        gaps[x] = (m, g)
    xc = max(gaps, key=lambda x: (gaps[x][1], -abs(x - (xl + peak * span)))) if gaps else (xl + xr) // 2
    return out, gaps, xc


# per view: how the face is turned.  front = toward the camera, three = 3/4, profile = side.
VIEWS = {
    0: dict(kind="front", side=0, peak=0.5),
    1: dict(kind="three", side=1, peak=0.62),
    7: dict(kind="three", side=-1, peak=0.38),
    2: dict(kind="profile", side=1, peak=0.80),
    6: dict(kind="profile", side=-1, peak=0.20),
}


def tongue_path(root, view, L, wag):
    """Centre line of the tongue, root -> tip, following the way the face points.
    front  : hangs straight down over the lower teeth and belly, the tip sways left/right;
    three  : out toward the side the face turns to, then down;
    profile: straight out of the snout, then droops; the tip flaps up/down."""
    side = view["side"]
    pts = []
    n = max(2, int(L * 1.5))
    for i in range(n + 1):
        s = i / n
        if view["kind"] == "front":
            x = root[0] + wag * 0.28 * L * s ** 1.6
            y = root[1] + L * s
        elif view["kind"] == "three":
            x = root[0] + side * 0.55 * L * s + wag * 0.22 * L * s ** 1.6
            y = root[1] + 0.82 * L * s
        else:
            x = root[0] + side * 0.80 * L * s
            y = root[1] + 0.12 * L * s + (0.55 + 0.18 * wag) * L * s * s
        pts.append((x, y))
    return pts


def draw_tongue(img, pts, r_root, r_tip):
    """Cute paddle tongue drawn flat like the sprite art: narrow at the mouth, wide round tip, near-black 1px
    outline, one pink + one shadow tone (no gradient, groove or shine — v3, user: too realistic)."""
    h, w = img.shape[:2]
    P = np.array(pts)
    n = len(P)
    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0 = P.min(0) - r_tip - 3
    x1, y1 = P.max(0) + r_tip + 3
    box = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    by, bx = np.nonzero(box)
    best = np.full(len(by), 1e9)
    bi = np.zeros(len(by), int)
    for i, (px, py) in enumerate(P):
        d = np.hypot(bx - px, by - py)
        sel = d < best
        best[sel] = d[sel]
        bi[sel] = i
    bt = bi / (n - 1)
    sm = np.clip(bt / 0.7, 0, 1)
    sm = sm * sm * (3 - 2 * sm)
    rad = r_root + (r_tip - r_root) * sm
    inside = best <= rad
    ring = (best <= rad + 1.0) & ~inside
    img[by[ring], bx[ring], :3] = T_OUT
    img[by[ring], bx[ring], 3] = 255
    tipx, tipy = P[-1]
    for j in np.nonzero(inside)[0]:
        yv, xv, i, t, r = by[j], bx[j], bi[j], bt[j], rad[j]
        i2 = min(n - 1, i + 1)
        tx, ty = P[i2] - P[max(0, i2 - 1)]
        nl = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / nl, tx / nl
        if nx > 0 or (nx == 0 and ny > 0):           # normal points to the upper-left side
            nx, ny = -nx, -ny
        off = (xv - P[i][0]) * nx + (yv - P[i][1]) * ny   # + = upper-left (lit) side
        # flat 2-tone cel shading like the sprite: one pink, one shadow band on the lower side
        col = T_DARK if off < -0.4 * r else T_MID
        img[yv, xv, :3] = col
        img[yv, xv, 3] = 255
    return img


def taunt_frame(c, open_amt, tongue_len, wag, view):
    T = teeth_mask(c)
    if T is None:
        return c
    out, gaps, xc = open_grin(c, T, open_amt, view["peak"])
    if tongue_len <= 0 or xc not in gaps:
        return out
    m, g = gaps[xc]
    xs = np.nonzero(T.any(0))[0]
    span = xs.max() - xs.min()
    side = view["side"]
    if view["kind"] == "front":
        r_root, r_tip, Lmax = span * 0.12, span * 0.19, span * 0.58
    elif view["kind"] == "three":
        r_root, r_tip, Lmax = span * 0.14, span * 0.21, span * 0.78
    else:
        r_root, r_tip, Lmax = span * 0.17, span * 0.25, span * 1.30
    root = (xc + side * 1.5, m + g * 0.45)
    L = Lmax * tongue_len
    pts = tongue_path(root, view, max(3.0, L), wag)
    # keep the whole tongue inside the cell (shorten instead of squashing)
    while L > 3 and (max(p[1] for p in pts) + r_tip > CH - 4 or max(p[0] for p in pts) + r_tip > CW - 4
                     or min(p[0] for p in pts) - r_tip < 3):
        L -= 1
        pts = tongue_path(root, view, L, wag)
    return draw_tongue(out, pts, max(3.5, r_root), max(5.0, r_tip))


def main():
    src = np.array(Image.open(SRC).convert("RGBA"))
    base_n = 8
    loops = 2
    n_front = base_n * loops + len(TAUNT)
    out = np.zeros((n_front * CH, 8 * CW, 4), np.uint8)
    out[:base_n * CH] = src[:base_n * CH]
    for q in FRONT:
        view = VIEWS[q]
        for L in range(1, loops):
            out[L * base_n * CH:(L + 1) * base_n * CH, q * CW:(q + 1) * CW] = src[:base_n * CH, q * CW:(q + 1) * CW]
        for j, (o, tl, wag, _) in enumerate(TAUNT):
            k = j % base_n
            c = src[k * CH:(k + 1) * CH, q * CW:(q + 1) * CW]
            f = taunt_frame(c, o, tl, wag, view)
            r = base_n * loops + j
            out[r * CH:(r + 1) * CH, q * CW:(q + 1) * CW] = f
    Image.fromarray(out).save(DST, optimize=True)

    n = [n_front if q in FRONT else base_n for q in range(8)]
    d = [([BASE_D] * (base_n * loops) + [t[3] for t in TAUNT]) if q in FRONT else [BASE_D] * base_n for q in range(8)]
    s = DIR8.read_text()
    i = s.index('"%s":{' % KEY)
    j = s.index('"idle":1', i)
    ent = s[i:j]
    ent2 = re.sub(r'"n":\[[^\]]*\]', '"n":' + json.dumps(n, separators=(",", ":")), ent)
    ent2 = re.sub(r'"d":\[\[.*?\]\]', '"d":' + json.dumps(d, separators=(",", ":")), ent2)
    s = s[:i] + ent2 + s[j:]
    k = s.index('"idle":1', i)
    e = s.index("}", k)
    tail = s[k:e]
    tail = re.sub(r',"v":"[^"]*"', "", tail) + ',"v":"%s"' % VER
    s = s[:k] + tail + s[e:]
    DIR8.write_text(s)
    print("frames per view", n, "sheet", out.shape[1], "x", out.shape[0])


if __name__ == "__main__":
    main()
