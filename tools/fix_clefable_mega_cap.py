"""Mega Clefable: hide the Clefable-style forehead curl under the white cap (2026-11-04, user request).

front.png / back.png show the white cap pulled down to just above the eyes; the 8-direction sheet
(0036_clefable-mega.png, 8 dir columns x 8 frame rows) still showed the forehead curl. For every cell with
visible eyes, the white cap is extended down to EYE_GAP px above the eyes with a soft arc, a pink-white
shade band on its lower rim and a thin skin shadow under it. Back views (no eyes) are untouched.
Re-running is idempotent-ish but always start from the backup:
  _backup_clefable_mega_cap_20261104/0036_clefable-mega.png
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SRC = sys.argv[1]
DST = sys.argv[2]
CW, CH, COLS, ROWS = 278, 196, 8, 8
EYE_GAP = 4      # cap rim stops this many px above the eye top (centre)
ARC = 6          # rim rises this many px toward the cap sides
CAP = np.array([253, 249, 247])
RIM = [np.array([250, 242, 238]), np.array([252, 236, 228])]   # sampled from the sheet's own cap rim


def fix_cell(c):
    a = c.astype(int)
    al = a[..., 3] > 200
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    eyes = al & (B > R + 30) & (G > R + 20)
    lab, n = ndi.label(eyes)
    if n == 0:
        return c, False
    sizes = ndi.sum(eyes, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 4])
    if not keep.any():
        return c, False
    ys, xs = np.nonzero(keep)
    etop, xc = ys.min(), (xs.min() + xs.max()) / 2
    white = al & (a[..., :3].min(-1) > 235) & ((a[..., :3].max(-1) - a[..., :3].min(-1)) < 22)
    white[etop - 2:] = False
    lab2, n2 = ndi.label(white)
    if n2 == 0:
        return c, False
    big = 1 + int(np.argmax(ndi.sum(white, lab2, range(1, n2 + 1))))
    cap = lab2 == big
    cols = np.nonzero(cap.any(0))[0]
    half = max(1.0, (cols.max() - cols.min()) / 2)
    # skin + curl strokes (also the darker anti-aliased curl pixels); grey outline stays out
    warm = al & (((R > 150) & (R - B > 15)) | ((R > 90) & (R - B > 25) & (R - G > 15)))
    hh, ww = warm.shape
    xx = np.arange(ww)
    u = np.clip((xx - xc) / half, -1.3, 1.3)
    nb = np.round(etop - EYE_GAP - ARC * u * u).astype(int)
    below = np.arange(hh)[:, None] <= nb[None, :]
    # flood from the cap through warm pixels above the new rim: stays inside the head outline
    grow = (warm & below) | cap
    lab3, _ = ndi.label(grow)
    ids = np.unique(lab3[cap])
    region = np.isin(lab3, ids[ids > 0]) & warm & ~cap
    region[:cap.nonzero()[0].min()] = False
    region[:, :cols.min() + 1] = False             # never wider than the cap itself (ears / wings stay)
    region[:, cols.max():] = False
    out = c.copy()
    for y, x in zip(*np.nonzero(region)):
        k = nb[x] - y
        out[y, x, :3] = RIM[1 - k] if k <= 1 else CAP
    for x in np.unique(np.nonzero(region)[1]):     # soft shadow under the rim
        for k, f in ((1, 0.90), (2, 0.95)):
            y = nb[x] + k
            if y < etop and warm[y, x] and not eyes[y, x]:
                out[y, x, :3] = (a[y, x, :3] * f).astype(np.uint8)
    return out, True


im = np.array(Image.open(SRC).convert('RGBA'))
done = 0
for r in range(ROWS):
    for q in range(COLS):
        sl = (slice(r * CH, (r + 1) * CH), slice(q * CW, (q + 1) * CW))
        im[sl], ok = fix_cell(im[sl])
        done += ok
Image.fromarray(im).save(DST)
print('cells fixed', done)
