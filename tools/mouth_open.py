"""Small mouth opening that follows the face (2026-11-04, user request).

The old mega rigs opened the mouth by shifting the whole lower-jaw layer 1-2 px down and painting a rectangular
"mouth interior" block behind it. On a 60-100 px sprite that reads as a rectangle and the chin juts out
(しゃくれ). This helper replaces that with a small lens-shaped opening drawn ALONG the mouth line that is
already in the art:

  * the jaw does not move (no chin drop); the opening is carved below the mouth line, inside the face,
  * both mouth corners stay closed; the gap grows smoothly toward `peak` (0 = hinge/back, 1 = lip/front),
  * every carved pixel must lie inside the opaque silhouette eroded by 1 px, so the face outline never breaks,
  * the gap is drawn with the sprite's own palette: `deep` (darkest, e.g. the outline colour) on the gap rows and
    an optional `inner` colour (tongue / throat, e.g. the darkest red in the palette) one row inside when the gap
    is >= 2 px.

Typical use inside a forge script, after the frame is composed (before palette snap / FX):

    import mouth_open
    line = mouth_open.transform_points(MOUTH_LINE[view], dx=head_dx, dy=head_dy)   # follow the head
    img = mouth_open.open_mouth(img, line, amount=open_px, deep=INK, inner=DARK_RED, peak=0.65)

`MOUTH_LINE[view]` is a list of (x, y) cell coordinates of the closed mouth line in that view, ordered from the
mouth corner / hinge (back of the head) to the lip tip (front). Measure it on the still. Views where the mouth
is not visible (back views) simply have no line and get no opening. `amount` should stay 1 (growl) to 2 (roar).
"""
import math

import numpy as np
from scipy import ndimage as ndi


def _dense(points):
    """Resample a polyline to 1 px steps (keeps order)."""
    out = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        n = max(1, int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)))))
        for i in range(n):
            u = i / n
            out.append((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u))
    out.append(tuple(points[-1]))
    return out


def transform_points(points, dx=0.0, dy=0.0, angle=0.0, pivot=None, mirror_w=None):
    """Move a mouth line with the head: rotate `angle` degrees (+ = clockwise on screen) about `pivot`,
    then shift by (dx, dy). `mirror_w` mirrors x as (mirror_w - 1 - x) first (for flipped views)."""
    pts = [(float(x), float(y)) for x, y in points]
    if mirror_w is not None:
        pts = [(mirror_w - 1 - x, y) for x, y in pts]
    if angle and pivot is not None:
        a = math.radians(angle)
        c, s = math.cos(a), math.sin(a)
        px, py = pivot
        pts = [(px + (x - px) * c - (y - py) * s, py + (x - px) * s + (y - py) * c) for x, y in pts]
    return [(x + dx, y + dy) for x, y in pts]


def open_mouth(img, line, amount, deep, inner=None, peak=0.6, normal=None):
    """Carve a small lens-shaped opening below `line` (hinge -> lip) into an RGBA uint8 frame.

    amount : max gap in px (0 = closed, 1 = growl, 2 = roar). Values are rounded per column.
    deep   : RGB of the gap (use the sprite's darkest palette colour / outline colour).
    inner  : optional RGB for the bottom row of the gap when the local gap >= 2 (tongue).
    peak   : where along the line the gap is widest (0 = hinge end, 1 = lip end).
    normal : (nx, ny) direction the jaw opens toward; default straight down (0, 1).
    Returns a new array; pixels outside the opaque silhouette (eroded by 1 px) are never touched.
    """
    if amount <= 0 or not line or len(line) < 2:
        return img
    out = img.copy()
    opaque = img[..., 3] > 0
    safe = ndi.binary_erosion(opaque, iterations=1)
    nx, ny = normal if normal is not None else (0.0, 1.0)
    pts = _dense(line)
    n = len(pts)
    peak = min(0.95, max(0.05, peak))
    h, w = opaque.shape
    for i, (x, y) in enumerate(pts):
        t = i / max(1, n - 1)
        # smooth lens: 0 at both corners, 1 at `peak`
        u = t / peak if t <= peak else (1 - t) / (1 - peak)
        g = amount * math.sin(0.5 * math.pi * max(0.0, min(1.0, u)))
        gap = int(round(g))
        for k in range(1, gap + 1):
            px, py = int(round(x + nx * k)), int(round(y + ny * k))
            if 0 <= px < w and 0 <= py < h and safe[py, px]:
                col = inner if (inner is not None and gap >= 2 and k == gap) else deep
                out[py, px, :3] = col
                out[py, px, 3] = 255
    return out
