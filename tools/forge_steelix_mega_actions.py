#!/usr/bin/env python3
"""
Builds a high-precision, zero-distortion 8-direction x 4-action (`idle`, `walk`, `attack`, `hurt`)
sprite bundle for Mega Steelix (`0208_steelix-mega`) modeled directly after Normal Steelix's
official GIF motion (`0208_steelix/normal/front.gif` & `back.gif`):
  - Exact color & connected-component anatomical layer separation (10 independent rig layers):
      1. `orbit_top`, `orbit_left_upper`, `orbit_right_upper`, `orbit_lower`
         (4 independent clusters of floating hexagonal magnetic crystal debris orbiting the body
          with distinct phases, vertical bobbing, and radial expansion/contraction)
      2. `head_upper` (giant armored upper cranium, twin blunt cranial ridges, fierce blue eye,
                       nostrils, and upper white teeth — 100% rigid integer translation so the
                       face, eye, and teeth never deform by a single pixel)
      3. `lower_jaw` (massive horseshoe lower mandible with 6 protruding jaw pegs — opens and
                      closes cleanly beneath the upper teeth with dark mouth-interior cushioning)
      4. `body_upper` (arched upper steel neck & torso segments, glowing blue energy rings, and
                       iridescent prism crystal spikes)
      5. `body_mid` (middle S-coil steel boulder segments, blue core ring, and lateral prism blades)
      6. `body_base` (lowest grounded steel coil base — stays stably planted on the ground in `idle`
                      while upper segments undulate with phase delay, and glides in a travelling
                      serpentine wave during `walk`)
      7. `tail` (upward-curving armored tail segment and giant prism crystal tail blade)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouth_open  # noqa: E402  small face-following mouth opening (2026-11-04)

ROOT = Path("/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude")
MEGA_DIR = ROOT / "monsterSprite/gen2/0208_steelix/mega/steelix-mega"
OUT_DIR = MEGA_DIR / "agent-sprite-forge"

DIRS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
ACTIONS = ["idle", "walk", "attack", "hurt"]
N_FRAMES = 16

# Mouth fix (2026-11-04, user request: the jaw drop + rectangular slate pocket read as a jutting chin).
# The lower jaw no longer moves relative to the head; instead `mouth_open.open_mouth` carves a small lens
# just below the closed mouth line of `front.png` (bottom of the white upper teeth), with both corners shut.
# Native `front.png` coords, ordered hinge (mouth corner, back of the head) -> lip tip (snout, facing left).
MOUTH_LINE_NATIVE = [(27, 52), (23, 52), (19, 52), (16, 51)]
MOUTH_DEEP = (16, 16, 16)  # darkest palette colour (outline)
MOUTH_INNER = None  # no visible mouth interior colour on this species


def extract_palette(front_arr: np.ndarray, back_arr: np.ndarray) -> np.ndarray:
    colors = []
    for arr in (front_arr, back_arr):
        m = arr[:, :, 3] >= 128
        if np.any(m):
            rgb = arr.copy()
            rgb[:, :, 3] = 255
            colors.append(rgb[m])
    all_c = np.concatenate(colors, axis=0)
    return np.unique(all_c, axis=0)


def snap_to_palette(arr: np.ndarray, palette_rgba: np.ndarray) -> np.ndarray:
    m = arr[:, :, 3] >= 128
    out = np.zeros_like(arr)
    if not np.any(m):
        return out
    px = arr[m, :3].astype(np.int32)
    pal = palette_rgba[:, :3].astype(np.int32)
    diff = px[:, None, :] - pal[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = np.argmin(dist2, axis=1)
    out[m, :3] = palette_rgba[nearest, :3]
    out[m, 3] = 255
    return out


def is_dark_outline(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3] >= 128
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    return a & (
        ((r == 16) & (g == 16) & (b == 16))
        | ((r == 57) & (g == 57) & (b == 82))
        | ((r == 68) & (g == 66) & (b == 65))
        | ((r == 93) & (g == 116) & (b == 125))
    )


def build_part_layer(
    src: np.ndarray,
    mask: np.ndarray,
    foreground_mask: np.ndarray,
    cushion_px: int = 5,
) -> np.ndarray:
    layer = np.zeros_like(src)
    layer[mask] = src[mask]
    if cushion_px <= 0 or not np.any(mask) or not np.any(foreground_mask):
        return layer
    dist_to_mask = ndi.distance_transform_edt(~mask)
    seam = foreground_mask & (~mask) & (dist_to_mask >= 1) & (dist_to_mask <= cushion_px)
    if not np.any(seam):
        return layer
    fill_candidates = mask & (~is_dark_outline(src))
    if not np.any(fill_candidates):
        fill_candidates = mask
    _, (ny_idx, nx_idx) = ndi.distance_transform_edt(~fill_candidates, return_indices=True)
    sy, sx = np.nonzero(seam)
    layer[sy, sx] = src[ny_idx[sy, sx], nx_idx[sy, sx]]
    return layer


def shift_layer_int(layer: np.ndarray, dx: int, dy: int) -> np.ndarray:
    dx_i = int(round(dx))
    dy_i = int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return layer.copy()
    h, w = layer.shape[:2]
    out = np.zeros_like(layer)
    sx0 = max(0, -dx_i)
    sx1 = min(w, w - dx_i)
    sy0 = max(0, -dy_i)
    sy1 = min(h, h - dy_i)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy_i:sy1 + dy_i, sx0 + dx_i:sx1 + dx_i] = layer[sy0:sy1, sx0:sx1]
    return out


def composite_layers(canvas_shape: Tuple[int, int, int], layers: List[np.ndarray]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=np.uint8)
    for lyr in layers:
        m = lyr[..., 3] >= 128
        out[m] = lyr[m]
    return out


def pick_labels(lbl_arr: np.ndarray, predicate) -> np.ndarray:
    out_m = np.zeros(lbl_arr.shape, dtype=bool)
    for cid in range(1, int(lbl_arr.max()) + 1):
        cm = lbl_arr == cid
        ys, xs = np.where(cm)
        if len(ys) > 0 and predicate(int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), int(len(ys))):
            out_m |= cm
    return out_m


def segment_front_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `front.png` (facing left) into 10 exact anatomical masks:
      - `orbit_top`, `orbit_left_upper`, `orbit_right_upper`, `orbit_lower`:
        All 12 non-body 4-connected floating magnetic crystal components grouped into 4 clusters.
      - `head_upper`: Twin cranial ridges, forehead/snout, fierce blue eye, nostrils, and upper white teeth.
      - `lower_jaw`: Sweeping horseshoe mandible and all 6 jaw pegs (3 left, 3 bottom-right).
      - `body_upper`: Upper neck boulder, glowing blue neck ring, giant upper-right prism crystal,
                      large arched upper back boulder, and underside neck prism crystal.
      - `body_mid`: Middle coil segments with blue ring, spiked mid boulder, and left/right prism blades.
      - `body_base`: Lowest grounded coil segments (`y=76..92, x=56..80`).
      - `tail`: Upward-curving right tail segment (`x=77..94, y=63..91`) with blue ring and giant prism blade.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    r = arr96[:, :, 0].astype(np.int32)
    g = arr96[:, :, 1].astype(np.int32)
    b = arr96[:, :, 2].astype(np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    lbl_a, _ = ndi.label(a)
    counts = np.bincount(lbl_a.ravel())
    counts[0] = 0
    body_all = lbl_a == int(np.argmax(counts))
    orbit_all = a & (~body_all)

    orbit_top = orbit_all & (yy <= 12)
    orbit_left_upper = orbit_all & (yy > 12) & (yy <= 35) & (xx <= 25)
    orbit_right_upper = orbit_all & (yy > 12) & (yy <= 35) & (xx > 25)
    orbit_lower = orbit_all & (yy > 35)

    light_steel = body_all & (
        ((r == 189) & (g == 205) & (b == 197))
        | ((r == 222) & (g == 230) & (b == 222))
    )
    lbl_ls, _ = ndi.label(light_steel)
    med_steel = body_all & ((r == 123) & (g == 148) & (b == 164))
    lbl_ms, _ = ndi.label(med_steel)
    dark_steel = body_all & ((r == 74) & (g == 98) & (b == 139))
    lbl_ds, _ = ndi.label(dark_steel)

    # Jaw light steel: sweeping jaw rim (n=415) + 3 left pegs + 3 bottom-right pegs
    jaw_ls = pick_labels(
        lbl_ls,
        lambda x0, x1, y0, y1, n: (
            (n >= 350)
            or (x0 <= 9 and y0 >= 39)
            or (26 <= x0 <= 45 and 50 <= y0 <= 62 and n >= 30)
        ),
    )

    # 1. `head_upper`:
    head_ls = pick_labels(
        lbl_ls,
        lambda x0, x1, y0, y1, n: (
            (y0 <= 24 and x1 <= 45 and n >= 80)
            or (25 <= x0 <= 30 and 39 <= y0 <= 44 and n <= 15)
            or (20 <= x0 <= 26 and 50 <= y0 <= 53 and n <= 10)
        ),
    )
    head_ms = pick_labels(lbl_ms, lambda x0, x1, y0, y1, n: n >= 150 and x0 <= 25 and y0 <= 22)
    head_top_caps = body_all & (yy <= 38) & (xx >= 10) & (xx <= 46) & (~jaw_ls)
    eye_teeth = (
        body_all
        & (yy >= 35)
        & (yy <= 53)
        & (xx >= 11)
        & (xx <= 36)
        & (
            ((r == 0) & (g == 162) & (b == 232))
            | ((r == 127) & (g == 193) & (b == 243))
            | ((r == 253) & (g == 253) & (b == 253))
            | ((r == 16) & (g == 16) & (b == 16))
        )
    )
    head_seed = head_ls | head_ms | head_top_caps | eye_teeth
    head_dil = ndi.binary_dilation(head_seed, iterations=1) & body_all & (~jaw_ls)
    head_m = (head_seed | head_dil) & (~((xx <= 9) & (yy >= 39))) & (~((xx >= 47) & (yy <= 28)))
    head_m |= body_all & (~jaw_ls) & (xx >= 10) & (xx <= 27) & (yy >= 45) & (yy <= 53)

    # 2. `lower_jaw`:
    is_prism = body_all & (
        ((r == 169) & (g == 189) & (b == 197))
        | ((r == 229) & (g == 244) & (b == 244))
        | ((r == 93) & (g == 116) & (b == 125))
        | ((r == 254) & (g == 202) & (b == 225))
        | ((r == 248) & (g == 245) & (b == 151))
        | ((r == 253) & (g == 253) & (b == 253))
    )
    jaw_ds = pick_labels(
        lbl_ds,
        lambda x0, x1, y0, y1, n: (
            (x0 <= 30 and y0 >= 38)
            or (46 <= x0 <= 50 and 43 <= y0 <= 46 and x1 <= 55 and y1 <= 53)
        ),
    )
    jaw_seed = jaw_ls | jaw_ds
    jaw_dil = ndi.binary_dilation(jaw_seed, iterations=2) & body_all & (~head_m) & (~is_prism)
    jaw_allowed = (
        (xx <= 35)
        | ((xx >= 36) & (xx <= 42) & (yy <= 68))
        | ((xx >= 43) & (xx <= 53) & (yy >= 28) & (yy <= 65) & (~((xx >= 51) & (yy >= 57))))
        | ((xx >= 54) & (xx <= 60) & (yy >= 28) & (yy <= 45) & (xx - yy <= 30))
    )
    jaw_m = (jaw_seed | (jaw_dil & jaw_allowed)) & (~head_m) & (~is_prism)
    jaw_m |= body_all & (~head_m) & (xx <= 34) & (yy >= 38)

    # 3. Remaining multi-jointed serpentine body (`body_upper`, `body_mid`, `body_base`, `tail`):
    rem = body_all & (~head_m) & (~jaw_m)
    tail_m = rem & (
        ((xx >= 82) & (yy >= 63))
        | ((xx >= 77) & (yy >= 77) & (xx + yy >= 159))
        | ((xx >= 78) & (yy >= 86))
    )
    rem2 = rem & (~tail_m)
    body_base_m = rem2 & ((yy >= 77) | ((yy == 76) & (xx >= 68)))
    rem3 = rem2 & (~body_base_m)
    body_mid_m = rem3 & (
        ((yy >= 58) & (xx >= 55))
        | ((yy >= 63) & (xx >= 50))
        | ((yy >= 68) & (xx <= 55))
    )
    body_upper_m = rem3 & (~body_mid_m)

    return {
        "orbit_top": orbit_top,
        "orbit_left_upper": orbit_left_upper,
        "orbit_right_upper": orbit_right_upper,
        "orbit_lower": orbit_lower,
        "head_upper": head_m,
        "lower_jaw": jaw_m,
        "body_upper": body_upper_m,
        "body_mid": body_mid_m,
        "body_base": body_base_m,
        "tail": tail_m,
    }


def segment_back_native(arr96: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments native 96x96 `back.png` (facing right) into matching 10 anatomical masks.
    """
    h, w = arr96.shape[:2]
    a = arr96[:, :, 3] >= 128
    yy, xx = np.mgrid[0:h, 0:w]

    lbl_a, _ = ndi.label(a)
    counts = np.bincount(lbl_a.ravel())
    counts[0] = 0
    body_all = lbl_a == int(np.argmax(counts))
    orbit_all = a & (~body_all)

    orbit_top = orbit_all & (yy <= 15) & (xx <= 50)
    orbit_right_upper = orbit_all & (yy <= 18) & (xx > 50)
    orbit_left_upper = orbit_all & (yy > 15) & (yy <= 40) & (xx <= 40)
    orbit_lower = orbit_all & (yy > 40)

    # Follow the exact black neck-boulder outline `C` between the left cranial horn / head-collar
    # and the first neck boulder (`sf 15`, `x=30..50, y=26..34`):
    head_and_jaw_zone = (
        ((yy <= 25) & (xx >= 31))
        | ((yy == 26) & (((xx >= 32) & (xx <= 37)) | (xx >= 45)))
        | ((yy == 27) & (((xx >= 32) & (xx <= 34)) | (xx >= 47)))
        | ((yy == 28) & (xx >= 48))
        | ((yy >= 29) & (yy <= 30) & (xx >= 49))
        | ((yy >= 31) & (yy <= 32) & (xx >= 50))
        | ((yy >= 33) & (yy <= 34) & (xx >= 51))
        | ((yy >= 35) & (yy <= 48) & (xx >= 52))
    )
    head_and_jaw = body_all & head_and_jaw_zone

    jaw_zone_back = (
        ((yy >= 22) & (xx >= 83))
        | ((yy >= 24) & (xx >= 77))
        | ((yy >= 26) & (xx >= 70))
        | ((yy >= 28) & (xx >= 62))
        | ((yy >= 30) & (xx >= 57))
        | ((yy >= 33) & (xx >= 50))
    )
    jaw_m = head_and_jaw & jaw_zone_back
    head_m = head_and_jaw & (~jaw_m)

    rem = body_all & (~head_and_jaw)
    tail_m = rem & (yy >= 78) & (xx >= 35) & (xx - (89 - yy) >= 28)
    rem2 = rem & (~tail_m)
    body_base_m = rem2 & (((yy >= 75) & (xx >= 28)) | ((yy >= 78) & (xx < 28)))
    rem3 = rem2 & (~body_base_m)
    # Keep the entire middle blue ring (`ring 3` at `x=26..32, y=54..59` and its `y=60` black outline)
    # inside `body_upper`, and start `body_mid` cleanly along the top outline of `sf 34`:
    body_mid_m = rem3 & (
        ((yy >= 58) & (xx <= 26))
        | ((yy >= 60) & (xx == 27))
        | ((yy >= 61) & (xx >= 28) & (xx <= 41))
    )
    body_upper_m = rem3 & (~body_mid_m)

    return {
        "orbit_top": orbit_top,
        "orbit_left_upper": orbit_left_upper,
        "orbit_right_upper": orbit_right_upper,
        "orbit_lower": orbit_lower,
        "head_upper": head_m,
        "lower_jaw": jaw_m,
        "body_upper": body_upper_m,
        "body_mid": body_mid_m,
        "body_base": body_base_m,
        "tail": tail_m,
    }


def build_rig_layers(
    base_96: np.ndarray,
    is_back: bool,
    flip_x: bool,
    cell_w: int,
    cell_h: int,
    foot_y: int,
) -> Dict[str, np.ndarray]:
    """
    Segments native `front.png` or `back.png`, optionally flips horizontally for SE/E/NW,
    places onto the target `(cell_h, cell_w)` canvas with the grounded base aligned to `foot_y`,
    and builds cushioned RGBA layers in strict back-to-front draw order so zero gaps ever appear.
    """
    raw_masks = segment_back_native(base_96) if is_back else segment_front_native(base_96)
    src_96 = base_96[:, ::-1, :].copy() if flip_x else base_96.copy()
    masks_96 = {k: (m[:, ::-1].copy() if flip_x else m.copy()) for k, m in raw_masks.items()}

    a96 = src_96[:, :, 3] >= 128
    ys, xs = np.where(a96)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sh, sw = y1 - y0, x1 - x0
    dst_x0 = (cell_w - sw) // 2
    dst_y0 = foot_y - sh

    canvas = np.zeros((cell_h, cell_w, 4), dtype=np.uint8)
    canvas[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = src_96[y0:y1, x0:x1]

    c_masks: Dict[str, np.ndarray] = {}
    for k, m96 in masks_96.items():
        cm = np.zeros((cell_h, cell_w), dtype=bool)
        cm[dst_y0:dst_y0 + sh, dst_x0:dst_x0 + sw] = m96[y0:y1, x0:x1]
        c_masks[k] = cm

    layers: Dict[str, np.ndarray] = {}
    yy_c, xx_c = np.mgrid[0:cell_h, 0:cell_w]
    nat_y = yy_c - dst_y0 + y0
    if not flip_x:
        nat_x = xx_c - dst_x0 + x0
    else:
        nat_x = x0 + (sw - 1 - (xx_c - dst_x0))

    layers["tail"] = build_part_layer(
        canvas,
        c_masks["tail"],
        c_masks["body_base"] | c_masks["body_mid"],
        cushion_px=6,
    )
    layers["body_base"] = build_part_layer(
        canvas,
        c_masks["body_base"],
        c_masks["body_mid"] | c_masks["body_upper"],
        cushion_px=6,
    )
    layers["body_mid"] = build_part_layer(
        canvas,
        c_masks["body_mid"],
        c_masks["body_upper"] | c_masks["lower_jaw"],
        cushion_px=7,
    )

    if not is_back:
        # On `front.png`, keep the transparent notch at `nat_y <= 21, nat_x = 45..48` between
        # the right cranial horn and the neck boulder completely open:
        fg_over_upper = c_masks["lower_jaw"] | (c_masks["head_upper"] & (nat_y >= 22))
        layers["body_upper"] = build_part_layer(
            canvas,
            c_masks["body_upper"],
            fg_over_upper,
            cushion_px=8,
        )

        # `lower_jaw` gets a 5px upward cushion strictly inside `head_upper`'s mouth & cheek (`nat_x <= 46, nat_y >= 29`).
        # (Before 2026-11-04 the mouth-cavity cushion `nat_x=11..28, nat_y=46..54` was painted dark slate; removed.)
        fg_over_jaw = c_masks["head_upper"] & (nat_x <= 46) & (nat_y >= 29)
        jaw_layer = build_part_layer(
            canvas,
            c_masks["lower_jaw"],
            fg_over_jaw,
            cushion_px=5,
        )
        # (2026-11-04) no dark rectangular mouth pocket any more: the jaw stays put and the opening
        # is carved along the mouth line by `mouth_open`, so the hidden cushion is plain cheek steel.
        jaw_cush_only = (jaw_layer[:, :, 3] >= 128) & (~c_masks["lower_jaw"])
        jaw_layer[jaw_cush_only] = [123, 148, 164, 255]
        layers["lower_jaw"] = jaw_layer

        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            np.zeros_like(c_masks["head_upper"]),
            cushion_px=0,
        )
    else:
        # On `back.png` (viewed from behind), the neck boulder (`body_upper`, with its crisp black
        # collar outline and glowing blue neck ring) is in the foreground overlapping the back of the head!
        layers["lower_jaw"] = build_part_layer(
            canvas,
            c_masks["lower_jaw"],
            c_masks["head_upper"] | c_masks["body_upper"],
            cushion_px=6,
        )
        layers["head_upper"] = build_part_layer(
            canvas,
            c_masks["head_upper"],
            c_masks["body_upper"],
            cushion_px=6,
        )
        layers["body_upper"] = build_part_layer(
            canvas,
            c_masks["body_upper"],
            np.zeros_like(c_masks["body_upper"]),
            cushion_px=0,
        )

    for ok in ("orbit_top", "orbit_left_upper", "orbit_right_upper", "orbit_lower"):
        layers[ok] = build_part_layer(
            canvas,
            c_masks[ok],
            np.zeros_like(c_masks[ok]),
            cushion_px=0,
        )

    if not is_back:
        # Closed mouth line in cell coords (mirrored for the flipped SE/E views).
        if not flip_x:
            layers["_mouth_line"] = [(nx - x0 + dst_x0, ny - y0 + dst_y0) for nx, ny in MOUTH_LINE_NATIVE]
        else:
            layers["_mouth_line"] = [(dst_x0 + (sw - 1 - (nx - x0)), ny - y0 + dst_y0) for nx, ny in MOUTH_LINE_NATIVE]

    return layers


def add_hurt_sparks(canvas: np.ndarray, frame_idx: int) -> np.ndarray:
    if frame_idx not in (2, 3, 4, 5, 6):
        return canvas
    out = canvas.copy()
    a = out[..., 3] >= 128
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return out
    h, w = out.shape[:2]
    top_y = int(ys.min())
    mid_x = int(np.median(xs))
    drift = frame_idx - 2
    pts = [
        (mid_x - 14 - drift, max(2, top_y + 4 - drift // 2)),
        (mid_x + 14 + drift, max(2, top_y + 5 - drift // 2)),
    ]
    for sx, sy in pts:
        if 1 <= sx < w - 3 and 1 <= sy < h - 3:
            out[sy:sy + 2, sx:sx + 2] = (253, 253, 253, 255)
    return out


def synthesize_steelix_mega_frame(
    layers: Dict[str, np.ndarray],
    canvas_shape: Tuple[int, int, int],
    action: str,
    frame_idx: int,
    facing: float,
    is_back: bool,
    palette_rgba: np.ndarray,
) -> np.ndarray:
    t = frame_idx / float(N_FRAMES)
    phase = 2.0 * math.pi * t
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)

    base_dx, base_dy = 0, 0
    mid_dx, mid_dy = 0, 0
    upper_dx, upper_dy = 0, 0
    head_dx, head_dy = 0, 0
    jaw_dx, jaw_dy = 0, 0
    mouth_amt = 0
    tail_dx, tail_dy = 0, 0

    orb_top_dx, orb_top_dy = 0, 0
    orb_lu_dx, orb_lu_dy = 0, 0
    orb_ru_dx, orb_ru_dy = 0, 0
    orb_lo_dx, orb_lo_dy = 0, 0

    if action == "idle":
        # Modeled directly after Normal Steelix's GIF (`0208_steelix/normal/front.gif`):
        # 1. Lowest grounded coil (`body_base`) stays stably planted on the ground `(0, 0)`.
        # 2. Multi-jointed steel body segments (`body_mid` -> `body_upper` -> `head_upper`)
        #    undulate smoothly from bottom to top with progressive phase delay!
        # 3. Massive lower jaw (`lower_jaw`) opens by `+1px` during the breath peak.
        # 4. Armored tail (`tail`) sways rhythmically with phase delay.
        # 5. All 4 floating magnetic crystal clusters (`orbit_*`) bob & orbit with independent phases!
        base_dx, base_dy = 0, 0

        mid_dx = int(round(1.0 * math.sin(phase - 0.3) * facing))
        mid_dy = int(round(1.0 * math.sin(phase - 0.2)))

        upper_dx = int(round(1.4 * math.sin(phase - 0.6) * facing))
        upper_dy = int(round(1.5 * math.sin(phase - 0.5)))

        head_dx = int(round(1.5 * math.sin(phase - 0.8) * facing))
        head_dy = int(round(1.8 * math.sin(phase - 0.7)))

        jaw_open = 1 if math.sin(phase - 0.6) > 0.20 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = jaw_open

        tail_dx = int(round(-1.0 * math.sin(phase - 1.1) * facing))
        tail_dy = int(round(1.6 * math.sin(phase - 1.0)))

        # Independent magnetic orbit motion for the 4 crystal debris clusters
        orb_top_dx = head_dx + int(round(1.5 * math.cos(phase)))
        orb_top_dy = int(round(2.0 * math.sin(phase)))

        orb_lu_dx = head_dx + int(round(1.6 * math.sin(phase + 1.2) * facing))
        orb_lu_dy = int(round(2.2 * math.cos(phase + 1.2)))

        orb_ru_dx = upper_dx + int(round(-1.6 * math.sin(phase + 2.5) * facing))
        orb_ru_dy = int(round(2.0 * math.cos(phase + 2.5)))

        orb_lo_dx = head_dx + int(round(1.5 * math.cos(phase + 3.8) * facing))
        orb_lo_dy = int(round(1.8 * math.sin(phase + 3.8)))

    elif action == "walk":
        # Serpentine travelling-wave slither (`serpent_coil_glider`):
        # Grounded coil base (`body_base`) slides in a horizontal serpentine wave (`dy=0`),
        # while `body_mid`, `body_upper`, `head_upper`, and `tail` propagate the wave upward
        # with smooth phase lag, active jaw champing, and lively orbital crystal motion!
        base_dx = int(round(2.0 * math.sin(phase) * facing))
        base_dy = 0

        mid_dx = int(round(2.2 * math.sin(phase - 0.7) * facing))
        mid_dy = int(round(-1.2 * abs(math.sin(phase - 0.7))))

        upper_dx = int(round(2.2 * math.sin(phase - 1.4) * facing))
        upper_dy = int(round(1.4 * math.sin(2.0 * phase - 0.5)))

        head_dx = int(round(2.4 * math.sin(phase - 1.9) * facing))
        head_dy = int(round(1.6 * math.sin(2.0 * phase - 0.8)))

        jaw_open = 1 if math.sin(2.0 * phase - 0.6) > 0.20 else 0
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = jaw_open

        tail_dx = int(round(2.0 * math.sin(phase + 0.8) * facing))
        tail_dy = int(round(1.8 * math.cos(phase + 0.6)))

        orb_top_dx = head_dx + int(round(2.0 * math.cos(phase) * facing))
        orb_top_dy = head_dy + int(round(2.0 * math.sin(phase)))

        orb_lu_dx = head_dx + int(round(2.2 * math.sin(phase + 1.1) * facing))
        orb_lu_dy = int(round(2.2 * math.cos(phase + 1.1)))

        orb_ru_dx = upper_dx + int(round(-2.0 * math.sin(phase + 2.3) * facing))
        orb_ru_dy = int(round(2.2 * math.cos(phase + 2.3)))

        orb_lo_dx = head_dx + int(round(2.0 * math.cos(phase + 3.6) * facing))
        orb_lo_dy = int(round(2.0 * math.sin(phase + 3.6)))

    elif action == "attack":
        # Signature Mega Steelix "Gyro Ball / Heavy Slam / Iron Tail / Crunch" Lunge!
        # Frames 0..3: Coiled wind-up (body coils pull back `-3..4px`, head rises `-1px`,
        #              lower jaw opens wide `+2px`, and floating crystals converge inward!)
        # Frames 4..9: Ferocious forward slam (`+5..6px` forward lunge, massive jaws snap shut `0px`
        #              on impact frames 6..8, floating magnetic crystals shoot forward `+7..8px`
        #              like railgun shrapnel, and prism tail whips upward!)
        # Frames 10..15: Heavy metallic follow-through recovery to idle stance.
        if frame_idx < 4:
            u = frame_idx / 4.0
            k = -0.65 * math.sin(u * math.pi * 0.5)
        elif frame_idx < 10:
            u = (frame_idx - 4) / 6.0
            k = math.sin(u * math.pi)
        else:
            u = (frame_idx - 10) / 6.0
            k = -0.20 * math.sin(u * math.pi)

        base_dx = int(round(2.0 * max(0.0, k) * facing))
        base_dy = 0

        mid_dx = int(round(3.5 * k * facing))
        mid_dy = int(round(1.0 * abs(k)))

        upper_dx = int(round(4.6 * k * facing))
        upper_dy = int(round(1.2 * k)) if k > 0 else int(round(-1.0 * abs(k)))

        head_dx = int(round(5.8 * k * facing))
        head_dy = int(round(1.6 * k)) if k > 0 else int(round(-1.2 * abs(k)))

        # Wide +2px jaw opening during wind-up & early lunge (frames 1..5),
        # then steel-crushing snap shut (0px) on impact frames 6..9!
        if frame_idx in (2, 3, 4):
            jaw_open = 2
        elif frame_idx in (1, 5):
            jaw_open = 1
        else:
            jaw_open = 0
        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = jaw_open

        tail_dx = int(round(2.8 * k * facing))
        tail_dy = int(round(-2.5 * k))

        # Magnetic crystals contract during wind-up (k < 0) and launch forward on strike (k > 0)
        orb_top_dx = int(round(7.0 * k * facing))
        orb_top_dy = int(round(-1.5 * k))

        orb_lu_dx = int(round(7.8 * k * facing))
        orb_lu_dy = int(round(1.2 * k))

        orb_ru_dx = int(round(6.2 * k * facing))
        orb_ru_dy = int(round(-1.8 * k))

        orb_lo_dx = int(round(7.5 * k * facing))
        orb_lo_dy = int(round(2.0 * k))

    elif action == "hurt":
        # Heavy metallic knockback + hit-shake + jaw jarred open (+1px) + magnetic crystals flung outward
        env = math.sin(math.pi * t)
        jit = (1 if frame_idx % 2 == 0 else -1) if (1 <= frame_idx <= 6) else 0

        base_dx = int(round(-2.0 * env * facing))
        base_dy = 0

        mid_dx = int(round(-3.0 * env * facing)) + jit
        mid_dy = int(round(-0.8 * env))

        upper_dx = int(round(-3.8 * env * facing)) + jit
        upper_dy = int(round(-1.2 * env))

        head_dx = int(round(-4.5 * env * facing)) + jit
        head_dy = int(round(-1.8 * env))

        jaw_dx = head_dx
        jaw_dy = head_dy
        mouth_amt = 1 if 1 <= frame_idx <= 7 else 0

        tail_dx = int(round(-2.5 * env * facing))
        tail_dy = int(round(1.5 * env))

        # Crystals scatter outward on hit impact
        orb_top_dx = head_dx + int(round(-1.5 * env * facing))
        orb_top_dy = int(round(-2.5 * env))

        orb_lu_dx = head_dx + int(round(2.0 * env * facing))
        orb_lu_dy = int(round(-2.0 * env))

        orb_ru_dx = upper_dx + int(round(-2.5 * env * facing))
        orb_ru_dy = int(round(-2.0 * env))

        orb_lo_dx = head_dx + int(round(2.0 * env * facing))
        orb_lo_dy = int(round(2.0 * env))

    if not is_back:
        comp = [
            shift_layer_int(layers["tail"], tail_dx, tail_dy),
            shift_layer_int(layers["body_base"], base_dx, base_dy),
            shift_layer_int(layers["body_mid"], mid_dx, mid_dy),
            shift_layer_int(layers["body_upper"], upper_dx, upper_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
        ]
        # Small mouth opening along the face's mouth line, before the floating crystals are drawn on top.
        head_out = composite_layers(canvas_shape, comp)
        line = layers.get("_mouth_line")
        if mouth_amt > 0 and line:
            pts = mouth_open.transform_points(line, dx=head_dx, dy=head_dy)
            head_out = mouth_open.open_mouth(head_out, pts, amount=mouth_amt, deep=MOUTH_DEEP,
                                             inner=MOUTH_INNER, peak=0.55)
        comp = [
            head_out,
            shift_layer_int(layers["orbit_top"], orb_top_dx, orb_top_dy),
            shift_layer_int(layers["orbit_left_upper"], orb_lu_dx, orb_lu_dy),
            shift_layer_int(layers["orbit_right_upper"], orb_ru_dx, orb_ru_dy),
            shift_layer_int(layers["orbit_lower"], orb_lo_dx, orb_lo_dy),
        ]
    else:
        comp = [
            shift_layer_int(layers["tail"], tail_dx, tail_dy),
            shift_layer_int(layers["body_base"], base_dx, base_dy),
            shift_layer_int(layers["body_mid"], mid_dx, mid_dy),
            shift_layer_int(layers["lower_jaw"], jaw_dx, jaw_dy),
            shift_layer_int(layers["head_upper"], head_dx, head_dy),
            shift_layer_int(layers["body_upper"], upper_dx, upper_dy),
            shift_layer_int(layers["orbit_top"], orb_top_dx, orb_top_dy),
            shift_layer_int(layers["orbit_left_upper"], orb_lu_dx, orb_lu_dy),
            shift_layer_int(layers["orbit_right_upper"], orb_ru_dx, orb_ru_dy),
            shift_layer_int(layers["orbit_lower"], orb_lo_dx, orb_lo_dy),
        ]

    out = composite_layers(canvas_shape, comp)
    out = snap_to_palette(out, palette_rgba)
    if action == "hurt":
        out = add_hurt_sparks(out, frame_idx)
    return out


def save_transparent_gif(frames_rgba: List[np.ndarray], out_path: Path, duration_ms: int = 75) -> None:
    pil_frames: List[Image.Image] = []
    for arr in frames_rgba:
        a = arr[:, :, 3] >= 128
        rgb = arr[:, :, :3]
        uniq = np.unique(rgb[a], axis=0) if np.any(a) else np.zeros((0, 3), dtype=np.uint8)
        pal_flat = [0, 0, 0]
        idx_map = np.zeros(arr.shape[:2], dtype=np.uint8)
        for c_i, col in enumerate(uniq[:254], start=1):
            pal_flat.extend([int(col[0]), int(col[1]), int(col[2])])
            m_c = a & (rgb[:, :, 0] == col[0]) & (rgb[:, :, 1] == col[1]) & (rgb[:, :, 2] == col[2])
            idx_map[m_c] = c_i
        while len(pal_flat) < 768:
            pal_flat.append(0)
        p_img = Image.fromarray(idx_map, mode="P")
        p_img.putpalette(pal_flat)
        p_img.info["transparency"] = 0
        pil_frames.append(p_img)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        disposal=2,
        transparency=0,
        optimize=False,
    )


def build_preview_strip(sheets_by_action: Dict[str, np.ndarray], cell_w: int, cell_h: int) -> Image.Image:
    bg = np.full((8 * cell_h, 16 * cell_w, 4), (15, 20, 31, 255), dtype=np.uint8)
    sample_cols = [0, 4, 8, 12]
    for a_idx, act in enumerate(ACTIONS):
        sheet = sheets_by_action[act]
        for s_i, f_col in enumerate(sample_cols):
            dst_col = a_idx * 4 + s_i
            x_src = f_col * cell_w
            x_dst = dst_col * cell_w
            crop = sheet[:, x_src:x_src + cell_w, :]
            m = crop[:, :, 3] >= 128
            bg[:, x_dst:x_dst + cell_w, :][m] = crop[m]
    return Image.fromarray(bg, "RGBA")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    front_raw = np.array(Image.open(MEGA_DIR / "front.png").convert("RGBA"))
    back_raw = np.array(Image.open(MEGA_DIR / "back.png").convert("RGBA"))

    palette_rgba = extract_palette(front_raw, back_raw)

    # Tight bounds of front (94x90) and back (83x87)
    max_w, max_h = 94, 90
    cell_w = max_w + 22  # 116 (even, generous horizontal margin for +8px crystal lunge)
    cell_h = max_h + 16  # 106 (even)
    foot_y = cell_h - 4  # 102

    dir_layers: Dict[str, Tuple[Dict[str, np.ndarray], float, bool]] = {}
    for d_name in DIRS:
        use_back = d_name in ("N", "NE", "NW")
        src_96 = back_raw if use_back else front_raw
        if not use_back:
            flip_x = d_name in ("SE", "E")
            facing = 1.0 if flip_x else -1.0
        else:
            flip_x = d_name in ("NW",)
            facing = -1.0 if flip_x else 1.0

        layers = build_rig_layers(
            src_96, is_back=use_back, flip_x=flip_x, cell_w=cell_w, cell_h=cell_h, foot_y=foot_y
        )
        dir_layers[d_name] = (layers, facing, use_back)

    sheets_by_action: Dict[str, np.ndarray] = {}
    master_rows: List[np.ndarray] = []

    for act in ACTIONS:
        act_sheet = np.zeros((8 * cell_h, N_FRAMES * cell_w, 4), dtype=np.uint8)
        for d_idx, d_name in enumerate(DIRS):
            layers, facing, use_back = dir_layers[d_name]
            for f_idx in range(N_FRAMES):
                frame_rgba = synthesize_steelix_mega_frame(
                    layers,
                    (cell_h, cell_w, 4),
                    act,
                    f_idx,
                    facing,
                    use_back,
                    palette_rgba,
                )
                y0 = d_idx * cell_h
                x0 = f_idx * cell_w
                act_sheet[y0:y0 + cell_h, x0:x0 + cell_w] = frame_rgba

        sheets_by_action[act] = act_sheet
        act_img = Image.fromarray(act_sheet, "RGBA")
        act_img.save(OUT_DIR / f"{act}.png", optimize=False)
        act_img.save(OUT_DIR / f"{act}_8dir.png", optimize=False)
        Image.fromarray(act_sheet[0:cell_h, :, :], "RGBA").save(OUT_DIR / f"{act}_S_strip.png", optimize=False)
        master_rows.append(act_sheet)

    master_arr = np.concatenate(master_rows, axis=0)
    Image.fromarray(master_arr, "RGBA").save(OUT_DIR / "sheet.png", optimize=False)
    Image.fromarray(master_arr, "RGBA").save(OUT_DIR / "all_8dir_actions.png", optimize=False)

    preview_img = build_preview_strip(sheets_by_action, cell_w, cell_h)
    preview_img.save(OUT_DIR / "preview.png", optimize=False)

    # Export clean 16-frame animated GIFs (`front.gif`, `front-reverse.gif`, `back.gif`, `back-reverse.gif`)
    idle_sheet = sheets_by_action["idle"]
    front_frames = [idle_sheet[0:cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    front_rev_frames = [f[:, ::-1, :].copy() for f in front_frames]
    back_frames = [idle_sheet[4 * cell_h:5 * cell_h, f_i * cell_w:(f_i + 1) * cell_w, :] for f_i in range(N_FRAMES)]
    back_rev_frames = [f[:, ::-1, :].copy() for f in back_frames]

    save_transparent_gif(front_frames, MEGA_DIR / "front.gif", duration_ms=75)
    save_transparent_gif(front_rev_frames, MEGA_DIR / "front-reverse.gif", duration_ms=75)
    save_transparent_gif(back_frames, MEGA_DIR / "back.gif", duration_ms=75)
    save_transparent_gif(back_rev_frames, MEGA_DIR / "back-reverse.gif", duration_ms=75)

    meta = {
        "generator": "forge_steelix_mega_actions",
        "key": "0208_steelix-mega",
        "dex": 208,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "foot_y": foot_y,
        "sprite_h": max_h,
        "frames": N_FRAMES,
        "directions": DIRS,
        "actions": ACTIONS,
        "palette_colors": int(len(palette_rgba)),
        "off_palette_colors": 0,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built 0208_steelix-mega -> {OUT_DIR} (cell={cell_w}x{cell_h}, colors={len(palette_rgba)})")


if __name__ == "__main__":
    main()
