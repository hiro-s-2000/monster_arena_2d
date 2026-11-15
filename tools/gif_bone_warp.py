#!/usr/bin/env python3
"""Animate one still pixel sprite into a front/back GIF by bending it with soft bones.

Same method as tools/build_charizard_mega_x_back_gif.py, made reusable: the motion and the bones live in a
small config file (a Python file), this file is the engine.

    <agent-sprite-forge>/.venv/bin/python -I tools/gif_bone_warp.py <config.py> --grid grid.png
    <agent-sprite-forge>/.venv/bin/python -I tools/gif_bone_warp.py <config.py> --out DIR [--sheet sheet.png --frames 0,5,9 --scale 6 --cols 3]
    <agent-sprite-forge>/.venv/bin/python -I tools/gif_bone_warp.py <config.py> --apply     # writes into monsterSprite + data/sprite_bounds.js

Config file (all coordinates are pixels of the still, x right, y down):

    FORM_REL = "gen1/0006_charizard/mega/charizard-mega-y"   # folder under monsterSprite/
    STILL = "front.png"            # file in that folder (PNG, or a GIF whose frame 0 is used)
    OUT_NAME = "front"             # writes front.gif and front-reverse.gif (mirror)
    N, CYCLE, ROAR0, DUR = 96, 18, 72, 100   # frames, idle-cycle length, first frame of the special section, ms per frame
    FLAME_COLOURS = [(r, g, b), ...]     # colours that only the flames use (flames are cut out by colour)
    OUTLINE_COLOURS = [(r, g, b), ...]   # 1px outline colours (kept from being voted away)
    HIGHLIGHT_COLOURS = [(r, g, b), ...] # tiny highlights such as claws / eye white
    BLUR = 1.6                           # softness of the joints in px
    PARTS = [
        dict(name="wingL", poly=[(x, y), ...], pivot=(x, y)),                     # soft bone
        dict(name="tailflame", poly=[...], pivot=(x, y), parent="tail", flame=True),  # flame layer, rides on its parent
    ]
    def pose(i):   # i = frame number; must return the rest pose (all zero / one) for i == 0
        return dict(lean=px, bob=px, fl=1.0, parts={"wingL": dict(rot=deg, sx=1, sy=1, dx=0, dy=0), ...})

Rules of the engine:
- A soft bone's polygon is a rough region. It may (and should) run out into empty space as far as the part will travel:
  where a moved part lands outside every polygon it is cut off. Polygons listed earlier win where two overlap.
  Pixels in no polygon belong to the body, which only follows lean / bob.
- rot > 0 turns counter-clockwise on screen: it lowers what is left of the pivot and lifts what is right of it (y grows downward).
- lean > 0 pushes the upper body to the right, bob < 0 lifts it; the feet stay planted.
- A flame part takes only FLAME_COLOURS pixels inside its polygon, moves rigidly with its parent, flickers by itself
  (pose "fl" scales its height) and is drawn on top.
- Frames CYCLE..ROAR0-1 are exact copies of frames 0..CYCLE-1 (like the Black/White GIFs). Frame 0 is the still, untouched.
"""
import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label, map_coordinates

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = ROOT / "data" / "sprite_bounds.js"
PAD = 24
SS = 3


def load_cfg(path):
    spec = importlib.util.spec_from_file_location("gifcfg", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_still(cfg):
    im = Image.open(ROOT / "monsterSprite" / cfg.FORM_REL / cfg.STILL)
    im.seek(getattr(cfg, "FRAME", 0))
    return np.array(im.convert("RGBA"))


def affine(pivot, rot=0.0, sx=1.0, sy=1.0, dx=0.0, dy=0.0):
    px, py = pivot[0] + 0.5, pivot[1] + 0.5
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    T1 = np.array([[1, 0, -px], [0, 1, -py], [0, 0, 1.0]])
    Sc = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    T2 = np.array([[1, 0, px + dx], [0, 1, py + dy], [0, 0, 1.0]])
    return T2 @ R @ Sc @ T1


def index_still(rgba):
    H, W = rgba.shape[:2]
    cols = [(0, 0, 0, 0)]
    idx = np.zeros((H, W), np.int32)
    for y in range(H):
        for x in range(W):
            c = tuple(int(v) for v in rgba[y, x])
            if c[3] < 128:
                continue
            c = c[:3] + (255,)
            if c not in cols:
                cols.append(c)
            idx[y, x] = cols.index(c)
    return idx, cols


def build(cfg):
    idx, cols = index_still(read_still(cfg))
    H, W = idx.shape
    N, CYCLE, ROAR0 = cfg.N, cfg.CYCLE, cfg.ROAR0
    flame_idx = [k for k, c in enumerate(cols) if c[:3] in [tuple(v) for v in getattr(cfg, "FLAME_COLOURS", [])]]
    foot_y = int(np.nonzero(idx.any(1))[0].max())
    body_h = foot_y - int(np.nonzero(idx.any(1))[0].min()) + 1
    OW, OH = W + 2 * PAD, H + 2 * PAD
    parts = cfg.PARTS
    bones = [p for p in parts if not p.get("flame")]
    names = [p["name"] for p in bones]
    flame_img = {}
    body_idx = idx.copy()
    for p in parts:
        if not p.get("flame"):
            continue
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(p["poly"], fill=1)
        m = (np.array(m) > 0) & np.isin(idx, flame_idx)
        flame_img[p["name"]] = np.where(m, idx, 0)
        body_idx[m] = 0
    taken = np.zeros((OH, OW), bool)
    hard = {}
    for p in bones:
        m = Image.new("L", (OW, OH), 0)
        ImageDraw.Draw(m).polygon([(x + PAD, y + PAD) for x, y in p["poly"]], fill=1)
        hard[p["name"]] = (np.array(m) > 0) & ~taken
        taken |= hard[p["name"]]
    blur = getattr(cfg, "BLUR", 1.6)
    wts = np.stack([gaussian_filter(hard[n].astype(np.float64), blur) for n in names]) if names else np.zeros((0, OH, OW))
    wts = wts / np.maximum(wts.sum(0), 1.0) if names else wts

    sub = (np.arange(SS) + 0.5) / SS
    sy_, sx_ = np.meshgrid(sub, sub, indexing="ij")
    oy = np.arange(OH)[:, None, None] + sy_.reshape(1, 1, -1) - PAD + np.zeros((1, OW, 1))
    ox = np.arange(OW)[None, :, None] + sx_.reshape(1, 1, -1) - PAD + np.zeros((OH, 1, 1))

    def weights_at(sx, sy):
        co = np.stack([(sy + PAD - 0.5).ravel(), (sx + PAD - 0.5).ravel()])
        return [map_coordinates(wts[k], co, order=1, mode="nearest").reshape(sx.shape) for k in range(len(names))]

    def sample(img, sx, sy):
        xi, yi = np.floor(sx).astype(int), np.floor(sy).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out = np.zeros(sx.shape, np.int32)
        out[ok] = img[yi[ok], xi[ok]]
        return out

    vote_w = np.ones(len(cols))
    for k, c in enumerate(cols):
        if c[:3] in [tuple(v) for v in getattr(cfg, "OUTLINE_COLOURS", [])]:
            vote_w[k] = 1.7
        elif c[:3] in [tuple(v) for v in getattr(cfg, "HIGHLIGHT_COLOURS", [])]:
            vote_w[k] = 2.5
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    # see-through spots the still itself has stay see-through (only new pinholes get closed)
    lab0, n0 = label(idx == 0, structure=cross)
    size0 = np.bincount(lab0.ravel())
    edge0 = set(np.unique(np.concatenate([lab0[0], lab0[-1], lab0[:, 0], lab0[:, -1]])))
    still_small = sum(1 for k in range(1, n0 + 1) if k not in edge0 and size0[k] <= 10)

    def close_holes(f):
        if still_small:
            return f
        lab, n = label(f == 0, structure=cross)
        if n <= 1:
            return f
        size = np.bincount(lab.ravel())
        edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        todo = [k for k in range(1, n + 1) if k not in edge and 0 < size[k] <= 10]
        if not todo:
            return f
        f = f.copy()
        for _ in range(4):
            hole = np.isin(lab, todo) & (f == 0)
            if not hole.any():
                break
            best = np.zeros(f.shape, np.int32)
            bestn = np.zeros(f.shape, np.int32)
            for c in range(1, len(cols)):
                m = (f == c).astype(np.int32)
                cn = np.zeros(f.shape, np.int32)
                cn[1:] += m[:-1]; cn[:-1] += m[1:]; cn[:, 1:] += m[:, :-1]; cn[:, :-1] += m[:, 1:]
                up = cn > bestn
                best[up] = c
                bestn[up] = cn[up]
            fill = hole & (bestn > 0)
            f[fill] = best[fill]
        return f

    frames = []
    for i in range(N):
        if CYCLE <= i < ROAR0:
            frames.append(frames[i % CYCLE].copy())
            continue
        P = cfg.pose(i)
        pp = P.get("parts", {})
        k_up = np.clip((foot_y - body_h * 0.1 - oy) / (body_h * 0.42), 0, 1)
        bx = ox - P.get("lean", 0) * np.clip((foot_y - oy) / (body_h * 0.86), 0, 1.2)
        by = oy - P.get("bob", 0) * k_up
        wob = math.sin(2 * math.pi * i / 6)
        mats = {}
        for p in parts:
            a = dict(pp.get(p["name"], {}))
            if p.get("flame"):
                grow = P.get("fl", 1.0) * a.pop("grow", 1.0) + 0.07 * math.sin(2 * math.pi * i / 9)
                M = affine(p["pivot"], a.get("rot", 0) + 5 * wob, a.get("sx", 1) * (1 + (grow - 1) * 0.5), a.get("sy", 1) * grow, a.get("dx", 0), a.get("dy", 0))
            else:
                M = affine(p["pivot"], a.get("rot", 0), a.get("sx", 1), a.get("sy", 1), a.get("dx", 0), a.get("dy", 0))
            if p.get("parent"):
                M = mats[p["parent"]] @ M
            mats[p["name"]] = M
        sx, sy = bx.copy(), by.copy()
        if i and names:
            for _ in range(8):
                w = weights_at(sx, sy)
                dx = np.zeros_like(sx)
                dy = np.zeros_like(sy)
                for k, n in enumerate(names):
                    M = mats[n]
                    dx += w[k] * (M[0, 0] * sx + M[0, 1] * sy + M[0, 2] - sx)
                    dy += w[k] * (M[1, 0] * sx + M[1, 1] * sy + M[1, 2] - sy)
                sx, sy = bx - dx, by - dy
        acc = sample(body_idx, sx, sy)
        for p in parts:
            if not p.get("flame"):
                continue
            inv = np.linalg.inv(mats[p["name"]])
            fx = inv[0, 0] * bx + inv[0, 1] * by + inv[0, 2]
            fy = inv[1, 0] * bx + inv[1, 1] * by + inv[1, 2]
            if i:
                tip = np.clip((p["pivot"][1] - fy) / 20.0, 0, 1)
                fx = fx + 1.4 * wob * tip * np.sin(fy / 4.0 + 2 * math.pi * i / 9)
            got = sample(flame_img[p["name"]], fx, fy)
            under = p.get("under")  # under=True: the flame shows only where the body is empty (flame behind the body)
            acc = np.where((got > 0) & ((acc == 0) if under else True), got, acc)
        cnt = np.zeros((OH, OW, len(cols)), np.float64)
        for k in range(len(cols)):
            cnt[..., k] = (acc == k).sum(-1)
        f = (cnt * vote_w).argmax(-1).astype(np.uint8)
        if i:
            f = close_holes(f)
            lab, n = label(f > 0, structure=np.ones((3, 3)))
            if n > 1:
                size = np.bincount(lab.ravel())
                for k in range(1, n + 1):
                    if size[k] < 12 and not np.isin(f[lab == k], flame_idx).all():  # loose flame sparks are part of the art
                        f[lab == k] = 0
        frames.append(f)

    f0 = frames[0][PAD:PAD + H, PAD:PAD + W]
    if not (f0 == idx).all():
        raise SystemExit(f"frame 0 differs from the still: {(f0 != idx).sum()} px (pose(0) must be the rest pose)")
    any_ = np.zeros((OH, OW), bool)
    for f in frames:
        any_ |= f > 0
    ys, xs = np.nonzero(any_)
    x0, x1, y0, y1 = xs.min() - 1, xs.max() + 2, ys.min() - 1, ys.max() + 2
    if x0 < 0 or y0 < 0 or x1 > OW or y1 > OH:
        raise SystemExit("motion leaves the padded canvas; reduce the motion")
    return [f[y0:y1, x0:x1] for f in frames], cols


def palette(cols):
    pal = []
    for c in cols:
        pal += list(c[:3])
    return pal + [0] * (768 - len(pal))


def save_gif(frames, cols, path, dur, flip=False):
    ims = []
    for f in frames:
        im = Image.fromarray(np.ascontiguousarray(f[:, ::-1] if flip else f), "P")
        im.putpalette(palette(cols))
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=dur, loop=0, transparency=0, disposal=2, optimize=False)


def rgba_of(f, cols):
    lut = np.array(cols, np.uint8)
    return Image.fromarray(lut[f], "RGBA")


def sheet(frames, cols, path, which, scale, ncols):
    h, w = frames[0].shape
    rows = (len(which) + ncols - 1) // ncols
    sh = Image.new("RGBA", (ncols * w * scale, rows * h * scale), (40, 44, 70, 255))
    d = ImageDraw.Draw(sh)
    for j, i in enumerate(which):
        x, y = (j % ncols) * w * scale, (j // ncols) * h * scale
        sh.alpha_composite(rgba_of(frames[i], cols).resize((w * scale, h * scale), Image.NEAREST), (x, y))
        d.text((x + 4, y + 4), str(i), fill=(255, 255, 0))
    sh.save(path)


def grid(cfg, path, k=10):
    idx, cols = index_still(read_still(cfg))
    H, W = idx.shape
    sh = Image.new("RGBA", (W * k + 30, H * k + 20), (60, 64, 90, 255))
    sh.alpha_composite(rgba_of(idx.astype(np.uint8), cols).resize((W * k, H * k), Image.NEAREST), (30, 20))
    d = ImageDraw.Draw(sh)
    for x in range(0, W + 1, 10):
        d.line([(30 + x * k, 20), (30 + x * k, 20 + H * k)], fill=(255, 255, 0, 255) if x % 20 == 0 else (140, 140, 60, 255))
        d.text((32 + x * k, 4), str(x), fill=(255, 255, 0))
    for y in range(0, H + 1, 10):
        d.line([(30, 20 + y * k), (30 + W * k, 20 + y * k)], fill=(255, 255, 0, 255) if y % 20 == 0 else (140, 140, 60, 255))
        d.text((4, 22 + y * k), str(y), fill=(255, 255, 0))
    tint = [(255, 0, 255), (0, 255, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255), (255, 0, 0), (120, 160, 255), (255, 200, 120)]
    for j, p in enumerate(getattr(cfg, "PARTS", [])):
        pts = [(30 + x * k, 20 + y * k) for x, y in p["poly"]]
        d.line(pts + [pts[0]], fill=tint[j % len(tint)] + (255,), width=2)
        px, py = 30 + (p["pivot"][0] + 0.5) * k, 20 + (p["pivot"][1] + 0.5) * k
        d.ellipse([px - 5, py - 5, px + 5, py + 5], outline=tint[j % len(tint)] + (255,), width=2)
        d.text((px + 7, py - 5), p["name"], fill=tint[j % len(tint)] + (255,))
    sh.save(path)
    print("colours:", [(c[:3], int((idx == i).sum())) for i, c in enumerate(cols) if i])


def bounds_of(path):
    im = Image.open(path)
    im.seek(0)
    a = np.array(im.convert("RGBA"))[..., 3] > 0
    ys, xs = np.nonzero(a)
    return [im.size[0], im.size[1], int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def publish_bounds(entries):
    text = BOUNDS.read_text()
    for rel, val in entries.items():
        pat = re.compile(re.escape(json.dumps(rel)) + r":\[[^\]]*\]")
        if len(pat.findall(text)) != 1:
            raise SystemExit(f"bounds entry not found exactly once: {rel}")
        text = pat.sub(lambda m: json.dumps(rel) + ":[" + ",".join(str(v) for v in val) + "]", text)
    BOUNDS.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--grid", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--sheet", default="")
    ap.add_argument("--frames", default="")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.grid:
        grid(cfg, a.grid)
        return
    frames, cols = build(cfg)
    print("frames", len(frames), "canvas", frames[0].shape[::-1], "colours", len(cols) - 1)
    if a.sheet:
        which = [int(v) for v in a.frames.split(",")] if a.frames else list(range(cfg.CYCLE)) + list(range(cfg.ROAR0, cfg.N))
        sheet(frames, cols, a.sheet, which, a.scale, a.cols)
    out = ROOT / "monsterSprite" / cfg.FORM_REL if a.apply else (Path(a.out) if a.out else None)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        names = [cfg.OUT_NAME + ".gif", cfg.OUT_NAME + "-reverse.gif"]
        save_gif(frames, cols, out / names[0], cfg.DUR)
        save_gif(frames, cols, out / names[1], cfg.DUR, flip=True)
        info = {n: bounds_of(out / n) for n in names}
        print(json.dumps(info))
        if a.apply:
            publish_bounds({f"{cfg.FORM_REL}/{n}": v for n, v in info.items()})
            print("bounds updated")


if __name__ == "__main__":
    main()
