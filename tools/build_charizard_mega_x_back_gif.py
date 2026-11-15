#!/usr/bin/env python3
"""Mega Charizard X: rebuild back.gif / back-reverse.gif so they move like front.gif.

[ユーザー指定 2026-11-15] 「back.gif と back-reverse.gif の動きを front.gif / front-reverse.gif にかなり近づける」。

front.gif = 96 コマ・100ms。18 コマの羽ばたきを 4 回（0〜71）、そのあと 24 コマの咆哮（のけぞる → 前へ踏み込む → 戻る）。
旧 back.gif = 20 コマ・60ms（口から橙の火と火花が出る型）。

ここでは旧 back.gif の 1 コマ目の絵だけを素材に、部位（翼・頭・尾・腕・炎）ごとの動きを
なめらかにつないで（関節でちぎれないよう重みをぼかす）、front と同じ時間割で動かす。
新しい絵は描かない。色は元の 12 色のまま。1 コマ目は元の 1 コマ目と同じ絵。

    <agent-sprite-forge>/.venv/bin/python -I tools/build_charizard_mega_x_back_gif.py [--src <old back.gif>] [--out <dir>] [--apply]

--apply なしでは --out（既定: 作業用フォルダ）へ書くだけ。--apply で monsterSprite と data/sprite_bounds.js を書き換える。
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REL = "gen1/0006_charizard/mega/charizard-mega-x"
FORM = ROOT / "monsterSprite" / REL
BOUNDS = ROOT / "data" / "sprite_bounds.js"

N, CYCLE, ROAR0, DUR = 96, 18, 72, 100  # same as front.gif
PAD = 20
SS = 3  # sub-samples per pixel side (majority vote keeps hard pixel edges)
FLAME = [(0, 157, 225), (96, 166, 200), (153, 217, 247), (0, 177, 230)]

# Bones of the back pose (old back.gif frame 0, 108x93). Each polygon is a rough region (it may run into empty space);
# the regions are blurred into soft weights, so the sprite bends at the joints instead of tearing into pieces.
BLUR = 1.6
PARTS = [
    dict(name="wingR", poly=[(80, 12), (88, -20), (128, -20), (128, 34), (101, 40), (100, 48), (89, 49), (85, 45), (83, 36), (84, 26)], pivot=(85, 40)),
    dict(name="tail", poly=[(-20, 56), (24, 60), (42, 63), (45, 70), (45, 86), (30, 86), (8, 82), (-20, 76)], pivot=(46, 74)),
    dict(name="tailflame", poly=[(0, 30), (26, 30), (26, 65.5), (12, 65.5), (9, 68), (0, 68)], pivot=(14, 65), parent="tail", flame=True),
    dict(name="arm", poly=[(74, 62), (88, 62), (88, 75), (74, 74)], pivot=(73, 68)),
    dict(name="head", poly=[(60, -20), (86, -20), (86, 13), (84, 24), (82, 34), (80, 43), (72, 46), (64, 44), (60, 38), (61, 30)], pivot=(68, 46)),
    dict(name="headflame", poly=[(52, 8), (66, 8), (70, 24), (77, 30), (77, 38), (66, 38), (52, 30)], pivot=(72, 34), parent="head", flame=True),
    dict(name="wingL", poly=[(-20, -20), (50, -20), (53, 10), (56, 19), (58, 28), (54, 36), (55, 44), (50, 50), (46, 55), (36, 60), (26, 60), (14, 50), (-20, 50)], pivot=(56, 29)),
]


def read_frame0(path):
    im = Image.open(path)
    im.seek(0)
    return np.array(im.convert("RGBA"))


def smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def keys(u, table):
    """table: [(u, {param: value})] sorted by u. Smooth interpolation."""
    for (u0, a), (u1, b) in zip(table, table[1:]):
        if u0 <= u <= u1:
            k = smooth((u - u0) / (u1 - u0))
            return {p: a[p] + (b[p] - a[p]) * k for p in a}
    return dict(table[-1][1])


REST = dict(lean=0, bob=0, wl=0, wr=0, wsx=1, head=0, hdx=0, hdy=0, tail=0, arm=0, fl=1)
def K(f, **kw):
    return ((f - ROAR0) / (N - ROAR0), dict(REST, **kw))


# front.gif の咆哮と同じ割り振り: 73〜79 後ろへためる → 80〜84 前へ踏み込む → 85〜93 頭と体を2回ふるう → 94〜95 戻る
ROAR = [
    K(72),
    K(76, lean=-5, bob=1, wl=8, wr=-7, wsx=0.96, head=-5, tail=-5, arm=-3),
    K(79, lean=-7, bob=1.5, wl=14, wr=-12, wsx=0.92, head=-8, hdy=-1, tail=-8, arm=-6, fl=1.1),
    K(84, lean=14, bob=-2, wl=-10, wr=10, wsx=1.0, head=5, hdx=2, tail=9, arm=12, fl=1.25),
    K(86, lean=11, bob=-1, wl=-7, wr=7, wsx=1.0, head=4, hdx=2, tail=6, arm=9, fl=1.2),
    K(93, lean=4, bob=0, wl=-2, wr=2, wsx=1.0, head=1, hdx=1, tail=2, arm=3, fl=1.05),
    K(96),
]


def pose(i):
    if i < ROAR0:
        p = (i % CYCLE) / CYCLE
        s = (1 - math.cos(2 * math.pi * p)) / 2  # 0 → 1 → 0: like the front, the body leans forward and up, wings swing, peak at 8〜9
        q = math.sin(2 * math.pi * p)
        return dict(lean=8.0 * s, bob=-2.5 * s, wl=22 * s, wr=-18 * s, wsx=1 - 0.2 * s, head=-4 * s + 1.5 * q, hdx=0, hdy=-1 * s,
                    tail=7 * math.sin(2 * math.pi * p - 0.9) + 7 * math.sin(0.9), arm=6 * s, fl=1 + 0.06 * s)
    P = keys((i - ROAR0) / (N - ROAR0), ROAR)
    if 85 <= i <= 93:  # two-beat shake after the roar
        env = math.sin(math.pi * (i - 84.5) / 9.0)
        w = math.sin(2 * math.pi * (i - 85) / 4.0 + 0.6)
        P["head"] += 6 * env * w
        P["lean"] += 2.5 * env * w
        P["bob"] += 1.2 * env * w
        P["wl"] += 4 * env * w
        P["wr"] -= 4 * env * w
    return P


def affine(pivot, deg=0.0, sx=1.0, sy=1.0, dx=0.0, dy=0.0):
    px, py = pivot[0] + 0.5, pivot[1] + 0.5
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    T1 = np.array([[1, 0, -px], [0, 1, -py], [0, 0, 1.0]])
    Sc = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])
    # deg > 0 lifts what is left of the pivot (y grows downward)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    T2 = np.array([[1, 0, px + dx], [0, 1, py + dy], [0, 0, 1.0]])
    return T2 @ R @ Sc @ T1


def build(src):
    rgba = read_frame0(src)
    H, W = rgba.shape[:2]
    cols = [(0, 0, 0, 0)]
    idx = np.zeros((H, W), np.int32)
    for y in range(H):
        for x in range(W):
            c = tuple(int(v) for v in rgba[y, x])
            if c[3] == 0:
                continue
            if c not in cols:
                cols.append(c)
            idx[y, x] = cols.index(c)
    flame_idx = [k for k, c in enumerate(cols) if c[:3] in FLAME]
    foot_y = int(np.nonzero(idx.any(1))[0].max())

    from scipy.ndimage import gaussian_filter, binary_dilation, map_coordinates, label

    OW, OH = W + 2 * PAD, H + 2 * PAD
    big = np.zeros((OH, OW), np.int32)
    big[PAD:PAD + H, PAD:PAD + W] = idx
    # Flames are their own layers (cut out by colour, moved rigidly, drawn on top): blended into the body weights
    # they got dragged along by the wing next to them. Everything else bends with soft weights.
    bones = [p for p in PARTS if not p.get("flame")]
    names = [p["name"] for p in bones]
    flame_img = {}
    body_idx = idx.copy()
    for p in PARTS:
        if not p.get("flame"):
            continue
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(p["poly"], fill=1)
        m = (np.array(m) > 0) & np.isin(idx, flame_idx)
        flame_img[p["name"]] = np.where(m, idx, 0)
        body_idx[m] = 0
    hard = {}
    taken = np.zeros((OH, OW), bool)
    for p in bones:
        m = Image.new("L", (OW, OH), 0)
        ImageDraw.Draw(m).polygon([(x + PAD, y + PAD) for x, y in p["poly"]], fill=1)
        hard[p["name"]] = (np.array(m) > 0) & ~taken
        taken |= hard[p["name"]]
    wts = np.stack([gaussian_filter(hard[n].astype(np.float64), BLUR) for n in names])
    tot = wts.sum(0)
    wts = wts / np.maximum(tot, 1.0)  # what is left (1 - sum) belongs to the body, which only follows the whole-body warp

    sub = (np.arange(SS) + 0.5) / SS
    sy_, sx_ = np.meshgrid(sub, sub, indexing="ij")
    oy = np.arange(OH)[:, None, None] + sy_.reshape(1, 1, -1) - PAD + np.zeros((1, OW, 1))
    ox = np.arange(OW)[None, :, None] + sx_.reshape(1, 1, -1) - PAD + np.zeros((OH, 1, 1))

    def weights_at(sx, sy):  # bilinear, coordinates in sprite pixels (pixel centre = +0.5)
        co = np.stack([(sy + PAD - 0.5).ravel(), (sx + PAD - 0.5).ravel()])
        return [map_coordinates(wts[k], co, order=1, mode="nearest").reshape(sx.shape) for k in range(len(names))]

    def sample(img, sx, sy):
        xi, yi = np.floor(sx).astype(int), np.floor(sy).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out = np.zeros(sx.shape, np.int32)
        out[ok] = img[yi[ok], xi[ok]]
        return out

    # the outline and the claw highlight are one pixel wide: give them extra say in the vote so they do not get voted away
    vote_w = np.ones(len(cols))
    for k, c in enumerate(cols):
        if c[:3] in ((0, 0, 0), (31, 31, 31)):
            vote_w[k] = 1.7
        elif c[:3] == (248, 248, 248):
            vote_w[k] = 2.5
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)

    def close_holes(f):
        """Fill pinholes: small see-through spots enclosed by the sprite (the still has none this small)."""
        lab, n = label(f == 0, structure=cross)
        if n <= 1:
            return f
        size = np.bincount(lab.ravel())
        edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        for _ in range(3):
            todo = [k for k in range(1, n + 1) if k not in edge and 0 < size[k] <= 10]
            if not todo:
                break
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
            f = f.copy()
            f[fill] = best[fill]
        return f

    frames = []
    for i in range(N):
        if CYCLE <= i < ROAR0:  # like the front: the flap cycle repeats as exact copies
            frames.append(frames[i % CYCLE].copy())
            continue
        P = pose(i)
        # whole-body warp: feet stay planted, the upper body leans / rises
        k_up = np.clip((foot_y - 8 - oy) / 34.0, 0, 1)
        bx = ox - P["lean"] * np.clip((foot_y - oy) / 70.0, 0, 1.2)
        by = oy - P["bob"] * k_up
        wob = math.sin(2 * math.pi * i / 6)  # zero on frames 0, 3, 6 … so the loop start is the still
        mats = {}
        for p in PARTS:
            n = p["name"]
            if n == "wingL":
                M = affine(p["pivot"], P["wl"], P["wsx"], 1)
            elif n == "wingR":
                M = affine(p["pivot"], P["wr"], P["wsx"], 1)
            elif n == "head":
                M = affine(p["pivot"], P["head"], 1, 1, P["hdx"], P["hdy"])
            elif n == "tail":
                M = affine(p["pivot"], P["tail"])
            elif n == "arm":
                M = affine(p["pivot"], P["arm"])
            else:  # flames: grow from the base, lean with the flicker
                grow = P["fl"] + 0.07 * math.sin(2 * math.pi * i / 9)
                M = affine(p["pivot"], 5 * wob, 1 + (grow - 1) * 0.5, grow)
            if p.get("parent"):
                M = mats[p["parent"]] @ M
            mats[n] = M
        # forward displacement D(s) = sum_k w_k(s) (M_k s - s); solve s = o - D(s) by fixed-point iteration
        sx, sy = bx.copy(), by.copy()
        if i:
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
        for p in PARTS:
            if not p.get("flame"):
                continue
            inv = np.linalg.inv(mats[p["name"]])
            fx = inv[0, 0] * bx + inv[0, 1] * by + inv[0, 2]
            fy = inv[1, 0] * bx + inv[1, 1] * by + inv[1, 2]
            if i:  # travelling ripple, stronger toward the tip
                tip = np.clip((p["pivot"][1] - fy) / 20.0, 0, 1)
                fx = fx + 1.4 * wob * tip * np.sin(fy / 4.0 + 2 * math.pi * i / 9)
            got = sample(flame_img[p["name"]], fx, fy)
            acc = np.where(got > 0, got, acc)
        # majority vote per pixel
        cnt = np.zeros((OH, OW, len(cols)), np.float64)
        for k in range(len(cols)):
            cnt[..., k] = (acc == k).sum(-1)
        f = (cnt * vote_w).argmax(-1).astype(np.uint8)
        if i:
            f = close_holes(f)
        if i:  # drop stray specks that came loose from the body
            lab, n = label(f > 0, structure=np.ones((3, 3)))
            if n > 1:
                size = np.bincount(lab.ravel())
                for k in range(1, n + 1):
                    if size[k] < 12:
                        f[lab == k] = 0
        frames.append(f)

    # frame 0 must be the original still
    f0 = frames[0][PAD:PAD + H, PAD:PAD + W]
    if not (f0 == idx).all():
        raise SystemExit(f"frame 0 differs from the still: {(f0 != idx).sum()} px")

    # crop: keep at least the original canvas, grow only as far as the motion needs (+1px)
    any_ = np.zeros((OH, OW), bool)
    for f in frames:
        any_ |= f > 0
    ys, xs = np.nonzero(any_)
    x0, x1 = min(PAD, xs.min() - 1), max(PAD + W, xs.max() + 2)
    y0, y1 = min(PAD, ys.min() - 1), max(PAD + H, ys.max() + 2)
    if x0 < 0 or y0 < 0 or x1 > OW or y1 > OH:
        raise SystemExit("motion leaves the padded canvas; raise PAD")
    frames = [f[y0:y1, x0:x1] for f in frames]
    return frames, cols, (PAD - x0, PAD - y0)


def save_gif(frames, cols, path, flip=False):
    pal = []
    for c in cols:
        pal += list(c[:3])
    pal += [0] * (768 - len(pal))
    ims = []
    for f in frames:
        a = f[:, ::-1] if flip else f
        im = Image.fromarray(np.ascontiguousarray(a), "P")
        im.putpalette(pal)
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=DUR, loop=0, transparency=0, disposal=2, optimize=False)


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
    ap.add_argument("--src", default=str(FORM / "back.gif"), help="old back.gif (frame 0 is the still)")
    ap.add_argument("--out", default="")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    frames, cols, off = build(a.src)
    out = FORM if a.apply else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_gif(frames, cols, out / "back.gif")
    save_gif(frames, cols, out / "back-reverse.gif", flip=True)
    info = {n: bounds_of(out / n) for n in ("back.gif", "back-reverse.gif")}
    print("frames", len(frames), "size", frames[0].shape[::-1], "still offset", off, "colours", len(cols) - 1)
    print(json.dumps(info))
    if a.apply:
        publish_bounds({f"{REL}/{n}": v for n, v in info.items()})
        print("bounds updated")


if __name__ == "__main__":
    main()
