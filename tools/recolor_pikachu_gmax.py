#!/usr/bin/env python3
"""キョダイマックスピカチュウの GIF を、同じフォルダの front.png / back.png の色合いと光沢に塗り替える。
（2026-11-15、ユーザー指示「キョダイマックスのピカチュウは front.png と back.png の色合いや光沢加減を採用」）

元の GIF（暗い茶色の体・ピンクの縁取り）は ../_backup_pikachu_gmax_recolor_20261115/ に置いたものを毎回読む（何度実行しても同じ結果）。
形と動き（輪郭・コマ数・表示時間）は変えない。変えるのは色だけ。使う色は front.png / back.png にある色だけ。
  体      : 元の帯状の明暗（上が暗く下が明るい）は使わず、PNG と同じ「左上から光が当たる」塗り分けを輪郭から作る。
            腕の下などの局所的な影と線は元の絵から拾う。
  ふち    : ピンクの縁取り → PNG と同じ橙〜茶のふち線。
  ほお    : くすんだ赤 → PNG の赤。
  しっぽ  : 白のまま。ふちを PNG と同じ黄色の光にする。
  耳の先・目: 黒〜灰。
  背中のしま（back）: 黒い線で囲まれた部分 → PNG の茶色。
-reverse は塗り替えたあとの左右反転。

使い方（向きごとに 1 回ずつ独立に実行する）:
  <agent-sprite-forge>/.venv/bin/python tools/recolor_pikachu_gmax.py --view front [--apply]
  <agent-sprite-forge>/.venv/bin/python tools/recolor_pikachu_gmax.py --view back  [--apply]
--apply なしのときは --out（既定 /tmp ではなく指定必須）に書くだけ。
このあと tools/build_dir8_from_gifs.py --gen gen1 --only 0025_pikachu-gmax --apply で 8 方向シートを作り直す。
"""
import argparse, colorsys
from pathlib import Path
import numpy as np
from PIL import Image, ImageSequence
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
FORM = ROOT / "monsterSprite/gen1/0025_pikachu/gigantamax/pikachu-gmax"
SRC = ROOT.parent / "_backup_pikachu_gmax_recolor_20261115"

# front.png / back.png の色
OUT_D, OUT_M, SH2, SH1, BASE, HL = (98, 49, 8), (156, 82, 0), (222, 148, 0), (246, 189, 32), (246, 230, 82), (255, 246, 164)
RAMP = [SH2, SH1, BASE, HL]
GLOW, WHITE = (248, 252, 75), (254, 254, 254)
K0, K1, K2 = (16, 16, 16), (41, 41, 41), (65, 65, 74)
RED, RED_L = (197, 32, 24), (230, 90, 65)

def classify(rgb):
    """元 GIF の 1 色 → 種類。"""
    r, g, b = [v / 255 for v in rgb]; h, s, v = colorsys.rgb_to_hsv(r, g, b); h *= 360
    if v > 0.95 and s < 0.08: return "white"
    if v < 0.12: return "k0"
    if s < 0.13: return "grey_l" if v > 0.6 else ("grey_m" if v > 0.4 else "k1")
    if 36 <= h <= 62 and v > 0.7 and s < 0.5:                         # しっぽの光（明るい順に 白 / うすい黄 / 黄）
        return "pale_w" if (s < 0.3 and v > 0.9) else ("pale" if v > 0.9 else "pale_d")
    if s < 0.25: return "grey_l" if v > 0.6 else "grey_m"
    if (h >= 300 or h < 5) and v < 0.5 and s < 0.6: return "fill"     # 背中上部の暗い小豆色（帯）
    if h >= 300 or h < 5: return "rose" if (v < 0.7 and h >= 340) else "rim"
    if h < 13 and v > 0.8: return "rim_lo"                            # 下からの赤い照り返し
    if v < 0.32: return "line"                                        # こげ茶の線
    return "fill"

def shift(m, dx, dy):
    o = np.zeros_like(m); H, W = m.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    o[yd, xd] = m[ys, xs]
    return o

def dens(m, r=2):
    return ndi.uniform_filter(m.astype(np.float32), size=2 * r + 1, mode="constant")

def recolor(rgba, view, cls_of):
    H, W = rgba.shape[:2]
    op = rgba[..., 3] > 127
    key = rgba[..., 0].astype(np.int32) << 16 | rgba[..., 1].astype(np.int32) << 8 | rgba[..., 2]
    C = np.full((H, W), "", dtype=object)
    for k in np.unique(key[op]):
        C[(key == k) & op] = cls_of[(int(k) >> 16, (int(k) >> 8) & 255, int(k) & 255)]
    is_ = lambda *n: np.isin(C, n)
    lum = (rgba[..., 0] * 0.299 + rgba[..., 1] * 0.587 + rgba[..., 2] * 0.114).astype(np.float32)

    tail = is_("white", "pale", "pale_w", "pale_d")
    tl, tn = ndi.label(tail, np.ones((3, 3)))
    if tn: tail &= np.isin(tl, [i + 1 for i, z in enumerate(ndi.sum(tail, tl, range(1, tn + 1))) if z >= 8])  # 目の白い点などは除く
    near_tail = ndi.binary_dilation(tail, np.ones((3, 3)), iterations=3)
    darkc = is_("k0", "k1")
    blob = darkc & (dens(darkc, 2) >= 0.42)                       # 耳の先・目の黒いかたまり
    ys = np.where(op.any(1))[0]; rows = np.arange(H)[:, None]
    blob &= rows < ys[0] + (ys[-1] - ys[0]) * 0.62                # 足の先は対象外（体の線として塗る）
    blob &= ~ndi.binary_dilation(tail, np.ones((3, 3)))           # しっぽの黒いふちは対象外（光にする）
    blob = ndi.binary_dilation(blob, np.ones((3, 3))) & (darkc | is_("line", "grey_m"))
    if view == "back":
        # 背面の黒は耳の先だけ。耳の先 = 胴の箱の外（左右の外か、頭のてっぺんより上）にある黒いかたまり。耳の中の線や背中の線は茶色の線にする
        rw0 = op.sum(1); hb = np.where(rw0 >= rw0.max() * 0.5)[0]; cols0 = np.where(op[hb[0]:hb[-1] + 1].sum(0) >= (hb[-1] - hb[0]) * 0.5)[0]
        xs_ = np.arange(W)[None, :]
        blob &= (rows < hb[0] + 5) & ((xs_ < cols0[0] + 6) | (xs_ > cols0[-1] - 10))
    body = op & ~tail                                             # 体（耳・足をふくむ）
    bodyf = ndi.binary_fill_holes(ndi.binary_closing(np.pad(body, 8), np.ones((11, 11)))[8:-8, 8:-8] | body)  # 体を横切るしっぽの切れ目はふさぐ
    edge = body & ~ndi.binary_erosion(op, np.ones((3, 3)))        # 輪郭の 1 ドット

    out = np.zeros((H, W, 3), np.uint8)
    # ---- 体の塗り: 輪郭から作る光（左上から。back は元の絵が右向きなので同じ左上） ----
    bw = max(1, int(np.ptp(np.where(bodyf.any(0))[0])) + 1)
    def inside(dx, dy): return shift(bodyf, dx, dy)               # (dx,dy) だけ動かした体。重ならない所 = その向きのふち
    k = bw / 100.0
    if view == "front":
        lvl = np.full((H, W), 2, np.int8)
        lvl[~inside(round(-14 * k), round(-10 * k))] = 1           # 右下の広い影
        lvl[~inside(round(-5 * k), round(-4 * k))] = 0             # 右下のきわ
        hl = ~inside(round(9 * k), round(9 * k)) & inside(round(4 * k), round(4 * k))
        lvl[hl & (lvl == 2)] = 3                                   # 左上のつや
    else:
        lvl = np.full((H, W), 1, np.int8)
        # 背中は PNG と同じ「左上に丸い明るい面、その中に小さいつや」。体（耳・足を除いた胴）の外接の箱を 0〜1 として楕円で置く
        rw_ = bodyf.sum(1); br = np.where(rw_ >= rw_.max() * 0.5)[0]; cw_ = np.where(bodyf[br[0]:br[-1] + 1].any(0))[0]
        yy_, xx_ = np.mgrid[0:H, 0:W]
        u = (xx_ - cw_[0]) / max(1, cw_[-1] - cw_[0]); v = (yy_ - br[0]) / max(1, br[-1] - br[0])
        lit = ((u - 0.30) / 0.36) ** 2 + ((v - 0.34) / 0.40) ** 2 <= 1
        lvl[lit] = 2
        lvl[~inside(round(-7 * k), round(-6 * k))] = 0             # 右下のきわ
        hl = ((u - 0.24) / 0.13) ** 2 + ((v - 0.13) / 0.07) ** 2 <= 1
        lvl[hl & (lvl == 2)] = 3
    # 元の絵の局所的な影（腕の下・あごの下など）: 周りの平均より暗い塗り
    fill = is_("fill") & body
    # 元の絵の明暗は横向きの帯なので、行ごとの中央値とくらべる（帯のさかい目を拾わない）
    lm = np.where(fill, lum, np.nan)
    import warnings
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); med = np.nanmedian(lm, axis=1)
    med = np.nan_to_num(med, nan=0.0); med0 = med.copy()
    med = ndi.maximum_filter1d(med, 5)                             # 帯のさかい目の行では明るいほうを基準にする
    d = np.where(fill, lum - med[:, None], 0)
    if view == "back": d = np.where(rows < br[0] + 3, 0, d)        # 背面の耳は影にしない（PNG の耳は体と同じ黄色）
    dark1 = ndi.binary_opening(d < -28, np.ones((2, 2)))           # 帯 2 段ぶんより暗い所だけ。1 ドットの点は影にしない
    lvl = lvl - dark1.astype(np.int8) - (dark1 & (d < -50)).astype(np.int8)
    lvl = np.clip(lvl, 0, 3)
    ramp = np.array(RAMP, np.uint8)
    out[body] = ramp[lvl[body]]

    # ---- 背中のしま: 黒い線で囲まれた小さい塗り ----
    if view == "back":
        # 元の絵のしまは、黒〜こげ茶（明るさ 60 未満）の線で囲まれた細いレンズ形。上 2 本は中もこげ茶、3 本目は中が 1 段暗いだけ。
        # 線をつないで（切れ目 4 ドットまで）囲まれた中を埋め、1 本の茶色のしまとして塗る。しまどうしの間は 5 ドット以上あくので、つながらない。
        by = np.where(bodyf.sum(1) >= bodyf.sum(1).max() * 0.5)[0]; y0, y1 = by[0] + (by[-1] - by[0]) * 0.15, by[0] + (by[-1] - by[0]) * 0.8
        zone = (rows >= y0) & (rows <= y1) & ~ndi.binary_dilation(~bodyf, np.ones((3, 3)), iterations=3) & ~near_tail  # 輪郭としっぽのそばは除く
        dk = (is_("k0") | (fill & (lum < 60))) & body & zone
        st = ndi.binary_closing(np.pad(dk, 8), np.ones((3, 5)))
        st = ndi.binary_fill_holes(st)[8:-8, 8:-8]
        sl, sn = ndi.label(st, np.ones((3, 3)))
        if sn: st &= np.isin(sl, [i + 1 for i, z in enumerate(ndi.sum(st, sl, range(1, sn + 1))) if z >= 45])  # 短い線はしまではない
        stripe = st & body & zone
        out[stripe] = OUT_M
    else: stripe = np.zeros((H, W), bool)

    # ---- 線・ふち ----
    out[is_("line") & body] = OUT_M
    out[is_("k0", "k1") & body & ~stripe] = OUT_D if view == "front" else OUT_M   # 背面は黒い点が散らないように 1 段明るい茶
    out[is_("grey_m", "grey_l") & body] = OUT_M
    near_edge = body & ndi.binary_dilation(~op, np.ones((3, 3)), iterations=2)
    out[is_("rim") & near_edge] = SH2                              # 体の内がわに入りこむ照り返しは塗らない（しみに見える）
    out[is_("rim_lo") & body] = SH1
    out[edge & is_("rim", "rim_lo", "line", "grey_m", "grey_l")] = OUT_M
    out[edge & is_("k0", "k1")] = OUT_D
    # ---- ほお ----
    rose = is_("rose") & body
    cheek = rose & (dens(rose, 2) >= 0.3)
    cl, cn = ndi.label(ndi.binary_dilation(cheek, np.ones((3, 3))))
    if cn:
        sz = ndi.sum(cheek, cl, range(1, cn + 1)); cy = [c[0] for c in ndi.center_of_mass(cheek, cl, range(1, cn + 1))]
        rw = bodyf.sum(1); by = np.where(rw >= rw.max() * 0.2)[0]; ymin = by[0] + (by[-1] - by[0]) * 0.2  # 頭のてっぺんの照り返しはほおではない
        keep = [i + 1 for i in np.argsort(sz)[::-1] if sz[i] >= 10 and cy[i] > ymin][:2]
        cheek &= np.isin(cl, keep)                                 # ほおは大きい順に 2 つまで
    out[rose & ~cheek] = SH2
    out[cheek] = RED
    out[cheek & ~shift(cheek, 1, 1) & shift(cheek, -1, -1)] = RED_L  # 左上に明るい赤
    # ---- 耳の先・目 ----
    out[blob & is_("k0")] = K0
    out[blob & is_("k1")] = K1
    out[blob & is_("line", "grey_m")] = K2
    # ---- しっぽ ----
    out[tail & is_("white")] = WHITE
    out[tail & is_("pale_w")] = WHITE
    out[tail & is_("pale")] = HL
    out[tail & is_("pale_d")] = GLOW
    glow = near_tail & ~tail & op & is_("grey_l", "grey_m", "k1", "k0", "line", "pale", "pale_w", "pale_d", "white") & ~blob
    out[glow] = GLOW
    out[tail & ndi.binary_dilation(~op, np.ones((3, 3)))] = GLOW   # しっぽの外がわの 1 ドットは黄色の光（PNG と同じ）
    # 目の白い点
    out[is_("white") & ~tail] = WHITE

    res = np.zeros((H, W, 4), np.uint8); res[..., :3] = out; res[..., 3] = np.where(op, 255, 0); res[~op] = 0
    return res

def save_gif(frames, durs, path):
    cols = sorted({tuple(c) for f in frames for c in np.unique(f[f[..., 3] > 0][:, :3], axis=0)})
    assert len(cols) < 255
    pal = [255, 0, 255] + [v for c in cols for v in c]; pal += [0] * (768 - len(pal))
    ps = []
    for f in frames:
        idx = np.zeros(f.shape[:2], np.uint8)
        for i, c in enumerate(cols): idx[(f[..., :3] == c).all(-1) & (f[..., 3] > 0)] = i + 1
        p = Image.fromarray(idx, "P"); p.putpalette(pal); ps.append(p)
    ps[0].save(path, save_all=True, append_images=ps[1:], duration=durs, loop=0, transparency=0, disposal=2, optimize=False)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--view", required=True, choices=["front", "back"]); ap.add_argument("--out"); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    g = Image.open(SRC / f"{a.view}.gif")
    src, durs = [], []
    for f in ImageSequence.Iterator(g): src.append(np.array(f.convert("RGBA"))); durs.append(f.info.get("duration", 30))
    cols = {tuple(int(v) for v in c) for f in src for c in np.unique(f[f[..., 3] > 127][:, :3], axis=0)}
    cls_of = {c: classify(c) for c in cols}
    frames = [recolor(f, a.view, cls_of) for f in src]
    allowed = {tuple(p[:3]) for n in ("front", "back") for p in Image.open(FORM / f"{n}.png").convert("RGBA").get_flattened_data() if p[3] > 127}
    used = {tuple(int(v) for v in c) for f in frames for c in np.unique(f[f[..., 3] > 0][:, :3], axis=0)}
    assert used <= allowed, used - allowed                                       # PNG に無い色は使わない
    for s, f in zip(src, frames): assert ((s[..., 3] > 127) == (f[..., 3] > 0)).all()  # 輪郭は変えない
    out = FORM if a.apply else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_gif(frames, durs, out / f"{a.view}.gif")
    save_gif([f[:, ::-1] for f in frames], durs, out / f"{a.view}-reverse.gif")
    # 読み直して確かめる
    for name, flip in ((f"{a.view}.gif", False), (f"{a.view}-reverse.gif", True)):
        r = Image.open(out / name); i = 0  # 同じ絵が続くコマは 1 コマにまとめて保存される（表示時間は合計）。時間どおりに並べ直して比べる
        for f in ImageSequence.Iterator(r):
            px = np.array(f.convert("RGBA")); left = f.info.get("duration")
            while left > 0:
                want = frames[i][:, ::-1] if flip else frames[i]; o = want[..., 3] > 0
                assert ((px[..., 3] > 0) == o).all() and (px[o] == want[o]).all(), (name, i)
                left -= durs[i]; i += 1
            assert left == 0
        assert i == len(frames)
    print(a.view, len(frames), "frames", g.size, "colours", len(used), "->", out)

if __name__ == "__main__":
    main()
