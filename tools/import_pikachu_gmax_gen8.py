#!/usr/bin/env python3
"""再実行しない。2026-11-16 の全キョダイマックス上書きは、表示用の v1/anim（256×256）を縮めずに GIF にしている。このスクリプトは orig の 512px を体高 155 に縮めるので、その4枚を壊す。

キョダイマックスピカチュウの GIF を、pokestats.gg の Generation VIII（Sword / Shield）のアニメで上書きする。
（2026-11-16、ユーザー指示「キョダイマックスピカチュウを https://pokestats.gg の Generation VIII から採用してきて上書き」）

素材（--src のフォルダに置く。https://pokestats.gg/sprites/pikachu-gmax の Generation VIII・通常色）:
  front_gen8.webp … assets.pokestats.gg/pokeapi/orig/pokemon/versions/generation-viii/sword-shield/10199.webp      （512x512・50 コマ）
  back_gen8.webp  … assets.pokestats.gg/pokeapi/orig/pokemon/versions/generation-viii/sword-shield/back/10199.webp （512x512・50 コマ）
作り方:
  - 大きさは、これまでの GIF と同じ（正面の体の高さ 155 ドット）になるよう、正面・背面とも同じ倍率で縮める（絵は描き足さない）。
  - 全コマを同じ枠で切り抜く（足もとがコマごとにずれない）。透明は 0 か 255。色は全コマ共通の 255 色以内。
  - 表示時間は元の 33/33/34 ms を GIF の単位（10 ms）に合わせ 30/30/40 ms のくり返しにする（平均 33.3 ms で元と同じ速さ）。
  - -reverse は左右反転。
書き出し先: monsterSprite/gen1/0025_pikachu/gigantamax/pikachu-gmax/{front,front-reverse,back,back-reverse}.gif と data/sprite_bounds.js の g。
色違いの GIF は作らない（このフォームには元から無い）。front.png / back.png は変えない。

使い方（向きごとに 1 回ずつ独立に実行する）:
  <agent-sprite-forge>/.venv/bin/python tools/import_pikachu_gmax_gen8.py --src <素材フォルダ> --view front [--apply]
  <agent-sprite-forge>/.venv/bin/python tools/import_pikachu_gmax_gen8.py --src <素材フォルダ> --view back  [--apply]
このあと tools/build_dir8_from_gifs.py --gen gen1 --only 0025_pikachu-gmax --apply で 8 方向シートを作り直す。
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parent.parent
REL = "gen1/0025_pikachu/gigantamax/pikachu-gmax"
FORM = ROOT / "monsterSprite" / REL
SCALE = 155 / 452  # 正面の全コマを合わせた体の高さ 452px → 155 ドット。背面も同じ倍率
DUR = [30, 30, 40]

def load(path):
    im = Image.open(path); out = []
    for f in ImageSequence.Iterator(im): out.append(f.convert("RGBA"))
    return out

def shrink(f, size):
    """透明部分の色がにじまないよう、アルファを掛けてから縮める。"""
    a = np.asarray(f, np.float32); al = a[..., 3:4] / 255
    pm = Image.fromarray(np.concatenate([a[..., :3] * al, a[..., 3:4]], -1).round().astype(np.uint8), "RGBA").resize(size, Image.LANCZOS)
    p = np.asarray(pm, np.float32); al2 = np.maximum(p[..., 3:4], 1) / 255
    rgb = np.clip(p[..., :3] / al2, 0, 255).round().astype(np.uint8)
    op = p[..., 3] >= 128
    res = np.zeros(p.shape, np.uint8); res[..., :3] = rgb; res[..., 3] = np.where(op, 255, 0); res[~op] = 0
    return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--view", required=True, choices=["front", "back"])
    ap.add_argument("--out"); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    src = load(Path(a.src) / f"{a.view}_gen8.webp")
    W0, H0 = src[0].size; size = (round(W0 * SCALE), round(H0 * SCALE))
    frames = [shrink(f, size) for f in src]
    op = np.any([f[..., 3] > 0 for f in frames], axis=0)
    ys, xs = np.where(op.any(1))[0], np.where(op.any(0))[0]
    x0, x1, y0, y1 = max(0, xs[0] - 1), min(size[0], xs[-1] + 2), max(0, ys[0] - 1), min(size[1], ys[-1] + 2)
    frames = [f[y0:y1, x0:x1] for f in frames]
    # 全コマ共通の色（255 色以内、ディザなし）
    allpx = np.concatenate([f[f[..., 3] > 0][:, :3] for f in frames])
    pal_img = Image.fromarray(allpx.reshape(1, -1, 3), "RGB").quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = pal_img.getpalette()[:255 * 3]
    n_ = len(pal) // 3
    pimg = Image.new("P", (1, 1)); pimg.putpalette(pal + pal[-3:] * (256 - n_))  # 余りは最後の色で埋める（黒で埋めると 255 番が選ばれて透明になる）
    full = [255, 0, 255] + pal; full += [0] * (768 - len(full))
    ps = []
    for f in frames:
        q = np.asarray(Image.fromarray(f[..., :3], "RGB").quantize(palette=pimg, dither=Image.Dither.NONE), np.uint8)
        idx = np.where(f[..., 3] > 0, np.minimum(q, n_ - 1).astype(np.int32) + 1, 0).astype(np.uint8)
        p = Image.fromarray(idx, "P"); p.putpalette(full); ps.append(p)
    durs = [DUR[i % 3] for i in range(len(ps))]
    out = FORM if a.apply else Path(a.out); out.mkdir(parents=True, exist_ok=True)
    def save(seq, name):
        seq[0].save(out / name, save_all=True, append_images=seq[1:], duration=durs, loop=0, transparency=0, disposal=2, optimize=False)
    save(ps, f"{a.view}.gif")
    save([p.transpose(Image.FLIP_LEFT_RIGHT) for p in ps], f"{a.view}-reverse.gif")
    # 読み直して確かめる: 合計時間、各時点の輪郭、左右反転
    def timeline(path):
        r = Image.open(path); tl = []
        for f in ImageSequence.Iterator(r): tl.append((f.info.get("duration"), np.asarray(f.convert("RGBA"))))
        return tl
    A, B = timeline(out / f"{a.view}.gif"), timeline(out / f"{a.view}-reverse.gif")
    assert sum(d for d, _ in A) == sum(durs) == sum(d for d, _ in B), (sum(d for d, _ in A), sum(durs))
    assert len(A) == len(B)
    for (da, fa), (db, fb) in zip(A, B):
        assert da == db and ((fa[..., 3] > 0) == (fb[:, ::-1, 3] > 0)).all() and (fa[fa[..., 3] > 0] == fb[:, ::-1][fa[..., 3] > 0]).all()
    i = 0
    for d, fa in A:                                                # 同じ絵が続くコマは 1 コマにまとまることがある。時間で追って輪郭を比べる
        left = d
        while left > 0: assert ((fa[..., 3] > 0) == (frames[i][..., 3] > 0)).all(), i; left -= durs[i]; i += 1
        assert left == 0
    assert i == len(frames)
    bb = lambda p: (lambda im: (lambda b: [im.size[0], im.size[1], b[0], b[1], b[2] - b[0], b[3] - b[1]])(im.convert("RGBA").getchannel("A").getbbox()))(Image.open(p))
    print(a.view, len(frames), "frames", (x1 - x0, y1 - y0), sum(durs), "ms ->", out, bb(out / f"{a.view}.gif"))
    if a.apply:
        bp = ROOT / "data/sprite_bounds.js"; raw = bp.read_text(encoding="utf-8"); i0, j0 = raw.index("{"), raw.rindex("}") + 1
        Bd = json.loads(raw[i0:j0])
        for v in (a.view, a.view + "-reverse"): Bd["g"][f"{REL}/{v}.gif"] = bb(FORM / f"{v}.gif")
        bp.write_text(raw[:i0] + json.dumps(Bd, separators=(",", ":"), ensure_ascii=False) + raw[j0:], encoding="utf-8")

if __name__ == "__main__":
    main()
