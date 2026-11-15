"""Turn the official-art Mega Latias image into a monsterSprite-style pixel sprite (normal + shiny).

usage: python make_sprite.py <ref.webp> <out_dir> [max_px]
Writes front/back(.png/.gif, -reverse, -shiny) into out_dir.
"""
import sys, colorsys
import numpy as np
from PIL import Image

ref, out = sys.argv[1], sys.argv[2]
MAXPX = int(sys.argv[3]) if len(sys.argv) > 3 else 90
CANVAS = 96
OUTLINE = (16, 16, 16)

src = Image.open(ref).convert("RGBA")
bb = src.getchannel("A").point(lambda a: 255 if a > 20 else 0).getbbox()
src = src.crop(bb)
sc = MAXPX / max(src.size)
w, h = max(1, round(src.width * sc)), max(1, round(src.height * sc))
small = src.resize((w, h), Image.LANCZOS)  # Pillow premultiplies alpha for RGBA

a = np.array(small).astype(np.float32)
alpha = a[..., 3] > 110
rgb = a[..., :3]

# Line-art preservation: pixels clearly darker than their 3x3 neighbourhood become dark line colour.
lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
pad = np.pad(lum, 1, mode="edge")
neigh = sum(pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
line = alpha & (lum < neigh - 26) & (lum < 120)

# Palette: quantize opaque pixels to a small palette (median cut).
img = Image.fromarray(np.dstack([rgb, np.full((h, w), 255, np.float32)]).astype(np.uint8), "RGBA").convert("RGB")
q = img.quantize(colors=18, method=Image.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
qa = np.array(q).astype(np.int32)

res = np.zeros((h, w, 4), np.uint8)
res[..., :3] = qa
res[..., 3] = np.where(alpha, 255, 0)
# interior lines: a darkened version of the local colour (keeps purple lines purple-ish, not pure black)
dark = (qa * 0.35).astype(np.uint8)
res[line, :3] = dark[line]

# 1px outer outline
op = np.pad(alpha, 1)
edge = alpha & ~(op[:-2, 1:-1] & op[2:, 1:-1] & op[1:-1, :-2] & op[1:-1, 2:])
res[edge, :3] = OUTLINE

def place(arr):
    c = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    ox = (CANVAS - w) // 2
    oy = CANVAS - h - 3  # feet near the bottom like the other sprites
    c[oy:oy + h, ox:ox + w] = arr
    return Image.fromarray(c, "RGBA")

def shiny(arr):
    """Purple/blue body -> the green of the existing shiny sprite (hue 264 deg -> 90 deg). Greys/eyes untouched."""
    o = arr.copy()
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r, g, b, al = arr[y, x]
            if not al:
                continue
            hh, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if s > 0.12 and 0.55 < hh < 0.85:
                hh = (hh - 174 / 360) % 1.0
                s = min(1.0, s * 1.9)
                v = v * 0.95
                r2, g2, b2 = colorsys.hsv_to_rgb(hh, s, v)
                o[y, x, :3] = (round(r2 * 255), round(g2 * 255), round(b2 * 255))
    return o

def save_gif(im, path):
    # single-frame transparent GIF (palette + transparent index)
    p = im.convert("RGBA")
    al = p.getchannel("A")
    pq = p.convert("RGB").quantize(colors=255, method=Image.MEDIANCUT, dither=Image.Dither.NONE)
    pal = pq.getpalette()[:255 * 3] + [0, 0, 0]
    pq.putpalette(pal)
    px = np.array(pq)
    px[np.array(al) < 128] = 255
    g = Image.fromarray(px.astype(np.uint8), "P"); g.putpalette(pal)
    g.save(path, transparency=255, disposal=2, loop=0, duration=100)

for tag, arr in (("", res), ("-shiny", shiny(res))):
    f = place(arr)
    m = f.transpose(Image.FLIP_LEFT_RIGHT)
    f.save(f"{out}/front{tag}.png")
    # back views: no back art exists -> reuse the front (mirrored so NE faces right), per the user's choice
    m.save(f"{out}/back{tag}.png")
    save_gif(f, f"{out}/front{tag}.gif")
    save_gif(m, f"{out}/front{tag}-reverse.gif")
    save_gif(m, f"{out}/back{tag}.gif")
    save_gif(f, f"{out}/back{tag}-reverse.gif")
    Image.fromarray(np.kron(np.array(f), np.ones((4, 4, 1), np.uint8)), "RGBA").save(f"{out}/preview{tag}_x4.png")
print("sprite", w, "x", h)
