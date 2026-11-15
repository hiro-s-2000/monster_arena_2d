"""Mega Latias v2: same pipeline as art_to_sprite.py, plus a readable face.

1. Enlarge the head in the source art (x1.3 around the neck) so the face gets more pixels.
2. Pixelize to MAXPX (default 112) on a CANVAS (default 120) canvas.
3. Stamp a hand-designed eye (outline almond, amber iris, black pupil, catch-light) and
   re-ink the mask edge / white brow patch so the face reads at sprite size.
usage: python make3.py <ref> <out_dir> [maxpx] [canvas]
"""
import sys, colorsys
import numpy as np
from PIL import Image, ImageDraw

ref, out = sys.argv[1], sys.argv[2]
MAXPX = int(sys.argv[3]) if len(sys.argv) > 3 else 112
CANVAS = int(sys.argv[4]) if len(sys.argv) > 4 else 120
OUTLINE = (16, 16, 16)
HEAD_POLY = [(283, 96), (296, 86), (326, 85), (346, 98), (352, 140), (334, 149), (300, 149), (283, 132)]
ANCHOR, HS = (303, 147), 1.3
EYE_SRC = (315, 126)  # pupil centre in the source art

src = Image.open(ref).convert("RGBA")

# 1. enlarge the head
mask = Image.new("L", src.size, 0)
ImageDraw.Draw(mask).polygon(HEAD_POLY, fill=255)
head = Image.new("RGBA", src.size, (0, 0, 0, 0))
head.paste(src, (0, 0), mask)
hb = head.getbbox()
hc = head.crop(hb)
hc = hc.resize((round(hc.width * HS), round(hc.height * HS)), Image.LANCZOS)
nx = round(ANCHOR[0] + (hb[0] - ANCHOR[0]) * HS)
ny = round(ANCHOR[1] + (hb[1] - ANCHOR[1]) * HS)
big = Image.new("RGBA", (src.width + 60, src.height + 60), (0, 0, 0, 0))
big.alpha_composite(src, (30, 30))
big.alpha_composite(hc, (nx + 30, ny + 30))
src = big
eye = (ANCHOR[0] + (EYE_SRC[0] - ANCHOR[0]) * HS + 30, ANCHOR[1] + (EYE_SRC[1] - ANCHOR[1]) * HS + 30)

# 2. pixelize
bb = src.getchannel("A").point(lambda a: 255 if a > 20 else 0).getbbox()
src = src.crop(bb)
eye = (eye[0] - bb[0], eye[1] - bb[1])
sc = MAXPX / max(src.size)
w, h = max(1, round(src.width * sc)), max(1, round(src.height * sc))
small = src.resize((w, h), Image.LANCZOS)
eye = (eye[0] * sc, eye[1] * sc)

a = np.array(small).astype(np.float32)
alpha = a[..., 3] > 110
rgb = a[..., :3]
lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
pad = np.pad(lum, 1, mode="edge")
neigh = sum(pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
line = alpha & (lum < neigh - 26) & (lum < 120)
img = Image.fromarray(rgb.astype(np.uint8), "RGB")
q = img.quantize(colors=20, method=Image.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
qa = np.array(q).astype(np.int32)
res = np.zeros((h, w, 4), np.uint8)
res[..., :3] = qa
res[..., 3] = np.where(alpha, 255, 0)
res[line, :3] = (qa * 0.35).astype(np.uint8)[line]
op = np.pad(alpha, 1)
edge = alpha & ~(op[:-2, 1:-1] & op[2:, 1:-1] & op[1:-1, :-2] & op[1:-1, 2:])
res[edge, :3] = OUTLINE

def to_sprite(pt):
    x = (ANCHOR[0] + (pt[0] - ANCHOR[0]) * HS + 30 - bb[0]) * sc
    y = (ANCHOR[1] + (pt[1] - ANCHOR[1]) * HS + 30 - bb[1]) * sc
    return (x, y)


def paint_poly(poly, fill, ink=True):
    mk = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mk).polygon([to_sprite(p) for p in poly], fill=255)
    m = (np.array(mk) > 0) & alpha
    res[m, :3] = fill
    if ink:
        mp = np.pad(m, 1)
        rim = ~m & alpha & (mp[:-2, 1:-1] | mp[2:, 1:-1] | mp[1:-1, :-2] | mp[1:-1, 2:])
        res[rim, :3] = (40, 34, 64)


# white streak on the mask, right of the eye (outlined), and the white jaw under the mask
paint_poly([(323, 109), (329, 109), (341, 124), (341, 131), (337, 131)], (232, 234, 242))
paint_poly([(301, 137), (318, 139), (340, 141), (348, 141), (336, 148), (304, 148)], (226, 228, 236), ink=False)

# 3. hand-made eye, 7x5, drawn with its pupil cell on the computed eye centre.
#    K outline, A amber iris, a dark amber, P pupil, W catch-light, . leave
EYE = [
    "KKKK...",
    "KAWPKK.",
    "KAAPAAK",
    ".KaPaK.",
    "..KKK..",
]
COL = {"K": (24, 20, 34), "A": (232, 178, 72), "a": (176, 120, 48), "P": (16, 12, 10), "W": (255, 250, 236)}
ex, ey = int(round(eye[0])) - 3, int(round(eye[1])) - 2
for j, row in enumerate(EYE):
    for i, ch in enumerate(row):
        if ch in COL and 0 <= ey + j < h and 0 <= ex + i < w:
            res[ey + j, ex + i, :3] = COL[ch]
            res[ey + j, ex + i, 3] = 255
print("eye at", ex + 3, ey + 2, "sprite", w, h)


def place(arr):
    c = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    ox, oy = (CANVAS - w) // 2, CANVAS - h - 3
    c[oy:oy + h, ox:ox + w] = arr
    return Image.fromarray(c, "RGBA")


def shiny(arr):
    o = arr.copy()
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r, g, b, al = arr[y, x]
            if not al:
                continue
            hh, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if s > 0.12 and 0.55 < hh < 0.85:
                hh = (hh - 174 / 360) % 1.0
                r2, g2, b2 = colorsys.hsv_to_rgb(hh, min(1.0, s * 1.9), v * 0.95)
                o[y, x, :3] = (round(r2 * 255), round(g2 * 255), round(b2 * 255))
    return o


def save_gif(im, path):
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
    m.save(f"{out}/back{tag}.png")
    save_gif(f, f"{out}/front{tag}.gif")
    save_gif(m, f"{out}/front{tag}-reverse.gif")
    save_gif(m, f"{out}/back{tag}.gif")
    save_gif(f, f"{out}/back{tag}-reverse.gif")
    Image.fromarray(np.kron(np.array(f), np.ones((4, 4, 1), np.uint8)), "RGBA").save(f"{out}/preview{tag}_x4.png")
