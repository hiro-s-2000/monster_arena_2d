# Mega Charizard Y, FRONT view (front.gif + front-reverse.gif): bones and motion for tools/gif_bone_warp.py.
# The still (front.png, 96x96) faces LEFT. This file is only the config; the engine is gif_bone_warp.py.
#
#   <agent-sprite-forge>/.venv/bin/python -I tools/gif_bone_warp.py tools/charizard_mega_y_gif_front.py --grid grid.png
#   <agent-sprite-forge>/.venv/bin/python -I tools/gif_bone_warp.py tools/charizard_mega_y_gif_front.py --out DIR --sheet sheet.png
#   (--apply instead of --out writes into monsterSprite/ and data/sprite_bounds.js)
#
# Timing = Mega Charizard X front.gif: 96 frames x 100 ms; frames 0-17 idle cycle (repeated 4x up to 71), 72-95 roar.
import math

FORM_REL = "gen1/0006_charizard/mega/charizard-mega-y"
STILL = "front.png"
OUT_NAME = "front"
N, CYCLE, ROAR0, DUR = 96, 18, 72, 100

# Only the tail flame uses these three colours (belly = (238,222,123)/(238,180,90), which are different).
FLAME_COLOURS = [(246, 164, 16), (230, 65, 16), (255, 213, 16)]
OUTLINE_COLOURS = [(16, 16, 16), (131, 49, 24)]
HIGHLIGHT_COLOURS = [(248, 248, 248), (205, 205, 205)]
BLUR = 2.4

# How this still is built (x right, y down):
# - head x31-61 / y5-33 (three horns: small left one x41-44/y11-16, long middle one up to (60,5), right one x52-55/y14-15),
#   neck down to y~39.
# - left wing: finger spike x20-28/y8-22, wrist ~(25,27), big outer lobe x0-19 hanging to y72, teal membrane x14-42/y31-52.
#   The left forearm with its fin (spike tip (24,38), hand/claws (46-52,61-63)) lies IN FRONT of that membrane.
# - right wing: bone/spike x65-73/y8-30, wrist ~(70,25), outer lobe x74-95 hanging to y71, teal membrane x70-80/y32-46;
#   the right forearm fin (x62-72/y28-50, teal x62-67) and hand (claws (71-75,55-56)) sit in front of the wing root.
# - tail x68-84/y62-68 behind the right thigh; its flame x78-88/y44-63 stands in the notch between the right wing's
#   membrane and outer lobe (in front of the lobe's inner edge).
# Bones: the two outer wing lobes hinge at the wing wrists (the crease runs through the flat dark-teal membrane, so the
# stretch does not show); the wing roots, the forearms + fins and the thighs stay with the body (lean / bob only).
PARTS = [
    dict(name="tail", poly=[(68.5, 59.5), (77.5, 59.5), (78.5, 57.5), (87, 57.5), (87, 66), (83, 71), (70, 72), (68.5, 69)],
         pivot=(69, 66)),
    dict(name="tailflame", poly=[(76, 42), (90, 42), (90, 64.6), (76, 64.6)], pivot=(83, 62), parent="tail", flame=True),
    dict(name="head", poly=[(29.5, -24), (62.5, -24), (62.5, 9), (60, 20), (59.5, 34), (58, 38.5), (47, 38.5), (46.5, 35.5),
                            (31, 35.5), (29.5, 30)], pivot=(52, 40)),
    dict(name="lobeL", poly=[(-24, -24), (19.5, -24), (19.5, 22.5), (22, 24.5), (25.5, 28), (26.3, 31), (26.5, 34), (25, 36.5),
                             (22.6, 38), (22, 40), (21, 42), (20.5, 44), (19.4, 45.4), (17, 47.6), (16.5, 49.5), (15.5, 53),
                             (14.5, 57), (14, 100), (-24, 100)], pivot=(25, 27)),
    dict(name="lobeR", poly=[(74.5, -24), (120, -24), (120, 100), (87, 100), (87, 57.5), (79.5, 57.5), (79.5, 48), (78.6, 46.6),
                             (78, 43.5), (77, 41), (75, 38), (73, 35), (72, 32), (72.5, 30.5), (73, 25), (74.5, 23)], pivot=(70, 25)),
]

REST = dict(lean=0, bob=0, ll=0, lr=0, lsx=1, head=0, hdx=0, hdy=0, tail=0, tdx=0, fl=1)


def smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def keys(u, table):
    for (u0, a), (u1, b) in zip(table, table[1:]):
        if u0 <= u <= u1:
            k = smooth((u - u0) / (u1 - u0))
            return {p: a[p] + (b[p] - a[p]) * k for p in a}
    return dict(table[-1][1])


def K(f, **kw):
    return ((f - ROAR0) / (N - ROAR0), dict(REST, **kw))


# Signs (measured with this engine): rot > 0 turns a part counter-clockwise on screen (what is right of the pivot rises,
# what is above it goes left). lean < 0 leans the body to the LEFT = forward for this still.
# ll < 0 / lr > 0 swing the wing lobes outward and up (spread); the tail follows the right lobe (tail > 0 lifts the tip,
# tdx > 0 moves it right) so that the gap between tail tip / flame and the lobe does not get crushed.
ROAR = [
    K(72),
    K(76, lean=5, bob=1, ll=2, lr=-1, lsx=1.0, head=-2, tail=-2, tdx=-0.5),
    K(79, lean=8, bob=1.8, ll=3, lr=-1.5, lsx=1.0, head=-3, hdy=0.5, tail=-3, tdx=-1, fl=1.05),
    K(84, lean=-16, bob=-2.5, ll=-13, lr=10, lsx=0.95, head=-2, hdx=-3, hdy=-1, tail=9, tdx=3.5, fl=1.02),
    K(86, lean=-13, bob=-1.5, ll=-11, lr=8.5, lsx=0.96, head=-1, hdx=-2.5, hdy=-0.5, tail=7.5, tdx=3, fl=1.02),
    K(93, lean=-4, bob=0, ll=-3, lr=2.5, lsx=1.0, head=0, hdx=-1, tail=2, tdx=0.8, fl=1.0),
    K(96),
]


def params(i):
    if i < ROAR0:
        p = (i % CYCLE) / CYCLE
        s = (1 - math.cos(2 * math.pi * p)) / 2  # 0 -> 1 -> 0, peak at frame 9
        q = math.sin(2 * math.pi * p)
        return dict(lean=-13.0 * s, bob=-2.5 * s, ll=-10 * s, lr=8 * s, lsx=1 - 0.1 * s, head=3 * s + 1.5 * q, hdx=-1 * s, hdy=0,
                    tail=6.5 * s + 2.5 * q, tdx=2.5 * s, fl=1 + 0.06 * s)
    P = keys((i - ROAR0) / (N - ROAR0), ROAR)
    if 85 <= i <= 93:  # two-beat shake after the roar
        env = math.sin(math.pi * (i - 84.5) / 9.0)
        w = math.sin(2 * math.pi * (i - 85) / 4.0 + 0.6)
        P["head"] += 4 * env * w
        P["lean"] -= 2.0 * env * w
        P["bob"] += 1.2 * env * w
        P["ll"] -= 3 * env * w
        P["lr"] += 2.5 * env * w
        P["tail"] += 2 * env * w
    return P


def pose(i):
    P = params(i)
    return dict(lean=P["lean"], bob=P["bob"], fl=P["fl"], parts={
        "head": dict(rot=P["head"], dx=P["hdx"], dy=P["hdy"]),
        "lobeL": dict(rot=P["ll"], sx=P["lsx"]),
        "lobeR": dict(rot=P["lr"], sx=P["lsx"]),
        "tail": dict(rot=P["tail"], dx=P["tdx"]),
        "tailflame": dict(rot=-0.6 * P["tail"]),  # the flame keeps pointing up while the tail tip tilts
    })
