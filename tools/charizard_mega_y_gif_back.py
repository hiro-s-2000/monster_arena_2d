# Mega Charizard Y, BACK view: config for tools/gif_bone_warp.py (bones + motion; the engine is in that file).
# Makes back.gif and back-reverse.gif from the still back.png: 96 frames, 100 ms, 18-frame idle x4, then a 24-frame roar.
#
#   <agent-sprite-forge>/.venv/bin/python -I tools/gif_bone_warp.py tools/charizard_mega_y_gif_back.py --grid grid.png
#   <agent-sprite-forge>/.venv/bin/python -I tools/gif_bone_warp.py tools/charizard_mega_y_gif_back.py --out DIR --sheet sheet.png
#
# The still is seen from behind and faces RIGHT (head at the right). What is where (pixels of back.png):
#   - big left wing: x4-41, y13-62, root behind the tail at about (45,41)
#   - tail: comes round under the body, up the left side (cream band) and ends at (45,28) with the flame on top (x38-54, y4-28)
#   - head x74-90 y24-52, looking out from behind the folded right wing (x60-73, y12-47: long spike, small spike, membrane)
#   - right arm with the forearm fin: x57-78, y48-68.  Feet: y78-88.
# Flame colours (230,66,16) (247,165,16) (255,214,16) are used by the tail flame only.
import math

FORM_REL = "gen1/0006_charizard/mega/charizard-mega-y"
STILL = "back.png"
OUT_NAME = "back"
N, CYCLE, ROAR0, DUR = 96, 18, 72, 100

FLAME_COLOURS = [(230, 66, 16), (247, 165, 16), (255, 214, 16)]
OUTLINE_COLOURS = [(0, 0, 0), (131, 49, 24)]
HIGHLIGHT_COLOURS = [(205, 205, 205), (242, 242, 242), (255, 255, 255), (33, 115, 148), (99, 99, 99)]
BLUR = 1.6

PARTS = [
    # tail tip (the thin end that crosses in front of the wing) - whips about the point where it crosses the wing
    dict(name="tail", poly=[(42.5, 10), (52, 10), (52, 34), (49, 36.5), (43, 36.5), (42, 41), (36, 41), (36.5, 37), (40.5, 33.5), (42.5, 30)], pivot=(40, 39)),
    dict(name="tailflame", poly=[(30, 0), (60, 0), (60, 28.9), (30, 28.9)], pivot=(45, 27), parent="tail", flame=True),
    # head + the folded right wing / spikes in front of it: one rigid group, turning at the neck base
    dict(name="head", poly=[(52, 34), (52, -24), (120, -24), (120, 57), (82, 56), (76, 52), (72, 46), (69.5, 44.5), (64, 40.5), (57, 36)], pivot=(68, 43)),
    # right arm with the forearm fin, turning at the shoulder
    dict(name="arm", poly=[(57.5, 47.6), (68.5, 47.6), (70, 58), (74, 60), (81, 62), (81, 69.5), (71, 70), (63, 69.6), (57.5, 68.5), (56.5, 60)], pivot=(58, 50)),
    # big left wing, root at (45,41)
    dict(name="wingL", poly=[(-24, -24), (52, -24), (52, 10), (42.5, 10), (42.5, 30), (40.5, 33.5), (36.5, 37), (33, 40.5), (29, 42), (21.5, 42), (20.5, 45), (19.5, 52), (14, 48), (12, 56), (12, 70), (-24, 70)], pivot=(45, 41)),
]


def smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def keys(u, table):
    for (u0, a), (u1, b) in zip(table, table[1:]):
        if u0 <= u <= u1:
            k = smooth((u - u0) / (u1 - u0))
            return {p: a[p] + (b[p] - a[p]) * k for p in a}
    return dict(table[-1][1])


# lean: + = toward the facing side (right).  wing: + = wingtip (left end) lifts.  wsx: wing width (fold).
# head: + = head lifts.  tail: + = tail tip and flame swing right.  arm: + = hand lifts.
REST = dict(lean=0, bob=0, wing=0, wsx=1, wsy=1, head=0, hdx=0, hdy=0, tail=0, arm=0, fl=1)


def K(f, **kw):
    return ((f - ROAR0) / (N - ROAR0), dict(REST, **kw))


ROAR = [
    K(72),
    K(76, lean=-5, bob=1, wing=-6, wsx=0.93, head=-4, tail=-6, arm=-2),
    K(79, lean=-7, bob=1.5, wing=-10, wsx=0.88, head=-6, hdy=1, tail=-9, arm=-3, fl=1.1),
    K(84, lean=14, bob=-2.5, wing=16, wsx=1.05, wsy=1.04, head=7, hdx=2, hdy=-1, tail=9, arm=9, fl=1.1),
    K(86, lean=11, bob=-1, wing=11, wsx=1.03, wsy=1.02, head=5, hdx=2, tail=6, arm=7, fl=1.08),
    K(93, lean=4, bob=0, wing=3, wsx=1.0, head=1, hdx=1, tail=2, arm=2, fl=1.05),
    K(96),
]


def params(i):
    if i < ROAR0:
        p = (i % CYCLE) / CYCLE
        s = (1 - math.cos(2 * math.pi * p)) / 2  # 0 -> 1 -> 0, peak at frames 8-9
        q = math.sin(2 * math.pi * p)
        return dict(REST, lean=9.5 * s, bob=-2.5 * s, wing=18 * s - 2.5 * q, wsx=1 - 0.16 * s, head=4 * s - 2 * q, hdy=-1 * s,
                    tail=8 * math.sin(2 * math.pi * p - 0.9) + 8 * math.sin(0.9), arm=6 * s, fl=1 + 0.06 * s)
    P = keys((i - ROAR0) / (N - ROAR0), ROAR)
    if 85 <= i <= 93:  # two-beat shake after the roar
        env = math.sin(math.pi * (i - 84.5) / 9.0)
        w = math.sin(2 * math.pi * (i - 85) / 4.0 + 0.6)
        P["head"] += 6 * env * w
        P["lean"] += 2.5 * env * w
        P["bob"] += 1.2 * env * w
        P["wing"] += 4 * env * w
        P["tail"] += 3 * env * w
    return P


def pose(i):
    P = params(i)
    # engine as built: rot < 0 lifts what is left of the pivot and lowers what is right of it (checked on the sheets)
    return dict(lean=P["lean"], bob=P["bob"], fl=P["fl"], parts={
        "wingL": dict(rot=-P["wing"], sx=P["wsx"], sy=P["wsy"]),
        "head": dict(rot=P["head"], dx=P["hdx"], dy=P["hdy"]),
        "tail": dict(rot=-P["tail"]),
        "arm": dict(rot=P["arm"]),
    })
