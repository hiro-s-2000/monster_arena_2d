# Task (image generation only — do not edit any other files)

Use your image generation tool. You MUST pass BOTH reference image files as input images to every generation call (absolute paths in this folder; check they are attached before generating): `ref_front.png` (front view pixel sprite, the creature faces the viewer, slightly lower-left) and `ref_back.png` (back view pixel sprite). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

The subject is ONLY the teal four-legged plant creature with the huge pink back flower shown in the reference images. It is NOT an orange lizard, has NO tail, NO flame, NO arms; it stands on four legs in every one of the four views.

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites (front view and back view) into larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a heavy, wide, low four-legged reptile-like plant creature, like a squat toad / small dinosaur, standing on all four short thick legs. Broad flat head at the front with a wide mouth, two red eyes with angry brows, small pointed ears.
2. Colour palette:
   - Skin (body, head, legs): teal blue-green (#3FB2AE) with light teal highlights (#7CC8C7), mid teal (#328D98, #298682) and dark teal shading (#19554D). Darker teal spots on the skin.
   - Eyes: red irises (#D02429) with white.
   - Claws: small white claws (#F8F8F8), three on each foot.
   - Flower petals: salmon pink (#F69499) with deep pink/red shading (#DF5B5F) and white highlight streaks (#F8F8F8).
   - Trunk in the centre of the flower: brown (#833100, #B36F50) with a yellow crown-like spiky top (#FFEE52).
   - Leaves: light green (#83DE7B) and mid green (#5A9C39) with dark teal-green veins (#19554D).
   - Outlines: crisp 1-2px near-black pixel-art outline (#101010), no anti-aliasing halo.
3. Back flower (1-Flower Rule, most important): exactly ONE very large pink flower sits on top of the back, wider than the body, made of big rounded pink petals with white streaks, spreading outward like an umbrella. In its centre stands a short brown palm-like trunk with a yellow spiky crown on top. Never two flowers on the back.
4. Leaves (2-Leaf Rule): exactly TWO large green fern-like leaves with jagged, saw-tooth edges, one sticking out to the left and one to the right, under the big flower, above the shoulders. Never more than two big leaves.
5. Forehead flower: exactly ONE small pink five-petalled flower on top of the head / forehead.
6. Legs (4-Leg Rule): exactly FOUR short, thick legs with white claws. No arms, no tail, no extra legs.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the head and face in front with both red eyes, forehead flower, the two front legs visible, the big back flower and the two leaves spread above and behind, one leaf on the left and one on the right.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: the head and mouth point to the left, one eye visible, all four legs seen from the side (two front legs on the left, two back legs on the right), the big flower on top of the back seen from the side, leaves sticking out forward and backward.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the teal back and rump, the head turned away to the upper right, the big flower on top and the two leaves to the sides.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: teal back and rump, two back legs, the big flower on top, one leaf on each side.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly (keep the brown trunk with the small yellow spiky crown in the flower centre small, as in the reference; the two leaves are broad jagged fern fronds sticking out sideways, not hanging grass; the mouth is closed) in design, colours and proportions: a heavy, low, teal blue-green four-legged plant creature with white claws, red eyes, one small pink flower on the forehead, exactly ONE huge pink flower on its back (pink petals with white streaks, a brown trunk with a yellow spiky crown in the centre) and exactly TWO large jagged green fern leaves under the flower, one on each side.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the head points diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the body extends back to the upper right, all four legs on the ground. Exactly four legs, one back flower, two big leaves. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
