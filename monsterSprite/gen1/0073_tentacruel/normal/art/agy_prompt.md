# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel-art sprites: front view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large floating jellyfish. It hovers in the air; there are no legs, no feet, no arms, no hands, no fins, no tail. A wide dome-shaped cap on top, a dark shadowed underside below it, and a curtain of tentacles hanging down from under the cap.
2. Colour palette (use only these, no new hues):
   - Cap: sky blue (#6AB4E6) with light highlights (#83CDE6), teal-blue shading (#4194AC) and a dark navy outline / seams (#205283).
   - Red orbs and gem: red (#B42920), dark red (#831010), bright red highlight (#F64152), pale pink-white shine (#FFBDAC, #FFFFFF).
   - Thin tentacles: grey-violet (#9C94A4), mid grey-violet (#736A7B), dark grey (#52525A), light grey tips (#BDBDBD).
   - Underside under the cap and the main outline: near-black (#101010).
   - Eyes: small white eyes (#FFFFFF) with a grey ring (#BDBDBD) and a dark pupil.
3. Cap (2-Orb Rule): the dome cap has exactly TWO large glossy red round orbs set into its top, one on the left half and one on the right half, separated by a blue ridge running front to back. Exactly ONE small red diamond-shaped gem sits on the front of the cap between the two orbs, low above the eyes (front views only). The cap rim flares out into exactly two short pointed side flaps (one left, one right), and a wavy hem at the front.
4. Eyes (2-Eye Rule): exactly two small eyes just under the front rim of the cap, in the dark underside. Not visible from the back.
5. Tentacles (most important): exactly TWO thick blue tentacles (sky blue #6AB4E6 with teal shading and a navy outline), hanging down from the front of the underside like a pair of long tapering blades, reaching the lowest point of the body. Around and behind them, a bundle of about TEN thin grey-violet tentacles with light grey rounded tips, hanging down and curling outward to the left and right. Never more than two blue tentacles; the grey tentacles stay thin and grey.

GRID LAYOUT (4 key poses, large and crisp, same size, all hovering at the same height, lowest tentacle tips on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the cap with both red orbs on top, the small red gem in the middle front, both eyes under the rim, the two blue tentacles hanging straight down in the centre, the grey tentacles spreading down to the left and to the right.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: the cap seen from the side with the near red orb large and the far orb partly visible behind the ridge, one eye under the front (left) rim, the two blue tentacles hanging at the front (left), the grey tentacles trailing down and to the right behind.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back of the cap with both red orbs, no eyes and no gem visible, the face turned away to the upper right; the grey tentacles hang down in front of the viewer, the blue tentacles mostly hidden behind them.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the back of the cap with both red orbs, the dark underside edge, the grey tentacles hanging down, parts of the two blue tentacles visible between them.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a floating jellyfish with a wide sky-blue dome cap carrying exactly two large glossy red orbs and one small red diamond gem at the front, two short pointed side flaps on the rim, two small white eyes under the front rim in a dark underside, exactly two thick blue blade-like tentacles hanging from the front, and about ten thin grey-violet tentacles with light grey tips curling down and outward. No legs, no arms.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the gem toward the left of the cap front, the two blue tentacles hanging down at the front-left, the grey tentacles spreading down to the left and right like the reference. Exactly two red orbs and two blue tentacles. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
