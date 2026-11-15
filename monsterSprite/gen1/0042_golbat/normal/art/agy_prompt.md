# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art front view: the creature faces the viewer, hovering with both wings raised high, its huge mouth wide open, two short legs dangling below) and `ref_back.png` (pixel-art back view: seen from behind, plain blue back, both wings spread up and out, the two legs dangling below). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a bat-like flying creature HOVERING in the air. Its body is one plump rounded egg-shaped blue blob (no separate neck, no arms): the head and the body are the same single shape. Most of the front of the body is one ENORMOUS gaping open mouth. It does not stand on the ground: its two short legs hang below the body.
2. Colour palette:
   - Body, legs, wing arm-bones: medium blue (#6283CD) with sky blue highlights (#5AACEE), pale blue light edges (#ACCDFF) and dark navy shadow (#39397B).
   - Wing membranes: purple (#AC52AC) with pink-mauve light panels (#CD6AB4), dark plum vein lines (#94397B) and a deep plum edge (#621052).
   - Mouth interior: black (#000000) with a dark grey tongue / throat shade (#626262).
   - Fangs: white (#FFFFFF) with light grey (#D5D5D5).
   - Inner ears: small plum-pink (#94397B).
   - Outlines: crisp 1px black outline (#000000), no anti-aliasing halo.
3. Head (1-Mouth Rule): exactly two small pointed triangular ears on top of the body, two small narrow eyes just above the mouth, and ONE huge wide-open mouth that fills most of the front of the body, black inside, with exactly two small white fangs hanging from the upper lip and two tiny white fangs at the bottom. No nose, no beak, no horns.
4. Wings (2-Wing Rule): exactly TWO very large bat wings growing from the upper sides of the body, raised high and spread wide: each wing has a thin blue arm-bone along its top edge that bends at a wrist point, and a purple membrane with jagged scalloped lower edges between dark plum vein lines. Never one wing, never three or four. No hands, no arms besides the wing bones.
5. Legs (2-Leg Rule): exactly two short thin blue legs hanging below the body, each ending in a small foot with exactly two toes. No tail. No extra limbs.

GRID LAYOUT (4 key poses, large and crisp, same size, all four hovering at the same height, lowest point (the feet) on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric like the front reference: both ears, both eyes, the huge black open mouth with its white fangs in the centre, both wings raised up and out to the left and to the right, both legs dangling below.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one ear, one eye, the huge open mouth seen from the side gaping to the left (upper and lower lips, fangs pointing down), the near wing raised up and back over the body (the far wing just visible behind it), both legs dangling below.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the plain blue back of the body, both ears seen from behind, the mouth hidden (face turned away to the upper right); both wings raised and spread up-left and up-right, legs dangling below.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: plain blue rounded back with both ears on top, no mouth visible, both wings spread up and out to the left and to the right, both legs dangling below.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a hovering plump blue egg-shaped bat creature whose front is one ENORMOUS open black mouth with two small white upper fangs and two tiny lower fangs, two small pointed ears, two small narrow eyes, TWO very large purple bat wings with blue arm-bone edges raised high, and TWO short blue legs with two-toed feet dangling below. No tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face and the huge open mouth look diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Exactly two wings, exactly two ears, exactly two legs and exactly one mouth, all clearly visible. Crisp 1px black outline, limited palette, no anti-aliasing.
