# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite, seen from behind). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

IMPORTANT: you MUST actually load and look at `ref_front.png` and `ref_back.png` and pass them as reference images to the image generator. The creature is a BUTTERFLY with broad pale lavender-white wings and a dark purple-navy body. It is NOT a bee or wasp: no yellow, no black-and-yellow stripes, no stinger, no spear arms, no six legs, no clear glassy wings.

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
0. Overall look: an open butterfly seen almost flat, the wide opaque pale lavender-white wings take up most of the sprite and the small dark body sits in the middle. NOT a bee: no yellow anywhere, no stripes, no stinger.
1. Body: a small flying butterfly-like insect hovering in the air (no feet touching the ground). A short round dark purple-navy body (thorax and a small round abdomen) with a round head on top.
2. Colour palette:
   - Body and head: dark purple-navy (#313152) with mid violet (#5A4A73) and violet highlight (#7362AC).
   - Wings: very pale lavender-white (#EEEEFF) with lavender shading (#BDBDE6) and periwinkle (#8383BD) near the body; thin dark navy vein lines (#313152 / #5A4A73) radiating from the body.
   - Eyes: two large round red compound eyes (#DE3131) with a pink highlight (#FF9CB4).
   - Hands: small light teal-blue (#6ABDCD). Feet: medium blue (#527BC5).
   - Mouth: a tiny small mouth with a light blue tooth-like fang below the eyes.
   - Outlines: crisp 1px near-black outline (#000000 / #313152), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one round head with exactly TWO huge round red eyes, one on each side of the face, and exactly TWO thin black antennae rising straight up from the top of the head, each ending in a small knob. Never one antenna, never three.
4. Wings (4-Wing Rule, most important): exactly FOUR wings — TWO large rounded upper wings (each with one dark navy crescent / comma-shaped mark near its upper outer part and dark radiating veins) and TWO small rounded lower wings below them. One upper and one lower wing on the left side, one upper and one lower wing on the right side, spread wide like an open butterfly. Never two wings, never six.
5. Limbs (2-Hand, 2-Foot Rule): exactly two tiny teal hands held at the chest and exactly two tiny blue feet hanging below the body. Do not add any other legs.

GRID LAYOUT (4 key poses, large and crisp, same size, all hovering at the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both big red eyes, both antennae, the dark body in the centre, the two teal hands and two blue feet, the four wings spread wide open to the left and right.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one big red eye, the face pointing left, both antennae pointing up and slightly forward, the wings raised up above and behind the back (seen edge-on-ish, upper wings tall above the body, lower wings behind), hands forward to the left, feet hanging down.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back of the head and the body, the face turned away to the upper right; the four wings spread up-left and up-right from the back, the near wings larger.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the back of the head, both antennae, the four wings spread wide to the left and right with their pale tops and dark veins, the feet just visible below.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a hovering butterfly-like insect with a dark purple-navy round body, TWO huge round red eyes, TWO thin black antennae, two tiny teal hands, two tiny blue feet, and exactly FOUR pale lavender-white wings (two large upper wings with a dark crescent mark and dark veins, two small lower wings) spread wide.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both red eyes visible in natural 3/4 perspective (not squished), the far wings partly behind the body, the near wings in front. Exactly four wings and exactly two antennae. Crisp 1px dark outline, limited palette, no anti-aliasing.
