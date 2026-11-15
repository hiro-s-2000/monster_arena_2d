# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite, seen from behind). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours. Do NOT draw the small loose floating green scale particles around it: only the creature itself, nothing floating separately.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a giant butterfly-like insect hovering in the air (no feet touching the ground). A small round dark purple body (thorax and a small round abdomen) with a round head on top. The body is small compared to the enormous wings.
2. Colour palette:
   - Body and head: dark purple (#313152) with mid violet (#5A4A73) and violet highlight (#7362AC); the belly has a light blue-grey oval patch (#9FBBBB) with a small teal-blue mark (#6ABDCD).
   - Wings: translucent glowing white (#FAFEFE) and pale mint (#B3F2DB) with soft sea-green areas (#8CCEBC) and bright light-green glowing patches (#92E996) along the outer upper parts; a crisp emerald green outline (#27A777) around every wing and thin light-mint vein lines radiating from the body. No dark marks on the wings.
   - Eyes: two large round red compound eyes (#DE3131) with a pink highlight (#FF9CB4).
   - Feet: two small blue feet (#527BC5) with light teal (#6ABDCD).
   - Outlines: body outline near-black (#101010 / #313152), wing outline emerald green (#27A777), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one round head with exactly TWO huge round red eyes, one on each side of the face, and exactly TWO thin black antennae rising from the top of the head, each ending in a light-green curled hook tip (#92E996). Never one antenna, never three.
4. Wings (4-Wing Rule, most important): exactly FOUR wings — TWO enormous, very tall, long oval upper wings that rise high up and outward in a wide V (each much taller than the body, glowing white-mint with light-green patches near the top), and TWO smaller rounded lower wings below the body, one on each side. One upper and one lower wing on the left side, one upper and one lower wing on the right side. Never two wings, never six.
5. Limbs (2-Foot Rule): exactly two tiny blue feet hanging below the body. Do not add any other legs, no hands, no tail.

GRID LAYOUT (4 key poses, large and crisp, same size, all hovering at the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both big red eyes, both antennae, the dark body in the centre with the light belly patch, the two blue feet, the two huge upper wings rising up-left and up-right in a wide V, the two lower wings below on the left and right.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one big red eye, the face pointing left, both antennae pointing up and slightly forward, the two huge upper wings raised tall above and behind the back (one in front of the other), the lower wings behind the body, feet hanging down.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back of the head and the body, the face turned away to the upper right; the four wings spread up-left and up-right from the back, the near wings larger.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the back of the head, both antennae, the light grey back of the body, the two huge upper wings rising in a wide V and the two lower wings below, all glowing white-mint with emerald outlines.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a hovering giant butterfly-like insect with a small dark purple round body, TWO huge round red eyes, TWO thin black antennae with light-green curled tips, two tiny blue feet, and exactly FOUR glowing white-mint wings with emerald green outlines and light-green patches (two enormous tall oval upper wings rising in a wide V, two smaller rounded lower wings). Do NOT draw any small loose floating particles around it.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both red eyes visible in natural 3/4 perspective (not squished), the far wings partly behind the body, the near wings in front. Exactly four wings and exactly two antennae. Crisp outline, limited palette, no anti-aliasing.
