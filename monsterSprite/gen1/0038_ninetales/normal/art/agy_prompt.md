# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel sprites: front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: an elegant, slender four-legged fox standing on all four legs (not sitting), long neck held high, head up, calm and graceful.
2. Colour palette:
   - Fur of the body, head, legs and tails: golden cream (#E6C552) with light pale-yellow highlights (#F6E67B) and ochre shadow (#CD9C4A).
   - Tail tips: orange (#EE8329) with light orange (#FFA439).
   - Eyes: red (#DE2010), narrow and calm.
   - Outlines: crisp 1-2px dark brown pixel-art outline (#836218) with black (#101010) on the outer edge, no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with a pointed fox snout, exactly two pointed upright ears, and a tall flowing crest of fur on top of the head (about five long spikes swept up and back). A fluffy fur ruff on the chest and neck.
4. Legs (4-Leg Rule): exactly four slender legs with small paws, all four standing on the ground. Never more, never fewer.
5. Tails (9-Tail Rule, most important): exactly NINE long, thick, fluffy tails, all growing from one point at the rump and fanning out up and back behind the body like a big fan, each tail ending in an orange tip. The tail fan is about as big as the body. Do not add or remove tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: face, both red eyes, both ears, the head crest and the chest ruff visible, both front legs in front, the nine tails fanned out behind the body to the left and to the right above the back.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, snout to the left, four legs seen from the side (two near, two far), the nine tails fanning up and back behind the body on the right side.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the head turned away to the upper right, the back of the head and crest, the nine tails fanned out toward the viewer covering much of the back.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the nine tails fanned out in a big symmetric fan in front of us, the back of the head with the two ears and crest peeking above them, the hind paws below.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design and colours: an elegant slender golden-cream fox with red eyes, two pointed ears, a tall flowing crest of fur on the head, a fluffy chest ruff, four slender legs, and exactly NINE long fluffy tails with orange tips fanning up and back behind the body.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left, standing on all four legs (not sitting): the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the tail fan behind the body to the right. Exactly four legs, exactly nine tails. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
