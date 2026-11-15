# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite, seen from behind). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (small pixel sprites: front 3/4 view and back view) into larger, crisp pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no shadows on the ground, no grid lines.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, squat, chubby four-legged reptile-like creature (like a little frog-toad), low to the ground, with a big wide round head as large as its body, standing on FOUR short stubby legs.
2. Colour palette:
   - Skin (head, body, legs): blue-green teal (#62D5B4) with light mint highlights (#83EEC5), mid teal shading (#399494) and dark teal shadow (#317373 / #184A4A).
   - Skin spots: several darker teal (#399494 / #317373) irregular patches on the forehead, cheeks, back and legs.
   - Bulb on the back: one large round plant bulb (a closed leaf bud), yellow-green (#A4D541) with bright lime highlights (#BDFF73), mid green (#73AC31) and olive shadow (#526229); its top ends in a short pointed tip of folded leaves.
   - Eyes: two large eyes with red irises (#EE2039, dark red #AC0031), a white highlight (#FFFFFF), light grey (#CDCDCD) eye whites.
   - Mouth: a wide mouth, slightly open in a smile, with a pink-red inside (#FF6A62) and dark red edge (#AC0031).
   - Outlines: crisp 1-2px very dark outline (#101010), no anti-aliasing halo.
3. Legs (4-Leg Rule): exactly FOUR short thick legs, each ending in a round foot with three tiny white claws; two front legs and two hind legs. Never more, never fewer.
4. Ears (2-Ear Rule): exactly two small pointed triangular ears on the top corners of the head.
5. Bulb (1-Bulb Rule, most important): exactly ONE large green bulb sitting on the middle of the back, behind the head, bigger than the head when seen from behind. No leaves, no vines, no flowers, no extra bulbs. No tail.

GRID LAYOUT (4 key poses, large and crisp, same size, same scale, all feet on the same baseline, each creature centred in its quarter with white margin around it):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, both ears, the wide mouth and both front legs visible in front; the green bulb rises behind the head; the hind legs barely visible at the sides.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the snout and mouth to the left, the head on the left, the bulb on the back on the right half above the body, front legs under the head on the left, hind legs on the right; far legs slightly visible behind the near legs.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the big green bulb dominates the view, the back of the head with both ears peeks out at the upper right, the face is turned away to the upper right (no eyes or mouth visible), the hind legs in front.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the large round green bulb in the centre, the teal back and the rear of the body below it, the two ears peeking out at the top sides of the bulb, two hind legs at the bottom.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a squat chubby teal (#62D5B4) four-legged creature with darker teal spots, a big wide head with two small pointed ears, two large red eyes, a wide open smiling mouth with pink-red inside, FOUR short stubby legs with tiny white claws, and exactly ONE large yellow-green (#A4D541 / #BDFF73) plant bulb on its back.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the bulb sits on the back to the upper right of the head. Exactly four legs, one bulb, two ears, no tail. Crisp 1-2px dark (#101010) outline, limited palette, no anti-aliasing, no ground shadow, no text.
