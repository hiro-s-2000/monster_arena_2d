# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite, seen from behind and from the upper left, the head on the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (small pixel sprites: front 3/4 view and back view) into larger, crisp pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no shadows on the ground, no grid lines.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, stocky, four-legged poison rodent standing on all FOUR legs (NOT upright), with a big head, a short blunt snout and a short thick pointed tail. Gentle, slightly hunched stance.
2. Colour palette:
   - Body, head, legs and tail: pale powder blue (#8BC5CD) with very light icy highlights (#C5E6EE), steel-blue shading (#5A94B4) and dark teal-grey shadow (#395A62).
   - Spots: a few small dark blue-grey irregular spots (#395A62) on the back, flank and haunch.
   - Inner ears: deep blue (#4A83F6 centre, #1062B4 and navy #183973 rim).
   - Eyes: two red eyes (#C52010, light red #EE4A39) with a tiny white highlight.
   - Mouth: small, with exactly TWO tiny white fangs (#FFFFFF, grey #D5D5D5) from the lower jaw, pink-red inside (#EE4A39 / #FF9C8B).
   - Claws: small white/grey claws (#FFFFFF, #D5D5D5, #838383) on each paw: three tiny claws per paw.
   - Outlines: crisp 1-2px very dark outline (#101010) and dark teal (#395A62) inner lines, no anti-aliasing halo.
3. Legs (4-Leg Rule): exactly FOUR short sturdy legs, two front legs and two hind legs, each ending in a paw with small white claws. Never more, never fewer.
4. Ears (2-Ear Rule): exactly two very large, round, upright ears on top of the head, each with a deep blue inner ear and a jagged spiky outer rim (a few small points along the edge).
5. Spikes: a row of small, short pale-blue pointed spikes running down the back (on the head between the ears and along the spine). Small, NOT a single big horn: there is NO horn on the forehead.
6. Tail (1-Tail Rule): exactly ONE short, thick, pointed tail growing from the rump, slightly raised. Never two tails.
7. No wings, no horn, no clothing, no stripes.

GRID LAYOUT (4 key poses, large and crisp, same size, same scale, all paws on the same baseline, each creature centred in its quarter with white margin around it, ears and tail fully inside the quarter):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, both big ears up with the blue inner ears visible, the snout with the two small fangs, both front legs in front; the hind legs barely visible at the sides; the tail hidden or just peeking behind.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the snout and fangs to the left, both ears on top (one in front of the other), the head on the left, front legs under the head, hind legs on the right; far legs slightly visible behind the near legs; the back spikes along the top; the short tail pointing right from the rump.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back with spots and spikes in front, the back of the head with both ears at the upper right, the snout pointing away to the upper right (no eyes visible), hind legs and the tail in front at the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the back with the row of small spikes down the middle, the back of the head with both big ears above, two hind paws at the bottom (front paws barely visible), the short tail in the centre.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a stocky pale powder-blue (#8BC5CD) four-legged rodent with a few dark blue-grey spots, two very large round spiky-rimmed ears with deep blue inner ears, two red eyes, a small mouth with exactly TWO tiny white fangs, small spikes along the back, FOUR short legs with small white claws, and exactly ONE short thick pointed tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left (the same view as the front reference): the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the body extends behind to the upper right and the tail sticks out from the rump on the right. Exactly four legs, one tail, two ears, no horn. Crisp 1-2px dark (#101010) outline, limited palette, no anti-aliasing, no ground shadow, no text.
