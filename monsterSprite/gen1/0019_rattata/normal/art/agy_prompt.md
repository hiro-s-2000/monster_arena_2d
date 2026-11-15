# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite, seen from behind and from the upper right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (small pixel sprites: front 3/4 view and back view) into larger, crisp pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no shadows on the ground, no grid lines.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, lean, agile four-legged rat, crouched low in a ready-to-pounce stance, with a large head, a pointed snout and a long thin tail. It stands on all FOUR legs (it is NOT standing upright).
2. Colour palette:
   - Fur (head, back, legs, tail): light violet-purple (#B473BD) with pale lilac highlights (#D59CD5), mid purple shading (#8B4A8B) and dark plum shadow (#4A2941).
   - Muzzle, cheeks and belly: cream (#EEDEB4) with tan shading (#CDAC62).
   - Paws: tan-yellow (#CDAC62 / #E6CD73) with ochre (#A47308) and dark brown (#624A08) shading; small pink pads (#E65A73) under the hind paws.
   - Eyes: two red eyes (#A41839, light red #E65A73) with a tiny white highlight.
   - Mouth: open, pink-red inside (#E65A73), with exactly TWO big white front incisors (#FFFFFF, grey #CDCDCD) pointing down from the upper jaw.
   - Whiskers: a few thin tan whiskers (#CDAC62) sticking out sideways from each cheek.
   - Outlines: crisp 1-2px very dark outline (#101010), no anti-aliasing halo.
3. Legs (4-Leg Rule): exactly FOUR thin legs, two front legs and two hind legs, each ending in a tan paw with three small toes. Never more, never fewer.
4. Ears (2-Ear Rule): exactly two large round ears on top of the head, purple with a cream-tan inner ear.
5. Tail (1-Tail Rule, most important): exactly ONE long, thin purple tail growing from the rump, rising up high above the back and ending in a single round curl at the tip (curling forward). Never two tails, never a fluffy tail.
6. No wings, no horns, no stripes on the back, no clothing.

GRID LAYOUT (4 key poses, large and crisp, same size, same scale, all paws on the same baseline, each creature centred in its quarter with white margin around it, the tail fully inside the quarter):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, both round ears, the cream muzzle with the two white incisors and the whiskers, both front legs in front; the hind legs barely visible at the sides; the tail rises straight up behind the body with its curl at the top.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, one ear, the pointed snout with the incisors to the left, the head on the left, the body behind it, front legs under the head on the left, hind legs on the right; far legs slightly visible behind the near legs; the tail rises from the rump on the right and curls forward over the back.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: the purple back and rump in front, the back of the head with both ears at the upper right, the snout tip pointing away to the upper right (no eyes visible), the hind legs in front; the tail rises from the rump up at the left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the purple back and rump, the back of the head with both round ears above, two hind paws at the bottom (front paws barely visible), the tail rising straight up from the rump in the centre with its curl at the top.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a lean violet-purple (#B473BD) four-legged rat crouched low, with a cream (#EEDEB4) muzzle and belly, two large round ears, two red eyes, an open mouth with exactly TWO big white incisors, thin tan whiskers, FOUR thin legs with tan paws, and exactly ONE long thin purple tail rising high from the rump and ending in a round curl.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left (the same view as the front reference): the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the body extends behind to the upper right and the tail rises above the rump on the right. Exactly four legs, one tail, two ears, two incisors. Crisp 1-2px dark (#101010) outline, limited palette, no anti-aliasing, no ground shadow, no text.
