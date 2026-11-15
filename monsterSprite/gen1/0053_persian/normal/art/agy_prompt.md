# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature's head faces the viewer and its body points to the lower-left) and `ref_back.png` (back 3/4 view, the creature faces up-right away from the viewer). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (a small pixel sprite: front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a tall, slender, elegant cat standing on ALL FOUR long thin legs (a quadruped, never standing upright). Long lean torso held level, small round head on a short neck, long thin tail.
2. Colour palette (use only these):
   - Fur (head, body, legs, tail): sandy cream (#DEB46A) with pale yellow highlights (#FFE683) on the back, the top of the head and the upper tail, darker tan shading (#B48329) and brown edge lines (#835200).
   - Inside of the ears: olive brown (#947B20 / #BDA44A).
   - Forehead jewel: exactly ONE small round red gem (#F65241 with dark red #A41000 rim and a tiny white highlight) in the middle of the forehead.
   - Eyes: narrow sly eyes, white (#FFFFFF) with small red pupils (#A41000).
   - Outlines: crisp 1px near-black pixel-art outline (#101010), no anti-aliasing halo.
3. Head (counting rules, never add more): exactly TWO pointed triangular ears, one on each side of the top of the head; exactly ONE red forehead jewel; exactly SIX long thin whiskers, THREE on each side of the face, cream with brown edges, sticking straight out sideways; a small muzzle with a tiny mouth.
4. Legs (4-Leg Rule): exactly FOUR long thin straight legs with small rounded paws, two front legs and two hind legs. Never more, never fewer. No arms, no hands.
5. Tail (1-Tail Rule): exactly ONE long thin tail rising from the rump, curving up and back and ending in ONE tight round spiral curl at the tip. Never two tails.
6. No clothes, no accessories, no extra gems, no collar, no wings, no stripes or spots.

GRID LAYOUT (4 key poses, large and crisp, same size and scale, paws on the same baseline):
- Top-Left: SOUTH (0°, front view) — the cat faces straight toward the viewer, symmetric: face with both eyes, the jewel and all six whiskers, both ears up, the chest and both front legs in front, the two hind legs visible behind and slightly apart, the tail rising behind the body with its spiral curl visible above one hip.
- Top-Right: WEST (270°, left side profile) — a sharp side profile facing LEFT: head and muzzle to the left, one eye, whiskers pointing left and right of the muzzle, long level body, the four legs seen from the side (near pair and far pair, far legs slightly darker), the tail rising from the rump on the right and ending in the spiral curl.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, walking away toward the upper right: we see the back of the body and the back of the head with both ears, whisker tips peeking out beside the head, the four legs, the tail rising from the rump on the lower left with the spiral curl.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the back of the head with both ears, whisker tips on both sides, the rump with the two hind legs in front and the two front legs behind them, the tail rising straight up from the rump with the spiral curl on top.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a tall slender sandy-cream cat (#DEB46A, highlights #FFE683, brown edges #835200) standing on exactly four long thin legs, a small round head with exactly two pointed ears (olive insides), exactly one round red forehead jewel, narrow sly eyes, exactly six whiskers (three per side), and exactly one long thin tail ending in one tight spiral curl.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the body points diagonally down-left, the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the tail rises from the rump at the upper right with its spiral curl. Exactly four legs, exactly one tail. Crisp 1px near-black outline (#101010), limited palette, no anti-aliasing.
