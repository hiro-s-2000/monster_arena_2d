# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back 3/4 view, the creature faces right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, compact, rabbit-like four-legged creature standing on all four short legs. A round head with a short muzzle, a stocky rounded body, a row of small sharp spines along the back. No wings, no arms, no armour.
2. Colour palette:
   - Whole body, head, legs, ears and tail: bright violet purple (#BD62C5) with light pink-lilac highlights (#E69CD5, #EEBDEE) and mid purple shading (#9C4AAC).
   - Spots: a few irregular dark purple spots (#62416A) scattered on the back and flanks.
   - Inner ears: deep teal green (#00A473) with dark green shading (#006241) and a light mint highlight (#18CD9C).
   - Eyes: small red eyes (#DE4129 with dark red #B41800) with a tiny white glint.
   - Claws and teeth: tiny white / light grey claws (#CDCDCD) on each foot and two tiny white teeth.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO very large, wide, upright ears, each with a jagged spiky back edge and teal-green inside. Never more than two ears.
4. Horn (1-Horn Rule): exactly ONE short pointed purple horn on the forehead between the ears, pointing up and forward. Never two horns.
5. Legs (4-Leg Rule): exactly FOUR short stubby legs (two front, two hind), each ending in a small paw with tiny white claws. All four stand on the ground. Never more than four legs.
6. Back spines: a row of small pointed purple spines runs from the back of the head down the spine to the rump.
7. Tail (1-Tail Rule): exactly ONE short pointed tail at the rump.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, the muzzle, the single horn on the forehead, both big ears spread up to the left and right showing their teal insides, both front legs in front, the hind legs just visible behind.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, muzzle pointing to the left, the horn pointing up-left from the forehead, both ears upright (the near one in front), the back spines along the top of the body, all four legs visible (near pair in front, far pair slightly behind), the single short tail at the right end.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: we see the purple back with the dark spots and the row of back spines, the back of the head with both ears (teal inside only on the right ear edge), the face turned away to the upper right; the single short tail at the rump in front, toward the viewer.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the purple back with spots and the row of spines down the middle, back of the head, both ears seen from behind, the two hind legs, the single short tail at the bottom centre.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small stocky violet purple four-legged creature with exactly two very large spiky-edged ears with teal-green insides, exactly one short horn on the forehead, small red eyes, a short muzzle, a row of small spines along the back, a few dark purple spots, exactly four short legs with tiny white claws, and one short pointed tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the horn on the forehead, both ears up, the body extends back to the right, all four legs on the ground. Exactly two ears, one horn, four legs, one tail. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
