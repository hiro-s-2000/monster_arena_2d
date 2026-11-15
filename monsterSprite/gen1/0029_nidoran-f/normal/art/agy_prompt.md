# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back 3/4 view, the creature faces right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, compact, rabbit-like four-legged creature standing on all four short legs. A round head with a short rounded muzzle, a plump rounded body. No horn, no wings, no arms, no armour.
2. Colour palette:
   - Whole body, head, legs, ears and tail: pale lavender blue (#D5D5FF) with very light highlights (#EEEEFF) and grey-lavender shading (#ACACCD) and slate blue shading (#7383AC).
   - Spots: a few small irregular dark indigo spots (#414183) scattered on the back and flanks.
   - Inner ears: deep teal (#189C94) with a dark steel-blue rim (#396A94) and dark indigo edge (#414183).
   - Eyes: small red eyes (#BD314A) with a tiny white glint.
   - Mouth: small open mouth, red inside (#BD314A, #FF5241), with two tiny white fangs on the lower jaw.
   - Whiskers: a few short dark whisker lines (#101010) on each cheek.
   - Claws: tiny white claws (#FFFFFF / #DEDEDE) on each foot.
   - Outlines: crisp 1-2px near-black outline (#101010) mixed with indigo (#414183), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO large, wide, rounded ears with a small spiky rim, each with a teal inside. Never more than two ears.
4. No horn (0-Horn Rule): the forehead is smooth; only a tiny pointed tuft between the ears. Never draw a horn.
5. Legs (4-Leg Rule): exactly FOUR short stubby legs (two front, two hind), each ending in a small paw with tiny white claws. All four stand on the ground. Never more than four legs.
6. Back: a few small pointed lavender bumps along the back of the head and neck.
7. Tail (1-Tail Rule): exactly ONE short pointed tail at the rump.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, the muzzle with the small open mouth, both big ears spread up to the left and right showing their teal insides, both front legs in front, the hind legs just visible behind.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, muzzle pointing to the left, both ears upright (the near one in front), all four legs visible (near pair in front, far pair slightly behind), the single short tail at the right end.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: we see the lavender back with the dark indigo spots, the back of the head with both ears, the face turned away to the upper right; the single short tail at the rump, toward the viewer.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the lavender back with spots, back of the head, both ears seen from behind, the two hind legs, the single short tail at the bottom centre.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small plump pale lavender-blue four-legged creature with exactly two large rounded ears with teal insides, no horn, small red eyes, a short muzzle with a small open red mouth and two tiny lower fangs, short dark whiskers, a few dark indigo spots, exactly four short legs with tiny white claws, and one short pointed tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), both ears up, the body extends back to the right, all four legs on the ground. Exactly two ears, no horn, four legs, one tail. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
