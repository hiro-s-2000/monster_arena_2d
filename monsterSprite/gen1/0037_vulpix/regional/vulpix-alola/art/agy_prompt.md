# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front 3/4 view, the creature faces left, tails on the right) and `ref_back.png` (back 3/4 view, the creature faces up-right away from the viewer, tails on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, slender young snow fox standing on four thin legs (a quadruped, never upright). A rounded head with a short pointed snout, a slim body, four thin straight legs with small pale-blue paws. No wings, no horns, no arms.
2. Colour palette:
   - Head, chest and legs: snow white (#FDFDFD) with pale icy grey-blue shading (#D9E9F4, #C1D1D2).
   - Back and flanks: pale grey-blue (#C1D1D2) with darker blue-grey shading (#97BDD2).
   - Paws: light ice blue (#97BDD2 / #6CB1DB) at the bottom of each leg.
   - Eyes: royal blue eyes (#2D57BB) with a lighter blue shine (#6CB1DB); a tiny blue nose tip (#2D57BB).
   - Ears: inner ear deep royal blue (#2D57BB).
   - Outlines: crisp 1-2px outline, near-black (#151515) on the lower body and legs, dark slate grey-blue (#648082) around the fluffy curls; no anti-aliasing halo.
3. Head (1-Head Rule): exactly ONE head with exactly TWO pointed ears (blue inside), and on top of the forehead ONE big fluffy cloud-like puff of white curly fur (a cluster of 3-4 round curls, pale blue shading inside).
4. Legs (4-Leg Rule): exactly FOUR thin legs — two front legs and two hind legs — each ending in ONE small pale-blue paw. Never more than four legs.
5. Tails (most important): a small number of soft fluffy curly tails (about six) that are BUNDLED TOGETHER into ONE large round cloud-like puff of white curly fur at the rump, with pale ice-blue shading (#97BDD2, #D9E9F4) inside the curls and a dark slate outline (#648082). It reads as one big fluffy cloud behind the body, about as large as the head. No flames, no red or orange anywhere.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both blue eyes, the snout and nose, both ears, the fluffy head puff, the white chest, both front legs in front and both hind legs behind them; the fluffy tail cloud shows behind the body above the hips on both sides.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile, like the front reference but fully sideways: one eye, snout pointing to the left, the near front leg and near hind leg in front of the far legs (all four legs visible), the fluffy tail cloud behind the body to the right, rising above the rump.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right (the mirror image of the back reference): we see the pale grey-blue back, the back of the head with both ears and the head puff, the face turned away to the upper right, no eyes; the big fluffy tail cloud is in the foreground at the lower left, partly covering the hind legs; all four legs visible.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the big fluffy tail cloud in the centre covering the rump, the back of the head with both ears and the head puff above it, the two hind legs below the tail cloud and the two front legs just visible beside them.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small slender snow-white young fox standing on exactly four thin legs with pale-blue paws, a rounded head with a short snout, blue eyes, a tiny blue nose, exactly two pointed ears with royal blue insides, one big fluffy white curly puff of fur on the forehead, pale grey-blue back, and one big round cloud of bundled fluffy white curly tails at the rump with ice-blue shading.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the white chest visible, the tail cloud behind the body to the right above the rump. Exactly one head, two ears, four legs, one tail cloud. Crisp 1-2px dark outline, limited palette, no anti-aliasing, no red or orange.
