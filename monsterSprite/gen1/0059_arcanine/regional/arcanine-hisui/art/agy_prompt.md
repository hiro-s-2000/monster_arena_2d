# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, seen from behind and to the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel sprites: front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large, powerful wolf-like dog standing on FOUR legs (quadruped), chest high, head held up. Rugged, rocky, battle-worn look.
2. Colour palette:
   - Fur (mane, head, tail, chest fur, leg cuffs): slate grey (#576460) with light grey-green highlights (#828E85) and dark shadow (#242F2F).
   - Body and legs under the fur: brick red (#AA2822) with bright red-orange (#E84937) and salmon highlights (#FA7861), crossed by thick black-grey tiger stripes (#242F2F).
   - Paws: red (#AA2822 / #E84937) with dark lines between the toes.
   - Outlines: crisp 1-2px near-black pixel-art outline (#101010), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with a long grey muzzle, a black nose, one jagged pointed tuft of fur standing straight up on top of the head, exactly two small ears; a red mask-like marking around one eye and a small red four-pointed star mark on the forehead; a red-brown spike (#620D00 / #AA2822) sticking out beside the face.
4. Mane: a big shaggy grey mane hangs from the neck over the chest like a jagged cape of pointed tufts, with a red diamond mark in its centre and a few red streaks.
5. Legs (4-Leg Rule): exactly FOUR legs — two front legs and two hind legs, all red with black stripes and red paws; the front legs have grey fur cuffs. Never three legs, never five.
6. Tail (1-Tail Rule, important): exactly ONE very large, bushy grey tail made of rounded cloud-like puffs with swirl lines, rising up and back from the rump like a big plume, as tall as the head. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, the whole body inside each cell):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the muzzle pointing at us, the mane over the chest with its red diamond, both front legs in front, the hind legs partly visible behind, the big tail plume rising behind the body.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, muzzle to the left, all four legs visible (two front, two hind, near legs in front of far legs), the mane hanging from the neck, the big tail plume rising from the rump on the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the back with the black stripes, the back of the mane and head, the face turned away to the upper right; the tail plume rises from the rump toward the viewer's lower left; four legs.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the rump with the tail plume rising in the middle, both hind legs, the back of the head and the mane above, both ears.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a large grey-maned wolf-like dog on FOUR red black-striped legs, a shaggy grey mane cape over the chest with a red diamond mark, a long grey muzzle, one jagged fur tuft on top of the head, a red mask marking around one eye, and exactly ONE huge bushy grey cloud-puff tail rising behind the body.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Exactly four legs, exactly one tail. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
