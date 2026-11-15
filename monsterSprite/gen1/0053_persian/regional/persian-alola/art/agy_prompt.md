# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a tall, slender, elegant cat standing on all four long thin legs, head held high on a slim neck. A rather small rounded head with a short muzzle, a long lean torso, a narrow chest. No wings, no horns, no armour, no clothes.
2. Colour palette:
   - Head, chest, front legs and lower body: slate blue-grey (#6C7791) with darker shading (#464457 / #45505F).
   - Face, top of the back, flank highlight and tail: light silver blue-grey (#9BA8B7).
   - Inner ears: dark teal grey (#627986).
   - Forehead gem: exactly ONE small round sapphire-blue jewel in the middle of the forehead (#1784D5 with dark blue #1149A8 and a bright cyan glint #11B8F7).
   - Eyes: two narrow sleepy eyes, white (#FDFDFD) with a dark pupil, under heavy dark grey brows.
   - Teeth: two tiny white fangs (#FDFDFD) showing at the bottom of the closed mouth.
   - Outlines: crisp 1-2px near-black outline (#101010) with dark navy (#262B3C) inner lines, no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO short rounded-triangular cat ears on top of the head, dark teal grey inside.
4. Whiskers: exactly THREE long thin dark whiskers on each side of the face (six in total), sticking out sideways from the cheeks.
5. Legs (4-Leg Rule): exactly FOUR long thin legs (two front, two hind), each ending in one small rounded paw with tiny toes. Never more than four legs, never three.
6. Tail (1-Tail Rule, most important): exactly ONE long thin light silver tail that grows from the rump, rises up behind the body and ends in ONE tight round spiral curl. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, paws on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the forehead jewel, the two fangs and all six whiskers visible, both front legs straight in front, the two hind legs just visible behind them; the single tail rises behind the body and curls on the right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, muzzle pointing left, the whiskers of the near side, the long torso horizontal, all four legs visible (near front and near hind leg in front, far legs slightly behind), the single tail behind the rump on the right, rising up and ending in the spiral curl.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: the back of the head with both ears, the face turned away to the upper right (only whisker tips visible), the silver back and rump, all four legs; the single tail comes from the rump and curls up to the left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the back of the head, both ears, whisker tips on both sides, the rump with both hind legs, both front legs just visible; the single tail comes out of the rump and rises straight up, ending in the spiral curl.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a tall slender slate blue-grey cat standing on four long thin legs, a small rounded head held high with two short ears, one small round sapphire-blue jewel on the forehead, narrow sleepy eyes under heavy brows, two tiny white fangs, three long dark whiskers on each side, a light silver face, back and tail, and exactly ONE long thin tail rising behind the body ending in a tight spiral curl.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the chest and front legs toward the viewer, the body going back to the upper right, the tail behind the rump on the right curling up. Exactly four legs, two ears, one tail, one forehead jewel. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
