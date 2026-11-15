# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art sprite, 3/4 front-side view, the creature faces left / lower-left) and `ref_back.png` (pixel-art sprite, back-side view, the creature faces right / upper-right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites (front-side view and back-side view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, sturdy puppy-like QUADRUPED (a young dog) standing on all four legs, chest forward, head held up, alert and playful.
2. Colour palette:
   - Body, legs and the top of the head: warm red-orange (#C55252) with bright orange patches (#EE8B4A, #FFB452) on the shoulders, face and paws.
   - Stripes: a few short black tiger stripes (#101010) on the back, flanks and legs, plus a dark grey (#6A6A6A) stripe on the back.
   - Fluffy fur: pale cream (#FFF6A4) with khaki shading (#DECD7B, #C5AC62, #736220): a fluffy cream mane tuft on top of the head, a big fluffy cream ruff on the chest and cheeks, and ONE big fluffy cream tail.
   - Muzzle: cream, small black nose. Eyes: dark with a white highlight.
   - Ears: two rounded ears, orange outside with a khaki-brown inner circle (#C5AC62 / #733910).
   - Outlines: crisp 1-2px near-black pixel-art outline (#101010), no anti-aliasing halo.
3. Legs (4-Leg Rule): exactly FOUR legs (two front legs, two hind legs), each ending in a small red-orange paw with tiny cream claws. Never more than four legs, never fewer.
4. Tail (1-Tail Rule): exactly ONE big round fluffy cream tail, curled up over the rump like a puff. Never two tails.
5. Head: exactly one head, two ears, one cream mane tuft on top. Do not add horns, wings or extra limbs.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the cream muzzle and the big cream chest ruff, both front legs in front, the hind legs mostly hidden behind the chest, the fluffy tail puff visible behind the body on one side, both ears up.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, muzzle pointing left, all four legs visible from the side (near legs in front, far legs slightly behind), the cream chest ruff at the front, the fluffy tail curled up at the right end over the rump.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the red-orange back with its black stripes, the back of the head with the cream mane tuft, the face turned away to the upper right; the fluffy tail puff in the foreground at the lower left over the rump; four legs.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: back of the head with the cream mane tuft and both ears, the red-orange back with black stripes, the fluffy cream tail puff in the centre over the rump, two hind legs in front with the front legs barely visible.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a sturdy red-orange puppy-like quadruped with black tiger stripes, a fluffy cream mane tuft on the head, a big fluffy cream chest ruff, two rounded ears, a cream muzzle with a black nose, exactly FOUR legs with red-orange paws, and exactly ONE big fluffy cream tail curled up over the rump.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the chest ruff toward the viewer, the body going back toward the upper right, the tail puff at the upper right. Exactly four legs and one tail. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
