# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the right) and `ref_back.png` (back 3/4 view, the creature faces right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large, proud, sturdy dog-like quadruped standing on FOUR legs, head held high. A big shaggy head with a short muzzle, a thick fluffy mane covering the head, neck and chest, a strong orange torso, four sturdy legs and one huge fluffy tail. No wings, no horns, no armour, no extra limbs.
2. Colour palette:
   - Mane, head fur, chest fur, tail and the fluffy tufts on the legs: sandy cream beige (#C6A573) with light cream highlights (#E7CE9C) and brown shading lines (#6B5229 / #634221).
   - Torso and legs: bright orange (#EF7331) with lighter orange highlights (#FF9C6B) and reddish-brown shading (#BD6B52).
   - Stripes: bold jagged black tiger stripes (#101010) across the orange back, flanks and legs.
   - Paws: reddish-brown (#BD6B52) with small toes.
   - Ears: inner ear orange (#EF7331).
   - Nose: small dark grey nose (#846B63); eyes: small dark eyes.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Head (2-Ear Rule): exactly ONE head with exactly TWO small pointed ears with orange insides, half hidden in the shaggy mane; spiky tufts of mane fur on top of the head.
4. Legs (4-Leg Rule): exactly FOUR legs — two front legs and two hind legs — each orange with black stripes and a cream fur tuft, each ending in one reddish-brown paw. Never more or fewer than four legs.
5. Tail (1-Tail Rule, most important): exactly ONE huge, bushy, cloud-like cream tail growing from the rump and curling up high behind the back. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the face with both eyes and the nose, the big cream chest mane, both front legs side by side in front, the hind legs barely visible behind; the single bushy tail rises behind the body above the back.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, muzzle pointing left, the mane on the left; the orange striped body stretches to the right; all four legs visible (near pair in front of the far pair); the single bushy tail rises from the rump on the right and curls up.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but turned: the orange striped back and rump in front, the back of the mane-covered head at the upper right, the face turned away; the single bushy tail on the left rising from the rump; the four paws on the ground.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the rump and the orange striped back, the single bushy tail rising straight up in the middle, the back of the head and both ears above it, the two hind legs in front and two front paws visible behind them.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a large proud dog-like quadruped with a thick shaggy sandy cream mane (#C6A573 / #E7CE9C) over the head, neck and chest, two small pointed ears with orange insides, a short muzzle with a dark nose, a bright orange (#EF7331) body with bold black tiger stripes, four sturdy orange striped legs with cream fur tufts and reddish-brown paws, and exactly ONE huge bushy cream tail curling up behind the back.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left, like the front reference: the head looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the chest mane in front, the body extending back to the upper right, the tail curling up above the rump on the right. Exactly one head, two ears, four legs and one tail. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
