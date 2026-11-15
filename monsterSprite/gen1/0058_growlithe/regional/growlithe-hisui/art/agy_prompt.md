# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the right) and `ref_back.png` (back 3/4 view, the creature faces right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, sturdy puppy standing on ALL FOUR legs (a quadruped dog, body horizontal, low stance). A big head covered by a thick fluffy fur cap, a short muscular torso, four short legs. No wings, no horns, no armour plates, no clothes.
2. Colour palette:
   - Torso and hind legs: brick red (#BD4139) with orange-red highlights (#D16845) and dark red-brown shadows (#7B3108).
   - Stripes: several short jagged black stripes (#101010) on the flanks, shoulders and hind legs, like a tiger; a band of dark grey stripes (#6A6A6A) along the top of the back.
   - Fur cap / mane: a big rounded fluffy pale grey fur mass (#D4D0C8, highlight #F0EBE2, shade #AD9D91, outline #594845) that covers the top and back of the head and the neck like a hood, with exactly ONE small pointed tuft sticking up on the top of the head.
   - Face: a pale tan muzzle (#EED3B4, shading #C4A17E) under the fur cap, small dark sleepy half-closed eyes, a small black nose.
   - Front legs: both front legs are wrapped in thick fluffy pale grey fur (#D4D0C8 / #AD9D91) down to big round grey paws with small swirl marks.
   - Belly: a small light tan patch (#C4A17E) under the torso.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO small ROUND ears on the sides of the head, brick red with a brown inside (#765638), sitting on the grey fur cap.
4. Legs (4-Leg Rule): exactly FOUR legs: two front legs with fluffy grey fur and big grey paws, two red hind legs with black stripes and red paws. Never more than four legs, never fewer.
5. Tail (1-Tail Rule, most important): exactly ONE big fluffy pale grey tail (#D4D0C8 / #AD9D91) that grows from the rump, stands up behind the body and curls into ONE spiral at its tip. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the fur cap with its single tuft, both round ears, both eyes, the tan muzzle, the two fluffy grey front legs side by side in front; the red body behind the head, the two red hind paws visible behind the front legs; the single tail rises behind the body and curls up on the right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the tan muzzle pointing to the left, the grey fur cap with its tuft, one round ear; the horizontal red striped body; two front legs with grey fur on the left, two red hind legs on the right; the single tail rises from the rump on the right and curls up.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: we see the red striped back with the grey stripe band and the back of the grey fur cap, the face turned away to the upper right; the single tail rises from the rump on the left and curls up.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the red striped back, the round back of the grey fur cap with the tuft, both round ears, the two red hind legs in front and the grey front paws peeking out; the single tail rises from the rump in the middle and curls up.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small sturdy puppy on all four legs, brick red body with black tiger stripes and a dark grey stripe band on the back, a big fluffy pale grey fur cap over the head with exactly one small pointed tuft, two small round red ears with brown insides, a pale tan muzzle with sleepy half-closed eyes, two front legs wrapped in fluffy grey fur with big grey paws, two red striped hind legs, and exactly ONE fluffy pale grey tail curling up into a spiral behind the body.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the grey front paws in front at the lower left, the red body behind going up to the right, the tail standing up above the rump on the right. Exactly four legs, two ears, one tail. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
