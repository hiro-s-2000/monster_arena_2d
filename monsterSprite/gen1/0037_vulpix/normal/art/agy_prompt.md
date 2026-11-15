# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces left) and `ref_back.png` (back view, the creature faces away to the upper right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (small pixel sprites: a 3/4 front view and a back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, slender young fox standing on all four legs (quadruped), with a slightly large head, a short neck and a compact body. Never standing upright, never bipedal.
2. Colour palette:
   - Body, legs, head and ears: warm light reddish-brown fur (#E6946A) with mid shade (#BD735A) and dark shade (#835231).
   - Paws: dark brown (#835231 / #5A3100) on all four feet.
   - Chest: a small cream tuft (#FFE6B4 / #F6BD7B) on the chest only.
   - Head curls and tails: bright red-orange (#DE7329) with deep red (#BD4A29) and dark maroon curl lines (#732000).
   - Eyes: brown (#A47341) with a dark outline.
   - Outlines: crisp 1px dark pixel-art outline (#101010), no anti-aliasing halo.
3. Head (2-Ear Rule): exactly two large upright pointed triangular ears with dark brown inner ears (#835231); a short pointed muzzle with a small black nose; a big curly red-orange forelock of exactly three curls on top of the head between the ears.
4. Legs (4-Leg Rule): exactly four thin legs, two front and two back, each ending in one dark brown paw. Never more, never fewer.
5. Tails (6-Tail Rule, most important): exactly SIX red-orange tails, each ending in a curled spiral tip, bundled together into one big fluffy curly cluster that grows from the rump and rises up and back behind the body. The cluster is about as big as the body. Do not add more tails, do not merge them into one plain tail.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, both ears, the forelock and the cream chest visible, both front legs in front, the curly tail cluster visible behind the body rising above the back on both sides.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, muzzle pointing left, all four legs visible from the side (front legs on the left, back legs on the right), the tail cluster behind the rump on the right, rising up.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the back, the rump with the big curly tail cluster in front of us, the back of the head with both ears, the face turned away to the upper right.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the big curly tail cluster covers the rump and lower back, the back of the head and both ears visible above it, two back legs visible below.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small slender four-legged young fox with warm light reddish-brown fur, a cream chest tuft, two large pointed ears with dark brown insides, a curly red-orange forelock of three curls, four thin legs with dark brown paws, and SIX red-orange curl-tipped tails bundled into one big fluffy curly cluster rising behind the rump.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the body angled so the rump and the tail cluster are behind on the upper right. Exactly four legs and exactly six tails. Crisp 1px dark outline, limited palette, no anti-aliasing.
