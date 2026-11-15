# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (a 3D render: front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, chubby bipedal electric rodent standing upright on two short legs, small arms held loosely in front of the chest, a small fluffy tuft of fur on top of the head.
2. Colour palette:
   - Body, head, arms and legs: golden orange-yellow (#F0B020) with light yellow highlight (#F8D040) and shadow ochre (#C88818).
   - Belly: an oval white patch (#F4F4F0) with light grey shading.
   - Eyes: blue (#3060C0) with white highlights. Cheeks: small yellow cheek pouches.
   - Back: three thin horizontal brown stripes (#7A5A3A) across the lower back.
   - Outlines: crisp 1-2px dark pixel-art outline (#2A2026), no anti-aliasing halo.
3. Head (Ear Rule): two short pointed ears with BLACK tips on top of the head; and on each side of the head ONE very large flat lightning-bolt shaped ear-wing / crest, bright yellow (#F8E040) with a light brown inner edge, each ending in a small spiral curl at its lower tip. Exactly two crests, one on the left side and one on the right side of the head, spreading up and outward like wings.
4. Tail (Tail Rule, most important): exactly ONE long thin dark brown whip-like tail (#6A5040) that grows from the rump, curls on the ground behind the feet and ends in ONE huge flat bright-yellow lightning-bolt shaped blade (#F8E040), bigger than the body, standing up behind the creature. Never two blades.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric face: both blue eyes, white belly visible, both ear-crests spread left and right, the tail blade standing up behind the body on the right side.
- Top-Right: WEST (90°, left side profile) — facing LEFT in a sharp side profile: one eye, snout to the left, arms forward to the left, the thin tail goes back to the right and the big tail blade stands behind the body on the right side.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the back with the brown stripes and the back of the head, face turned away to the upper right; both ear-crests and the tail blade visible.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, like the back reference: back stripes, back of the head with both black-tipped ears and both ear-crests, the thin tail curling at the feet and the big tail blade standing on the left side.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image (a 3D render) exactly in design, colours and pose: a chubby golden-yellow electric rodent with a white oval belly, blue eyes, a fluffy head tuft, two short black-tipped ears, ONE very large yellow lightning-bolt ear-crest on each side of the head ending in a spiral curl, small arms in front of the chest, and ONE long thin dark brown tail that ends in ONE huge yellow lightning-bolt blade standing behind the body on the right.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Crisp 1-2px dark outline, limited palette, no anti-aliasing. The whole creature, crests and tail blade fit inside the image with a margin.
