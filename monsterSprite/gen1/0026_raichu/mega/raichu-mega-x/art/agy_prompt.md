# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (a 3D render: front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, chubby, round-bellied bipedal electric rodent standing upright on two short legs, fists raised in a boxing guard.
2. Colour palette:
   - Body, head, arms and legs: bright lemon yellow (#F8E030) with mid yellow (#E8C020) and shadow ochre (#C89818).
   - Belly: white (#F4F4F0) with light grey shading.
   - Hands (boxing-glove-like fists) and feet: reddish brown (#A05A3A).
   - Cheeks: round red cheek pouches (#E8303A).
   - Back: two horizontal brown stripes (#A0603A) across the middle of the back.
   - Outlines: crisp 1-2px dark pixel-art outline (#2A2026), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly two long ears, yellow with black tips, curving up and outward like horns; a small dark curl ornament sits beside one ear.
4. Tails (2-Tail Rule, most important): exactly TWO very large flat lightning-bolt-shaped tails, golden yellow blending to orange (#F8E030 -> #F0A030), each with a dark grey jagged base (#505058). Both tails grow from the lower back / rump and spread up and outward in a wide V, one on each side of the body. Both tails are clearly visible in every view, each is bigger than the body. Never one tail, never three.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, both red cheeks and the white belly visible, fists raised, both ears up, the two tails spread behind the body to the left and to the right.
- Top-Right: EAST (90°, right side profile) — facing right in a sharp side profile: one eye, one cheek, snout to the right, fists forward to the right, both tails behind the body on the left side, one slightly in front of the other, both rising from the rump.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the back with the brown stripes and the back of the head, the face turned away to the upper right; both tails rise from the rump and spread up-left and up-right.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: back stripes, back of the head, both ears, both tails rising from the rump in a wide V.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image (a 3D render) exactly in design, colours and pose: a chubby yellow electric rodent with a white belly, red cheeks, brown boxing-glove fists in a guard, two long black-tipped ears, and TWO very large flat lightning-bolt tails (yellow to orange with dark grey jagged bases) spreading behind the body to the left and to the right.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the near (right) cheek fully visible. Exactly two tails, both clearly visible. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
