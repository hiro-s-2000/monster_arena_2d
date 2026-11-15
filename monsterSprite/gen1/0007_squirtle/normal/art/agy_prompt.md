# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the lower right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, shell on its back, tail on the lower left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small young bipedal turtle standing upright on two short stubby legs, with a big round head (about as large as the torso), large eyes, a small smiling mouth, two short arms, and a rounded turtle shell on its back. No ears, no horns, no hair, no wings, no spikes.
2. Colour palette:
   - Skin (head, arms, legs, tail): light sky blue (#8BC5CD) with mid teal-blue shading (#5A9CA4), dark teal shadows (#297383) and a pale highlight (#B4E6EE).
   - Shell on the back: rich brown-orange (#BD6A00) with a darker brown hexagonal plate pattern (#622900) and a light tan rim (#D59452) running all around the edge of the shell. A thin pale grey-white band (#D5CDCD) separates the shell rim from the belly plate at the sides.
   - Belly plate (front of the shell): pale cream yellow (#FFE69C) with yellow shading (#FFD56A), divided into a few horizontal segments by thin orange-tan lines (#E6AC5A / #CD7B29).
   - Eyes: large eyes with a dark red-brown iris (#832900), a black pupil and one white highlight.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Limbs (exact counts): exactly TWO arms with small rounded hands (no claws), exactly TWO legs with small rounded feet. Never more than two arms or two legs.
4. Tail (1-Tail Rule, most important): exactly ONE short blue tail that grows from the bottom of the shell at the rump and curls up into ONE spiral curl at its tip. Never two tails. The tail is the same light blue as the skin.
5. Shell (1-Shell Rule): exactly ONE shell. From the back, the shell covers the whole back from the shoulders to the rump; from the front, only the cream belly plate and the shell's tan rim at the sides are visible.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the mouth, the cream belly plate visible, both arms held out a little to the left and right, both feet visible; the curled tail peeks out behind the body at the lower right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, face pointing to the left, the near arm reaching forward to the left, the cream belly plate on the left edge of the body, the brown shell on the right (back) side of the body; the single curled tail sticks out behind at the lower right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: we see the big brown shell with its hexagonal pattern and tan rim, the back of the head, the face turned away to the upper right (only the edge of the cheek visible), no belly; the single curled tail comes from the rump below the shell and curls out to the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the big brown shell with its hexagonal pattern and tan rim covering the back, the back of the head above it, both arms showing at the sides, both feet; the single curled tail comes out below the shell at the centre and curls to the right.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small light-blue bipedal turtle with a big round head, large red-brown eyes, a pale cream segmented belly plate, a brown shell with a tan rim on its back, two short arms, two stubby legs, and exactly ONE short blue tail ending in ONE spiral curl.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream belly plate visible, the shell rim visible behind the body on the right, the curled tail behind the body at the lower right. Exactly one tail, one shell, two arms and two legs. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
