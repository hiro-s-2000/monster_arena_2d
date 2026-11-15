# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, its curled tail on the lower right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, the spiral tail on the lower left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a plump, pear-shaped pink fairy creature standing upright on two short stubby legs. The head and the round body are one soft blob (no neck), wider at the hips. Small dark eyes, a small open smiling mouth, one curled lock (a single spiral curl) on the forehead. Two short stubby arms held out to the sides, each ending in a small rounded hand with tiny pointed fingers. No horns, no hair, no claws, no feathers.
2. Colour palette:
   - Face, front of the body, arms and legs: light peach pink (#FFC5B4) with salmon pink shading (#EE9494) and rose shadows (#D57B8B).
   - Back, top of the head and the tail: salmon pink (#EE9494) with darker mauve shading (#BD5A7B).
   - Forehead curl outline and mouth: coral red lines (#DE6262), the mouth inside red (#DE395A).
   - Ear tips: dark brown-grey (#524139) with grey-brown highlight (#7B6A62).
   - Wings: magenta pink (#D57BA4) with dark plum shading (#8B4A52 / #52314A).
   - Outlines: crisp 1-2px near-black outline (#101010) on the lower body, plum outline (#8B4A52) elsewhere, no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO large pointed ears that rise from the top corners of the head and point up and outward in a wide V, pink with dark brown-grey tips. Never more than two.
4. Wings (2-Wing Rule): exactly TWO small wings on the upper back, one on each side behind the head, each wing a small fan of three short pointed magenta feathers sticking out sideways. Never four wings, never large wings.
5. Limbs (exact counts): exactly TWO short arms and exactly TWO short stubby legs. Never more.
6. Tail (1-Tail Rule): exactly ONE short pink tail on the rump that curls up into ONE flat spiral (like a snail shell drawn with a dark plum spiral line), seen big on the back views and only as a curved edge behind the body on the front views. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the smiling mouth, the forehead curl, both ears up in a V, both wings peeking out behind the head on the left and right, both arms held out a little to the left and right, both feet visible.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the face pointing to the left, the near arm reaching forward to the left, both ears up (one slightly behind the other), one wing visible on the back (right side), the single spiral tail on the rump at the lower right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: the salmon-pink back, both wings on the upper back, both ears, the face turned away to the upper right (no eyes visible), the single spiral tail big on the rump at the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the salmon-pink back and the back of the head, both ears up in a V, both wings spread to the left and right of the upper back, both feet; the single spiral tail in the centre of the rump.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a plump pear-shaped light-pink fairy creature with two big pointed ears with dark brown-grey tips, one curl on the forehead, small dark eyes and a small open smile, two small magenta three-feather wings behind the head, two short arms held out, two stubby legs, and exactly ONE spiral-curled tail behind the body at the lower right.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), both ears up, the near arm held out to the left. Exactly two ears, two wings, two arms, two legs and one tail. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
