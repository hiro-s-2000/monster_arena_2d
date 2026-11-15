# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front view, the creature faces the viewer, head slightly to the lower-left) and `ref_back.png` (back view, seen from behind, head at the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (a small pixel sprite: front view and back view) into larger, crisp, detailed pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a big, heavy, low, wide quadruped like a squat toad / dinosaur, standing on FOUR short thick legs, belly close to the ground, broad flat head with a wide closed mouth. A huge flower grows on its back.
2. Colour palette:
   - Skin (body, head, legs): teal (#20B49C) with light teal highlights (#5AD5C5) and dark teal shading (#107B6A, #105241).
   - Eyes: two narrow angry eyes, red-brown iris (#DE4141) with a white glint.
   - Ears: exactly two small triangular ears on top of the head, teal outside with a red inside (#DE4141).
   - Claws: each foot has exactly three small white claws (#FFFFFF).
   - Teal skin carries a few darker teal spots (#107B6A) on the back and flanks.
   - Leaves: large fern-like palm fronds, light green (#83DE7B) with mid green (#5A9C39) and dark green veins / serrated edges (#105241).
   - Flower petals: coral pink (#FF7B73) with pale pink highlights (#FFBDBD), red shading (#DE4141) and dark red-brown edges (#833100).
   - Flower centre: a yellow crown-like pistil (#FFEE52 with #DEBD29) on top.
   - Trunk: a short brown bark trunk (#BD6A31 with #833100 scales) under the flower centre.
   - Outlines: crisp 1-2px near-black pixel-art outline (#101010), no anti-aliasing halo.
3. Flower (Flower Rule, most important): exactly ONE huge flower on the middle of the back with exactly FIVE big rounded petals spreading out like an umbrella, wider than the body, held above the back by the short brown trunk, the yellow crown in the middle. Never two flowers, never more than five petals.
4. Leaves (Leaf Rule): exactly FOUR big fern fronds growing from under the flower and draping down over the back and shoulders: one to the front-left, one to the front-right, one to the back-left, one to the back-right. Never more than four.
5. Legs (4-Leg Rule): exactly four legs, two front and two hind, each foot with three white claws. No tail, no arms, no wings.

GRID LAYOUT (4 key poses, large and crisp, same scale, feet on the same baseline, nothing touching the grid cell borders):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: wide head low in front, both eyes and both ears visible, both front legs visible at the sides, the flower and fronds spread above and behind.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: head and mouth on the left, one eye, front legs on the left and hind legs on the right (near legs in front of far legs), the flower above the middle of the back seen from the side as a wide umbrella, fronds hanging over the sides.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the rump and back with spots, the fronds and the flower from behind, the head turned away to the upper right (only the back of the head and ears visible, no face).
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: rump, both hind legs, the flower and four fronds from behind, the back of the head barely visible above the back. No face.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design and colours: a heavy low teal quadruped with four short legs (three white claws each), a broad head with two small red-lined ears and narrow red eyes, exactly four big light-green fern fronds on its back, and exactly ONE huge coral-pink flower with exactly FIVE petals, a yellow crown centre and a short brown trunk on the middle of its back.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the head looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the near (left) front leg in front, the body extending back to the upper right. Exactly four legs, one flower, five petals, four fronds. Crisp 1-2px dark outline, limited palette, no anti-aliasing, the whole creature fully inside the image with white margin around it.
