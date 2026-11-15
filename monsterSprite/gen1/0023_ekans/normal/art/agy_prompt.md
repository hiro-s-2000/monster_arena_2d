# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, its thin tail with a rattle rises up on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, its thin tail with a rattle rises up on the left). Use ONLY these two reference images from this folder; ignore any other image you may have seen. Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small slim snake with NO legs, NO arms, NO wings and NO hood. The front part of the body rises up as a curved neck; ONE rounded head with a blunt snout sits on top, its jaws open. The lower body lies on the ground in ONE round spiral coil, and it ends in ONE thin tail that rises up beside the coil; the tail tip is a short rattle made of FOUR small stacked golden rings.
2. Colour palette:
   - Head, back of the neck, coil and tail: bright pink-purple (#CD62B4) with light pink highlights (#EEA4D5), mid purple shading (#A44A8B), dark plum shading (#7B316A) and a dark plum edge (#5A104A).
   - Collar: ONE yellow band (#FFD56A with light #FFE69C and ochre #E6AC5A) around the neck just below the head; the underside (belly) of the coil is the same pale yellow, with brown shading (#B47B31, #835210) at the bottom.
   - Rattle: FOUR small rings at the tail tip, golden ochre (#E6AC5A / #B47B31) with dark brown lines (#835210).
   - Mouth: open jaws with a dark red inside (#9C1000) and an orange-red tongue area (#C54118, #F6734A). No fangs, no forked tongue sticking out.
   - Eyes: one eye on each side of the head, yellow-ochre (#E6AC5A) with a black slit pupil (#101010).
   - Outlines: crisp 1-2px black outline (#101010), no anti-aliasing halo.
3. Counts (most important): exactly ONE head, ONE neck, ONE yellow collar band, ONE coil, ONE thin tail with ONE rattle (four rings), ZERO legs, ZERO arms, ZERO wings, NO hood. Never add legs, feet, arms, wings, horns, spikes, a hood, a second head or a second tail.

GRID LAYOUT (4 key poses, large and crisp, same size, the bottom of the coil on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — the head faces straight toward the viewer, symmetric: both eyes, the open mouth in the middle, the yellow collar band across the front of the neck; the neck goes down into the round coil on the ground; the thin tail with the rattle rises up on the right side of the coil.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the snout and the open jaws pointing left, the yellow collar and the yellow belly line on the front (left) side of the curved neck; the coil lies on the ground and the thin tail with the rattle rises up at the far right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — like the attached back reference: we see the purple back of the neck and the back of the head, the face turned away to the upper right (only the edge of the snout visible), the yellow collar seen from behind; the coil lies at the bottom and the thin tail with the rattle rises up on the left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the purple back of the head and of the neck, the yellow collar band around the neck, no face visible; the neck goes down into the coil on the ground; the thin tail with the rattle rises up on the left side.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small slim pink-purple snake rising from ONE round spiral coil, ONE rounded head with open jaws (dark red inside, orange-red tongue area), yellow eyes with black slit pupils, ONE yellow collar band below the head, a pale yellow belly, and ONE thin tail rising up beside the coil that ends in a rattle of FOUR small golden rings.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the head looks diagonally down-left with the open jaws toward the lower left, the yellow collar and belly visible in natural 3/4 perspective (not squished), the coil on the ground, the thin tail with the rattle rising up on the right. Zero legs, zero arms, no hood, one head, one tail. Crisp 1-2px black outline, limited palette, no anti-aliasing.
