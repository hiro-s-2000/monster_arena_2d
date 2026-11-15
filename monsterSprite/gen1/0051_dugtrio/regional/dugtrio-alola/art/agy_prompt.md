# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a cluster of exactly THREE (3) rounded, bullet-shaped burrowing mole heads poking up out of one small pile of grey rocks. One head stands higher at the back-top centre, two heads sit lower in front, side by side. There are NO arms, NO legs, NO feet, NO hands, NO tails, NO wings: only the three heads and the rock pile. The heads touch each other.
2. Each head: smooth reddish-brown skin (#C57341 light, #A45A5A shadow), two tiny black dot eyes with a 1-pixel white shine, and one big oval pink-red nose (#D5394A, highlight #FF6A5A, dark #730018) below the eyes. No mouth, no teeth.
3. Hair: each head wears long, flowing, glossy golden-yellow hair (#FFC903 highlight, #D3920A mid, #5A3118 dark brown shading) that falls over the top and down the sides like long straight hair, with pointed tips. From behind, the hair of the three heads forms one big golden mound that hides almost all the skin; a few thin strands curl out to the left and to the right at the bottom.
4. Rock pile: a low ring of rough grey stones (#968B95 light, #665867 mid, #302528 dark cracks) around the bottom of the heads, as wide as the cluster.
5. Outlines: crisp 1-px dark outline (#101010), no anti-aliasing halo.
COUNT RULE: exactly 3 heads, 3 noses, 6 eyes in front views. Never add a 4th head, never add limbs.

GRID LAYOUT (4 key poses, large and crisp, same size, rock piles on the same baseline):
- Top-Left: SOUTH (0°, front view) — the cluster faces straight toward the viewer: the top head centred at the back, the two front heads left and right below it; all three faces, noses and eyes look at the viewer.
- Top-Right: WEST (90°, left side profile) — the cluster faces left in a sharp side view: the heads' noses point to the left, each face shows one eye; the hair flows to the right/back; the top head is behind and above, the front heads overlap one another.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: mostly the golden hair mound and the rock pile, the faces turned away to the upper right, only a sliver of brown skin and one nose tip visible at the right edge.
- Bottom-Right: NORTH (180°, pure back view) — straight from behind, like the back reference: one big golden hair mound over three heads, no faces, rock pile below.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: exactly three rounded reddish-brown mole heads (one higher at the back, two lower in front) with long glossy golden-yellow hair, tiny black eyes and big pink-red oval noses, rising from one low pile of grey rocks. No arms, no legs, no tail, no mouth.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: all three faces look diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Exactly three heads. Crisp 1-px dark outline, limited palette, no anti-aliasing.
