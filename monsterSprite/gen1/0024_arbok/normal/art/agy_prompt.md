# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, its thin tail curls up on the left) and `ref_back.png` (back 3/4 view, the creature faces upper-right, its thin tail curls up on the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a big cobra-like serpent with NO legs, NO arms and NO wings. The front part of the body rises straight up like a neck; at the top it spreads into ONE very wide flat round hood (wider than the head, about as wide as the coiled body), and ONE small head sits on top of the hood with its jaws open. The lower part of the body lies on the ground in ONE thick coil made of ring segments, and it ends in ONE long thin tail that curls up beside the coil with a pointed tip.
2. Colour palette:
   - Head, hood back, neck, coil and tail: lavender purple (#7B62AC) with light lilac (#A483C5), pale lilac highlights (#C5A4EE) and dark violet shading (#52397B). Thin dark violet lines separate the ring segments of the coil and of the tail.
   - Hood front pattern (seen only from the front, a scary face-like marking): TWO large slanted oval spots, yellow centre (#EEE652) inside an orange ring (#F6734A) and a dark red edge (#C54118), outlined in black, one on each side of the hood; below them ONE wide black (#101010) jagged grin-shaped band across the lower hood. The hood back has NO pattern (plain purple).
   - Mouth: open jaws with a dark red inside (#9C1000) and an orange-red tongue area (#C54118); exactly TWO small white fangs (#FFFFFF with grey #C5C5C5) in the upper jaw.
   - Eyes: two small fierce eyes on the head, white with a black pupil.
   - Outlines: crisp 1-2px black outline (#101010), no anti-aliasing halo.
3. Counts (most important): exactly ONE head, ONE hood, ONE coil, ONE thin tail, TWO hood spots, ONE black grin band, TWO fangs, ZERO legs, ZERO arms. Never add legs, feet, arms, wings, horns or a second head or tail. Do not add a rattle, spikes or fins.

GRID LAYOUT (4 key poses, large and crisp, same size, the bottom of the coil on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — the hood faces straight toward the viewer, symmetric: the hood pattern (two yellow-orange spots and the black grin band) fully visible, the small head on top looking at the viewer with the mouth open and both fangs; the neck goes down into the coil on the ground; the thin tail curls up on the left side of the coil.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: the head points left with the open jaws to the left, the hood seen from the side as a thin tall flap behind the head, the neck rising from the coil; the coil lies on the ground and the thin tail curls up at the far right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — like the attached back reference: we see the plain purple back of the round hood and the back of the head, the head turned away to the upper right; the coil lies in front at the bottom and the thin tail curls up on the right.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the plain purple back of the round hood, the back of the small head on top, no face and no hood pattern visible; the neck goes down into the coil on the ground; the thin tail curls up on the right side.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a big lavender-purple legless cobra-like serpent rising upright from ONE thick segmented coil, ONE very wide round hood with TWO slanted yellow-orange oval spots and ONE wide black jagged grin band on its front, ONE small head on top with open jaws (dark red inside, TWO white fangs), and ONE long thin segmented tail curling up beside the coil.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the head looks diagonally down-left with the open jaws toward the lower left, the hood front pattern clearly visible in natural 3/4 perspective (not squished), the coil on the ground, the thin tail curling up on the left. Zero legs, zero arms, one head, one tail. Crisp 1-2px black outline, limited palette, no anti-aliasing.
