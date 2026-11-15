# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel-art front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a legless carnivorous plant shaped like a big hanging bell / upside-down pear. The narrow rounded top of the bell is up, the wide open mouth end is down-forward. It has NO legs, NO feet, NO arms and NO hands. It floats slightly above the ground.
2. Colour palette:
   - Bell body: pale yellow (#FFE66A) with light highlight (#FFF6AC), mid olive-yellow (#D5C552) and shadow brown-gold (#A47B31).
   - Small green spots: a few tiny green dots (#6AC552) on the upper front of the bell only.
   - Mouth: one large round open mouth at the wide end of the bell, with thick pink lips (#EE9CAC, #CD6A83), dark red inside (#B45A4A -> #7B2929) and a black throat (#000000).
   - Eyes: exactly two small round white eyes with black pupils and a thin brown rim (#624A31), just above the mouth.
   - Leaves: green (#9CDE83 light, #6AC552 mid, #399C52 dark, #085A31 deep shadow).
   - Stem: brown (#624A31, #A47B31).
   - Outlines: crisp 1px black pixel-art outline (#000000), no anti-aliasing halo.
3. Leaves (2-Leaf Rule): exactly TWO broad pointed leaves, one on each side of the bell at the height of the eyes, like big ears / wings, tips pointing up and outward. Never one leaf, never three or four.
4. Stem (1-Stem Rule): exactly ONE short thin brown stem growing from the very top of the bell, curling over into a small hook. No vines, no tendrils elsewhere.
5. Do not add any extra parts: no legs, no arms, no roots, no teeth, no tongue, no flower.

GRID LAYOUT (4 key poses, large and crisp, same size, bottom of the body on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the round open pink-lipped mouth at the bottom centre facing the viewer, both eyes above it, one leaf on the left and one on the right, the hooked stem on top.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: the bell tilts forward so the open mouth points to the left-down, one eye visible, the near leaf in front of the body and the far leaf peeking out behind it, the stem hook on top.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the plain olive-yellow back of the bell, the mouth turned away to the upper right (only a bit of pink lip edge visible at the lower right), both leaves visible, the stem on top.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the plain olive-yellow back of the bell, one leaf on each side, the stem hook on top, no face visible.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a legless pale-yellow bell-shaped plant (no legs, no arms) with one big round open mouth with thick pink lips at its wide lower end, two small round eyes just above the mouth, a few tiny green dots on the upper front, exactly TWO broad green pointed leaves (one on each side, tips up and outward) and exactly ONE short brown hooked stem on the top.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the open mouth points diagonally down-left, both eyes visible in natural 3/4 perspective (not squished), the far leaf on the upper left and the near leaf on the right. Exactly two leaves, one stem. Crisp 1px black outline, limited palette, no anti-aliasing.
