# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, seen from behind, the creature faces right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel-art front view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a big, heavy, bulky bipedal turtle standing upright on two short thick legs, with a huge round shell on its back. Short thick neck, wide head with a flat snout and a fierce frown.
2. Colour palette:
   - Skin (head, arms, legs): royal blue (#1959B0) with light blue highlights (#5181D1) and dark navy shadows (#043465).
   - Eyes: two small fierce red eyes (#D13D28) under heavy brows.
   - Belly / chest plate: pale cream (#F1D198) with ochre shading (#CEA84B) and thin brown horizontal segment lines (#5A3918).
   - Shell: dark brown plates (#5A3918) with lighter brown highlights (#8B6241), separated by dark seams; a thick white-and-light-grey rim (#F8F8F8 / #CDCDD5) around the shell edge.
   - Cannons: light grey metal (#CDCDD5, #949494) with dark grey (#4A4A4A) and a black bore (#101010).
   - Claws: white (#F8F8F8), three small claws on each foot.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Cannons (3-Cannon Rule, most important): exactly THREE cannons in total, never more, never fewer:
   - ONE very large long cylindrical grey cannon mounted on top of the shell, rising diagonally from behind the right shoulder and pointing up-forward over the head, with a black round muzzle.
   - TWO short arm cannons, one on each forearm: each is a short white-grey gun barrel with a black muzzle, held in a brown shell-coloured housing on the forearm, pointing forward past the fist.
   No cannons anywhere else (none on the shoulders, none on the legs, none on the head).
4. Limbs: exactly two arms and two legs. Arms are thick and blue, fists blue. Legs are short and thick, blue, each foot with three white claws. A tiny blue tail may peek out under the shell at the back, only one.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, the cream belly with segment lines, both arm cannons pointing toward the viewer at the left and right side of the belly, the big back cannon visible rising above the head behind one shoulder, the brown shell edge with white rim visible behind the body on both sides.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one red eye, snout to the left, the near arm cannon pointing left, the shell on the right side (the back), the big cannon rising from the shell diagonally up-left over the head.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference image mirrored: we see mostly the big brown shell with its white rim, the head turned away to the upper right, the big cannon on top of the shell, one arm cannon sticking out at the right.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the big brown round shell with plate seams and white rim covers the back, the big cannon on top pointing up (foreshortened), the back of the head above the shell, both arms and both legs peeking out at the sides.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a big bulky blue bipedal turtle with a cream belly with brown segment lines, fierce red eyes, a huge brown shell with a white rim, exactly THREE cannons (one very large long grey cannon on top of the shell pointing up-left over the head, and one short white-grey arm cannon in a brown housing on each forearm), two thick legs with three white claws each.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the far arm cannon pointing left, the near arm cannon in front of the belly, the big cannon rising over the head to the upper left. Exactly three cannons. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
