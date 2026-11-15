# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature faces lower-left, its shell and the right cannon are behind it on the right) and `ref_back.png` (pixel-art back 3/4 view, the creature faces right, the big brown shell fills the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no grid lines, no shadows on the ground, no water effects.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a big, heavy, upright bipedal turtle standing on two thick legs in a wide, sturdy stance, carrying a large hard shell on its back. Broad head with a short blunt snout, a fierce narrow eye with a dark brown iris and a white highlight, and exactly TWO small rounded triangular ears on top of the head.
2. Colour palette (use only these families):
   - Skin of head, arms and legs: medium blue (#5A8BCD) with royal blue shading (#2062AC), deep navy shadow (#083962) and pale lavender-blue highlights (#94ACE6).
   - Chest plate (plastron) and lower jaw: golden tan (#D5AC4A) with light cream (#F6D59C) and pale gold (#E6C573), divided by dark horizontal lines into segments.
   - Shell on the back: chocolate brown (#8B6241) with dark brown (#5A3918), hexagon-like plates outlined in near-black, and a white rim (#FFFFFF) running around the shell edge.
   - Cannons and claws: white (#FFFFFF), light grey (#CDCDD5), grey (#949494) and dark grey (#4A4A4A).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Cannons (2-Cannon Rule, most important): exactly TWO short, thick, silver-white cylindrical water cannons, one sticking out of the top of the shell above each shoulder (one left, one right), both pointing FORWARD, each with a dark round hole at its front end. Never one cannon, never three or four.
4. Limbs (2-Arm / 2-Leg Rule): exactly TWO thick, muscular blue arms, each hand with exactly THREE small white claws; exactly TWO very thick blue legs, each foot with exactly THREE small white claws. No tail visible from the front except a tiny stub; no wings, no spikes, no extra limbs.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all fully inside their quadrant with white margin):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, both ears, the golden segmented chest plate fully visible, both arms bent at the sides with claws visible, both legs apart; the two cannons rise from behind both shoulders, pointing toward the viewer so their round dark holes show; the shell's edge shows on both sides behind the body.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the snout pointing left, the golden chest plate on the left edge of the body, the big round brown shell on the back on the RIGHT side, both cannons on top of the shell pointing LEFT (forward) above the head height, one in front of the other, both arms forward to the left.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the big brown hexagon-plated shell with its white rim filling most of the body, the back of the head peeking above it, the face turned away to the upper right; both cannons on top of the shell point up-right (away from the viewer); the legs and arms peek out at the sides.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the big brown hexagon-plated shell with its white rim covering the whole back, the back of the head and both ears above it, the two cannons on the shell's top left and top right pointing away from the viewer, the thick arms and feet peeking out at the sides and bottom.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a big upright blue turtle with a golden segmented chest plate, a brown hexagon-plated shell with a white rim on its back, two small ears, two thick arms with three white claws each, two thick legs with three white claws each, and exactly TWO silver-white cylindrical cannons coming out of the top of the shell above the shoulders.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with the eye visible in natural 3/4 perspective (not squished), the golden chest plate visible, the brown shell behind the body to the RIGHT, both cannons pointing diagonally forward-left (down-left) with their dark holes visible. Exactly two cannons, two ears, two arms, two legs. Crisp 1-2px near-black outline, limited palette, no anti-aliasing, no text, no ground shadow, no water, the whole creature fully inside the image with white margin.
