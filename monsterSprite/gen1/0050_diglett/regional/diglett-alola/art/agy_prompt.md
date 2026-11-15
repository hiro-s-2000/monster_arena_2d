# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front view, the creature faces the viewer, slightly to the lower-left) and `ref_back.png` (back view). They are small pixel sprites enlarged 8x. Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: ONE small upright rounded body shaped like a capsule / bullet with a domed top, poking straight up out of the ground. Only the upper body is visible; the bottom is hidden in the ground. NO arms, NO legs, NO feet, NO hands, NO tail, NO wings. Never add limbs.
2. Colour palette:
   - Body: warm reddish brown (#C57341) on the lit upper-left, dusty rose-brown (#A45A5A) on the shaded side, a light tan highlight (#DE9C5A) on the top-left of the dome.
   - Outline: crisp 1px near-black outline (#101010) with dark brown (#5A3118) inner shading. No anti-aliasing halo.
3. Face (front views only): exactly TWO tiny black dot eyes (#101010) with a tiny white glint, side by side in the upper middle of the body; below them exactly ONE big oval glossy nose, pinkish red (#D5394A) with a light coral highlight (#FF6A5A, #FFAC94) and a dark crimson rim (#730018). No mouth, no teeth, no ears.
4. Hair (3-Strand Rule): from the very top of the dome grow exactly THREE short stiff golden hair strands (#FFC903 with dark brown #5A3118 edges) fanning out like a small three-pronged sprout (one left, one up, one right). Never more than three, never a full wig.
5. Ground: ONE low ring-shaped pile of grey stones around the base of the body (#968B95 light grey, #665867 mid grey, #302528 / #443533 dark gaps), about as wide as 1.5x the body, flat on the bottom. The body sinks into it. The stone pile is part of the sprite.

GRID LAYOUT (4 key poses, large and crisp, same size, stone pile on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer: both eyes and the big nose centred, three hair strands on top, stones all around the base.
- Top-Right: WEST (90°, left side profile) — facing LEFT in a sharp side profile: one eye, the big nose sticking out to the left from the front of the body, back of the dome on the right, three hair strands on top, stones around the base.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the plain brown back of the body, no face (only the edge of the nose may just peek out on the upper right), three hair strands, stones around the base.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind exactly like the back reference: plain brown back with the tan highlight, no face, three hair strands, stones around the base.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design and colours: ONE small upright capsule-shaped warm reddish-brown body (#C57341 / #A45A5A, tan highlight #DE9C5A) poking out of a low ring of grey stones (#968B95 / #665867 / #302528), exactly two tiny black dot eyes, exactly one big oval glossy pinkish-red nose (#D5394A with #FF6A5A highlight, #730018 rim), exactly three short stiff golden hair strands (#FFC903) on top of the dome. NO arms, NO legs, NO mouth, NO tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face (two eyes and the nose) is shifted toward the lower-left side of the body in natural 3/4 perspective (not squished), the nose pokes out a little to the left. The whole creature is centred and large. Crisp 1px dark outline (#101010), limited palette, no anti-aliasing.
