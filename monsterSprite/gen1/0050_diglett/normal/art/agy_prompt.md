# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the small creature from the two attached reference images (a low-resolution pixel sprite: front view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: exactly ONE upright capsule / egg-shaped body (rounded dome top, straight sides, slightly taller than wide) poking up out of a hole in the ground. Only the upper part shows; the lower part is hidden in the ground. NO arms, NO legs, NO hands, NO feet, NO tail, NO ears, NO hair, NO mouth. Never add limbs.
2. Colour palette:
   - Body: warm brown (#C57341) with a light tan highlight on the upper-left of the dome (#DE9C5A), a dusty mauve shade on the right side and lower body (#A45A5A), and a dark brown accent (#5A3118) at the very top.
   - Eyes: exactly TWO tiny black eyes (#101010), each a short vertical oval with a single white pixel highlight, set close together in the upper third of the face.
   - Nose: exactly ONE big oval pink-red nose (#D5394A) with a lighter pink-salmon centre (#FF6A5A, #FFAC94) and a dark maroon edge (#730018), set below the eyes, wider than the two eyes together.
   - Ground: exactly ONE low ring of about 6-7 round pebbles / dirt clods hugging the base of the body, pale khaki (#E6E6B4, #B4AC73) with olive shading (#837B4A, #5A5220), plus a few loose khaki dirt specks scattered under the ring.
   - Outlines: crisp 1-2px near-black pixel-art outline (#101010), no anti-aliasing halo.
3. Count rule: one body, two eyes, one nose, one pebble ring. Do not add more of anything; no second creature.

GRID LAYOUT (4 key poses, large and crisp, same size, pebble ring on the same baseline):
- Top-Left: SOUTH (0°, front view) — the face points straight at the viewer, symmetric: both eyes and the big nose centred on the body, pebble ring all around the base.
- Top-Right: WEST (90°, left side profile) — the body seen from its left side, the face turned fully to the LEFT: one eye near the left edge and the nose sticking out to the left as a bump on the profile; the pebble ring around the base.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: plain brown back of the dome with the mauve shade, the face turned away to the upper right so no eyes are visible (at most the edge of the nose at the right side); pebble ring around the base.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: plain brown dome back, no face, pebble ring around the base.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: one upright brown capsule-shaped body (#C57341, tan highlight #DE9C5A, mauve shade #A45A5A) poking out of the ground, two tiny black eyes with white highlights, one big oval pink-red nose (#D5394A / #FF6A5A, maroon edge #730018), no arms, no legs, no mouth, and one low ring of round khaki pebbles (#E6E6B4 / #B4AC73 / #837B4A) around the base with a few dirt specks.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left, both eyes visible in natural 3/4 perspective (not squished), the nose shifted toward the left side of the body. Exactly one body, two eyes, one nose. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
