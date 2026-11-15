# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (a small pixel sprite: front view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a trio of THREE identical burrowing mole heads poking up out of one shared pile of round rocks on the ground. Each head is a tall rounded capsule / bullet shape (like a thumb sticking out of the soil). No arms, no legs, no hands, no feet, no tail, no ears, no hair — only the three capsule heads and the rock pile.
2. Three-Head Rule (most important): exactly THREE heads, never two, never four. One head is taller and stands at the back in the middle; the other two are shorter and stand in front of it, one on the left and one on the right. All three emerge from the same rock pile, their bottoms hidden by the rocks.
3. Colour palette:
   - Heads: warm reddish brown (#C57341) with a light tan highlight on the top (#DE9C5A) and a dusty mauve-brown shadow on the lower and back side (#A45A5A); dark brown edge (#5A3118).
   - Face on each head (front side only): two tiny black vertical slit eyes (#101010) with a 1px white glint, and one big horizontal oval pink-red nose (#D5394A with light #FF6A5A and pale #FFAC94 shine, dark rim #730018) below the eyes. No mouth.
   - Rock pile: a low wide mound of round pebbles in pale olive beige (#E6E6B4, #B4AC73) with olive shadow (#837B4A, #5A5220), dark outlines between the stones.
   - Outlines: crisp 1-2px near-black pixel-art outline (#101010), no anti-aliasing halo.
4. Do not add any limbs, claws, whiskers, mouths, hair or extra heads. Keep the heads smooth and simple.

GRID LAYOUT (4 key poses, large and crisp, same size, rock piles on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer: the tall head in the middle at the back, one short head in front-left and one short head in front-right, all three faces (two eyes + one nose each) looking at the viewer.
- Top-Right: WEST (270°, left side profile) — the trio seen from its left side, all heads facing left: each head shows one eye and its nose sticking out on the left edge; the three heads overlap in a row (one short head nearest the viewer, the tall head and the other short head partly behind it).
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the smooth backs of the three heads (no faces, only a small bit of one nose peeking at the upper-right edge), the tall head in the middle, the two short heads at the sides.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, like the back reference: three smooth mauve-brown head backs with tan tops, the tall one in the middle, the two short ones left and right, no faces.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: exactly THREE reddish-brown capsule-shaped mole heads (tan tops, mauve shadows) poking out of one low pile of pale olive round rocks, the tallest head at the back in the middle and two shorter heads in front; each head has two tiny black slit eyes and one big pink-red oval nose. No limbs, no mouth, no hair, no tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: all three faces look diagonally down-left with both eyes visible in natural 3/4 perspective, noses pointing down-left. Exactly three heads. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
