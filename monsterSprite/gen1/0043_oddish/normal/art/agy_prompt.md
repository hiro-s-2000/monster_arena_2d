# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Both are enlarged low-resolution pixel sprites on white. Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small round walking plant bulb. The body is ONE round slightly egg-shaped ball, slate blue (#7394A4) with a light blue-grey highlight (#94B4C5) on the upper left and dark slate shading (#5A6A83, #29394A). No arms, no hands, no tail, no wings.
2. Face (front views only): exactly two tiny red dot eyes (#EE3931) and one short straight dark mouth line, all small, on the front of the ball. The back has no face.
3. Legs (2-Leg Rule): exactly two very short stubby legs with flat rounded feet, slate blue like the body, directly under the ball. Never more, never fewer.
4. Leaves (5-Leaf Rule, most important): exactly FIVE long pointed leaves grow from one point on the top of the ball and fan up and outward like a palm-leaf crown: one tall centre leaf pointing up, two leaves angled up-left and up-right, two leaves spreading lower to the left and right. Leaves are leaf green (#8BAC31) with a lighter yellow-green (#9CD54A, #C5E67B) upper face, dark olive veins and shadow (#4A5A20), each with a visible centre vein. The crown of leaves is about as tall as the ball and much wider. Never four, never six or more leaves.
5. Outlines: crisp 1px dark pixel-art outline (#101010), no anti-aliasing halo, flat limited palette, no gradients.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: two red eyes and the mouth line visible, both feet visible, the five leaves fanned symmetrically above.
- Top-Right: EAST (90°, right side profile) — facing right in a sharp side profile: one red eye near the right edge of the ball, both feet stepping right, the five leaves fan up and sweep backward to the left.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the plain back of the ball (no face), feet under it, the five leaves fanned above.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: plain back of the ball, two feet, five leaves fanned symmetrically.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: one round slate-blue ball body (#7394A4 with #94B4C5 highlight and #5A6A83 shade), two tiny red dot eyes and a short dark mouth line, exactly two short stubby feet, and exactly FIVE long pointed green leaves (#8BAC31 / #9CD54A / #C5E67B with #4A5A20 veins) fanning up from the top of the ball.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Exactly five leaves, exactly two feet, no arms. Crisp 1px dark outline (#101010), limited palette, no anti-aliasing.
