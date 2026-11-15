# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front view pixel sprite, the creature faces the viewer, slightly down-left) and `ref_back.png` (back view pixel sprite). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. Do not read or reuse any image other than the two attached references in this folder. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (front view and back view pixel sprites) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small plant creature. Under a gigantic flower sits one small round dark navy-blue body (#394162 mid, #5A6283 light, #18294A dark shadow), much smaller than the flower. Two tiny short stubby arms stick out sideways from the body, one on each side. Two short thin legs end in two small flat oval feet. No neck, no tail, no leaves.
2. Face (front only): on the upper part of the small blue body, just under the flower rim: two tiny dark eyes with small red pupils (#DE4A5A) and a small dark smiling mouth line. No nose, no teeth.
3. Flower (1-Flower Rule, most important): exactly ONE gigantic flower sits directly on the head like a huge umbrella hat, about TWICE as wide as the body and taller than the body. It is made of exactly FIVE big round, thick, puffy petals, coral red (#DE4A5A main, #F68B8B light, #F67373 highlight, #9C3941 shadow, #731810 dark-red outline between petals), each covered with many small pale-pink spots (#FFCDD5, #F6ACB4). The five petals spread outward in a ring, seen from slightly above. In the middle of the ring there is one round orange pistil (#FF8329 bright, #CD6218 mid, #944A00 dark) with a dark navy hole in its centre (#18294A). Never four petals, never six; exactly five.
4. Outlines: crisp 1px dark pixel-art outline (#101010), limited palette, no anti-aliasing, no gradients, no glow.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, the whole flower fully inside each quarter with white space around it):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the five-petal flower on top seen slightly from above with the orange pistil visible, the small blue body below with both eyes and the mouth, both tiny arms sticking out left and right, both legs and feet.
- Top-Right: WEST (90°, left side profile) — facing left in a sharp side profile: the small body with one eye and the mouth on the left edge, one arm pointing left, both feet pointing left, the giant five-petal flower on top seen from the side (a thick wide red disc, three or four petals visible, pistil hidden or only its rim).
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the plain navy back of the body (no face), the giant five-petal flower on top tilted away, both tiny arms, both feet.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, like the back reference: plain navy back with no face, the giant five-petal flower on top with the orange pistil peeking at the top, both tiny arms left and right, both feet.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small round dark navy-blue body with two tiny stubby arms and two short thin legs with small flat feet, two tiny eyes with red pupils and a small smiling mouth, and exactly ONE gigantic coral-red flower on its head, twice as wide as the body, made of exactly FIVE puffy petals covered with small pale-pink spots, with one round orange pistil with a dark navy centre hole.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the feet pointing down-left, the flower seen slightly from above. Exactly one flower, exactly five petals, two arms, two legs, no leaves. Crisp 1px dark outline, limited palette, no anti-aliasing. The whole creature fully inside the image with white space around it.
