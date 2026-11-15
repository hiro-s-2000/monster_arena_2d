# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front view pixel sprite, the creature faces the viewer, slightly down-left) and `ref_back.png` (back view pixel sprite). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. Do not read or reuse any image other than the two attached references. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (front view and back view pixel sprites) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, round, plump plant creature. The body is one round ball, slate blue-grey (#7B83A4 light, #4A6A7B mid, #10314A dark shadow). Two very short stubby arms stick out low on each side of the ball. Two short thin legs end in two flat, wide, paddle-like feet. No neck, no tail.
2. Face (front only): two half-closed sleepy eyes drawn as short dark curved lines near the top of the ball, and one wide flat mouth with a purple lip line (#B45AC5) and a single cream drip of nectar (#F6EEBD) hanging from the mouth. No nose, no teeth.
3. Flower (1-Flower Rule): exactly ONE large flower sits directly on top of the round body, as wide as the body. It is made of exactly FOUR big round rust-brown petals (#AC5239 light, #8B3918 mid, #5A2900 dark) arranged like a closed four-leaf clover seen from above: one petal at the front, one at the back, one left, one right, with a small orange (#FF7300) centre gap on top. Every petal has a few small cream-yellow spots (#FFBD41, #F6EEBD). The flower is larger than the head area; it covers the top of the body like a big hat.
4. Leaves (4-Leaf Rule): exactly FOUR long curling leaves, orange (#FF7300 bright, #C57329 mid, #8B3918 dark edge, #FFBD41 highlight), grow from under the flower petals: two on the left side and two on the right side. Each leaf is a long thin curved blade that curls at its tip like a tendril; some curl downward beside the body, one curls upward. Never more than four leaves, never fewer.
5. Outlines: crisp 1px dark pixel-art outline (#101010), limited palette, no anti-aliasing, no gradients, no glow.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both sleepy eyes, the purple mouth with the cream drip, both stubby arms, both feet pointing slightly outward, the four-petal flower on top, two leaves curling on each side.
- Top-Right: WEST (90°, left side profile) — facing left in a sharp side profile: the mouth on the left edge of the ball, one arm in front, both feet pointing left, the flower on top seen from the side (three petals visible), leaves curling out behind and below the flower.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the plain blue-grey back of the ball (no face), the flower on top, the leaves spreading left and right, the feet pointing up-right.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, like the back reference: plain blue-grey back with no face, the four-petal flower on top, two leaves on each side, both stubby arms, both feet.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a round slate blue-grey ball body with two stubby arms and two short legs with flat paddle feet, two half-closed sleepy eyes, a wide purple-lipped mouth with one cream drip, exactly ONE big flower on top made of exactly FOUR round rust-brown petals with small cream-yellow spots and a small orange centre, and exactly FOUR long orange curling leaves under the flower (two each side).
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the mouth shifted to the left, the feet pointing down-left. Exactly one flower, exactly four petals, exactly four leaves. Crisp 1px dark outline, limited palette, no anti-aliasing.
