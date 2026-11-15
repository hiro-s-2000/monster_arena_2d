# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, the creature faces away). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (small pixel sprites: a 3/4 front view and a back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small low crab-like bug standing low on the ground: a round, dome-shaped orange body made of a few segments with dark seam lines, a round head at the front, low and wide, never standing upright.
2. Colour palette:
   - Body, head, claws and legs: orange (#E68329) with light peach highlight (#FFAC62), mid brown-orange (#B46220) and dark brown shade (#734A18 / #392910).
   - Mushrooms: red caps (#BD4141, light #F66A6A, dark #622929) with round yellow spots (#C5B420 / #FFD552) and a short grey stalk (#5A5A5A).
   - Eyes: large round pale grey-white eyes (#FFF6FF / #CDCDCD) with a dark grey ring and a dark pupil.
   - Outlines: crisp 1px dark pixel-art outline (#101010), no anti-aliasing halo.
3. Mushrooms (2-Mushroom Rule, most important): exactly TWO round red mushroom caps with yellow spots grow from the top of the back, one on the left and one on the right, side by side, each about as big as the head. Never one, never three.
4. Claws (2-Claw Rule): exactly two big orange front claws (pincers), one on each side of the head, each ending in two blunt pincer tips, held forward and low near the ground.
5. Legs (4-Leg Rule): exactly four thin jointed orange walking legs, two on each side of the body behind the claws, bent like a crab's legs, tips touching the ground. Never more, never fewer.
6. Face: exactly two large round eyes on the front of the head and a small mouth with two tiny white fangs under them.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the small fanged mouth, both claws spread forward low on the left and right, the legs visible at both sides, both mushrooms on top of the back, one on each side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the head and both claws reaching forward to the left, the legs underneath, the round body behind on the right with both mushrooms on top (one slightly behind the other).
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the round segmented back with both mushrooms on top, the face turned away to the upper right, the claws only partly visible at the front.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the round segmented back, both mushrooms on top side by side, the legs visible at both sides.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small low orange crab-like bug with a round segmented body, two large round pale eyes, a small mouth with two tiny white fangs, exactly two big orange front claws with two pincer tips each, exactly four thin jointed legs, and exactly TWO round red mushroom caps with yellow spots growing side by side from the top of the back.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the claws reaching forward-left, the body and the two mushrooms behind on the upper right. Exactly two mushrooms, two claws and four legs. Crisp 1px dark outline, limited palette, no anti-aliasing.
