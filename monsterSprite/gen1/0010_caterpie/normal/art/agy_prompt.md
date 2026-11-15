# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, its tail curls up on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, its tail curls up on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes. (Retry note: if `out/sw_single.png` already exists, keep it and only generate `out/keys_2x2.png`; also attach `out/sw_single.png` to prompt A as a third reference — it shows the correct design: round green head, big cream-ringed eye, red forked antenna, cream belly, cream tail bulb.)

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small plump caterpillar-like larva with NO legs and NO arms. The front half of the body is raised upright (head on top), the back half lies on the ground and curls up at the end into a short tail. The body is made of round ring segments: the head plus FIVE body segments, each segment a little smaller toward the tail. No wings, no claws, no feet, no fins.
2. Colour palette:
   - Head and the back / sides of every segment: bright leaf green (#6AD531 / #4AC529) with mid green (#31AC41) and dark green shading (#207B4A, #185A41). Thin dark green lines separate the segments.
   - Underside (belly side) of every segment, from the chin down to the tail: pale cream (#E6CD94) with light yellow (#F6EE8B).
   - Side spots: one small round pale-yellow ring spot (#F6EE8B with #7B834A edge) on the side of each body segment.
   - Eye: ONE big round eye on each side of the head: cream ring (#E6CD94), black pupil (#000000) with a dark grey shade (#52624A). The eye is large, about one third of the head height.
   - Antenna: ONE short forked red horn on top of the head shaped like the letter Y / a V with two prongs: red (#FF4141) with orange-red (#FF7B41) and an orange tip (#FFB439), outlined in dark maroon (#7B394A).
   - Tail tip: the last segment ends in a round cream-yellow bulb (#E6CD94 / #F6EE8B) pointing upward.
   - Outlines: crisp 1-2px black outline (#000000), no anti-aliasing halo.
3. Counts (most important): exactly ONE head, exactly ONE forked red antenna (two prongs, never two antennas), exactly ONE tail bulb, ZERO legs, ZERO arms. Never add legs, feet or arms. Never add a second antenna or a second tail.
4. Must NOT look like a cocoon or a pointed pod: the head is ROUND (not pointed, not a cone), the red forked antenna is clearly visible on top of the head in ALL FOUR views (also from behind), the cream belly segments are clearly visible in the front and side views, and the round cream tail bulb is visible in all four views. Copy the round head, the big eye and the red antenna from the attached front reference.

GRID LAYOUT (4 key poses, large and crisp, same size, the bottom of the body on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — the raised front of the body faces straight toward the viewer, symmetric: both big eyes on the left and right of the head, the forked red antenna on top, the cream belly segments stacked down the middle; the back half of the body curls behind and the cream tail bulb sticks up to the right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one big eye, the face pointing left, the cream belly on the left edge of the raised front part; the back half of the body lies flat to the right on the ground and the tail bulb curls up at the far right end.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: we see the green back of the segments and the back of the head, the face turned away to the upper right (only the edge of one eye visible), the antenna on top; the back half of the body curls toward the viewer and the tail bulb curls up on the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the green back of the raised segments, the back of the head with the antenna on top, no face and no belly visible; the back half of the body lies on the ground toward the viewer and the tail bulb curls up at the bottom right.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a plump green legless caterpillar-like larva with its front half raised upright, round ring segments (head + five segments), cream belly segments, small pale-yellow ring spots on the sides, ONE big eye with a cream ring and black pupil on each side of the head, ONE short forked red Y-shaped antenna on top of the head, and the back half curling into a tail that ends in ONE round cream bulb pointing up.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with the near (right) eye fully visible and the far eye partly visible in natural 3/4 perspective (not squished), the cream belly segments visible down the front, the tail bulb curling up behind the body on the right. Zero legs, zero arms, one antenna, one tail bulb. Crisp 1-2px black outline, limited palette, no anti-aliasing.
