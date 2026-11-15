# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite, seen from behind). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large, slim, armoured hornet-like insect hovering upright in the air (it never stands on the ground): a round head on top, a short thorax, and a long, tapering, segmented abdomen hanging straight down below, ending in ONE sharp yellow stinger pointing down.
2. Colour palette:
   - Head, thorax and abdomen: golden yellow (#FFCD4A) with orange (#E69420), dark amber shading (#B46210, #834A00) and pale yellow highlights (#FFF6A4).
   - Abdomen stripes: alternating thick near-black (#0C0C0C) and slate blue-grey (#525A7B) bands across the yellow abdomen.
   - Eyes: two large angular red compound eyes (#D51831) with dark red (#9C0008) and a small salmon highlight (#FF946A).
   - Lances, leg spikes and wings: pale lavender-white (#DECDF6, #ECE1FF, #F8F8F8) with lavender-grey shading (#BDACC5, #8B7B94).
   - Antennae: slate blue-grey (#525A7B) with near-black outline.
   - Outlines: crisp 1px near-black outline (#0C0C0C), no anti-aliasing halo.
3. Head (1-Head Rule, 2-Antenna Rule): exactly one head with two red compound eyes and exactly TWO thin dark antennae sticking straight up from the top of the head.
4. Arms (2-Lance Rule, most important): exactly TWO arms, each forearm is one very large, long, cone-shaped pale lavender lance (like a jousting lance) pointing down and outward from the shoulders, one on the left and one on the right. Never one lance, never three or four.
5. Legs (2-Leg Rule): exactly two thin legs hanging down beside the abdomen, striped yellow and slate blue-grey, each ending in one white needle spike. The creature floats; the legs dangle, they do not stand.
6. Wings (4-Wing Rule): exactly FOUR translucent pale lavender-white insect wings (two pairs: two upper, two lower) with thin lavender-grey vein outlines, growing from the upper back and spreading up and outward. Never two wings, never six.

GRID LAYOUT (4 key poses, large and crisp, same size, the stinger tips on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, both antennae up, the two big lances pointing down-left and down-right from the shoulders, the striped abdomen hanging down the middle with the stinger at the bottom, the two thin legs dangling beside it, the four wings spread behind the shoulders to the left and right.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one red eye, antennae pointing up and forward to the left, the near lance held forward pointing down-left, the abdomen hanging down and slightly back to the right with the stinger at the bottom, the wings raised behind the back to the upper right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back of the head and thorax, the face turned away to the upper right, the four wings spread up-left and up-right from the back, the two lances pointing down and outward, the striped abdomen hanging down with the stinger.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: back of the head with both antennae, the four wings spread wide to the left and right, both lances pointing down-left and down-right, the striped abdomen hanging down the middle with the stinger.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a golden yellow hornet-like insect hovering upright, two large red compound eyes, exactly two thin dark antennae, exactly TWO huge pale lavender cone-shaped lance forearms pointing down and outward, exactly two thin striped legs ending in white needle spikes dangling below, a long abdomen with black and slate blue-grey stripes ending in one yellow stinger, and exactly FOUR translucent pale lavender wings spread behind.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both red eyes visible in natural 3/4 perspective (not squished). Exactly two lances, two legs, four wings, one stinger, all clearly visible. Crisp 1px dark outline, limited palette, no anti-aliasing.
