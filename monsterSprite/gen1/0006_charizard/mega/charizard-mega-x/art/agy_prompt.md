# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite, seen from behind). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large bipedal dragon standing upright on two thick legs, with a long neck, a broad chest and a long thick tail. Arms held forward at chest height with clawed hands ready to strike.
2. Colour palette:
   - Body, head, neck, arms, legs, tail and wing frames: charcoal black-grey (#5A5A5A) with dark shading (#383838, #1F1F1F).
   - Belly, throat and the underside of the tail: pale sky blue (#99D9F7) with mid blue shading (#60A6C8).
   - Wing membranes: deep navy blue (#2B629C) with darker navy (#1E4167).
   - Flames: bright cyan-blue fire (#009DE1, #00B1E6) with pale blue cores (#99D9F7).
   - Eyes: red (#CE1010). Mouth interior: pink (#E481B5). Claws and teeth: white (#F8F8F8) with grey (#92A09C).
   - Outlines: crisp 1px near-black outline (#080808), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with a long snout, red eyes, and two horns swept straight back from the top of the head. Blue flames stream backward from both corners of the mouth and from the horns.
4. Wings (2-Wing Rule): exactly TWO large bat-like wings growing from the upper back / shoulders, each with a charcoal frame (arm bone with 3 finger spines) and navy blue membranes with jagged lower edges. Wings spread wide and up. Never one wing, never three or four.
5. Arms and legs (2-Arm, 2-Leg Rule): exactly two arms, each hand with three white claws; exactly two legs, each foot with three white claws, both feet planted on the ground. Small dark-blue spikes on the shoulders.
6. Tail (1-Tail Rule): exactly ONE long thick tail curving out behind the body, its underside pale blue, ending in a large blue flame at the tip.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, the snout pointing at the viewer, the pale blue belly in the centre, both arms forward, both wings spread wide to the left and to the right behind the body, the tail curling out to one side behind the legs with its blue flame.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, snout pointing left, the blue mouth flames streaming right, arms forward to the left, wings raised behind the back, the tail extending to the right with the blue flame on its tip.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the dark back, the back of the head and the horns, the face turned away to the upper right; both wings spread up-left and up-right from the shoulders, the tail coming toward the viewer at the bottom left with its flame.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: dark back, back of the head with both horns, both wings spread wide to the left and right, the tail hanging down the middle and curving to one side with its blue flame.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a charcoal black bipedal dragon with a pale sky-blue belly, red eyes, two backswept horns, blue flames from the mouth corners, two clawed arms forward, two clawed legs planted, TWO large wings with navy blue membranes spread behind it, and ONE long tail ending in a blue flame.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the pale blue belly visible. Exactly two wings and exactly one tail, all clearly visible. Crisp 1px dark outline, limited palette, no anti-aliasing.
