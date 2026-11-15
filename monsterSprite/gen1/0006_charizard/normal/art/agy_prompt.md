# Task (image generation only — do not edit any other files)

Retry note: the previous 2x2 drew the wings as flames and the belly white. That is WRONG. The wings are solid bat wings with teal blue skin membranes (NOT fire), and the belly is cream yellow (NOT white). The only fire on the whole creature is ONE small flame on the tail tip. `out/sw_single.png` is already correct: do not regenerate it; attach it as a third style reference for prompt A and only make `out/keys_2x2.png`.

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature faces lower-left, its wings spread behind it, its tail with the flame goes up behind the body) and `ref_back.png` (pixel-art back 3/4 view, seen from behind, the creature faces right, its tail curls to the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large bipedal fire dragon standing upright on two thick legs, with a long neck, a round pot belly and a long thick tail. Arms held forward at chest height with clawed hands ready to strike.
2. Colour palette:
   - Body, head, neck, arms, legs and tail: bright orange (#EE8329) with red-orange shading (#CD5241) and dark brown-red shadows (#833118).
   - Belly and the underside of the tail: pale cream yellow (#EEDE7B) with light ochre shading (#EEB45A).
   - Wing membranes (inner side): solid teal blue skin (#207394) with dark teal (#084152). The back of the wings is solid orange skin like the body. Wings are NEVER made of fire, have no flames, no yellow.
   - Tail flame: red-orange fire (#E64110) with a yellow core (#FFD510) and amber (#F6A410).
   - Eyes: small blue-teal eye with a white highlight. Mouth interior: red (#E64110). Claws and teeth: white (#FFFFFF) with light grey (#CDCDCD).
   - Outlines: crisp 1px black outline (#000000), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with a long blunt snout and two short horns pointing straight back from the back of the head. Mouth may be slightly open.
4. Wings (2-Wing Rule): exactly TWO large bat-like wings growing from the upper back / shoulders, each with an orange frame (arm bone with finger spines) and teal membranes with scalloped lower edges. Wings spread wide and up. Never one wing, never three or four.
5. Arms and legs (2-Arm, 2-Leg Rule): exactly two arms, each hand with three small white claws; exactly two legs, each foot with three small white claws, both feet planted on the ground. Do not add extra limbs.
6. Tail (1-Tail Rule): exactly ONE long thick tail curving out behind the body, its underside cream yellow, ending in ONE flame at the tip.
7. Fire Rule: the tail-tip flame is the ONLY fire. No flames on the wings, head, horns, shoulders or back. No white belly.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the snout pointing at the viewer, the cream belly in the centre, both arms forward, both wings spread wide to the left and to the right behind the body (teal membranes visible), the tail curling out to one side behind the legs with its flame.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, snout pointing left, arms forward to the left, wings raised behind the back, the tail extending to the right and up with the flame on its tip.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the orange back, the back of the head and the horns, the face turned away to the upper right; both wings spread up-left and up-right from the shoulders (orange backs of the wings), the tail coming toward the viewer at the bottom left with its flame.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: orange back, back of the head with both horns, both wings spread wide to the left and right, the tail hanging down the middle and curving to one side with its flame.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: an orange bipedal fire dragon with a cream yellow belly, two short horns pointing back, two clawed arms forward, two clawed legs planted, TWO large wings with teal blue membranes spread behind it, and ONE long tail ending in a flame.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream belly visible. Exactly two wings and exactly one tail, all clearly visible. Crisp 1px black outline, limited palette, no anti-aliasing.
