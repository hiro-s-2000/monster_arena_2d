# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite, seen from behind, the head turned to the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large, heavy bipedal orange dragon standing upright on two thick legs, with a long neck, a big round pot belly and a long thick tail. Arms held out at chest height with clawed hands ready to strike.
2. Colour palette:
   - Body, head, neck, arms, legs and tail: warm orange (#EF8429) with brick red shading (#CE5242) and dark red-brown shadows (#A2362B, #843119).
   - Belly: a large white oval (#FCFCFC) shading to cream (#D5CFAE) at the bottom, with a few small glowing yellow (#FFD610) dots on it.
   - Fire: red-orange flames (#FF4200) with golden (#F7A510), bright yellow (#FFD610) and pale yellow (#FFE877) highlights.
   - Eyes: yellow with a red pupil. Claws: white (#FCFCFC) with cream (#D5CFAE).
   - Outlines: crisp 1px near-black outline (#101010), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with a long snout and two short horns pointing back; a crown of flames burns up from the top and back of the head like wild hair.
4. Fire wings (2-Wing Rule, most important): exactly TWO huge wings made entirely of fire, NOT bat membranes: each wing is a big, ragged mass of red-orange flame with yellow streaks, growing from the shoulders / upper back and spreading wide and up, one on each side. A collar of flame runs over the shoulders and down the chest between the wings. Never one wing, never three or four, no membrane, no wing bones.
5. Arms and legs (2-Arm, 2-Leg Rule): exactly two arms, each hand with three white claws; exactly two legs, each foot with three white claws, both feet planted on the ground. Do not add extra limbs.
6. Tail (1-Tail Rule): exactly ONE long thick tail curving out behind the body, its underside cream, ending in a large tall flame at the tip.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the snout pointing at the viewer, the white belly in the centre, both arms out, both fire wings spread wide to the left and to the right behind the body, the tail curling out to one side behind the legs with its flame.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, snout pointing left, the head flames streaming right, arms forward to the left, the fire wings raised behind the back, the tail extending to the right with the flame on its tip.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the orange back, the back of the head and its crown of flames, the face turned away to the upper right; both fire wings spread up-left and up-right from the shoulders, the tail coming toward the viewer at the bottom left with its flame.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind: orange back, back of the head with its flames, both fire wings spread wide to the left and right, the tail hanging down the middle and curving to one side with its flame on the tip.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a heavy orange bipedal dragon with a big white pot belly with small yellow glowing dots, a crown of flames on the head, two clawed arms out, two clawed legs planted, TWO huge wings made entirely of red-orange fire (no membranes) spread behind it, and ONE long tail ending in a tall flame.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the white belly visible. Exactly two fire wings, two arms, two legs and exactly one tail, all clearly visible. Crisp 1px dark outline, limited palette, no anti-aliasing.
