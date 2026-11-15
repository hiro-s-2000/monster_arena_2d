# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature faces lower-left, it holds a spoon up on the LEFT side of the picture, its tail goes to the right) and `ref_back.png` (pixel-art back 3/4 view, the creature faces right, the spoon is on the right, the tail to the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no grid lines, no shadows on the ground.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a slim, upright bipedal psychic fox-like humanoid standing on two thin legs with a slight crouch. Fox-like head with a long narrow pointed snout, narrow slanted eyes (white eye with a small dark pupil), and a long thin droopy moustache: exactly TWO thin whiskers hanging down from the sides of the snout.
2. Colour palette (use only these families):
   - Skin of head, arms, legs and tail: bright yellow (#FFEE29) with golden shading (#D5BD08), olive-ochre shadow (#8B7310, #624A08) and pale cream highlight (#FFFFBD).
   - Armour-like plates on the shoulders, upper chest and the whole back: brown (#A47329) with dark brown shading (#7B5208, #5A3908) and a light tan highlight (#C5A452).
   - Forehead mark: one small red five-pointed star (#E63929) in the middle of the forehead.
   - Belly mark: a small red zigzag symbol (#E63929) on the lower belly (front views only).
   - Spoon: one silver spoon (#FFFFFF, #DEDEDE, #B4B4B4, #5A5A5A): an oval bowl on a short straight handle.
   - Toe claws: white (#FFFFFF) with grey (#B4B4B4).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO large pointed triangular ears on top of the head, sticking up and outward like a fox. Never more.
4. Arms (2-Arm Rule, 1-Spoon Rule): exactly TWO thin arms. One hand holds exactly ONE silver spoon raised up beside the head, bowl up; the other hand is open with three thin fingers, held out to the side. Never two spoons.
5. Legs (2-Leg Rule): exactly TWO thin legs with knobby knees, each foot with three toes tipped by small white claws.
6. Tail (1-Tail Rule): exactly ONE big bushy fox tail, yellow with one brown stripe (#A47329) along it, coming out of the lower back and curving up to the side. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all fully inside their quadrant with white margin):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, nearly symmetric: both eyes, the snout pointing at the viewer, the red star on the forehead, both ears up, the brown shoulder plates on both sides, the red zigzag on the belly; the spoon is raised beside the head on the LEFT side of the picture, the open hand is held out on the RIGHT side; both legs apart; the tail is behind the body and sweeps out to the RIGHT.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the long snout pointing left with the whiskers hanging down, the ears pointing up and back, the spoon raised in front of the chest pointing up-left, the brown back plate on the right edge of the body, the tail stretching back to the RIGHT and curving up.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the big brown back plates and the back of the head with both ears, the face turned away to the upper right; the spoon raised on the RIGHT side beside the head; the tail comes out of the lower back and sweeps to the LEFT.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the brown back plates covering the back and shoulders, the back of the head with both ears, both arms peeking out at the sides (spoon raised on the RIGHT side); the tail comes down from the lower back and sweeps to the LEFT.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a slim upright yellow psychic fox-like humanoid with two pointed ears, a red star on the forehead, a long pointed snout with two thin droopy whiskers, brown armour plates on the shoulders and chest, a red zigzag on the belly, two thin arms (one raising ONE silver spoon beside the head on the LEFT of the picture, the other with three open fingers on the right), two thin legs with white toe claws, and ONE big bushy yellow tail with a brown stripe sweeping out behind to the RIGHT.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Exactly two ears, one spoon, two arms, two legs, one tail. Crisp 1-2px near-black outline, limited palette, no anti-aliasing, no text, no ground shadow, the whole creature fully inside the image with white margin.
