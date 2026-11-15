# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature faces lower-left) and `ref_back.png` (pixel-art back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no grid lines, no shadows on the ground.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a slim, upright, humanoid psychic creature with a fox-like face, standing on two thin legs. Narrow slanted eyes (dark slits), a pointed snout, a slim waist.
2. Colour palette (use only these families):
   - Skin of head, ears, hands, upper arms, belly and legs: bright golden yellow (#FFEE18) with mid yellow shading (#CDB410) and pale yellow highlight (#FFF6AC).
   - Shoulder pads, forearms, chest plate and knee pads: brown armour (#B48329) with dark brown shading (#7B5A08, #523108).
   - Spoons and toe claws: silver (#DEDEDE, #B4B4B4, #FFFFFF highlight, #5A5A5A shadow).
   - Moustache tips: pale cream yellow (#FFF6AC).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Head (2-Ear Rule): exactly TWO long sharp pointed ears/horns rising up and outward from the top of the head in a V, like a star. No other horns.
4. Moustache (2-Whisker Rule): exactly TWO long thin moustache whiskers growing from under the snout, hanging down and outward to the left and to the right beside the belly, each ending in a pale cream-yellow tufted tip at hip height.
5. Arms and spoons (2-Arm / 2-Spoon Rule): exactly TWO arms; each arm has a big round brown shoulder pad and a brown forearm, and each yellow hand holds exactly ONE silver spoon raised upward beside the head. Exactly two spoons, one in each hand. Never three.
6. Legs (2-Leg Rule): exactly TWO thin legs with round brown knee pads, yellow shins and feet with small silver claws. No tail, no wings.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all fully inside their quadrant with white margin):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the snout pointing at the viewer, both ears up in a V, the brown chest plate visible, both arms bent upward with one spoon on the left and one spoon on the right beside the head, both whiskers hanging down to the left and to the right, legs apart.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the snout pointing left, both ears pointing up and back, both hands held forward to the left at chest height each with its spoon pointing up, the whiskers hanging down in front of the body on the left.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back of the head with both ears, the brown shoulder pads and brown back, the face turned away to the upper right; both arms raised with the spoons visible above the shoulders; whisker tips peek out at the sides.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: back of the head and both ears, the big brown shoulder pads and brown back, both arms raised with a spoon on each side, whisker tips peeking out to the left and right, two legs.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a slim upright golden-yellow humanoid with a fox-like face, two long pointed ears in a V, two long moustache whiskers with pale cream tips hanging down to both sides, brown round shoulder pads, brown forearms, a brown chest plate, brown knee pads, and two hands each holding ONE silver spoon raised beside the head.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Exactly two ears, two whiskers, two arms, two spoons, two legs, no tail. Crisp 1-2px near-black outline, limited palette, no anti-aliasing, no text, no ground shadow, the whole creature fully inside the image with white margin.
