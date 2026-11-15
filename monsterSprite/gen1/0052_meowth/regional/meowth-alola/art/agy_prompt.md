# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the right) and `ref_back.png` (back view, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, slim cat standing upright on its two hind legs, with a very big round head (as wide as the whole body with the whiskers), a slim short torso, two thin arms held out to the sides with small round paws, two thin legs with flat white feet. Sly half-closed eyes and a smug look. No wings, no horns, no armour.
2. Colour palette:
   - Head, body, arms, legs and tail: dusty blue-grey (#91A3BF) with slate shading (#627986) and dark slate shadows (#45505F).
   - Ears: dark navy-black outer ears (#262B3C) with light grey inside.
   - Forehead: ONE oval gold coin set in the forehead, gold (#E3CC2B) with orange (#CA833C) and brown (#995433) shading and dark horizontal grooves.
   - Eyes: white eyes (#F0F0F0) with small black pupils, heavy dark eyelids.
   - Mouth: a thin smug mouth with two tiny white fangs.
   - Feet: white (#F0F0F0) with light grey shading (#B5B6B9).
   - Tail tip: one round white-grey curl (#F0F0F0 / #B5B6B9 / #798071).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO pointed cat ears at the top corners of the head, dark navy-black.
4. Whiskers and hair (exact counts): exactly THREE long thin blue-grey whiskers on EACH side of the face (six in total), sticking out sideways; two thin hair strands rise from the top of the head between the ears, beside the coin. Never more.
5. Limbs (exact counts): exactly TWO thin arms, each ending in one small round paw; exactly TWO thin legs, each ending in one white foot. Never more than two arms or two legs.
6. Tail (1-Tail Rule): exactly ONE long thin blue-grey tail from the rump that ends in ONE round white curl. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the gold coin on the forehead, three whiskers on each side, both arms held out to the sides, both white feet visible; the single tail is behind the body and curls out on the right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the face pointing to the left, the coin seen from the side on the forehead, the near arm reaching forward to the left; the single tail extends behind to the right and ends in the round curl.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the blue-grey back and the back of the big round head with both ears, the face turned away to the upper right (only the edge of the coin and the whiskers visible); the single tail comes from the rump and curls out to the left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the back of the round head, both ears, the whiskers sticking out on both sides, both arms out to the sides, both legs; the single tail comes out of the rump and curls out to the left.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a slim dusty blue-grey cat standing upright on two thin legs with white feet, a very big round head with two dark navy pointed ears, ONE oval gold coin on the forehead, sly half-closed eyes, three long whiskers on each side of the face, two thin arms held out to the sides with small round paws, and exactly ONE long thin tail ending in one round white curl.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the coin on the forehead, the tail behind the body to the right ending in the round curl beside the right leg. Exactly one tail, exactly two ears, two arms and two legs, three whiskers per side. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
