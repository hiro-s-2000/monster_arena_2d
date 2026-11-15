# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature faces lower-left, its left fist raised high on the left, its other fist low on the right) and `ref_back.png` (pixel-art back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no grid lines, no shadows on the ground.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: an angry ape-like brawler whose head and body are ONE big round shaggy ball of fur with many short spiky tufts all around its outline (no separate neck, no separate torso). Thin limbs stick out of the fur ball. Fighting stance: knees bent, legs wide apart, fists raised in a boxing guard.
2. Face (front views only): two small furious eyes (white with a black pupil) under a heavy angry brow, a big round pig-like snout (orange-pink #FF945A with darker #BD4A31 nostrils and outline) in the middle of the face, and a small brown cross-shaped anger mark (#9C6A6A) on the upper right of the forehead.
3. Colour palette (use only these families):
   - Fur ball: pale peach (#E6BD9C) with light cream highlights (#FFE6B4, #FFFFD5) on the upper fur and tan shading (#CD8B5A); the tuft tips are outlined in dark brown (#624A20 / #5A3100).
   - Arms and legs: dusty mauve-brown (#9C6A6A).
   - Fists and feet: big rounded tan-brown (#CD8B5A) with mauve shading (#9C6A6A).
   - Wrist and ankle cuffs: thick dark grey rings (#414141) with grey highlight (#7B7B7B).
   - Ear insides: orange-pink (#FF945A).
   - Outlines: crisp 1-2px black outline (#000000), no anti-aliasing halo.
4. Ears (2-Ear Rule): exactly TWO pointed ears on top of the fur ball, with orange-pink insides. Never more.
5. Limbs (2-Arm / 2-Leg Rule): exactly TWO thin arms, each ending in exactly ONE big round fist with ONE dark grey cuff at the wrist; exactly TWO thin bent legs, each with ONE dark grey cuff at the ankle and ONE big rounded foot. No tail, no wings, no extra limbs, no extra cuffs.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all fully inside their quadrant with white margin):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the snout in the centre of the face, both ears up, both fists raised at the sides of the fur ball at face height, both legs bent and wide apart.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the snout sticking out to the left, one ear on top, the near fist punched forward to the left at face height, the far fist raised close to the body, both legs visible in a wide fighting stance (one forward to the left, one back to the right).
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the furry back of the ball (no face, no snout), both ears from behind, the face turned away to the upper right; both fists raised at the sides, both legs wide apart.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the furry back (lighter cream on top, peach below), both ears from behind, both fists raised at the sides, both legs wide apart. No face.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: one big round shaggy pale-peach fur ball (head and body in one) with spiky tufts, two pointed ears with orange-pink insides, two small furious eyes, a big round orange-pink pig snout, a brown anger mark on the forehead, two thin mauve arms with big tan fists and dark grey wrist cuffs, two thin bent mauve legs with dark grey ankle cuffs and big tan feet, in a boxing stance.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the snout toward the lower left, one fist raised high on the left, the other fist lower on the right. Exactly two ears, two arms, two fists, two legs, no tail. Crisp 1-2px black outline, limited palette, no anti-aliasing, no text, no ground shadow, the whole creature fully inside the image with white margin.
