# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature faces lower-left, its tail goes to the lower left behind it) and `ref_back.png` (pixel-art back 3/4 view, the creature faces right, its tail to the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no grid lines, no shadows on the ground.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a heavy, stocky, upright bipedal armoured reptile-like beast with a broad barrel chest, thick neck, short powerful arms held out to the sides and two thick sturdy legs. Small head with a short blunt muzzle, a cream lower jaw, one narrow dark eye with a white highlight on each side of the face.
2. Colour palette (use only these families):
   - Hide of head, back, arms, legs and tail: steel blue (#628BAC) with bright cyan-blue highlights (#5AB4CD, #9CD5DE) and dark teal shadow / seams (#314A4A).
   - Lower jaw / chin, the two chest plates and the belly band: pale cream (#EEE6A4) and sand (#E6CD8B) with olive-tan shading (#BDAC5A) and brown (#735A29).
   - Inside of the ears: brown (#735A29) with tan (#BDAC5A).
   - Claws: white (#FFFFFF) with light grey (#D5D5D5).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Head (1-Horn / 2-Ear Rule): exactly ONE short pointed horn on the forehead, and exactly TWO large rounded ears with brown insides on top of the head, pointing up and outward. A few small pointed spikes on the top of the head between the ears.
4. Chest: exactly TWO rounded cream chest plates side by side on the upper chest, and ONE wide cream belly band across the lower belly. The rest of the front is blue.
5. Limbs (2-Arm / 2-Leg Rule): exactly TWO short thick arms, each hand ending in exactly THREE small white claws; exactly TWO thick legs with big feet, each foot with two or three small white claws. No wings, no extra limbs.
6. Back spikes: one row of about FIVE large blue triangular spikes running down the back from the neck to the base of the tail (biggest on the shoulders, smaller toward the tail). They point backward and up.
7. Tail (1-Tail Rule): exactly ONE thick, heavy, segmented armoured tail, blue with dark teal segment lines and small spiky ridges, coming from the lower back and lying on the ground, tapering to a point. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all fully inside their quadrant with white margin):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the cream jaw, the two cream chest plates and the cream belly band fully visible, both arms held out to the left and right with claws visible, both legs apart; the tail is behind the body and pokes out on the RIGHT side near the ground.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, one ear, the muzzle and horn pointing left, both arms forward to the left, the cream chest and belly on the left edge of the body, the row of back spikes on the right edge, the tail stretching back to the RIGHT along the ground.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the blue back with the row of spikes, the back of the head and both ears, the face turned away to the upper right; the tail comes out of the lower back toward the viewer and lies on the ground to the LEFT.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the blue back with the row of spikes down the middle, the back of the head with both ears, both arms peeking out at the sides; the tail comes down from the lower back and lies on the ground curving to the LEFT.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a heavy stocky upright steel-blue armoured beast with a cream jaw, two cream chest plates and a cream belly band, one short horn on the forehead, two large rounded ears with brown insides, two thick arms held out with three white claws each, two thick legs with white claws, spikes down the back and ONE thick segmented armoured tail lying on the ground.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream chest plates and belly band visible, the tail coming out behind to the RIGHT along the ground. Exactly one horn, two ears, two arms, two legs, one tail. Crisp 1-2px near-black outline, limited palette, no anti-aliasing, no text, no ground shadow, the whole creature fully inside the image with white margin.
