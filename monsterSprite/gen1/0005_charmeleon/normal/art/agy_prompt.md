# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature faces lower-left, its tail goes to the right) and `ref_back.png` (pixel-art back 3/4 view, the creature faces right, tail to the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no grid lines, no shadows on the ground.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a slim, upright bipedal fire lizard, standing on two legs with a slight forward hunch. Long head with a rounded blunt snout, a small mouth line, fierce narrow eye with a blue iris (#005AFF) and a white eye highlight.
2. Colour palette (use only these families):
   - Skin of head, back, arms, legs and tail top: crimson red (#FF524A) with darker red shading (#CD3939) and deep maroon shadow (#942010), light coral highlight (#FF836A).
   - Belly and chest, and the underside of the tail: pale cream (#E6CDAC) with tan shading (#CDAC7B, #B48B5A).
   - Claws: white (#FFFFFF) with light grey shading (#B4B4B4) and grey (#6A6A6A).
   - Tail flame: orange-red (#FF4100), orange (#FFA400) and yellow (#FFDE29).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Horn (1-Horn Rule): exactly ONE single short horn on the back of the head, pointing backward and upward like a hooked crest. Never two horns.
4. Limbs (2-Arm / 2-Leg Rule): exactly TWO short arms held bent in front of the chest, each hand ending in exactly THREE small white claws; exactly TWO legs with thick thighs and small feet, each foot with three small white claws. No wings, no extra limbs, no spikes on the back.
5. Tail (1-Tail Rule): exactly ONE long thick tail, red on top and cream underneath, curving out from the lower back; the tail tip carries a burning flame (orange-red outside, orange and yellow inside). Never two tails, never two flames.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all fully inside their quadrant with white margin):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the snout pointing at the viewer, the cream belly fully visible, both arms bent in front of the chest with claws visible, both legs apart; the tail is behind the body and curls out to the RIGHT side with the flame tip beside the body at hip-to-shoulder height.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the snout pointing left, the horn pointing back to the right, both arms forward to the left, the cream belly on the left edge of the body, the tail stretching back to the RIGHT and curving up with the flame tip.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the red back, the back of the head and the horn, the face turned away to the upper right; the tail comes out of the lower back toward the viewer and curls to the LEFT with the flame tip.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the red back, the back of the head with the single horn, both arms only peeking at the sides; the tail comes down from the lower back and curls to the LEFT with the flame tip.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a slim upright red fire lizard with a cream belly, one backward hooked horn on the head, two bent arms with three white claws each, two legs with small white claws, and ONE long tail (red on top, cream underneath) with a burning flame at its tip (orange-red, orange, yellow).
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with the eye visible in natural 3/4 perspective (not squished), the cream belly visible, the tail curling out behind to the RIGHT with the flame tip at shoulder height. Exactly one horn, one tail, one flame, two arms, two legs. Crisp 1-2px near-black outline, limited palette, no anti-aliasing, no text, no ground shadow, the whole creature fully inside the image with white margin.
