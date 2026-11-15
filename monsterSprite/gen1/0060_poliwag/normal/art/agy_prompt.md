# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the lower right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, tail on the lower left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small round tadpole creature. Head and body are ONE single round ball (no neck, no separate head), standing on two short stubby legs with rounded feet. NO arms, NO hands, no ears, no horns, no hair, no fins on the body, no wings.
2. Colour palette:
   - Body ball and legs: medium blue (#3973AC) with lighter blue (#5283C5), a pale periwinkle highlight on the top of the ball (#8BA4E6), and dark navy shadows (#183973 / #395A83).
   - Belly spiral (front only): a large round patch on the lower front of the ball with a clockwise spiral drawn in black (#101010) lines on white and light grey (#FFFFFF / #CDCDCD). Exactly ONE spiral, centred on the belly.
   - Mouth: ONE small round puckered pink-red lips (#FF6A6A, darker rim #BD5A4A, light pink #FFBDAC) in the middle of the face, between and just below the two eyes.
   - Eyes: exactly two large eyes on the upper front of the ball, white with a big black pupil.
   - Tail fin: white and pale lavender-grey (#FFFFFF / #B4ACBD / #CDCDCD) with a dark grey edge (#626262) and a few thin navy-blue streaks (#183973) inside the fin.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Limbs (exact counts): exactly TWO short blue legs with rounded feet under the ball. Never arms, never more than two legs.
4. Tail (1-Tail Rule, most important): exactly ONE tail. It is a thin dark-blue stalk that grows from the lower back of the ball near the ground and ends in ONE big flat oval translucent white fin (like a tadpole tail), about as long as the ball is wide. Never two tails, never two fins.
5. The back of the ball is plain blue with no spiral, no face and no markings.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the pink lips in the centre, the white-and-black spiral belly, both legs visible; the single tail fin peeks out behind the ball at the lower right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the pink lips sticking out on the left edge of the ball, the edge of the spiral belly on the lower left; the single tail with its white fin sticks out behind at the lower right, near the ground.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: a plain blue ball seen from behind, only the edge of one eye and the pink lips visible on the far upper-right edge, no spiral; the single tail comes from the lower back and its white fin stretches out to the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: a plain blue ball with a light highlight on top, both legs below; the single tail comes out from the lower back at the centre and its white fin is fully visible, sticking out to the lower left near the ground.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a round blue ball-shaped tadpole creature with no arms, two big white eyes with black pupils, small round pink puckered lips, a black-and-white spiral on its belly, two short stubby blue legs, and exactly ONE tail: a thin dark-blue stalk ending in ONE big oval white translucent fin.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the lips slightly left of centre, the spiral belly visible, the tail fin behind the body at the lower right. Exactly one tail, two legs, no arms. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
