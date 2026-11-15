# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front view, the creature faces the viewer, slightly to the lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front view and back view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a round frog-like tadpole creature. Head and body are ONE single round ball (no neck), standing upright on two short legs, with two arms held out to the sides like a boxer.
2. Colour palette:
   - Body ball, arms and legs: medium blue (#5283C5) with darker blue (#3973AC), a pale periwinkle highlight (#8BA4E6) and dark navy shadows (#395A83 / #183973).
   - Belly spiral (front only): a very large round white patch covering most of the front of the ball, with ONE clockwise spiral drawn in thick black (#101010) lines on white and light grey (#FFFFFF / #D5D5DE / #BDBDBD). Exactly ONE spiral.
   - Hands: two big round white fists like boxing gloves (#FFFFFF / #D5D5DE / #BDBDBD with grey #A4A4A4 / #525252 shading), at the end of short blue arms.
   - Eyes: exactly two eyes on two small bumps on top of the ball, white with a big black pupil.
   - Mouth: no visible lips; only one tiny pale periwinkle bump (#8BA4E6) between the eyes.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Limbs (exact counts): exactly TWO short blue arms each ending in ONE white fist, and exactly TWO short blue legs with flat rounded blue feet. Never more arms or legs.
4. NO tail, no fins, no wings, no ears, no horns, no hair.
5. The back of the ball is plain dark blue (#395A83) with a lighter blue top (#5283C5); no spiral and no face on the back. Two eye bumps are visible on top from behind.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes on top, the big spiral belly in the centre, both white fists held out at the left and right sides, both legs visible.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye bump on top, the edge of the spiral belly on the left side of the ball, one white fist in front on the left and the other fist partly visible behind, both legs below.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: a plain blue ball seen from behind, the two eye bumps on top turned away to the upper right, no spiral visible, both white fists at the sides, both legs below.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: a plain dark-blue ball with a lighter top, two eye bumps on top, both white fists at the left and right sides, both legs below.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a round blue ball-shaped frog-like creature, two eyes on two small bumps on top, a very large white belly with one black spiral, two short blue arms ending in big round white fists held out like a boxer, two short blue legs with flat feet, and no tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: both eyes visible in natural 3/4 perspective (not squished), the spiral belly slightly left of centre, both white fists visible (the near fist on the left, the far fist on the right). Exactly two arms, two fists, two legs, no tail. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
