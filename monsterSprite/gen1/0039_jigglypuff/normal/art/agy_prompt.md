# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, one big eye winking) and `ref_back.png` (back 3/4 view, the creature faces upper-right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, perfectly round, balloon-like creature: one single round ball that is head and body in one (no neck, no separate torso), standing on two very short stubby feet. Face on the front of the ball: two very large round blue eyes, a tiny smiling mouth, no nose, no teeth, no claws, no wings, no tail.
2. Colour palette:
   - Whole body, ears, arms and feet: soft pink (#FFACA4) with a pale pink highlight (#FFCDC5) on the upper left and rose shading (#E67383) on the lower part and the back.
   - Eyes: large blue irises (#1873C5) with dark blue (#104A8B), a light blue lower glint (#10B4EE) and a white highlight (#FFFFFF), ringed by a thin dark red line.
   - Inside of the ears: black (#101010) with a dark grey-mauve (#6A5262).
   - Outlines: crisp 1-2px dark red outline (#A41020) on the lit side and near-black (#101010) on the shadow side, no anti-aliasing halo.
3. Head features (exact counts): exactly TWO pointed triangular cat-like ears on the top-left and top-right of the ball, pink with a black inside; exactly ONE tuft of hair on the forehead between the ears that curls down into a single spiral (drawn as a pink curl outlined in dark red on the front of the ball). Never more ears, never more curls.
4. Limbs (exact counts): exactly TWO tiny stubby nub arms (small rounded pink stubs, no fingers) sticking out from the left and right sides at the lower middle of the ball; exactly TWO very short pink feet (small rounded ovals) under the ball, spread slightly apart. Never more than two arms or two feet.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both big blue eyes open, the tiny smile, the forehead curl in the middle, both ears up, both nub arms out to the left and right, both feet visible.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one big blue eye near the left edge of the ball, the tiny mouth on the left edge, the forehead curl on the front-top of the ball, one ear visible in front and the other ear just peeking behind it, the near nub arm pointing forward to the left, the two feet one in front of the other.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: the plain rose-pink round back, the back of both ears, no eyes, no face, no curl; one nub arm at the right side, two feet.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the plain rose-pink round back, both ears (back side), both nub arms at the sides, both feet; no face, no eyes.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design and colours: a perfectly round soft-pink balloon-like ball creature (head and body one ball) with two pointed ears with black insides, one forehead hair curl, two very large round blue eyes (BOTH eyes OPEN in this sprite, no wink), a tiny smile, two tiny nub arms, two very short feet.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the near nub arm on the left, the far nub arm on the right. Exactly two ears, one curl, two eyes, two arms, two feet, no tail, no wings. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
