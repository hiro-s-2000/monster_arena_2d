# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front view, the creature faces the viewer, slightly to the lower-left) and `ref_back.png` (back 3/4 view, the creature faces away, its curled tail on the lower left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small scruffy cat standing upright on two thin hind legs. Head and body are ONE big round shaggy ball of fur (no visible neck), covered in short spiky fur tufts all around its outline, like a thick unkempt beard on the cheeks and chin. Two short arms, two thin legs, one tail. No wings, no horns, no armour.
2. Colour palette:
   - Fur (head, body, arms, legs): warm grey-brown (#ADA09A) with mid grey-brown (#84726F) and dark brown-grey shadows (#5B4E4D).
   - Dark iron parts: dark slate grey (#3D4547, #272D2E, #181A1D).
   - Outlines: crisp 1-2px black outline (#010101), no anti-aliasing halo.
3. Face: two big bright yellow eyes (#F3D91D) with thin black slit pupils and a mean, cocky look; one very wide grin showing one row of small white teeth (#F3F3F3). The spiky fur tufts stick out sideways from both cheeks.
4. Head top (exact counts): exactly ONE dark slate oval coin-like plate (#3D4547 with black stripes) on the forehead between the ears, and exactly TWO thin black curved hair tufts rising from the top of the head like little hooks. Exactly TWO pointed triangular cat ears with white insides (#F3F3F3) and dark tips.
5. Limbs (exact counts): exactly TWO short fuzzy arms ending in small paws (held out to the sides, slightly raised); exactly TWO thin legs, each ending in ONE dark slate grey foot (#3D4547 / #272D2E). Never more than two arms or two legs.
6. Tail (1-Tail Rule): exactly ONE tail: a dark slate grey tail (#3D4547 / #272D2E) curled into a round coil at the rump, low behind the body. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both yellow eyes, the wide toothy grin, the forehead plate, both ears and both hair tufts visible; both arms held out to the sides; both legs and dark feet visible; the tail hidden behind the body.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the grin opening to the left, one ear forward and one behind, the near arm reaching forward to the left, both legs visible; the single dark coiled tail at the rump on the right side.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: the shaggy grey-brown back of the round head-body, both ears and both black hair tufts seen from behind, the face turned away (no eyes, no teeth), both arms at the sides; the single dark coiled tail low at the rump.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the shaggy grey-brown back, both ears, both hair tufts, both arms at the sides, both legs and feet; the single dark coiled tail low in the middle of the rump.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small scruffy upright cat whose head and body are one round shaggy grey-brown fur ball with spiky tufts all around, two big yellow eyes with slit pupils, a very wide grin with white teeth, one dark oval plate on the forehead, two thin black hook-like hair tufts on top, two pointed ears with white insides, two short fuzzy arms held out to the sides, two thin legs with dark slate feet, and exactly ONE dark coiled tail at the rump.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the grin visible, the coiled tail peeking out behind the body on the right. Exactly one tail, two ears, two hair tufts, two arms and two legs. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
