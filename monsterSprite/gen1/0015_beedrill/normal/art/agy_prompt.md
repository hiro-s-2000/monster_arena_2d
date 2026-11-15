# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front pixel-art view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. Other jobs run at the same time on this machine: NEVER write to /tmp or any shared folder and never reuse an image you did not generate in this session; write intermediate files only inside `out/` with names starting `wasp15_` and delete them at the end. Before saving, look at each generated image and check it shows the yellow-and-black striped flying insect with two silver lances and four wings; if it shows any other creature, generate again. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference images (small pixel-art sprites: front 3/4 view and back view) as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large flying wasp-like insect hovering in the air (it never stands on the ground). Three segments: a round head, a small thorax, and a big oval striped abdomen hanging down and slightly back, ending in a pointed stinger.
2. Colour palette:
   - Head, thorax and abdomen: golden yellow (#FFCD4A) with orange (#E69420) and brown shadow (#B46210, #834A00).
   - Abdomen stripes: exactly two thick black bands (#181818) across the yellow abdomen.
   - Abdomen tip: one short yellow cone stinger (#FFCD4A / #FFF6A4) pointing down.
   - Eyes: two large red compound eyes (#D51831 with #9C0008 shadow and #FF946A highlight), one on each side of the head.
   - Lances: pale lavender-silver (#DECDF6, #BDACC5, shadow #8B7B94), outlined dark.
   - Legs and antennae: thin black / dark slate (#181818, #525A7B).
   - Wings: translucent white with pale lavender veins and outline (#FFFFFF, #EEE6FF, #BDACC5, #8B7B94).
   - Outlines: crisp 1px dark pixel-art outline (#181818), no anti-aliasing halo.
3. Antennae (2-Antenna Rule): exactly two thin black antennae, bent at an angle, sticking up and forward from the top of the head.
4. Arms (2-Lance Rule, most important): exactly TWO front arms; each arm is a thin black limb ending in one very large cone-shaped drill lance (long sharp spike, as long as the head is wide x3), pale lavender-silver. One lance on each side of the body. Never one, never three, never more.
5. Legs (4-Leg Rule): exactly FOUR thin black jointed legs hanging down from the thorax (two on each side), bent at the knee. Never six.
6. Wings (4-Wing Rule): exactly FOUR translucent white wings (two large upper wings, two smaller lower wings) on the back, spread out behind the body.
7. Do not add a mouth, teeth, hair, extra limbs or extra lances.

GRID LAYOUT (4 key poses, large and crisp, same size, the stinger tips at about the same height in every cell):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, both antennae, both lances raised one on the left and one on the right, the four wings spread behind, the four legs hanging, the striped abdomen and stinger below.
- Top-Right: EAST (90°, right side profile) — facing right in a sharp side profile: one red eye, antennae and both lances pointing forward to the right, the abdomen hanging down-back to the left, wings up behind, legs hanging below.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the four wings on the back, the back of the head turned away to the upper right, the lances partly visible at the sides, the striped abdomen and stinger below.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: four wings spread, back of the head, the two lances at the sides, abdomen with the two black stripes and stinger below.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a hovering golden-yellow wasp-like insect with a round head, two big red compound eyes, two bent black antennae, exactly TWO front arms each ending in one huge pale lavender-silver cone drill lance, exactly FOUR thin black hanging legs, exactly FOUR translucent white wings, and a big oval yellow abdomen with exactly two black stripes ending in a yellow cone stinger.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the head looks diagonally down-left with both red eyes visible in natural 3/4 perspective (not squished), one lance pointing up-left and one lance pointing down-left. Crisp 1px dark outline, limited palette, no anti-aliasing.
