# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Both are low-resolution pixel sprites enlarged 8x. Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. Do not read or reuse any image outside this folder. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: one small round fluffy insect, a single big ball of shaggy spiky fur (head and body are the same ball, no neck, no torso), standing on two short feet. No wings, no tail, no extra legs.
2. Colour palette:
   - Fur ball: violet purple (#8B6AA4) with light lilac tufts (#A48BBD), dark purple shading (#624A7B) and deepest purple (#312952). The outline of the fur is jagged and spiky all around.
   - Eyes (2-Eye Rule): exactly TWO very large round red compound eyes (#FF5A5A with pink highlight #FFACB4, rim #B4415A and dark red #521010), set side by side on the upper front of the ball, slanted like a frown. Back views show no eyes.
   - Mouth: one small pink-red mouth (#B4415A) in the middle under the eyes with exactly two tiny white fangs (#FFFFFF) pointing down.
   - Antennae (2-Antenna Rule): exactly TWO thin antennae on top of the head, each a pale grey stalk (#D5D5D5 with #7B7B7B shading) ending in a small white oval loop tip, spreading up in a V.
   - Hands (2-Hand Rule): exactly TWO small tan claws (#BD946A, light #EEBD9C, dark #734129), each with two little hooked fingers, poking out of the lower front of the fur.
   - Feet (2-Foot Rule): exactly TWO short tan feet (#BD946A, light #EEBD9C, dark #734129) under the ball, pointing outward.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Never add more eyes, antennae, hands or feet than listed. Never add wings, a tail or a visible neck.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, the mouth with two fangs, both tan claws in front, both feet, both antennae up in a V.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one red eye on the left side of the ball, the mouth at the far left, one claw forward to the left, two feet (one stepping slightly ahead), both antennae leaning back.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: mostly the furry back, the face turned away to the upper right (only the edge of one red eye may show at the right edge), both antennae, two feet.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the furry back with a lighter lilac band across the upper back, both antennae, two feet below. No eyes, no mouth.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: one round shaggy violet-purple fur ball with jagged spiky fur outline, TWO very large round red compound eyes, one small pink mouth with two tiny white fangs, TWO thin pale grey antennae with white loop tips, TWO small tan two-fingered claws at the lower front, and TWO short tan feet.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the near (right) eye larger. Exactly two eyes, two antennae, two claws, two feet. Crisp 1-2px near-black outline, limited palette, no anti-aliasing.
