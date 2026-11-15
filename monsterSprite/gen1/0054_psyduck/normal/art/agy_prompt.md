# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, both hands holding the sides of its head, tail on the lower right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, tail on the lower left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, plump, upright bipedal duck-like creature standing on two short legs with flat webbed feet. Pear-shaped body, a big round head (about as wide as the body) sitting directly on the body with no visible neck, a wide flat rounded duck bill, two blank round eyes with small black dot pupils, and a confused, dazed expression. No ears, no horns, no wings, no feathers on the body, no teeth.
2. Arms (2-Arm Rule): exactly TWO short arms, both raised so that the hands press against the two sides of the head (holding its head as if it has a headache). Each hand has three short round fingers.
3. Hair (3-Hair Rule): exactly THREE thin short black hairs (#101010 with grey #7B7B7B highlights) sticking straight up from the top of the head. Never more, never fewer.
4. Tail (1-Tail Rule): exactly ONE short stubby triangular duck tail at the lower back.
5. Feet (2-Foot Rule): exactly two flat cream webbed feet, each with three toes, both on the ground.
6. Colour palette:
   - Body, head, arms and tail: warm golden yellow (#FFD54A) with orange-tan shading (#E6A452), dark ochre-brown edge shading (#945A10 / #835A00) and a pale yellow highlight (#F6DE8B / #FFE6A4) on the top of the head.
   - Bill and feet: pale cream (#FFEEBD) with tan shading (#DEB452).
   - Eyes: white (#FFFFFF) with a small black (#101010) dot pupil.
   - Outlines: crisp 1-2px near-black pixel-art outline (#101010) mixed with dark brown (#835A00) on lit edges, no anti-aliasing halo.
   - The back of the body is mostly the orange-tan shade (#E6A452).

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, same body size in all four):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the wide bill in the centre, both hands on both sides of the head, the three hairs on top, both feet; the tail hidden or just peeking behind.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the bill pointing to the left, one arm and hand on the near side of the head, the three hairs on top, both feet (one slightly behind the other), the stubby tail sticking out to the right at the lower back.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: the back of the head and body, the face turned away to the upper right, the edge of the bill just visible on the right, both elbows/hands at the sides of the head, the three hairs, the tail on the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the back of the head, both elbows/hands at the sides of the head, the three hairs, the tail centred at the lower back, both feet.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a plump golden-yellow duck-like creature with a wide cream bill, blank round eyes with dot pupils, exactly two arms with both hands pressed to the sides of its head, exactly three thin black hairs on top, two flat cream webbed feet and one short stubby tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the bill pointing down-left, the tail just visible on the lower right. Crisp 1-2px dark outline, limited palette, no anti-aliasing. Never add extra hairs, arms, feet or tails.
