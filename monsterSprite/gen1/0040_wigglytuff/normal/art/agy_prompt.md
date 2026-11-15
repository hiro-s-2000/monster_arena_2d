# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a big, soft, egg-shaped / pear-shaped balloon-like body (wider at the bottom), no neck — head and body are one round mass. Two very short stubby arms stick out sideways from the upper sides of the body. Two short stubby rounded feet at the bottom. No tail, no wings, no fingers, no claws.
2. Colour palette:
   - Body, ears, arms and feet: soft pink (#FFACBD) with deeper pink shading (#F67B94) and dusty red-brown shadow lines (#CD6A62, #8B4141).
   - Belly: one large white oval (#FFFFFF, shaded #E6DEE6) covering the lower front of the body.
   - Eyes: exactly two big round teal-green eyes (#205A62 with #52A48B highlight and a white glint), each eye ringed by a thin dusty-red circle (#CD6A62).
   - Mouth: a small open mouth just below the eyes, brown (#734118) with a little red tongue (#E63131).
   - Ear tips: white with a grey outline (#737373); inner ears dark brown-black (#101010 / #734118).
   - Outlines: crisp 1-2px near-black outline (#101010) on the front, dusty red-brown outline (#8B4141) on the back, no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly two long, tall, pointed rabbit-like ears rising up and slightly outward from the top of the head, pink outside, dark inside, white tips. Never one, never three.
4. Forehead curl (1-Curl Rule): exactly ONE soft round pink tuft of fur on top of the head between the ears, curled into a small spiral. Only one.
5. Limbs (count rule): exactly two short arms and exactly two short feet. Do not add more limbs, do not add a tail.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both big eyes, the small open mouth, the white belly, both arms out to the sides, both ears up, the curl on top.
- Top-Right: EAST (90°, right side profile) — facing right in a sharp side profile: one eye, mouth on the right edge of the round face, one arm forward, the white belly visible on the right side, both ears up (one slightly behind the other), the curl on top.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the plain pink back and the back of the head, the face turned away to the upper right; both ears and the curl visible, only a sliver of white belly at the right edge.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: plain pink rounded back, both ears with white tips, the curl, both arms out to the sides, both feet at the bottom.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a big round egg-shaped soft pink balloon-like body with one large white belly oval, exactly two big teal eyes with dusty-red rings, a small open brown mouth with a red tongue, exactly two long pointed pink ears with dark insides and white tips, exactly one spiral curl of fur on the top of the head, two short stubby arms and two short stubby feet. No tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Crisp 1-2px dark outline, limited palette, no anti-aliasing.
