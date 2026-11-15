# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, the creature faces right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel sprites: front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: an elegant, slender four-legged snow fox standing on all four legs (not sitting), long neck held high, head up, calm and graceful, wrapped in a big soft cloud of flowing icy fur.
2. Colour palette:
   - Fur of the body, head, legs and tails: icy white (#EAFEFE) with pale ice blue (#B0D3EA) and grey-blue shading (#BDCBD7, #86A8C0, #7A8A9C).
   - A soft mint-teal patch of fur on the flank / chest, under the flowing hair (#81C2C5 with light mint #A1E1DE and dark teal #38858D).
   - Eyes: blue (#007AB5) with light blue (#66B3D7), narrow and calm. Small dark nose.
   - Outlines: crisp 1-2px slate outline (#516077) with black (#101010) on the outer edge, no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with a slim pointed fox snout, exactly two pointed upright ears (small blue inside), and a large soft puff of fur on top of the head flowing back like long hair, with long wavy locks hanging down beside the neck.
4. Legs (4-Leg Rule): exactly four slender grey-blue legs with small paws, all four standing on the ground. Never more, never fewer.
5. Tails (9-Tail Rule, most important): exactly NINE long, wavy, hair-like tails, each ending in a thin curly wisp, all growing from the rump and flowing up and back behind the body like a big soft cloud of icy hair. The tail cloud is about as big as the body. Do not add or remove tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: face, both blue eyes, both ears, the fur puff on the head and the wavy locks beside the neck visible, both front legs in front, the nine tails flowing out behind the body to the left and to the right above the back.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, snout to the left, four legs seen from the side (two near, two far), the nine tails flowing up and back behind the body on the right side.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the head turned away to the upper right, the back of the head and fur puff, the nine tails flowing toward the viewer covering much of the back.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the nine tails spread in a big symmetric cloud in front of us, the back of the head with the two ears and fur puff peeking above them, the hind paws below.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design and colours: an elegant slender icy-white snow fox with pale blue shading, blue eyes, two pointed ears, a big soft puff of fur on the head with long wavy locks beside the neck, a mint-teal patch on the flank, four slender grey-blue legs, and exactly NINE long wavy hair-like tails with curly wisp tips flowing up and back behind the body.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left, standing on all four legs (not sitting): the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the tail cloud behind the body to the right. Exactly four legs, exactly nine tails. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
