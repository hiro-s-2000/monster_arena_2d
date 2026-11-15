# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, seen from behind-left, the head turned to the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel-art front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large, heavy, armoured bipedal reptile-like brute standing upright in a wide crouching stance on two thick legs, leaning slightly forward, two strong arms held out in front. Massive chest and shoulders, a very thick heavy tail.
2. Colour palette (limited, no other hues):
   - Body, head, arms, legs and tail: purple-magenta (#A45294) with light lilac highlights (#D573EE, #DE94F6) and dark plum shading (#5A296A).
   - Chest and belly: one large cream plate (#C5C5A4) with off-white highlight (#E6E6D5, #FFFFFF) and olive-grey shading (#73735A).
   - Inner ears: teal blue (#297B94, #4194C5, deep #004A62).
   - Mouth inside: red (#AC1000).
   - Claws: cream/olive (#C5C5A4, #73735A).
   - Outlines: crisp 1px near-black outline (#101010), no anti-aliasing halo.
3. Head: one single long sharp horn pointing forward-up from the forehead (exactly ONE horn), two large pointed ears with teal insides (exactly TWO ears), small spikes on top of the head, eyes small and fierce, mouth slightly open.
4. Arms (2-Arm Rule): exactly TWO arms, each ends in a hand with exactly THREE short cream claws.
5. Legs (2-Leg Rule): exactly TWO thick legs, each foot with cream claws.
6. Back: a row of many sharp lilac spikes running down the back from the head to the base of the tail.
7. Tail (1-Tail Rule): exactly ONE very thick, long, segmented tail (dark plum ring lines across it), curving up from the rump. Never two tails.
8. Body segmented by dark plum armour lines (shoulder plates, leg plates). Do not add wings, extra limbs, extra horns or extra tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the horn pointing up in the centre, both ears with teal insides, the cream chest plate, both arms held out to the sides with claws, both legs apart, the tail visible behind the body.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, horn pointing forward-left, snout to the left, arms forward to the left, the back spikes along the right edge, the thick tail extending behind to the right and curling up.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back with its row of spikes, the back of the head, the face turned away to the upper right, the thick tail coming toward the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the row of back spikes down the centre, both ears, the tip of the horn, the thick segmented tail hanging down in the centre to the ground.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a heavy purple armoured biped with a cream chest plate, exactly one forehead horn, two large ears with teal insides, a row of lilac back spikes, two arms with three cream claws each, two thick legs in a wide crouching stance, and exactly one very thick segmented tail curving up behind the body to the right.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream chest plate visible, the near arm reaching forward-left. Exactly one horn, two ears, two arms, two legs, one tail. Crisp 1px dark outline, limited palette, no anti-aliasing.
