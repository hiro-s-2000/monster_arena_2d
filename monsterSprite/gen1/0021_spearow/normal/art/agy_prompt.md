# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, seen from behind, it faces up-right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Save only inside the `out/` folder next to this prompt. Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel-art sprites: a 3/4 front view and a back view) into larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, plump, scruffy SPARROW-LIKE BIRD standing upright on two thin legs. Round body, big head, short neck. It is a bird, not a mammal: no fur, no ears, no arms, no hands.
2. Colour palette:
   - Head and upper back: warm brown (#9C5218) with dark brown shading (#6A4118).
   - Crest: a short shaggy crest of exactly THREE pointed spikes on top of the head, light tan-orange (#EEAC52) with orange (#C56A20) shading.
   - Beak: one short, thick, slightly hooked beak, pale pink (#FFD5CD / #FFA48B) with a dark-red underside (#AC414A).
   - Eyes: two small fierce dark eyes under angry brown brows.
   - Wings: exactly TWO wings, folded against the sides of the body, coral red (#EE524A) with darker red (#AC414A) and light pink highlights (#FFA48B).
   - Belly and thighs: grey-brown (#8B7B62) with dark grey horizontal stripes (#4A4139) and light beige (#C5B49C) highlights.
   - Tail: a short fan of exactly THREE stiff brown-orange tail feathers (#C56A20 / #9C5218) sticking out behind the body.
   - Legs and feet: exactly TWO thin coral-pink legs (#EE524A), each foot with three forward toes and grey-white claws.
   - Outlines: crisp 1px black pixel-art outline (#000000), no anti-aliasing halo.
3. Count rules (most important): exactly one head, one beak, three crest spikes, two wings, two legs, one tail fan of three feathers. Never add arms, extra wings, extra legs, extra tails or a long neck. Do not invent new colours or markings.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, each bird centred in its quarter with white space around it):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the beak pointing toward the viewer, the striped belly between the two folded red wings, both legs visible, the tail feathers peeking out behind on both sides.
- Top-Right: EAST (90°, right side profile) — facing right in a sharp side profile: one eye, the hooked beak pointing right, one folded red wing on the side of the body, the tail fan sticking out to the left, both legs visible under the body.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the brown back of the head with the three crest spikes, the beak tip just visible on the upper right, both red wings on the back, the tail fan pointing down-left toward the viewer.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: back of the brown head with the crest, both red wings folded on the back, the tail fan of three feathers centred at the bottom of the back, both legs below.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small plump scruffy brown sparrow-like bird with a shaggy crest of three light tan-orange spikes, a short thick pale-pink hooked beak, fierce dark eyes, exactly two folded coral-red wings, a grey-brown belly with dark stripes, a short fan of three brown-orange tail feathers, and exactly two thin coral-pink legs with three-toed feet.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left (like the front reference): the head and beak look diagonally down-left, both eyes visible in natural 3/4 perspective, the near wing on the right side of the body, the tail feathers sticking out to the right behind the body. Exactly two wings, two legs. Crisp 1px black outline, limited palette, no anti-aliasing. The bird is centred with white space around it.
