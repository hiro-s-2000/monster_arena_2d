# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (official illustration, 3/4 front view, head turned to the right) and `ref_pose2.png` (another 3/4 front illustration of the same creature; ignore its checkerboard background). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

If a result breaks a rule, regenerate (up to 3 tries each) and keep the best under these names. Do not write or modify any other file. When both PNGs are saved, reply with their paths and image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference illustrations into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a sleek floating jet-dragon creature, no legs. Upright floating posture: a slim white neck rising from a rounded torso, the lower body tapers down into a pointed tail with two small pointed side fins.
2. Colours: lavender-purple plates (#9C88C8, highlight #B8A8DC, shadow #7464A8); white-grey chest, neck and face (#ECECF2, shading #B8B8C8); grey joints (#9898A8); eyes amber/brown (#B07850) with a dark pupil; crisp 1-2px dark outline (#2A2433), no anti-aliasing halo.
3. Head: rounded lavender cap on top of a white face, two short pointed horn-like ear fins pointing up-back, a short blunt snout.
4. Wings/arms (most important): exactly TWO huge flat swept jet-wings, one on each shoulder, each joined to the body by a short grey arm connector. Each wing is a broad flat triangle with a saw-toothed trailing edge (3 short points). A small grey three-claw hand with white claws sits at the leading tip of each wing. The front face of each wing has a small grey triangle marking. Never more than two wings.
5. Chest: white-grey front with jagged white markings running down over the lavender belly sides.

GRID LAYOUT (4 key poses, large and crisp, same size, body bottom on the same baseline):
- Top-Left: SOUTH (front view) — facing straight toward the viewer, symmetric: both amber eyes, white chest, both wings spread left and right and slightly up, tail tip pointing down.
- Top-Right: WEST (left side profile) — facing LEFT in a sharp side profile: one eye, snout pointing left, the near wing swept back up to the right, the far wing peeking behind it, body leaning slightly forward, tail pointing down-right.
- Bottom-Left: NORTH-EAST (back-right 3/4 view) — seen from behind and to the right: lavender back and the back of the head, face turned away to the upper right, both wings spread, their back (plain lavender, no triangle mark) visible; the wing roots and shoulder joints are on the camera side, in front of the back.
- Bottom-Right: NORTH (pure back view) — straight from behind: lavender back, back of the head with both ear fins, both wings spread symmetrically left and right showing their plain backs, tail tip down. No face, no white chest visible.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature: sleek floating lavender-purple and white jet-dragon, no legs, white neck and chest with jagged white markings, rounded lavender head cap with two short ear fins, amber eyes, TWO huge flat swept saw-toothed jet-wings on grey arm connectors with a small grey three-claw hand at each wing tip and a grey triangle marking on each wing front, pointed tail with two small fins.
View: SOUTH-WEST 3/4 view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective, the near (left) wing spread to the left, the far wing spread up to the upper right behind the body, tail tip pointing down. Crisp 1-2px dark outline, limited palette, no anti-aliasing. Whole creature inside the image with margin.
