# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (the creature seen from behind and to the left: the huge shell faces the viewer, the small head is at the upper left, it faces left) and `ref_back.png` (the creature seen from the front-right: the orange belly, the face and one arm are visible, it faces right). Both are enlarged low-resolution pixel sprites. Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images into crisp, detailed pixel-art sprites, keeping its exact design, proportions and colours. Do not add any part that is not in the references.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a huge, heavy, bulky bipedal turtle standing upright on exactly TWO very thick short legs. A small rounded head with a blunt snout on a short neck (no ears, no horns), exactly TWO short thick arms with small white claws, a heavy round torso and exactly ONE short thick tail behind. The shell on its back is gigantic, much bigger than the body.
2. Colour palette (use only these families):
   - Skin (head, arms, legs, tail): royal blue (#2062AC) with light blue highlights (#5A8BCD), pale blue (#94ACE6) and dark navy shadow (#083962).
   - Glowing markings: small cyan (#6CE8D6) eyes, and a few small cyan and pale-blue triangle markings on the legs and the tail.
   - Belly: a rust-orange plated front (#C75435, light #EA7957) crossed by several dark horizontal grooves.
   - Shell: a massive angular armoured shell in dark grey-brown (#5F5647 mid, #494136 dark, #352E27 deepest), built from big blocky plates, edged all around by a thick white rim (#FDFDFD with light grey #CDCDD5) where it meets the body.
   - Cannons: grey metal (#949494, #4A4A4A) with white highlights. Exactly ONE huge boxy main cannon block on top of the shell with a big rectangular dark muzzle, and exactly SIX short grey multi-barrel nozzles poking out around the edge of the shell. Several square hatch openings on the shell plates outlined in white.
   - Claws: three small white (#FDFDFD) claws on each foot.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Counting rules: exactly two legs, exactly two arms, exactly one tail, exactly one head, exactly one huge main cannon block, exactly six small nozzles. Never add extra legs, arms, heads or cannons. Do not increase any count. No water effects.

GRID LAYOUT (4 key poses, large and crisp, same scale, both feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the small blue head with both cyan eyes in the centre top, the orange grooved belly in the middle, the two arms at the left and right of the belly, the two thick legs below, the white-rimmed shell spreading wide behind the body on both sides with the nozzles sticking out to the left and right and the main cannon block rising above the head.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: the head and snout at the left, one eye, the orange belly at the left front, one arm in front, the two legs (near and far), the giant shell on the right side of the body with its nozzles, the main cannon block on top, the tail at the lower right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the first reference mirrored: the giant shell with the main cannon block and nozzles faces the viewer, the small head barely visible at the upper right, the legs and the tail at the bottom.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the giant shell fills the view with the main cannon block on top in the centre and three nozzles on each side, the white rim around it, the tail at the bottom centre, the two legs below at the left and right. No face visible.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached references exactly in design and colours: a huge bulky blue bipedal turtle with a small head with cyan eyes, a rust-orange grooved belly, two short arms, two very thick legs with cyan triangle markings and white claws, one short tail, and a gigantic angular dark grey-brown shell with a thick white rim, one huge boxy main cannon block on top and six short grey multi-barrel nozzles around its edge.
View: SOUTH-WEST 3/4 quarter view: the face and the orange belly turned diagonally down-left toward the viewer, both eyes visible in natural 3/4 perspective, the giant shell behind the body to the upper right. Exactly two legs, two arms, one tail. Crisp 1-2px dark outline, limited palette, no anti-aliasing. No extra parts.
