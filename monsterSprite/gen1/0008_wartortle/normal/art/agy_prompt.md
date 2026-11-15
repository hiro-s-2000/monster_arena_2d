# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature faces lower-left, its fluffy tail is on the right) and `ref_back.png` (pixel-art back 3/4 view, the creature faces right, the fluffy tail is on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no grid lines, no shadows on the ground.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small upright bipedal turtle standing on two short thick legs, wearing a round shell. Rounded head with a short blunt snout, a small mouth line, a determined brown eye (#6A4A18 / #8B5A20) with a white highlight, a tiny pink cheek blush (#946262).
2. Colour palette (use only these families):
   - Skin of head, arms and legs: periwinkle blue (#8BACF6) with mid blue shading (#627BC5), deep navy shadow (#29416A) and pale blue highlight (#ACC5FF).
   - Belly plate (front of the shell): pale sand yellow (#DEC58B) with tan lines (#AC8B62) and brown edge (#8B5A20).
   - Back shell: reddish brown plates (#946262) separated by dark lines, a warm orange-brown top plate (#CD8320) near the neck, a thin white rim (#FFFFFF) and a dark brown rim (#6A4A18) around the shell.
   - Ears and tail: very pale blue-white fur (#D5EEF6, #BDC5E6) with light blue lines (#8BACF6) and navy curls (#29416A).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO large feathery, wing-like ears, pale blue-white with blue streaks, rising up and outward from the top sides of the head. Never more than two.
4. Limbs (2-Arm / 2-Leg Rule): exactly TWO short stubby blue arms with rounded hands (small fingers only as tiny lines), exactly TWO short thick blue legs with rounded feet. No claws, no extra limbs, no spikes on the shell.
5. Tail (1-Tail Rule): exactly ONE big fluffy cloud-like tail made of pale blue-white fur with two or three swirl curls, growing from the lower back behind the shell. It is about as tall as the shell. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all fully inside their quadrant with white margin):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the snout pointing at the viewer, the sand-yellow belly plate fully visible, both arms held out slightly to the sides, both legs apart, both ears up; the fluffy tail peeks out behind the body on the RIGHT side.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the snout pointing left, both arms forward to the left, the belly plate on the left edge of the body, the brown shell on the back (right), the fluffy tail behind the shell on the RIGHT.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the brown back shell with its plates and white rim, the back of the head with both ears, the face turned away to the upper right; the fluffy tail comes out below the shell on the LEFT.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the round brown back shell with its plates and white rim fills the middle, the back of the head with both ears, the arms only peeking at the sides; the fluffy tail hangs down from below the shell in the middle-left.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a small upright blue turtle with a sand-yellow belly plate, a brown back shell, two large feathery pale-blue ears, two short blue arms (the near arm stretched forward), two short thick legs, and ONE big fluffy pale blue-white swirled tail behind the shell on the right.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the belly plate visible, the fluffy tail behind the body to the RIGHT. Exactly two ears, one tail, two arms, two legs. Crisp 1-2px near-black outline, limited palette, no anti-aliasing, no text, no ground shadow, the whole creature fully inside the image with white margin.
