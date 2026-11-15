# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view: the bird faces lower-left, hovering with both wings raised high and spread, talons hanging below) and `ref_back.png` (pixel-art back view: seen from behind, the bird turned slightly right, both wings raised, the fan tail at the bottom). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large, majestic bird of prey HOVERING in the air, with a big rounded chest, a short neck and a small head. It does not stand on the ground: its two legs hang below the body with the talons spread.
2. Colour palette:
   - Back, head and wings: warm tan / light orange-brown (#EFC074) with ochre shading (#D89049), rust brown (#AC5F2C) and dark brown feather lines (#7B4335).
   - Chest and belly: pale cream yellow (#FFF29E) with lemon yellow (#FAEF5D).
   - Wing tips: the long primary flight feathers at the end of each wing have bright blue tips (#0094CA, dark navy #184A7B in shadow).
   - Crest and streamers: vivid red (#FF4C5B) with crimson (#E7003A) and dark red (#98292E).
   - Head top: a spiky flame-like yellow crest tuft (#FAEF5D, #FFF29E) on top of the head.
   - Beak and feet: pale pink (#D1867C, #F9B09D) with grey talon tips (#C5B4C5).
   - Eyes: a sharp black eye inside a dark brown mask marking (#7B4335).
   - Outlines: crisp 1px near-black outline (#101010), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with one short hooked pink beak, a spiky yellow crest tuft on top, and long thin RED ribbon-like crest plumes streaming back from the head and curving over the body. No horns, no ears.
4. Wings (2-Wing Rule): exactly TWO very large feathered bird wings growing from the shoulders, raised high and spread wide, tan with dark brown feather lines, long separated primary feathers with blue tips. Never one wing, never three or four. No arms, no hands.
5. Legs (2-Leg Rule): exactly two short pink legs, each foot with spread talons, hanging below the belly. No extra limbs.
6. Tail (1-Tail Rule): exactly ONE fan of tail feathers at the bottom of the body, red with blue tips, plus long thin red ribbon streamers trailing from the lower body. Do not add extra streamers, extra wings or extra heads.

GRID LAYOUT (4 key poses, large and crisp, same size, all four birds hovering at the same height, lowest point on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the beak pointing at the viewer, the cream chest in the centre, both wings raised up and out to the left and to the right, both legs hanging below with talons spread, the red-blue tail fan below / behind the legs, the red crest plumes rising behind the head.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the beak pointing left, the near wing raised up and back over the body (the far wing just visible behind it), legs hanging below and slightly forward to the left, the tail fan pointing right / down-right behind the body, the red crest plumes streaming back to the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the tan back, the back of the head with the yellow crest and red plumes, the face turned away to the upper right; both wings raised and spread up-left and up-right, the tail fan at the bottom toward the viewer.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: tan back, back of the head with the yellow crest and red plumes, both wings spread up and out to the left and to the right, the red-blue tail fan hanging down the middle.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a large hovering tan bird of prey with a pale cream-yellow chest, a spiky yellow crest tuft and long thin red ribbon crest plumes, a short pink hooked beak, TWO very large wings raised high (tan with blue-tipped primary feathers), TWO pink legs with spread talons hanging below, and ONE red tail fan with blue tips.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream chest visible. Exactly two wings, exactly two legs and exactly one tail fan, all clearly visible. Crisp 1px near-black outline, limited palette, no anti-aliasing.
