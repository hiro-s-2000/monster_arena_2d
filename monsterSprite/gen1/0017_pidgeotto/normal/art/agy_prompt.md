# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view: the bird faces lower-left, hovering with both wings raised high, talons hanging below) and `ref_back.png` (pixel-art back view: seen from behind, both wings spread up and out, the fan tail at the bottom). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a medium-sized bird of prey (a hawk-like bird) HOVERING in the air, flapping, with a compact round body, a short neck and a small round head. It does not stand on the ground: its two legs hang below the body with the talons spread, ready to grab.
2. Colour palette:
   - Back, top of the head, upper side of the wings: warm brown (#835231) with light brown highlights (#AC7B5A) and dark brown shadow (#412920).
   - Chest, belly, face and the long flight feathers of the wings: pale cream yellow (#FFE68B) with golden ochre shading (#E6BD5A).
   - Head crest: a tuft of red feathers on the forehead swept back over the head (#C53129, #EE6241, dark red #7B3120).
   - Tail: a fan of long tail feathers in alternating red (#C53129, #EE6241) and cream yellow (#FFE68B) stripes.
   - Beak and legs / feet: pink (#DE73CD) with light pink (#F69CEE). Talon tips: white-grey (#D5D5DE, #7B7B83).
   - Eyes: black eye with a dark stripe behind it, tiny white highlight.
   - Outlines: crisp 1px near-black outline (#101010), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with one short hooked pink beak and ONE red crest tuft. No horns, no ears.
4. Wings (2-Wing Rule): exactly TWO large feathered bird wings growing from the shoulders, raised high and spread wide, brown on the upper part with long separated cream-yellow primary feathers at the tips. Never one wing, never three or four. No arms, no hands.
5. Legs (2-Leg Rule): exactly two thin pink legs, each foot with exactly three forward talons and one back talon, hanging below the belly. No extra limbs.
6. Tail (1-Tail Rule): exactly ONE fan of tail feathers at the bottom of the body, red and cream striped.

GRID LAYOUT (4 key poses, large and crisp, same size, all four birds hovering at the same height, lowest point on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the beak pointing at the viewer, the cream chest in the centre, both wings raised up and out to the left and to the right, both legs hanging below with talons spread, the red-cream tail fan below / behind the legs.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the beak pointing left, the near wing raised up and back over the body (the far wing just visible behind it), legs hanging below and slightly forward to the left, the tail fan pointing right / down-right behind the body.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the brown back, the back of the head with the red crest, the face turned away to the upper right; both wings raised and spread up-left and up-right (brown upper sides), the tail fan at the bottom toward the viewer.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: brown back, back of the head with the red crest, both wings spread up and out to the left and to the right, the red-cream tail fan hanging down the middle.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a hovering brown-and-cream hawk-like bird with a red crest tuft, a short pink hooked beak, TWO large wings raised high (brown upper part, long cream-yellow flight feathers), TWO pink legs with spread talons hanging below, and ONE red-and-cream striped tail fan.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream chest visible. Exactly two wings, exactly two legs and exactly one tail fan, all clearly visible. Crisp 1px near-black outline, limited palette, no anti-aliasing.
