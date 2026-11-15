# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view: the bird faces left / lower-left, flying with both wings raised high, a very long thin beak pointing left, talons hanging below) and `ref_back.png` (pixel-art back view: seen from behind, both wings spread up and out, the cream neck ruff, the tail feathers at the lower left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large, lean bird of prey FLYING / HOVERING in the air with a long curved neck, a narrow head and a VERY LONG, straight, thin pointed beak (as long as the head and neck together). It does not stand on the ground: its two legs hang below the body with the talons spread.
2. Colour palette:
   - Body, neck, back and upper side of the wings: warm brown (#8B5231) with dark brown shadow (#623920).
   - Wing coverts and flight feathers: ochre (#B47320) and light tan (#D5A45A), the long primary feathers tan with ochre stripes.
   - Neck ruff (a collar of shaggy feathers around the base of the neck): pale cream (#EED5B4, #E6D5B4) with olive-grey shading (#625A39, #ACA46A).
   - Head crest: a spiky tuft of red feathers on top of the head pointing up and back (#E64141, #B42020, dark red #7B2020).
   - Beak and legs / feet: salmon pink (#FF836A) with dark red shading (#7B2020).
   - Eyes: small, with a white eye patch (#FFFFFF) and a black pupil.
   - Outlines: crisp 1px black outline (#000000), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with ONE very long straight pink beak and ONE red spiky crest. No horns, no ears, no second beak.
4. Wings (2-Wing Rule): exactly TWO large feathered bird wings growing from the shoulders, raised high and spread wide, with long separated feathers at the tips. Never one wing, never three or four. No arms, no hands.
5. Legs (2-Leg Rule): exactly two short pink legs, each foot with three talons, hanging below the belly. No extra limbs.
6. Tail (1-Tail Rule): exactly ONE short fan of brown tail feathers with tan tips behind / below the body.

GRID LAYOUT (4 key poses, large and crisp, same size, all four birds hovering at the same height, lowest point on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the long beak pointing down toward the viewer (foreshortened, slightly downward), the cream neck ruff around the neck, the brown chest in the centre, both wings raised up and out to the left and to the right, both legs hanging below with talons spread, the tail fan below / behind the legs.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the long beak pointing straight left, the red crest up and back, the near wing raised up and back over the body (the far wing just visible behind it), legs hanging below, the tail fan pointing right / down-right behind the body.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the brown back, the cream neck ruff, the back of the head with the red crest, the face and beak turned away to the upper right; both wings raised and spread up-left and up-right (brown upper sides), the tail fan at the bottom toward the viewer.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: brown back, the cream neck ruff, the back of the head with the red crest, both wings spread up and out to the left and to the right, the tail fan hanging down the middle.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a lean brown flying bird of prey with a long curved neck, a cream shaggy neck ruff, a red spiky crest, ONE very long straight salmon-pink beak, TWO large wings raised high (brown with ochre and tan feathers), TWO pink legs with spread talons hanging below, and ONE short brown tail fan.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the head and the long beak point diagonally down-left, both eyes visible in natural 3/4 perspective (not squished), the brown chest visible. Exactly two wings, exactly two legs, one beak and one tail fan, all clearly visible. Crisp 1px black outline, limited palette, no anti-aliasing.
