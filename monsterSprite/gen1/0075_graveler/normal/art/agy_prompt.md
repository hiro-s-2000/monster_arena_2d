# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel-art front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: one big round lumpy boulder (a rock ball) that is head and body in one. No neck, no separate head, no tail. It stands on two short stubby rock feet and has two short thick rock arms with clenched fists, raised up and outward at the sides like a strongman.
2. Colour palette (stone greys with an olive tint only):
   - Rock body: light khaki grey (#B4B49C) with highlight (#D5D5AC), mid warm grey (#948B7B), olive grey (#6A6A52) and dark brown-grey shadow cracks (#4A4131).
   - Eyes: two narrow angry slanted white eyes (#FFFFFF, #DED5DE) with small dark pupils.
   - Mouth: one wide grin full of white square teeth (#FFFFFF) just below the eyes, dark gap (#4A4131).
   - Outlines: crisp 1px black pixel-art outline (#000000), no anti-aliasing halo.
   - No other colours: no red, no blue, no green, no brown-orange.
3. Arms (2-Arm Rule): exactly TWO short thick rock arms, one on the left side and one on the right side of the boulder, each ending in ONE blocky clenched fist held at shoulder height. Never three, never four arms.
4. Feet (2-Foot Rule): exactly TWO short flat oval rock feet under the boulder, no visible legs.
5. Face and surface: the face is on the upper front of the boulder; dark cracks radiate around it like rock ridges. The back is a plain lumpy rock surface with a few rounded bumps and no face.
6. Do not add any extra parts: no tail, no horns, no spikes, no extra arms, no fingers spread open.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both angry eyes and the toothy grin in the middle of the boulder, one fist raised on the left and one on the right, both feet below.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye and half of the grin at the left edge of the boulder, the near fist in front of the body punching forward to the left, the far fist peeking out behind the top right, one foot in front and one behind.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the plain lumpy rock back, the face turned away to the upper right (no eyes visible), both fists raised at the sides, both feet below.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the plain lumpy rock back with rounded bumps, one fist raised on each side, both feet below, no face visible.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: one big round lumpy khaki-grey rock boulder with two narrow angry white eyes and a wide grin of white square teeth, dark cracks radiating around the face, exactly TWO short thick rock arms with clenched fists raised at the sides and exactly TWO short flat rock feet below. No tail, no extra arms.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the near (right) fist fully visible and the far (left) fist slightly in front at the left. Exactly two arms, two feet. Crisp 1px black outline, limited stone-grey palette, no anti-aliasing.
