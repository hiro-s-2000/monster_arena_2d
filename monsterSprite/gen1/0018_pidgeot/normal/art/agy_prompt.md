# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back 3/4 view pixel sprite, seen from behind, the head turned to the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

IMPORTANT: you MUST actually load and look at `ref_front.png` and `ref_back.png` and pass them as reference images to the image generator. The creature is a large standing BIRD of prey / pigeon-like bird with a tan-and-cream body, brown folded wings, a red tail fan and a very long flowing yellow-and-red head crest. It is NOT a dragon, NOT a mammal, NOT a chick: no arms, no hands, no horns, no fur.

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large proud bird standing upright on two legs, plump rounded chest, wings folded against the sides of the body, tail fan behind and below. Standing on the ground (feet on the baseline).
2. Colour palette:
   - Chest and throat: pale cream yellow (#FFE68B) blending into warm tan (#E6B462).
   - Body, lower body and the long flight feathers of the folded wings: warm tan (#E6B462) with dark brown feather lines (#73524A).
   - Upper wings (shoulders) and back of the head / back: brown (#AC7B5A) and dark brown (#73524A, #5A3929).
   - Head crest: very long plume, cream yellow (#FFE68B) and tan (#E6B462) striped with bright red (#D53118) and orange-red (#EE6241), flowing from the forehead backward and down over the back, about as long as the body, ending in 2-3 pointed tips.
   - Tail: a fan of bright red feathers (#D53118, dark red #8B3920) with tan (#E6B462) feather edges.
   - Beak: short hooked pink-mauve beak (#BD7B7B, #E6A49C).
   - Eyes: a black stripe (#101010) running from the beak back over the eye, with a small white eye highlight (#FFFFFF).
   - Legs and feet: pink-mauve (#BD7B7B, #E6A49C) with small white claws.
   - Outlines: crisp 1px near-black outline (#101010), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head, one beak, two eyes (only one visible in profile), exactly ONE long crest plume (it may split into 2-3 pointed tips at its end). No horns.
4. Wings (2-Wing Rule): exactly TWO wings, both FOLDED against the body (not spread), brown at the shoulder and tan long feathers below. No arms, no hands.
5. Legs (2-Leg Rule, most important): exactly TWO legs, each with ONE foot of three front toes with white claws. Never four legs.
6. Tail (1-Tail Rule): exactly one red tail fan at the rear, low behind the legs.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes with the black eye stripes, the beak in the centre, the cream chest, both folded wings at the left and right sides, both legs and feet on the ground; the long crest flows up and back over the top of the head (visible above the head), the red tail fan peeking out low behind the legs.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, beak pointing left, cream chest to the left, one folded wing on the body side, the long crest streaming back to the right behind the head, the red tail fan to the right at the bottom, both legs below (one slightly behind the other).
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the brown back and both folded wings, the back of the head with the beak just visible pointing to the upper right, the long crest flowing down the back toward the lower left, the red tail fan low at the bottom left, legs below.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the brown back, both folded wings, the back of the head, the long crest flowing straight down the middle of the back, the red tail fan spread low at the bottom centre, both feet just visible.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a large standing bird with a cream-yellow chest, tan body, brown folded wings, a short hooked pink beak, a black eye stripe, ONE very long yellow-and-red striped head crest flowing back over its back, ONE red tail fan, and exactly TWO pink legs with three-toed feet and white claws.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the head and beak look diagonally down-left, the cream chest toward the viewer, the near (right) folded wing on the body side, the crest streaming up and back to the right, the red tail low behind on the right, both feet on the ground. Exactly two legs, exactly two wings, exactly one crest. Crisp 1px dark outline, limited palette, no anti-aliasing.
