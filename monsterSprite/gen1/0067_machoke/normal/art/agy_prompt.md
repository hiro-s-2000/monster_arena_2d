# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature faces lower-left, flexing both arms) and `ref_back.png` (pixel-art back view, flexing both arms). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no grid lines, no shadows on the ground.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a very muscular, humanoid bipedal fighter with broad shoulders, a big chest, huge bulging arms and thick legs, standing in a wide boxing guard: knees slightly bent, feet apart, both arms bent with both fists raised in front of the chest at chin height. Small head sitting low between the shoulders, no visible neck, a stern face with fierce narrow red eyes (#C50000) and a short mouth line.
2. Colour palette (use only these families):
   - Skin of the whole body (head, arms, chest, legs): grey-lilac purple (#A48BCD) with light lilac highlight (#BD9CE6, #CDBDE6), mid purple shading (#8B73B4) and deep purple shadow (#624A83).
   - Head crest: a short flat crest of pale golden yellow (#EEE68B, #CDBD62) with olive shading (#837329) on top of the head, made of exactly THREE ridges running from front to back like a short mohawk.
   - Muscle markings: dark crimson-red (#C50000) and plum (#9C397B) short parallel stripes on each upper arm and forearm (three stripes per group).
   - Briefs: dark slate-grey wrestling briefs (#4A4A62) around the hips.
   - Belt: a dark belt around the waist; on the FRONT a large oval golden championship buckle (#EEE68B, #CDBD62, #837329, #523900); on the BACK a row of exactly FIVE small square gold studs.
   - Outlines: crisp 1-2px black outline (#000000), no anti-aliasing halo.
3. Limbs (2-Arm / 2-Leg Rule): exactly TWO arms, each ending in exactly ONE big closed fist; exactly TWO thick legs, each with one broad foot with three blunt toes. No tail, no wings, no extra arms, no weapons.
4. Head (1-Head Rule): exactly ONE small head with ONE golden crest. No horns, no ears sticking out.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all fully inside their quadrant with white margin):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, the golden crest, the broad chest, the golden belt buckle at the centre of the waist, both fists raised in front of the chest in a guard, both legs apart with knees bent.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the face pointing left, the crest running front to back, both fists held forward to the LEFT at chin height (the near fist slightly lower), the chest on the left edge of the body, one leg forward and one leg back.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the broad muscular back, the back of the head with the crest, the face turned away to the upper right, the dark belt with gold studs at the waist, the briefs, both legs apart.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the broad muscular back, the back of the head with the crest, both elbows out to the sides with the fists hidden in front, the dark belt with exactly five gold studs, the briefs, both legs apart.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design and colours: a very muscular grey-lilac purple humanoid fighter with a small head, a short pale-golden three-ridge crest, fierce red eyes, crimson stripe markings on the arms, dark slate-grey briefs and a golden oval championship belt buckle, two huge arms and two thick legs. Pose: a wide boxing guard, knees slightly bent, both fists raised in front of the chest at chin height.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the chest and the belt buckle visible, the fists held forward to the lower-left. Exactly one head, two arms with one fist each, two legs, no tail. Crisp 1-2px black outline, limited palette, no anti-aliasing, no text, no ground shadow, the whole creature fully inside the image with white margin.
