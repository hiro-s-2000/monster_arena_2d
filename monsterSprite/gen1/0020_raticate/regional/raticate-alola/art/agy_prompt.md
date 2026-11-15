# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a very fat, round, pear-shaped rat sitting/standing upright on its two hind feet, with a shaggy spiky fur coat, a big head merged into the body (almost no neck), huge puffy cheeks and a short pointed snout. No horns, no wings, no armour, no clothes.
2. Colour palette:
   - Fur (head top, back, sides, arms): very dark charcoal green-grey (#41443C) with mid grey (#575856), dark shadow (#282E25) and grey-brown (#6C6462); the fur edge is jagged and spiky.
   - Cheeks and muzzle: two very large puffy cream-white cheeks (#EDDDCD) with beige shading (#B9A599), covering the lower half of the face like a big moustache.
   - Belly: dull maroon brown (#98615A) with darker maroon shading (#764949, #5A2F2E), a round patch on the lower front of the body.
   - Ears: exactly TWO rounded rat ears, dark grey outside with a dusty maroon inner ear (#764949).
   - Eyes: narrow, half-closed sleepy eyes with a red iris (#D22D31) and a white highlight.
   - Teeth: exactly TWO big white front incisors (#F8F8F8 with light grey #B1B1BC) at the tip of the snout, pointing down.
   - Hands, feet and tail: pale cream-beige skin (#EDDDCD / #B9A599).
   - Outlines: crisp 1-2px near-black outline (#0A0A0A), no anti-aliasing halo.
3. Limbs (exact counts): exactly TWO small arms with small cream hands held in front of the belly, exactly TWO hind feet with long cream toes flat on the ground. Never more than two arms or two feet.
4. Tail (1-Tail Rule, most important): exactly ONE long, thin, hairless cream-beige rat tail (#EDDDCD with #B9A599 shading) that grows from the rump and curves up in a big arc behind the body. Never two tails.
5. Whiskers: a few thin cream whiskers stick out sideways from the cheeks.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both sleepy red eyes, both huge cream cheeks, the two white incisors in the middle, the maroon belly, both small hands in front of the belly, both hind feet; the single tail curves up behind the body on the right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the snout and incisors pointing left, one big cheek, the round fat body, the hands in front on the left, the feet on the ground; the single tail curves up behind the body on the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: the dark spiky furry back, the back of the head, both ears, the face turned away to the upper right (only the edge of a cheek visible), no belly; the single tail comes from the rump and curves up on the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the round dark spiky furry back, the back of the head, both ears, both feet below; the single tail comes from the rump at the centre and curves up to one side.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a very fat dark charcoal-grey shaggy rat sitting upright, with huge puffy cream cheeks, sleepy red eyes, two white incisors, two rounded ears with maroon insides, a maroon belly, two small cream hands, two cream hind feet with long toes, and exactly ONE long hairless cream tail curving up behind it.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), both cheeks and the incisors visible, the maroon belly visible, the tail arcing up behind the body on the right. Exactly one tail, two ears, two arms and two feet. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
