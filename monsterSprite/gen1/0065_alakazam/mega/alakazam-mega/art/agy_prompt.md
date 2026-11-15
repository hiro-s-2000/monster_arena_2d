# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `ref_back.png` (back view pixel sprite, seen from behind). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

IMPORTANT: you MUST actually load and look at `ref_front.png` and `ref_back.png` and pass them as reference images to the image generator. The creature is an old wise humanoid FLOATING in the air while SITTING CROSS-LEGGED (meditation pose), with a huge flowing white beard and five silver spoons hovering in a fan above its head. It does not stand and has no tail.

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a slim humanoid psychic sage that hovers in mid-air, sitting cross-legged in a meditation pose (no feet touching the ground, no standing legs). The torso is covered by brown plated armour with two big rounded shoulder pads and two wide bell-shaped brown sleeves.
2. Colour palette:
   - Head, hands and legs: pale gold yellow (#F7E273) with ochre shading (#D1A445) and dark ochre (#A87B37), a light cream highlight (#FFF6AC).
   - Armour, shoulder pads, sleeves and thighs: reddish brown (#884E4D) with light brown (#AA7166) and dark brown (#5B3737), plus a little warm brown (#865531).
   - Beard: white (#FBFBFB) with light grey (#DEDEDE) and grey (#B4B4B4) strands.
   - Spoons: silver grey (#B4B4B4) with dark steel grey (#868B91).
   - Gem: one small red gem (#EF383D) in the middle of the forehead.
   - Outlines: crisp 1px near-black outline (#101010), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one large spiky star-shaped yellow head with five short points like a crown (three on top, one on each side), a long pointed muzzle, two narrow slanted eyes and exactly ONE small red gem on the forehead.
4. Beard (1-Beard Rule, most important): exactly one huge white beard growing from under the muzzle, falling over the chest down to the lap and spreading out to the left and the right in two long wing-like white streams beside the knees. It is part of the creature, not a cloth.
5. Arms (2-Hand Rule): exactly two arms in wide brown sleeves, both hands raised up beside the head at shoulder height, palms open, each hand with exactly three long thin yellow fingers pointing up. Never more than two hands.
6. Legs: exactly two yellow legs folded in a cross-legged lotus pose under the body, brown thighs, the yellow shins crossed in front. No other legs.
7. Spoons (5-Spoon Rule): exactly FIVE silver spoons float in a half-circle fan above and around the head, not touching it and not held: one straight up above the head, two up-left, two up-right, each spoon's round bowl pointing outward away from the head. Never four, never six.
8. Back: the back is covered by the brown armour plates; a small gold Y-shaped ornament sits low on the back above the crossed legs.

GRID LAYOUT (4 key poses, large and crisp, same size, all floating at the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the star head with the red gem, the white beard falling down the middle and spreading left and right, both hands raised beside the head, the crossed legs below, the five spoons fanned above the head.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: the long muzzle pointing left, one eye, the beard flowing down and back from the muzzle, the near hand raised in front of the chest, the far hand raised behind, the crossed legs below, the five spoons fanned above the head.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back of the star head and the brown armoured back with its shoulder pads, the face turned away to the upper right; the beard's white streams just visible at the sides, the hands raised, the five spoons above.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the back of the star head, the brown armour plates, both sleeves and raised hands, the gold ornament low on the back, the crossed yellow legs under it, the five spoons fanned above the head.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a hovering humanoid sage sitting cross-legged, a large spiky star-shaped yellow head with ONE small red gem on the forehead, a long muzzle, a huge white beard spreading left and right like two wings, brown plated armour with big shoulder pads and wide brown sleeves, exactly TWO raised yellow hands with three thin fingers each, two crossed yellow legs, and exactly FIVE silver spoons floating in a fan above the head.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the beard spreading to both sides, both hands raised. Exactly five spoons, exactly two hands. Crisp 1px dark outline, limited palette, no anti-aliasing.
