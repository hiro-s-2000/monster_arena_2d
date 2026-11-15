# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, angry, monkey-like creature whose head and body are ONE single round fluffy ball of shaggy fur (no neck, no separate torso), with a ragged spiky fur outline all around. Two small pointed triangular ears stick out on top of the ball, one on the left and one on the right. In the middle of the face: two narrow angry eyes with a frowning brow, and a large round pink pig-like snout with two nostrils right under the eyes.
2. Colour palette:
   - Fur ball: pale cream (#F6DEB4) with very light highlights (#FFF6CD) on the top of the head and dusty pinkish-beige (#D5AC9C) shading; ragged fur edges drawn with dark brown strokes (#6A4A29).
   - Arms, legs and tail: thin, dusty pinkish-beige (#D5AC9C) with cream highlights (#F6DEB4).
   - Fists, feet and the tail's curled tip: light brown (#C5734A) with orange-tan highlights (#FFC583 / #DEA473) and dark brown shadow (#734100).
   - Snout: salmon pink (#EE7362) with dark red nostrils and shading (#B44131).
   - Inner ears: dark brown (#734100).
   - Eyes: white with a small black pupil.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Limbs (exact counts): exactly TWO thin arms, each ending in ONE large round brown fist (like a boxing fist), raised in a fighting stance; exactly TWO thin bent legs, each ending in a brown foot with long splayed toes. Never more than two arms, two fists or two legs.
4. Tail (1-Tail Rule): exactly ONE long thin tail that grows from the lower back, sweeps out sideways and curls up into a single hook / spiral at its brown tip. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both angry eyes, the pink snout in the centre, both ears; both arms out to the left and right with the fists raised at head height; both legs apart; the tail behind the body sticking out to the right with its curled tip.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one angry eye, the pink snout sticking out on the left edge of the ball, one ear on top; the near fist raised forward to the left in front of the face, the far fist raised behind it; two legs, one forward, one back; the single tail extends behind to the right and curls up at its tip.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: the shaggy back of the fur ball (dusty beige, no face, no snout, only the back edges of both ears), both arms with fists raised at the sides, both legs; the single tail comes from the lower back and curls out to the left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the shaggy back of the fur ball, both ears from behind, both arms with fists raised at the sides, both legs; the single tail comes from the lower back and curls out to the right.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small angry monkey-like creature whose head and body are one round shaggy cream fur ball, two small pointed ears, narrow angry eyes, a large round pink pig snout, two thin arms ending in big brown fists raised in a fighting stance, two thin bent legs with brown splayed-toe feet, and exactly ONE long thin tail curling up at its brown tip.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes and the pink snout visible in natural 3/4 perspective (not squished); one fist raised up high on the right, the other fist out to the left; the tail behind the body to the right, curling up at its tip. Exactly one tail, two arms, two fists and two legs. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
