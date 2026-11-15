# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation — use ONLY these two files, never any other image: `/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_revenge_claude/pokeSprite/gen1/0006_charizard/mega/charizard-mega-y/art/ref_front.png` (3/4 front view pixel sprite, the creature faces lower-left) and `/Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_revenge_claude/pokeSprite/gen1/0006_charizard/mega/charizard-mega-y/art/ref_back.png` (back view pixel sprite, seen from behind and to the right). Save the results exactly as (inside /Users/honmahiroshi/Desktop/ai_workspace/pokemon_arena_revenge_claude/pokeSprite/gen1/0006_charizard/mega/charizard-mega-y/art):

The creature is a winged orange FIRE DRAGON (not a turtle, not a reptile without wings). Its two wings are solid bat wings with teal membranes, NOT made of fire; the only fire is the single flame on the tail tip.


1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large bipedal fire dragon standing upright on two thick legs, slim long neck, a broad round chest and belly, and a long tail. Arms held forward at chest height with clawed hands.
2. Colour palette:
   - Body, head, neck, arms, legs, tail and wing frames: bright orange (#EE8329) with red-orange shading (#CD5241) and dark brown-red (#833118).
   - Belly, throat, lower jaw and the underside of the tail: pale cream yellow (#EEDE7B) with tan shading (#EEB45A).
   - Wing membranes: deep teal blue (#207394) with dark teal (#084152).
   - Tail flame: red-orange fire (#E64110) with orange (#F6A410) and a bright yellow core (#FFD510).
   - Eyes: blue (#207394). Claws: white (#F8F8F8) with light grey (#CDCDCD).
   - Outlines: crisp 1px near-black outline (#101010), no anti-aliasing halo.
3. Head (1-Head Rule): exactly one head with a long snout and exactly THREE horns: one very long pointed horn swept straight back from the top of the head, and two short pointed horns beside it. No mouth flames.
4. Wings (2-Wing Rule): exactly TWO very large bat-like wings growing from the upper back / shoulders, each with an orange frame (one arm bone with 3 finger spines) and teal membranes, the outer edges jagged and pointed. Wings spread wide and up, each wing about as tall as the body. Never one wing, never three or four.
5. Arms (2-Arm Rule): exactly two arms, each hand with three small white claws. Each forearm carries one small pointed orange wing-fin with a teal inner side. Exactly two legs, each foot with three white claws, both feet planted on the ground.
6. Tail (1-Tail Rule): exactly ONE long tail curving out behind the body, its underside cream, ending in a large orange-red flame with a yellow core at the tip.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both blue eyes, the snout pointing at the viewer, the cream belly in the centre, both arms forward, both wings spread wide to the left and to the right behind the body, the tail curling out to one side behind the legs with its flame.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, snout pointing left, the long horn sweeping back to the right, arms forward to the left, wings raised behind the back, the tail extending to the right with the flame on its tip.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the orange back, the back of the head and the horns, the face turned away to the upper right; both wings spread up-left and up-right from the shoulders, the tail coming toward the viewer at the bottom left with its flame.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: orange back, back of the head with the long horn, both wings spread wide to the left and right, the tail hanging down the middle and curving to one side with its flame.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a bright orange bipedal dragon with a pale cream belly, blue eyes, one very long backswept horn plus two short horns, two clawed arms forward each with a small wing-fin on the forearm, two clawed legs planted, TWO very large wings with teal membranes and jagged edges spread behind it, and ONE long tail ending in an orange-red flame with a yellow core.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream belly visible. Exactly two wings and exactly one tail, all clearly visible. Crisp 1px dark outline, limited palette, no anti-aliasing.
