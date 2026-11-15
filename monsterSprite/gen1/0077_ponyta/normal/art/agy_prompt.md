# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (3/4 back view, the creature faces right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel sprites: 3/4 front view and 3/4 back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a slender young pony (foal) standing on four long thin legs, small horse head with a short muzzle, long upright neck, compact barrel body. The body is clearly longer than it is tall in side view.
2. Colour palette:
   - Body, head, neck and legs: pale cream (#FFEECD) with sand beige shading (#E6CD8B) and tan shadow (#C59C6A).
   - Hooves: dark navy blue (#414A83) with a slate highlight (#737BA4).
   - Ears and muzzle details: brown (#8B5231) and dark maroon (#733131).
   - Eyes: dark maroon with a small red glint.
   - Flames: bright orange-red (#E65A29), orange (#FF9C18), yellow (#FFDE52) and scarlet tips (#EE3131).
   - Outlines: crisp 1px black pixel-art outline (#000000), no anti-aliasing halo.
3. Legs (4-Leg Rule): exactly FOUR thin legs, each ending in one small dark navy hoof. Each leg has a small orange flame tuft at the top of the leg (shoulder / hip). Never three or five legs.
4. Ears (2-Ear Rule): exactly two small pointed ears with brown insides.
5. Flames (2-Flame Rule, most important): exactly ONE flaming mane running from the top of the head (a fiery forelock) down the back of the neck, and exactly ONE large flaming tail at the rump that streams backward and upward. The flames are made of jagged pixel tongues, yellow at the core, orange in the middle, scarlet at the tips. No other flames on the body except the four small leg tufts.

GRID LAYOUT (4 key poses, large and crisp, same size, hooves on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: face, both eyes, both ears, the chest and both front legs in front; the flaming mane on top of the head; the hind legs visible behind the front legs; the tail flames peeking up behind the body.
- Top-Right: EAST (90°, right side profile) — facing right in a sharp side profile: one eye, muzzle to the right, the mane flowing back along the neck, the tail flames streaming to the left, all four legs visible (near legs and far legs slightly offset).
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the rump and tail flames toward the viewer, the head turned away to the upper right, the mane flowing over the neck, four legs visible.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the rump with the big tail flame rising in the centre, the back of the head with the mane and both ears above it, the two hind legs in front and the two front legs partly visible between them.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a slender pale cream pony foal with four thin legs and dark navy hooves, small flame tufts at the top of each leg, two small pointed ears, ONE fiery mane (yellow, orange, scarlet) on the head and neck, and ONE big fiery tail streaming back.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the head looks diagonally down-left with the face in natural 3/4 perspective (not squished), the body and tail extend back to the upper right, exactly four legs visible. Crisp 1px black outline, limited palette, no anti-aliasing.
