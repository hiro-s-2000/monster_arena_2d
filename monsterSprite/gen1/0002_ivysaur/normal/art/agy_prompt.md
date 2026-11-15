# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (3/4 back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Create the folder `out/` if it does not exist. Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: a 3/4 front view and a 3/4 back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, stocky four-legged amphibian-like creature that stands low on all four short, thick legs (quadruped, never standing upright). Wide round head with a short blunt snout, the body is about as long as the head is wide.
2. Colour palette:
   - Skin (head, body and legs): teal blue-green (#5ACDBD) with light highlights (#83E6D5) and darker teal shading (#209483). A few darker teal-blue patches / spots (#10414A) on the forehead, back and legs.
   - Eyes: red irises (#D5415A) with a white highlight, dark outline; mouth slightly open showing two tiny white fangs.
   - Ears (2-Ear Rule): exactly two small pointed triangular ears on top of the head, teal with dark inner.
   - Feet (4-Leg Rule): exactly four legs, each foot ends in three small white claws.
   - Back plant: on the middle of the back sits ONE large closed flower bud, pink-red (#FF7B7B, light #FFACA4, shadow #D5415A, dark brown-red outline #7B3129), shaped like a tall rounded teardrop with a few pointed petal tips at the top.
   - Leaves (4-Leaf Rule): exactly FOUR broad flat leaves grow from the base of the bud and spread outward horizontally like a collar: one to the front-left, one to the front-right, one to the back-left, one to the back-right. Leaves are light green (#7BD573) with mid green (#62AC5A) and dark green (#317B52, #104A39) veins / shading, with slightly jagged edges.
   - Outlines: crisp 1-2px very dark outline (#101010), no anti-aliasing halo.
3. No tail, no arms, no extra leaves, no vines, no flowers other than the single bud. Never add or remove legs, ears or leaves.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all standing on four legs):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes and both ears visible, the two front legs in front, the bud on top of the back behind the head, two leaves spreading to the left and two to the right.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, snout pointing to the left, the near front leg and the near hind leg in front, the two far legs slightly visible behind them, the bud on the middle of the back, leaves spreading forward and backward from the bud base.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the back with the bud and the leaves, the back of the head turned away to the upper right, the hind legs nearest to the viewer.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the round rump and the two hind legs, the bud in the centre of the back, two leaves spreading left and two right, the back of the head and both ear tips peeking above.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a stocky four-legged teal blue-green creature with darker teal spots, red eyes, two small pointed ears, four short legs with three white claws each, ONE large closed pink-red flower bud on its back and exactly FOUR broad light-green leaves spreading horizontally around the base of the bud.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), all four legs on the ground. Exactly one bud and four leaves. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
