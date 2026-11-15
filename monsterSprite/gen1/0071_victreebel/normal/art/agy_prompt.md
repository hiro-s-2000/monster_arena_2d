# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature's mouth faces the viewer and slightly to the left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel sprites: front view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a carnivorous pitcher-plant creature. ONE large upright pear-shaped / bell-shaped pitcher body, narrow at the top and round and heavy at the bottom. The top of the pitcher is ONE very large wide-open round mouth.
2. NO LEGS, NO ARMS (No-Limb Rule): zero legs, zero feet, zero hands, zero eyes. The creature hovers / rests on its round bottom. Do not add any limb.
3. Colour palette:
   - Pitcher body: golden mustard yellow (#DEBD52) with pale cream-yellow highlight (#F6EE94), light yellow (#EED56A) and olive-brown shading (#A47B31, #625229).
   - Spots: about 6 to 8 small rounded green spots (#4AA47B) scattered on the yellow body.
   - Mouth rim: ONE thick round lip ring, orange (#EE8B52 highlight, #C55A20 mid, #831800 dark), around a dark brown-black throat (#625229 / #000000).
   - Teeth: exactly TWO small white pointed teeth (#FFFFFF with grey #BDC5C5) hanging from the lower edge of the lip ring, one left and one right.
4. Leaves (3-Leaf Rule, most important): exactly THREE big leaves, all leaf green (#4AA47B) with pale green highlights (#8BC57B) and dark green veins / shading (#106239):
   - ONE leaf lid on top of the mouth, like a cap tilted over the top of the rim (it does not close the mouth).
   - TWO large leaves on the sides of the body like arms, one on the left side and one on the right side, at the middle of the body, pointing outward and slightly down.
   Never two side leaves on one side, never four leaves.
5. Vine tail (1-Vine Rule): exactly ONE thin brown vine (#A47B31 with #625229 shading) that starts at the stem of the top leaf lid, arcs over and down behind the body, and ends below the body in exactly ONE small yellow pointed leaf-bud tip (#DEBD52 / #F6EE94) hanging near the ground.
6. Outlines: crisp 1-2px black pixel-art outline (#000000), no anti-aliasing halo. Do not add extra parts or change the number of anything.

GRID LAYOUT (4 key poses, large and crisp, same size, bottoms on the same baseline):
- Top-Left: SOUTH (0°, front view) — the mouth faces straight toward the viewer: the round orange lip ring and the dark throat seen from the front at the top, the leaf lid above it, the two white teeth under the rim, one side leaf to the left and one to the right, the yellow spotted belly below, the vine arcing on one side down to the yellow bud tip under the body.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: the mouth opening points to the left (the lip ring seen edge-on as an oval on the left of the top), the leaf lid on top, the body bulging round at the bottom, one side leaf in front pointing left-down and the other leaf partly visible behind on the right, the vine arcing behind on the right down to the bud tip.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the plain yellow spotted back of the pitcher, the back of the orange rim and the leaf lid at the top, mouth turned away to the upper right, both side leaves visible, the vine running down the back to the bud tip.
- Bottom-Right: NORTH (180°, pure back view) — exactly like the back reference: the yellow spotted back, the back of the rim and the leaf lid on top, one side leaf to the left and one to the right, the brown vine curving down over the back in an S shape to the yellow bud tip below the body.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design and colours: ONE upright pear-shaped golden-yellow pitcher body with green spots, ONE big round mouth on top with a thick orange lip ring, dark throat and exactly two small white teeth under the rim, ONE leaf lid on top, exactly TWO big green side leaves (one left, one right), and ONE thin brown vine from the top that curves down to ONE small yellow pointed bud tip below the body. NO legs, NO arms, no eyes.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the mouth opening faces diagonally down-left toward the viewer (the lip ring seen as a slightly tilted oval), the near (left) side leaf in front, the far (right) side leaf partly behind, the vine arcing on the right side down to the bud tip. Crisp 1-2px black outline, limited palette, no anti-aliasing.
