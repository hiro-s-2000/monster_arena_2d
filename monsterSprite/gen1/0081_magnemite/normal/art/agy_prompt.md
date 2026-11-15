# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces to the lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel sprites: front view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a single small floating metal ball (a perfect sphere), pale steel-green. It floats in the air: it has NO legs, NO arms, NO feet, NO wings, NO tail, NO mouth.
2. Colour palette:
   - Sphere: pale steel green (#8BB4A4) with a light mint highlight (#B4D5BD, #D5F6DE) on the upper left and darker green shading (#5A8362) on the lower edge.
   - Magnets and screw sockets: very dark warm grey (#524A4A, #393131).
   - Magnet tips: one bright red tip (#EE1800) and one bright blue tip (#2983AC) on each magnet.
   - Screw heads: light grey / silver (#D5D5D5, #B4B4B4, #8B8B8B).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Eye (1-Eye Rule): exactly ONE large round eye in the middle of the front of the sphere: a big white disc (#FFFFFF) with a thin grey ring and a single small black dot pupil. Never two eyes. In back views no eye is visible.
4. Magnets (2-Magnet Rule, most important): exactly TWO dark grey U-shaped horseshoe magnets, one attached to the left side and one to the right side of the sphere, each joined to the sphere by a short dark stub. Each magnet's open end points outward and away from the ball, and each of its two prongs ends in a coloured tip: one red, one blue. Never one magnet, never three or more.
5. Screws (3-Screw Rule): exactly THREE screws. One large flat silver screw head sticks straight up out of the top of the sphere like an antenna (a short dark shaft with a wide flat silver cap). Two small round silver cross-head screws (with a dark "+" slot) sit on the lower front of the sphere, one lower-left and one lower-right. In back views the two small screws are hidden or only peek at the edge.
6. Do not add extra parts, wires, sparks, limbs or patterns. Do not change the number of anything.

GRID LAYOUT (4 key poses, large and crisp, same size, all floating at the same height with the ball centred in each quadrant):
- Top-Left: SOUTH (0°, front view) — the ball faces straight toward the viewer, symmetric: the one big eye in the centre, the two small cross screws below it to the left and right, the top screw straight up, one horseshoe magnet sticking out to the left and one to the right, both opening outward.
- Top-Right: WEST (270°, left side profile) — the ball faces left: the single eye seen from the side as a narrow white oval on the left edge of the ball, the top screw straight up, one magnet in front of the ball (pointing toward the viewer and down-left) and the other magnet behind the ball (partly hidden, poking out on the right), one small screw on the lower left.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the plain steel-green back of the ball, no eye visible, the top screw straight up, one magnet to the left and one to the right.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: the plain back of the ball, no eye, the top screw straight up, one magnet to the left and one to the right, both opening outward.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a single floating pale steel-green metal ball (#8BB4A4 with mint highlights) with exactly ONE big round white eye with a small black dot pupil, exactly THREE silver screws (one big flat-headed screw sticking up from the top, two small cross-head screws on the lower front), and exactly TWO dark grey horseshoe magnets on the left and right sides, each with one red tip and one blue tip. NO legs, NO arms, NO mouth.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the eye sits left of centre on the ball looking diagonally down-left, one magnet sticks out to the lower-left in front and the other to the upper-right behind, exactly as in the front reference. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
