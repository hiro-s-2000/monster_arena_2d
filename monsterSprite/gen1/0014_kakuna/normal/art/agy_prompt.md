# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front view, the head turned slightly to the right) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel sprites: front view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a single hard insect cocoon standing upright like an upside-down teardrop / pointed spindle. On top a smooth ROUND DOMED HEAD (about one third of the total height, like a helmet). Under the head a collar of 2 to 3 wrapped, overlapping horizontal shell bands, whose ends stick out as ONE small blunt pointed nub on the left side and ONE on the right side (exactly two nubs). Below that, the lower shell tapers down to ONE sharp point at the very bottom; the creature stands balanced on that point.
2. NO LIMBS (No-Limb Rule, most important): zero legs, zero arms, zero hands, zero feet, zero wings, zero antennae, zero horns, zero tail, no mouth. Do not add any limb or appendage. The two collar nubs are short shell edges, not arms.
3. Colour palette (golden yellow, no other hues):
   - Shell and head: golden yellow (#F6CD31) with mid gold (#CD9C10) and shadow ochre (#9C7B00); a pale cream highlight (#FFF6CD) on the upper left of the dome.
   - Plate seams: thin dark brown lines (#4A3900 / #7B5A00) dividing the lower shell into several angular wrapped plates, and the collar into horizontal bands.
   - Outlines: crisp 1-2px near-black (#101010) pixel-art outline, no anti-aliasing halo.
4. Eyes (2-Eye Rule): exactly two small round dark grey eyes (#393939 with a #6A6A6A rim and one white #FFFFFF highlight dot) on the lower part of the domed head, just above the collar. In a side view only ONE eye is visible. In back views no eye is visible (at most the edge of one eye peeking at the side).
5. Do not add extra parts, spikes, leaves, patterns or colours. Do not change the number of anything.

GRID LAYOUT (4 key poses, large and crisp, same size, bottom points on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: round dome on top with both eyes, the collar bands with one nub to the left and one to the right, the shell tapering to the single bottom point in the middle.
- Top-Right: EAST (90°, right side profile) — facing right: the dome with ONE eye on the right side, the collar bands with one nub pointing right, the lower shell tapering to the point; the back of the shell on the left is plain plated gold.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back of the dome (no face), the collar bands wrapping around, one nub at the right, the tapering plated back of the shell; at most the edge of one eye at the upper right.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: back of the dome, the horizontal collar bands, one nub on each side, the lower shell tapering to the bottom point. No eyes.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design and colours: a single rigid golden-yellow cocoon (#F6CD31 with #CD9C10 / #9C7B00 shading and a #FFF6CD highlight), shaped like an upside-down teardrop: a round domed head on top with two small round dark grey eyes (white highlight dot) just above a collar of 2 to 3 wrapped horizontal shell bands with exactly one short blunt nub on each side, then the lower plated shell (thin dark brown seams) tapering to ONE sharp point at the bottom on which it stands. NO legs, NO arms, NO wings, NO antennae, NO tail, no mouth.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the dome and both eyes look diagonally down-left (the left eye near the left edge of the dome, the right eye closer to the centre, both clearly visible, not a side profile), the left collar nub points to the lower-left. Same proportions as the front reference. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
