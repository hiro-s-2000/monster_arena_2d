# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, the creature faces right, seen from behind). They are small pixel-art sprites enlarged 8x. Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below (ALREADY DONE in an earlier run: if `out/sw_single.png` exists, do not regenerate or touch it)

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

IMPORTANT: use ONLY the two reference images in THIS folder (`ref_front.png`, `ref_back.png`) and `out/sw_single.png` as references. The creature is ORANGE with a salmon-red nose and grey-white stingers. It is NOT green, it has NO red antenna/horn, NO yellow belly, NO bulb tail. Match the colours of `out/sw_single.png` exactly.

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (front view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, plump, legless larva (a caterpillar-like grub). One large round head, then a soft body made of exactly EIGHT round, bead-like segments in a row that gets thinner toward the tail. The front part of the body rears up so the head is held high; the rear segments lie on the ground and curl to one side, ending in the tail tip.
2. Colour palette:
   - Head and body segments: warm orange (#CD7310) with light amber highlights (#EEAC41, #FFE67B on the top of the head) and brown shading and segment lines (#945A08, #6A3908).
   - Nose: one big round bulbous nose at the front of the head, salmon red (#DE5A52) with a pale pink highlight (#EE9C8B, #FFDECD) and dark red shade (#9C2920).
   - Eyes: two small black dot eyes (#181818) with no white.
   - Feet: tiny round salmon-red nubs (#DE5A52 / #9C2920) on the underside of each segment, like little suction pads; NOT legs.
   - Stingers: light grey-white (#FFFFFF, #C5C5C5) with grey outline (#8B8B94, #5A5A62).
   - Outlines: crisp 1-2px dark pixel-art outline (#181818 and dark brown), no anti-aliasing halo.
3. Stingers (2-Stinger Rule, most important): exactly TWO short pointed cone-shaped stingers. ONE stinger on the top of the head (pointing straight up), ONE stinger on the very tip of the tail (pointing up). Never more, never fewer. No horns, no antennae, no ears.
4. No arms, no legs, no wings, no hair. Only head, segmented body, tiny foot nubs, nose, two stingers.

GRID LAYOUT (4 key poses, large and crisp, same size, bottom of the body on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the round head with both eyes and the big red nose in the centre, the head stinger on top; the reared-up front segments under the head; the rear segments curl around behind to one side so the tail-tip stinger peeks out beside the body.
- Top-Right: EAST (90°, right side profile) — facing right in a sharp side profile: head on the right with one eye and the nose pointing right, the front segments rearing up, the rear segments lying flat behind to the left, tail stinger at the far left.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: we see the back of the head and the back of the segments, the face turned away to the upper right (only the edge of the nose visible); the rear segments curl toward the lower left with the tail stinger.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind: the back of the round head with the head stinger on top, the segments below it, the tail curling to one side with the tail stinger. No face visible.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a plump orange legless larva with a large round head, a big round salmon-red nose, two small black dot eyes, eight bead-like body segments with tiny red foot nubs underneath, exactly ONE grey-white cone stinger on top of the head and exactly ONE on the tail tip.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the nose pointing down-left; the front segments rear up under the head, the rear segments curl behind to the right ending in the tail stinger. Exactly two stingers. No legs, no arms. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
