# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front 3/4 view, the creature faces lower-left, tail on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small slim cat standing upright on its two hind legs like a person, with a big round head (wider than the torso), a slim body, a small round belly, two thin arms held slightly forward at chest height with small round paws, and two thin legs ending in large flat paws. No wings, no horns, no mane, no clothes.
2. Colour palette:
   - Head, body, arms and legs: pale cream yellow (#FFE683) with golden shading (#DEBD39), darker gold (#CDA400) and olive-brown shadows (#8B6A00). Small white highlights (#FFFFFF / #FFFFB4).
   - Paws (all four) and the end of the tail: warm brown (#C57B08) with orange (#EE9C31) and dark rust shadows (#944100).
   - Ears: insides pink-orange (#FF7352 / #D55A39), with black outer tips (#101010).
   - Forehead charm: ONE round gold coin (#FFD500 with #CDA400 shading and a white glint) set in the middle of the forehead between the ears.
   - Eyes: two large almond eyes, white with a thin black vertical slit pupil. A small open mouth with a pink tongue (#FF7352) and two tiny fangs.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Counts (exact, never more): exactly TWO pointed ears; exactly ONE coin; exactly SIX whiskers (three thin straight whiskers sticking out to each side of the face); exactly TWO thin hair strands standing straight up on top of the head between the ears; exactly TWO arms and TWO legs.
4. Tail (1-Tail Rule, most important): exactly ONE long thin tail that grows from the lower back / rump, sweeps out sideways and up, and ends in ONE brown curl (a spiral coil, brown #C57B08 / #944100) at the tip. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the coin in the middle of the forehead, both ears, three whiskers on each side, both arms held slightly forward, both feet visible; the single tail is behind the body and curls up beside the right hip.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the muzzle pointing to the left, the coin seen from the side at the front of the forehead, the near arm reaching forward to the left, one leg stepping forward; the single tail extends behind to the right and rises to the brown curl.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: the yellow back and back of the head, the face turned away to the upper right (only the edge of the coin and whiskers visible), no belly; the single tail comes from the rump and curves out to the lower left, ending in the brown curl.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the yellow back, back of the head with both ears, the whiskers sticking out at the sides, both arms showing at the sides, both feet; the single tail comes out of the rump and curves to the right, ending in the brown curl.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a slim pale-yellow cat standing upright on two legs, big round head with two pointed ears (black tips, pink insides), ONE gold coin on the forehead, two thin hair strands on top, three whiskers on each side, a small open mouth with a pink tongue, two thin arms with brown paws, two legs with large brown paws, and exactly ONE long thin tail ending in ONE brown curl.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the coin on the forehead, the tail behind the body to the right with the brown curl rising beside the right hip. Exactly one tail, one coin, two ears, two arms and two legs. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
