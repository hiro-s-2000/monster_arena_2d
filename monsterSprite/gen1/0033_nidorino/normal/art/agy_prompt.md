# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, the creature faces right and away). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (small pixel sprites: front view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a stocky, aggressive four-legged poison rodent, low to the ground, head held low and forward, back arched. It stands on FOUR short thick legs (4-Leg Rule: exactly four legs, never more, never fewer), each foot with small white-grey claws (#C5C5C5).
2. Colour palette:
   - Body, head, legs and ears: violet (#945AB4) with light lilac highlights (#CD83DE, brightest #E6ACEE) and deep purple shadow (#5A3962).
   - Spots: irregular dark purple spots (#5A3962) scattered over the back and flanks.
   - Inner ear: emerald green (#009462) with a lighter mint highlight (#4ACDA4).
   - Eyes: small, narrow and fierce, red-brown pupil with a white glint.
   - Mouth: open, with a red inside (#DE4129, dark #B41800) and one small white lower fang.
   - Outlines: crisp 1px near-black outline (#101010), no anti-aliasing halo.
3. Horn (1-Horn Rule): exactly ONE long sharp pointed horn on the forehead, pointing forward and slightly up, same violet colour. Never two horns.
4. Ears (2-Ear Rule): exactly TWO very large, tall, pointed ears with jagged spiky outer edges, standing up from the top of the head; the inner ear face is green. Never more than two ears.
5. Back spines: a row of 5 to 6 sharp violet spikes along the spine from the neck to the rump, pointing up and back.
6. Tail: exactly ONE short, thick, pointed tail at the rump.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the head and the single horn point at the viewer, both eyes visible, both big ears up on the left and right of the head with green insides visible, both front legs in front, the back spines rising behind the head.
- Top-Right: EAST (90°, right side profile) — facing right in a sharp side profile: horn and snout pointing right, one eye, ears up, the row of back spines along the top, all four legs on the ground (two near legs in front of two far legs), short tail on the left.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the spotted back, the row of back spines and the rump with the short tail, the head turned away to the upper right, the backs of the two ears visible.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: rump and short tail toward the viewer, the back spines in a line up the middle, both ears visible from behind, two hind legs on the ground.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a stocky violet four-legged poison rodent (#945AB4, highlights #CD83DE, shadows #5A3962) with dark purple spots, ONE long forehead horn, TWO very large spiky-edged ears with green insides (#009462 / #4ACDA4), a row of sharp back spines, ONE short pointed tail, an open red mouth with one small fang, and four short legs with white-grey claws.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the head and horn point diagonally down-left, both eyes visible in natural 3/4 perspective (not squished), the body extends back to the upper right. Exactly four legs, exactly one horn, exactly two ears. Crisp 1px near-black outline (#101010), limited palette, no anti-aliasing.
