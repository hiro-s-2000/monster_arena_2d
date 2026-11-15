# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small young bipedal lizard standing upright on two short stubby legs, with a big round head (about as large as the torso), a short rounded snout, a plump rounded belly, and two short arms held out to the sides at chest height. No wings, no horns, no spikes, no ears, no crest.
2. Colour palette:
   - Head, back, arms, legs and tail: warm orange (#FF9441) with red-orange shading (#DE5239) and dark rust shadows (#8B2900).
   - Belly and the underside of the tail: pale cream yellow (#FFD57B) with tan shading (#E6AC5A). The cream belly runs from the throat down to between the legs.
   - Eyes: large eyes with a blue iris (#31ACEE / #188BB4), a dark navy pupil (#08398B) and one white highlight.
   - Tail flame: a flame burning on the tail tip, bright yellow core (#FFD508) with orange (#F6A400) and red-orange (#E63900) outer tongues.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Limbs (exact counts): exactly TWO arms, each ending in a small hand with three tiny pointed claws; exactly TWO legs with small feet (three tiny white claws on each foot). Never more than two arms or two legs.
4. Tail (1-Tail Rule, most important): exactly ONE thick tail that grows from the lower back / rump, curves out sideways and up, and ends in ONE flame at the tip. The flame is a single teardrop of fire, about as tall as the head is half-tall. Never two tails, never two flames, no flame anywhere else on the body.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the snout, the cream belly visible, both arms held out to the left and right, both feet visible; the tail is behind the body and sticks out to the right side, its flame visible above the right hip.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, snout pointing to the left, the near arm reaching forward to the left, the cream belly on the left edge of the body; the single tail extends behind to the right and curves up with the flame at its tip.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: we see the orange back and the back of the head, the face turned away to the upper right (only the edge of the snout visible), no belly; the single tail comes from the rump and curves out to the lower left with the flame at its tip.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the orange back, back of the head, both arms showing at the sides, both feet; the single tail comes out of the rump and curves to the right with the flame at its tip.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small orange bipedal lizard with a big round head, short snout, large blue eyes, a pale cream belly, two short arms held out to the sides, two stubby legs, and exactly ONE thick tail ending in ONE yellow-orange flame at its tip.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream belly visible, the tail behind the body to the right with the flame rising above the right hip. Exactly one tail and one flame. Exactly two arms and two legs. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
