# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (front view, the creature faces the viewer, tail on the left) and `ref_back.png` (back 3/4 view, the creature faces upper-left, tail on the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, chubby, round-bellied bipedal electric mouse standing upright on two short legs, a big round head, small arms raised a little at the sides with open palms. It stands ON its own tail tip like a surfer on a surfboard. No wings, no horns, no armour, no clothes.
2. Colour palette:
   - Head, body, arms and legs: warm orange-brown fur (#E3882D) with mid brown (#B45F25), dark brown shadow (#965821) and deep red-brown (#672816) for the darkest parts.
   - Belly: a large oval cream-white patch (#FFFDFB) with beige shading (#ECD8B7) on the front of the body.
   - Ears: exactly TWO big ears, bright yellow (#FEF443 with #E9BE14 shading) and dark olive outline (#646124); each ear is wide and ends in an inward-curling rounded tip (like a curl), not a black tip.
   - Eyes: round bright blue eyes (#1A73CC with dark blue #174680), small smile.
   - Cheeks: round bright yellow cheek pouches (#FEF443), not red.
   - Hands and feet: white palms and toes (#FFFDFB) with grey-brown lines (#846B5B) and a yellow tint (#FEF443) at the wrists and ankles.
   - Back: two cream-white horizontal stripes (#ECD8B7) across the middle of the brown back.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Limbs (exact counts): exactly TWO short arms with small white hands, exactly TWO short legs with white feet. Never more than two arms or two feet.
4. Tail (1-Tail Rule, most important): exactly ONE long, thin, dark brown whip tail (#993C20 / #672816) that grows from the rump, makes one big loop on the ground, and ends in ONE large flat yellow lightning-bolt-shaped tip (#FEF443 / #E9BE14, olive outline #646124). The creature's two feet stand ON TOP of this flat yellow lightning tip, which lies on the ground under its feet like a surfboard. Never two tails, never two lightning tips.

GRID LAYOUT (4 key poses, large and crisp, same size, the surfboard tip on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both blue eyes, both yellow cheeks, the cream belly, both small hands raised at the sides, both feet standing on the yellow lightning tip under them; the thin tail loops out to one side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, one cheek, the snout to the left, the hands in front on the left, both feet standing on the yellow lightning tip (seen from the side, pointing left like a surfboard); the thin tail loops behind the body on the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the brown back with the two cream stripes, the back of the head, both yellow ears, the face turned away to the upper right, no belly; the feet stand on the yellow lightning tip; the thin tail loops out on the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the brown back with the two cream stripes, the back of the head, both yellow ears, both feet standing on the yellow lightning tip; the thin tail comes from the rump and loops out to one side.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a chubby orange-brown electric mouse with a cream belly, round blue eyes, yellow cheeks, two big yellow ears with curled tips, two small arms with white hands, two short legs with white feet, and exactly ONE thin dark brown tail that loops on the ground and ends in ONE flat yellow lightning-bolt tip that the creature stands on like a surfboard.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), both cheeks visible, the belly visible, both feet on the yellow lightning tip, the thin tail looping behind on the right. Exactly one tail, two ears, two arms and two feet. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
