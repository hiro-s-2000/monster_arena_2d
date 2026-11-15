# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the right) and `ref_back.png` (back 3/4 view, the creature faces right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, slim rat standing upright on its two hind legs in a slightly crouched, alert pose. A big head (about as large as the torso) with a pointed snout, a slim torso, two thin arms, two short hind legs with long flat feet. No wings, no horns, no armour.
2. Colour palette:
   - Head, back, arms, legs and tail: dark charcoal grey-green (#4A4F48) with lighter grey (#646C60) highlights and very dark grey (#323832) shadows.
   - Cheeks and belly: a big fluffy pale tan cheek ruff (#DCC8A3, shading #BF9B7B) that covers the lower face and cheeks like a thick beard, and a pale tan patch on the chest / belly.
   - Hands and feet: light brown (#BF9B7B / #86694F) with tiny toes.
   - Ears: inner ear pale blue-grey (#95A8A7 / #748792).
   - Eyes: small sharp red eyes (#E73D3D with dark red #BF0D0D).
   - Teeth: two small white front teeth (#F3F3F3) under the snout tip.
   - Outlines: crisp 1-2px near-black outline (#0D0D0D), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO large triangular ears on top of the head, each with a small pointed tuft at the tip, pale blue-grey inside.
4. Limbs (exact counts): exactly TWO arms, each ending in a small light-brown hand; exactly TWO hind legs, each with one long light-brown foot. Never more than two arms or two legs.
5. Tail (1-Tail Rule, most important): exactly ONE long thin dark grey tail that grows from the rump, curves up behind the body and ends in ONE fluffy curled spiral tuft of fur. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both red eyes, the snout, the two white teeth, the tan cheek ruff and the tan belly visible, both arms at the sides, both feet visible; the single tail is behind the body and curls up on the right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, snout pointing to the left, the near arm reaching forward to the left, the tan ruff and belly on the left edge of the body; the single tail extends behind to the right and curls up with the tuft at its tip.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: we see the dark grey back and the back of the head with both ears, the face turned away to the upper right (only the edge of the cheek ruff visible), no belly; the single tail comes from the rump and curls up to the left with the tuft at its tip.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the dark grey back, back of the head, both ears, both arms showing at the sides, both feet; the single tail comes out of the rump and curls up to the right with the tuft at its tip.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small slim dark charcoal grey rat standing upright on two hind legs, a big head with two large triangular ears (pale blue-grey inside), small red eyes, a pointed snout with two white front teeth, a thick fluffy pale tan cheek ruff, a tan belly patch, two thin arms with light-brown hands, two long light-brown feet, and exactly ONE long thin tail curling up behind the body ending in a curled tuft of fur.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cheek ruff and belly visible, the tail behind the body to the right curling up above the right hip. Exactly one tail, exactly two ears, two arms and two legs. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
