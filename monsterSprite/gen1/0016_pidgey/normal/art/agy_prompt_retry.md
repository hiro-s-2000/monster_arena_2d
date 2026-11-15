# RETRY: only regenerate out/keys_2x2.png (prompt A). Keep out/sw_single.png as it is. The previous keys_2x2 drew a completely different creature (a grey rodent); it MUST be the small brown-and-cream bird from the attached reference images, all four views.

# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the bird faces lower-left, tail fan on the right) and `ref_back.png` (back 3/4 view, the bird faces right, tail fan on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. (sw_single.png already done — do not regenerate it)

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small plump round songbird standing on the ground on two thin legs, with a round head, a short thick hooked beak, a round chubby body and a short fan-shaped tail. The two wings are folded against the sides of the body (not spread). No arms, no hands, no ears, no horns, no long tail.
2. Colour palette:
   - Head top, back, wings and tail: warm brown (#A4625A) with light rusty brown (#CD835A) highlights and dark brown (#734A31 / #412918) shading and feather lines.
   - Face, throat, chest and belly: pale cream yellow (#FFEE9C / #FFF6BD) with golden tan shading (#E6BD62).
   - Beak: red-orange (#EE6241) with dark red (#BD2920) shading.
   - Eyes: small black eyes with a thick black stripe running back from each eye toward the back of the head (a dark eye-mask line, #101010 / #412918).
   - Head crest: a small tuft of 3-4 short brown feathers sticking up on top of the head, with a cream band across the forehead just under it.
   - Legs and feet: short red-orange legs (#BD2920) with small grey-white toes (#BDBDBD / #FFFFFF).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Limbs (exact counts): exactly TWO wings (folded, one on each side of the body), exactly TWO thin legs with small three-toed feet. Never more than two wings or two legs, never arms.
4. Tail (1-Tail Rule): exactly ONE short brown tail fan made of 3-4 stiff feathers, sticking out straight behind the body from the rump. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four birds the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes with their black eye-stripes, the beak in the middle pointing toward the viewer, the cream chest and belly, both folded wings at the left and right sides, both legs and feet; the tail fan is hidden behind the body (only its tips may show).
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye with its eye-stripe, the beak pointing to the left, the cream chest on the left edge of the body, the near folded wing covering the side of the body; the single tail fan sticks out behind to the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: we see the brown back, both folded wings and the back of the head with the crest, the beak pointing to the upper right (only its edge visible), no belly; the single tail fan sticks out behind toward the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the brown back of the head with the crest, the brown back, both folded wings at the sides, both legs; the single tail fan points straight down toward the viewer from the rump, centred.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small plump brown songbird with a cream face and chest, a short red-orange hooked beak, black eye-stripes, a small brown crest tuft on top of the head, two folded brown wings, two thin red-orange legs with grey toes, and exactly ONE short brown tail fan behind the body.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream chest visible, the tail fan behind the body to the right. Exactly one tail, exactly two wings (folded), exactly two legs. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
