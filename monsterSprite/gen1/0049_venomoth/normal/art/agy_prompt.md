# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel-art sprites: front view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large flying moth hovering in the air. No legs touch the ground, no arms, no hands, no feet. The body is small compared with the wings.
2. Colour palette (use only these, no new hues):
   - Wings, head and antennae: pale lilac (#F6DEFF) and light lilac (#EEC5FF), mid lilac (#D5ACDE), dusty violet veins and shading (#9C8BBD).
   - Outline: dark plum (#624A73) on the wings, near-black (#101010) around the head and body.
   - Abdomen (lower body): small egg-shaped furry brown abdomen (#A4837B) with dark brown speckles (#625239) and a tan highlight (#C5AC8B).
   - Eyes: large round pale grey-cyan compound eyes (#ACC5C5, rim #E6E6E6) with a dark centre (#101010).
3. Wings (4-Wing Rule, most important): exactly FOUR wings: two large rounded upper forewings and two smaller lower hind wings, each hind wing ending in a short rounded tail lobe pointing down. Wings are pale lilac with thin dusty-violet parallel veins radiating from the body. Never more than four wings, never two.
4. Antennae (2-Antenna Rule): exactly two long, thin, flat, pale lilac feathery antennae rising up from the top of the head in a V. Never more, never fewer.
5. Head: small lilac head with two big round pale eyes and a tiny dark mouth area; no horns, no hair.
6. Abdomen: exactly one small brown speckled abdomen hanging below the head between the wings. No tail, no stinger.

GRID LAYOUT (4 key poses, large and crisp, same size, all hovering at the same height, lowest points on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both big eyes, both antennae in a V, the head and the brown abdomen in the centre, the two forewings spread up-left and up-right, the two hind wings below them on each side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, head and antennae pointing to the left, the brown abdomen below and behind the head, the wings raised above and behind the body (near forewing and near hind wing in front, the far wings partly visible behind).
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back of the head and the tops of the antennae, the face turned away to the upper right; the backs of all four wings spread wide, the thin brown abdomen line in the centre.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric like the back reference: back of the head, both antennae, the four wings spread wide, a thin brown abdomen strip in the centre.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a hovering pale lilac moth with exactly four wings (two large rounded forewings and two smaller hind wings with short rounded tail lobes, dusty violet veins, dark plum outline), exactly two long thin feathery antennae, two big round pale grey-cyan eyes with dark centres, and one small egg-shaped brown speckled furry abdomen. No legs, no arms.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the brown abdomen below the head, the near wings on the left and the large far forewing spread up on the right like the reference. Exactly four wings and two antennae. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
