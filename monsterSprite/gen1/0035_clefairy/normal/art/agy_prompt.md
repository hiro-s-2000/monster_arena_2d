# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, one arm raised in a wave, a small wing showing on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, the curled tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, very round, chubby fairy-like creature: one big round body that is head and torso in one (like a soft ball, slightly wider at the bottom), standing on two very short stubby legs. Happy face: two small black oval eyes, small rosy cheek marks, a small open smiling mouth with a red tongue. No neck, no nose, no horns, no claws.
2. Colour palette:
   - Body, ears, arms, legs, wings and tail: soft pink (#FFACAC) with rose shading (#E67B7B) and dark rose shadows (#9C5252).
   - Face and belly: a large pale peach area (#FFD5BD) covering the face and the whole front of the body.
   - Ear tips: dark brown (#5A3120) with mid brown (#734A39) and a grey-brown highlight (#9C8373).
   - Mouth: red (#EE4139) with a dark red inside (#B43929).
   - Small white-grey toe / finger nubs (#FFFFFF, #949494).
   - Outlines: crisp 1-2px near-black outline (#101010) mixed with dark rose (#9C5252), no anti-aliasing halo.
3. Head features (exact counts): exactly TWO large pointed ears on the top-left and top-right of the round body, each with a dark brown tip; exactly ONE single curl of hair on the forehead between the ears, curling down into a spiral (drawn as a pink spiral line on the body). Never more ears, never more curls.
4. Limbs (exact counts): exactly TWO short stubby arms with small rounded paws (two tiny nubs each), held out from the sides at mid-body; exactly TWO very short legs with small rounded feet. Never more than two arms or two legs.
5. Back (exact counts): exactly TWO tiny pink wings (small rounded leaf-shaped, with a scalloped edge) on the upper back, one on each shoulder; exactly ONE short tail on the lower back that curls into a flat spiral like a snail shell (pink with a dark rose spiral line). Never more wings, never more tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the smiling mouth, the peach face and belly, the forehead curl, both ears up, both arms held out to the left and right, both feet visible; the wing tips just peek out behind the shoulders; the tail is hidden behind the body.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the mouth on the left edge, the peach belly on the left side of the round body, the near arm reaching forward to the left, the forehead curl on the front of the head, one ear leaning back; one small wing on the upper back on the right; the single spiral tail on the lower back on the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: we see the pink back and the back of the head, the face turned away to the upper right, no belly, no eyes; both small wings on the upper back; the single spiral tail on the lower back toward the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the pink round back, both ears with brown tips, both arms showing at the sides, both feet; the two small wings on the upper back, one left and one right of the centre; the single spiral tail in the middle of the lower back.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a very round chubby soft-pink creature (head and body one ball) with a large pale peach face and belly, two large pointed ears with dark brown tips, one forehead hair curl, two small black eyes, a small open smiling mouth, two short stubby arms (the left one raised in a wave), two very short legs, two tiny pink wings on the upper back.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the peach belly visible, one small wing showing behind the right shoulder. Exactly two ears, one curl, two arms, two legs, two wings, one tail (the tail is hidden behind the body in this view). Crisp 1-2px dark outline, limited palette, no anti-aliasing.
