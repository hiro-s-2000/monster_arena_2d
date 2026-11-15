# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view: the rodent faces lower-left, sitting up, mouth open showing its big front teeth, tail curling up behind it on the right) and `ref_back.png` (pixel-art back view: seen from behind and slightly from the right, the round brown back, the face turned to the right, the long tail curling up on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Redraw the creature from the two attached reference sprites as larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a large, plump, pear-shaped rat sitting half-upright on its haunches. Big round body, a large head merged with the body (almost no neck), a short blunt snout. It stands on the ground on its two hind feet; its two small front paws are held in front of the chest.
2. Colour palette:
   - Body, back, head: tan brown (#A47329) with golden ochre (#C59439), dark brown shadow (#945210) and deep brown (#6A3900).
   - Face, cheeks, chest and belly: pale cream yellow (#EECD73, #F6F6A4).
   - Paws, feet and tail: pinkish tan (#C58352) with light peach highlights (#FFCD9C).
   - Inner ears: dark brown (#945210).
   - Teeth: two big white front incisors (#FFFFFF with grey #B4B4B4 shading) in the open mouth.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Head: exactly TWO small rounded ears on top of the head; a ragged spiky tuft of fur between them; small narrow fierce eyes; mouth open showing exactly TWO large front teeth (upper incisors) — never more teeth, never fangs.
4. Whiskers: three thin cream whiskers on each side of the snout (six in total), sticking out sideways.
5. Legs and paws (Count Rule): exactly TWO small front paws (three toes each) held in front of the chest, and exactly TWO hind feet (long, flat, pinkish tan, three toes each) flat on the ground. Never more than four limbs.
6. Tail (1-Tail Rule): exactly ONE long, thin, hairless pinkish-tan tail with faint segment rings, growing from the lower back / rump, curving up behind the body in a big arc like a hook. Never two tails.
7. Back: the back has a jagged lighter tan fur pattern (zig-zag tufts); no stripes, no wings, no spikes on the back.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, both ears, the open mouth with the two front teeth in the centre, whiskers out to both sides, cream belly, both front paws in front of the chest, both hind feet on the ground, the tail curling up behind the body on one side.
- Top-Right: WEST (90°, left side profile) — facing left in a sharp side profile: one eye, one ear, snout and teeth to the left, front paws forward to the left, both hind feet on the ground, the tail behind the body on the right rising from the rump in a big arc.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the round brown back, the back of the head turned away to the upper right, ears and a little of the whiskers visible, the tail curving up on the left side.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind: the round brown back with the jagged lighter fur pattern, the back of the head, both ears, both hind feet at the bottom, the tail rising from the rump and curving up to one side.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a plump tan-brown rat sitting half-upright, cream face and belly, two small ears with a spiky tuft between them, narrow fierce eyes, mouth open with exactly two big white front teeth, three cream whiskers on each side, two small pinkish front paws at the chest, two long flat pinkish hind feet on the ground, and exactly one long thin pinkish tail curling up behind the body to the right.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Exactly one tail, four limbs. Crisp 1-2px near-black outline, limited palette, no anti-aliasing.
