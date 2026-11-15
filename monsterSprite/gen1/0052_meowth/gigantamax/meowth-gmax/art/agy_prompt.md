# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, the creature faces right, away from the viewer). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference pixel images (front 3/4 view and back view) into crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a strange, extremely elongated cat. The torso is a very long, thick, tube-like body that rises from the hips and arches over in a tall upside-down U, so the head hangs low in front of the body at chest height. It stands on its two hind legs only; it never walks on four legs.
2. Colour palette:
   - Fur of the whole body: pale cream (#F0DEA2) on the top and lit side, with light highlight (#FBF7E6) and dusty pinkish-brown shading (#C89F8C) on the underside and shadow side; dark brown inner lines (#7F5745).
   - Paws (2 hind feet) and the tail tip: orange-brown (#EA9F38 / #C5810B) with dark brown (#944100) toe lines.
   - Coin: one round gold coin (#F9D400 / #FCED87 / #CCA700 / #986100) on the forehead.
   - Inner ears: red-orange (#D46140).
   - Outlines: crisp 1px near-black outline (#101010), no anti-aliasing halo.
3. Head (1 head): a cat face with exactly two pointed ears with black rims and red-orange insides, two narrow mischievous eyes, a wide toothy grin (white teeth #F6F6F6), exactly one gold coin on the forehead, and long thin whiskers: two long straight whiskers sticking up from the top of the head and a few short whiskers on each cheek.
4. Arms (2-Arm Rule): exactly two arms. One very long thin arm reaches down from the head/shoulder all the way to the ground, ending in a small hand with three thin claw-fingers held up. The other arm is short and tucked under the chin. Never more than two arms.
5. Legs (2-Leg Rule): exactly two short thick hind legs, slightly apart, each with one rounded orange-brown paw with toe lines. Never four legs.
6. Tail (1-Tail Rule): exactly one tail, grows from the rump behind the hind legs, curls up into a spiral; the curled tip is orange-brown.

GRID LAYOUT (4 key poses, large and crisp, same size, both feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer: the arched body rises behind and the head hangs in front of it, centred, with both eyes, both ears, the coin and the grin visible; the long arm reaches to the ground on one side, the two hind paws are below, the tail curls out to the other side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile like the front reference: the long body arches from the hips on the right up and over to the left, the head hangs at the left with one eye, the long arm reaches down to the ground at the far left, two hind legs at the right, the tail curls up behind at the far right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — like the back reference: seen from behind and to the right, the arched back with the shaded fur faces us, the head is turned away to the upper right (we see the back of the head, the ears, the upright whiskers and only the edge of the coin), the long arm reaches down on the right, the tail curls up on the left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind: the arched back of the long body, the back of the hanging head with both ears and the two upright whiskers, the two hind legs below, the tail rising from the rump and curling up.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: an extremely elongated pale cream cat whose long tube-like body arches over in a tall upside-down U so the head hangs low in front, one gold coin on its forehead, two pointed ears, a wide toothy grin, long thin whiskers, one very long thin arm reaching down to the ground with three claw-fingers, one short arm under the chin, two short hind legs with orange-brown paws, and one spiral-curled tail with an orange-brown tip.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left, like the front reference: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished). Exactly two arms, two hind legs, one tail, one coin. Crisp 1px dark outline, limited palette, no anti-aliasing.
