# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, the lightning-bolt tail tip on the left, the black tail cord loops on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-left, the black tail cord loops on the left, the lightning-bolt tip on the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a chubby, round-bellied bipedal electric rodent standing upright on two short legs with big flat feet, a round head about as wide as the torso, two short arms held up at the chest with small paws. No wings, no horns, no mane, no armour, no gloves.
2. Colour palette:
   - Head, body, arms, legs: warm orange (#F6AC29) with darker orange shading (#DE7B31) and rust shadows (#BD5A31).
   - Belly: a large pale cream oval (#FFEED5) on the chest and belly.
   - Paws, feet and the outer side of the ears: dark brown (#734131 / #5A2929); the soles of the feet pale tan (#E6BD83).
   - Cheeks: one round yellow cheek pouch on each side of the face (#FFDE5A / #FFBD00). Yellow, not red.
   - Eyes: black eyes with one white highlight; a small open mouth (#BD1808 inside).
   - Back: two short thin brown stripes (#944141) across the lower back.
   - Outlines: crisp 1px near-black outline (#000000 / #5A2929), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO large pointed ears, brown on the outside with a yellow inner side (#FFDE5A), each ear tip ending in a small curl. Never more than two ears.
4. Limbs (exact counts): exactly TWO arms and exactly TWO legs. Never more.
5. Tail (1-Tail Rule, most important): exactly ONE tail. It is a long, thin, dark grey-black whip-like cord (#393941 / #62626A, about as thick as a finger) that grows from the lower back, makes one loose loop on the ground beside the body, and ends in ONE flat yellow lightning-bolt-shaped tip (#FFDE5A / #FFBD00 with ochre #8B6210 shading). The lightning-bolt tip is about as tall as the head. Never two tails, never two bolts, no second bolt anywhere.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, both yellow cheeks, the cream belly, both arms raised at the chest, both feet visible, both ears up; the single tail cord comes from behind the body on the right and the lightning-bolt tip stands up beside the right hip.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, one cheek, snout pointing left, the near arm forward to the left, the cream belly on the left edge of the body; the single black tail cord trails behind to the right and the lightning-bolt tip stands up at its end on the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the orange back with the two brown stripes, the back of the head, the face turned away to the upper right (no belly, no eyes); the single black tail cord comes from the lower back and curves to the lower left, ending in the lightning-bolt tip on the left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the orange back with the two brown stripes, back of the head, both ears, both arms at the sides, both feet; the single tail cord comes from the lower back and curves out to the right, ending in the lightning-bolt tip on the right.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a chubby orange electric rodent with a cream belly, round yellow cheeks, two large brown-and-yellow ears with curled tips, two short arms with brown paws, two legs with brown feet, and exactly ONE long thin black tail cord that ends in ONE yellow lightning-bolt tip.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the near (right) cheek fully visible, the cream belly visible; the tail cord loops behind the body on the right and the lightning-bolt tip stands up beside the right hip. Exactly one tail and one bolt. Exactly two arms, two legs, two ears. Crisp 1px dark outline, limited palette, no anti-aliasing.
