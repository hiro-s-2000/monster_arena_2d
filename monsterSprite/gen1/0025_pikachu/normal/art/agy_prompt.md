# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the upper right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, chubby, pear-shaped bipedal electric mouse standing upright on two short stubby legs, with a round head joined directly to the body (almost no neck), small black eyes with one white highlight, a tiny mouth, and two short arms held in front of the chest with small rounded paws.
2. Colour palette:
   - Fur (head, body, arms, legs, ears, tail): bright lemon yellow (#F6E652) with golden yellow shading (#F6BD20), orange-ochre shadows (#DE9400) and a pale yellow highlight (#FFF6A4) on the top of the head.
   - Cheeks: exactly two round red cheek patches (#C52018 with a lighter #E65A41 highlight), one on each side of the face.
   - Ear tips: dark charcoal grey to black (#41414A / #292929).
   - Back: exactly TWO horizontal brown stripes (#9C5200) across the lower back, and the base of the tail is brown (#9C5200).
   - Outlines: crisp 1px dark brown outline (#623108) mixed with black (#000000) on the darker sides, no anti-aliasing halo.
3. Limbs (exact counts): exactly TWO short arms with small rounded paws (no claws, no gloves), exactly TWO short legs with small rounded feet. Never more than two arms or two legs.
4. Ears (2-Ear Rule): exactly TWO long pointed yellow ears with black tips, standing up and slightly outward from the top of the head.
5. Tail (1-Tail Rule, most important): exactly ONE flat yellow tail shaped like a zig-zag lightning bolt, with a short brown base where it grows from the rump, rising up and outward behind the body; the tail is about as tall as the body, NOT bigger. Never two tails. No wings, no horns, no extra spikes.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, both red cheeks, the mouth, both ears up, both arms held in front of the chest, both feet visible; the lightning-bolt tail peeks out behind the body at the upper right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, one red cheek, the small snout pointing to the left, the near arm in front of the chest pointing left, both ears up (one behind the other); the single lightning-bolt tail grows from the rump on the right (back) side and zig-zags up and to the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference but mirrored: we see the yellow back with its two brown stripes, the back of the head, the face turned away to the upper right (only the edge of one red cheek visible), both ears; the single tail grows from the rump and zig-zags up to the upper left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the yellow back with exactly two brown horizontal stripes, the back of the head, both ears, both feet; the single lightning-bolt tail grows from the rump at the centre and zig-zags straight up behind the back.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small chubby lemon-yellow bipedal electric mouse with two long black-tipped ears, small black eyes, two round red cheeks, two short arms in front of the chest, two stubby feet, and exactly ONE flat zig-zag lightning-bolt tail with a brown base.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), both red cheeks visible, the lightning-bolt tail behind the body rising to the upper right. Exactly one tail, two ears, two arms and two legs. Crisp 1px dark outline, limited palette, no anti-aliasing.
