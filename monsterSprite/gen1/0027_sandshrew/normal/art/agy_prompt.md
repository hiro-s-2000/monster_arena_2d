# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, tail on the right) and `ref_back.png` (back 3/4 view, the creature faces right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, round, stocky bipedal burrowing rodent standing upright on two short thick legs, slightly hunched, with a rounded head that sits low on the shoulders, a short pointed muzzle, and two short arms held in front of the chest. The whole back, head, arms, legs and tail are covered in a hard armour of small rectangular plates (like a brick pattern) outlined with brown lines. No spikes, no quills, no horns, no wings.
2. Colour palette:
   - Armour plates (head, back, arms, legs, tail): sand yellow (#CDB400) with bright yellow highlights (#EEDE10), ochre shading (#B48310) and brown plate lines (#734A18 / #836210).
   - Belly, throat and muzzle: smooth pale cream (#E6D5C5) with grey-beige shading (#CDBDB4). The cream belly is one oval from the throat to between the legs.
   - Eyes: two eyes, dark blue (#10397B) with a lighter blue (#297BCD) and a tiny white highlight.
   - Claws: small white-grey claws (#EEEEDE / #838383).
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly TWO small, short, rounded triangular ears on top of the head, yellow plated with a small pale inner side. Never long ears.
4. Limbs (exact counts): exactly TWO short arms, each ending in a small paw with two tiny white claws; exactly TWO short thick legs with feet that show two or three tiny white claws each. Never more than two arms or two legs.
5. Tail (1-Tail Rule): exactly ONE short, thick, pointed tail that grows from the lower back / rump, plated like the back, pointing out and slightly up. It is short (shorter than the body is tall). Never two tails.
Do not add any new part. Do not increase the number of ears, arms, legs, claws or tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the cream muzzle and the cream belly visible, both ears up, both arms held in front of the chest to the left and right, both feet visible; the short tail is behind the body and sticks out to the right side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the muzzle pointing to the left, the near arm in front to the left, the cream belly on the left edge of the body, the plated back curved on the right; the single short tail extends behind to the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the plated back and the back of the head, the face turned away to the upper right (only the edge of the muzzle visible), no belly; the single tail comes from the rump and points out to the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the plated back, the back of the head with both ears, both arms showing at the sides, both feet; the single short tail comes out of the rump and points down and slightly out.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small round sand-yellow bipedal rodent covered in brick-like plates with brown plate lines, a pale cream belly and muzzle, dark blue eyes, two small rounded ears, two short arms with tiny white claws held in front of the chest, two short thick legs, and exactly ONE short thick pointed tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream belly visible, the short tail behind the body to the right. Exactly one tail, two ears, two arms and two legs. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
