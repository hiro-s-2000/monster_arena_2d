# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, the creature faces right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (pixel sprites: front 3/4 view and back view) into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a hunched, crouching bipedal ice creature with a short snout, leaning forward, with a big rounded back; it stands on two short thick legs, arms held forward and down, ready to slash.
2. Colour palette:
   - Skin (head, arms, legs, tail): pale icy blue (#B7E3E7) with greyish blue shading (#77A2BB) and slate shadow (#434F6F).
   - Face mask / chest plate: light lavender grey (#BDBDCD), muzzle tip white (#F1F1F4).
   - Back spikes: translucent ice crystal blue — highlight #B0E5F8, light #7DCFF2, mid #52B0CF, deep #3C88B4, shadow #3A6B8C, darkest edge #29397B.
   - Claws: white (#F1F1F4) with mid grey (#606060) edges.
   - Eyes: small, black (#101010) with a white glint; outlines crisp 1-2px near-black (#101010), no anti-aliasing halo.
3. Head spikes (2-Spike Rule): exactly TWO tall pointed ice spikes rise from the top of the head, the front one the tallest. Plus exactly two small pointed ears at the sides of the head.
4. Back spikes (most important): the whole back is covered by a crest of large flat pointed ice-crystal spikes — about 7 big spikes, all pointing up and backward, fanning from the neck down to the rump. They sit ONLY on the back; the belly, arms and legs have no spikes. Do not turn them into fur.
5. Arms (2-Arm Rule): exactly two arms, each ending in exactly THREE long curved white claws.
6. Legs (2-Leg Rule): exactly two short legs, each foot with exactly THREE small white claws.
7. Tail (1-Tail Rule): exactly one short stiff tail with a few small pointed ice spikes, sticking out behind the rump. Never two tails.
Never add extra arms, legs, heads, tails or wings.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: face with both eyes, lavender chest, both arms with three claws each held forward, both legs, the two head spikes up, the back spikes visible behind the shoulders on both sides.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, snout to the left, both arms forward to the left with claws, the back spikes fanning up and to the right along the back, the tail behind on the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the big back full of ice spikes and the back of the head with its two spikes, the face turned away to the upper right; one arm's claws peek out on the right; the tail at the rump.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the back covered by the ice-spike crest, the two head spikes at the top, the tail at the bottom centre, both feet.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a hunched pale-icy-blue biped with a lavender chest, two tall ice spikes on the head and two small ears, a big back crest of about 7 translucent blue ice-crystal spikes, two arms each with exactly three long white claws, two short legs each with three small white claws, and one short spiky tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), both arms forward with claws, the back spikes rising behind the body to the right. Exactly one tail, two arms, two legs. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
