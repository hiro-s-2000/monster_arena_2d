# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back 3/4 view, the creature faces upper-right, one claw visible at the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a low, crouching crab-like bug creature almost completely covered by ONE huge mushroom cap. The cap is a tall rounded bell / dome shape (a little like a rounded triangle), much bigger than the rest of the creature, sitting on the bug's back and hanging down over it to just above the legs. Under the front edge of the cap is a small dark orange-brown bug body with a face.
2. Colour palette:
   - Mushroom cap: bright red (#EE3131) with darker crimson (#BD3141, #B44139) on the shadow side and the back, a pale salmon-pink shine (#FF6A6A) on the top, and about ten irregular flat yellow spots (#FFD552 and olive yellow #D5CD31) scattered over it.
   - Bug body, claws and legs: orange-brown (#DE6A31) with light orange (#FF8352) highlights and dark brown (#621000, #7B0808) shading.
   - Eyes: two pale, blank, white-grey eyes (#FFF6FF with grey #9C8BA4 / #D5D5D5), no pupils, no colour.
   - Outlines: crisp 1-2px near-black outline (#101010) and dark red-brown (#7B0808) inner lines, no anti-aliasing halo.
3. Face (exact count): exactly TWO small round pale blank eyes side by side under the front rim of the cap, no mouth visible, no nose, no teeth.
4. Claws (2-Claw Rule): exactly TWO large flat pincer claws (big orange-brown blade-like claws, each about as long as the bug body is wide), one on the left front and one on the right front, pointing down to the ground like legs. Never more than two claws.
5. Legs (4-Leg Rule): exactly FOUR thin, bent, spidery orange-brown walking legs with dark tips, two on each side, coming out from under the cap behind the claws. Never more than four thin legs. No arms, no tail, no wings, no extra mushrooms.

GRID LAYOUT (4 key poses, large and crisp, same size, legs on the same baseline, all four creatures the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the cap on top, under its front rim the two pale eyes side by side in the middle, the two big claws standing at the left and right front, two thin legs visible on each side behind the claws.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: the cap seen from the side (front rim on the left, the dome leaning back to the right), one pale eye under the left rim, the near claw in front at the left (the far claw just peeking behind it), the four thin legs in a row under the cap.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: mostly the back of the big spotted cap, no eyes, no face; one claw sticking out at the right front, the four thin legs under the cap rim.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the back of the big spotted cap covering everything, no eyes, no face; the four thin legs under the rim (two each side) and the tips of the two claws just peeking out at the far left and right.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a low crab-like bug almost covered by one huge red bell-shaped mushroom cap with about ten yellow spots, a small orange-brown body under the front rim with two pale blank white eyes (no pupils), exactly two large flat orange-brown pincer claws at the front, and exactly four thin spidery orange-brown legs.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both pale eyes visible, the near claw at the front left, the far claw at the front right, the four thin legs under the cap. Exactly two eyes, two claws, four thin legs, one cap. Crisp 1-2px dark outline, limited palette, no anti-aliasing.
