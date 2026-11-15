# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left) and `ref_back.png` (back view, seen from behind and above). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the flying creature from the two attached pixel-art reference images (front view and back view) into larger, crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small round blue bat-like flying creature. The body is a small rounded ball with NO eyes at all (the face is blank, smooth blue). It hovers in the air with wings spread; it never stands.
2. Colour palette:
   - Body, ears and wing bones: periwinkle blue (#8BB4EE) with light highlight (#BDCDFF), mid blue shading (#627BB4) and dark indigo shading (#4A417B).
   - Wing membrane underside (seen from the front and from below): magenta purple (#B4529C) with dark plum shading (#73205A).
   - Wing membrane top side (seen from behind/above): the same periwinkle blue as the body with darker blue shading.
   - Inner ears: pink (#D573BD) with plum (#73205A).
   - Mouth: one wide open dark mouth (#101010) with exactly 4 small white fangs (#FFFFFF, #D5D5D5): 2 on the upper jaw and 2 on the lower jaw.
   - Outlines: crisp 1px dark outline (#101010 / #4A417B), no anti-aliasing halo.
3. Ears (2-Ear Rule): exactly two large pointed ears on top of the head, blue outside with pink insides.
4. Wings (2-Wing Rule): exactly two large bat wings with 2-3 long finger bones each and a scalloped trailing edge, attached to the sides of the body, spread wide. Never more than two wings.
5. Legs (2-Leg Rule): exactly two very long thin straight blue legs (no feet, no toes) trailing behind/below the body like two streamers, slightly apart. No arms, no tail.
6. No eyes anywhere. Do not add eyes, pupils, a nose, hands or extra limbs.

GRID LAYOUT (4 key poses, large and crisp, same size, body centres on the same height):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: the blank face with the wide open mouth and 4 fangs in the centre, both ears up, both wings spread to the left and to the right showing their magenta undersides, both long legs hanging down and slightly apart.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: the open mouth points left, one ear in front of the other, both wings raised above the back (one nearer, one farther), both long legs trailing back to the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: the back of the round body and both ears, the face turned away to the upper right (mouth hidden), both wings spread showing their blue top sides, both legs trailing down-left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the round back, both ears, both wings spread left and right showing their blue top sides, both long legs hanging down.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small round periwinkle-blue flying bat creature with NO eyes, a wide open dark mouth with exactly 4 small white fangs, exactly two large pointed ears with pink insides, exactly two large bat wings with magenta-purple undersides, and exactly two very long thin straight blue legs trailing behind.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the open mouth looks diagonally down-left, both ears visible, both wings raised and spread (the far wing behind, the near wing showing its magenta underside), both long legs trailing back to the right. No eyes. Crisp 1px dark outline, limited palette, no anti-aliasing.
