# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front view, the creature faces lower-left, left arm reaching forward-left, long tail rising on the right) and `ref_back.png` (back 3/4 view, the creature faces upper-right, tail on the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (low-resolution pixel sprites: front 3/4 view and back 3/4 view) into crisp, clean pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a tall, slim, athletic upright bipedal duck-like amphibian standing in a wide crouched fighting stance on two long legs. Narrow head with a long flat duck bill, a slim torso with a light blue chest, long thin arms. No wings, no feathers, no ears, no teeth.
2. Head crest (4-Spike Rule): exactly FOUR sharp pointed spikes sweeping up and back from the top/back of the head like a spiky crown. Never more, never fewer.
3. Forehead gem: exactly ONE small round red gem (#DE1818 with dark red #8B0000 ring and a white glint) in the middle of the forehead, above the eyes.
4. Eyes: two narrow fierce red eyes (#DE1818) with white, just above the bill.
5. Bill: one long flat pale yellow bill (#FFFFA4 / #D5D56A, dark olive outline #736A10).
6. Arms (2-Arm Rule): exactly TWO long thin arms. Each hand is a webbed hand with three short pale yellow claws (#FFFFA4 / #D5D56A).
7. Legs (2-Leg Rule): exactly TWO long legs with bent knees, each ending in a large flat webbed foot with three pale yellow claws, both feet flat on the ground.
8. Tail (1-Tail Rule): exactly ONE long, thick, pointed tail growing from the lower back and curving up behind the body, about as long as the torso.
9. Colour palette:
   - Body, head, arms, legs and tail: slate blue-violet (#7B7BB4) with a lighter sky blue (#8BACD5) on the chest, the front of the head, the thighs and the top of the arms, and a pale highlight blue (#A4C5FF).
   - The back of the body, head and tail is mostly the slate blue-violet (#7B7BB4).
   - Bill and claws: pale yellow (#FFFFA4) with olive shading (#D5D56A / #A4A439 / #736A10).
   - Outlines: crisp 1-2px dark teal-navy pixel-art outline (#31526A) mixed with near-black (#101010) on the shadow side, no anti-aliasing halo.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, same body size in all four):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the red forehead gem, the bill pointing toward the viewer, the four crest spikes on top, both arms held out to the sides with claws open, both legs in a wide stance, the tail just peeking behind one side.
- Top-Right: WEST (270°, left side profile) — facing left in a sharp side profile: one eye, the bill pointing to the left, the four crest spikes sweeping back to the right, one arm reaching forward to the left (the other arm behind the body), both legs (one slightly behind the other), the long tail sticking out and curving up on the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: the back of the head and body, the face turned away to the upper right, the tip of the bill just visible on the right, both arms, the four crest spikes, the long tail rising on the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the back of the head with the four crest spikes, the back, both arms at the sides, the long tail rising from the lower back in the centre, both legs.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a tall slim slate blue-violet duck-like amphibian with a light blue chest, a long pale yellow bill, exactly one red gem on the forehead, narrow red eyes, exactly four sharp crest spikes on the back of the head, exactly two long arms with webbed yellow-clawed hands, exactly two long legs with large webbed yellow-clawed feet in a wide crouched stance, and exactly one long pointed tail curving up behind the body on the right.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the bill pointing down-left, the tail visible on the right. Crisp 1-2px dark outline, limited palette, no anti-aliasing. Never add extra spikes, arms, legs, gems or tails.
