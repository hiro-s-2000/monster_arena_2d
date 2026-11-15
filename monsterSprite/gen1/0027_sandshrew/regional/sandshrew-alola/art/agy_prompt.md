# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (3/4 front pixel sprite, the creature faces lower-left) and `ref_back.png` (back pixel sprite, seen from behind and to the right). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. Do not reuse any image from another folder or another task: only the two reference images in this folder. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference images (a small pixel sprite: front 3/4 view and back view) into larger crisp pixel-art sprites, keeping its exact design, proportions and colours.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a small, chubby, round, squat ground rodent standing upright on two short stubby legs, with a big round head merged into a round body (no neck), leaning slightly forward. It is a little armoured digger.
2. Armour: the top of the head, the whole back, the outer arms, the legs and the tail are covered in a hard shell made of many small faceted ice-crystal plates, like a mosaic of pale icy blue tiles.
   - Plates: pale ice blue (#B6DBE7) with light blue (#84B3CE), mid teal-blue (#518D9F) shading, white glints (#FEFEFE).
   - Plate seams / grooves: dark slate blue (#354E73).
3. Face and belly: the snout / lower face (muzzle) is pale cream yellow (#FEFEA9); the big round belly and chest are cream beige (#EBE2B1) with tan shading (#D1C592). There is NO shell on the belly.
4. Eyes: exactly two large round eyes, deep blue (#297BCD / #10397B) with a white highlight, set just above the cream muzzle. Small dark mouth line on the muzzle.
5. Ears: exactly two small pointed ears made of the same icy plates, on top of the head, short and close to the head.
6. Arms (2-Arm Rule): exactly two short stubby arms held in front of the belly, outer side plated in icy blue. Each hand ends in two small grey-brown claws (#897E67 with #CFD3D8 tips). Never more than two arms.
7. Legs (2-Leg Rule): exactly two short thick legs, icy plated, each foot with small grey-brown claws (#897E67, #CFD3D8). Never more than two legs.
8. Tail (1-Tail Rule): exactly ONE short, thick, pointed tail made of icy plates, sticking out from the lower back and pointing up and outward behind the body (like a small triangular wedge with a zig-zag plated edge). Never two tails.
9. Outlines: crisp 1-2px near-black pixel-art outline (#101010), no anti-aliasing halo, flat colours, no gradients, no blur. Do not add wings, horns, spikes or extra parts.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the cream muzzle and the cream belly visible, both arms in front of the belly, both feet forward, both small ears up, the one tail hidden or just peeking out behind one side.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the muzzle pointing left, the cream belly visible on the left edge, the plated back curving on the right, one arm forward to the left, the one short pointed tail sticking out behind to the right.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right, like the back reference: the big rounded plated back, the back of the head with the face turned away to the upper right, the one tail sticking out at the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: the round plated back, the back of the head, both small ears, the one tail in the middle of the lower back pointing down/up behind, both feet.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference image exactly in design, colours and pose: a small round squat rodent whose head, back, outer arms, legs and tail are covered in faceted pale icy blue crystal plates with dark slate-blue seams, a cream yellow muzzle, a cream beige round belly, two big deep-blue eyes, two small pointed ears, two short arms with small grey-brown claws held in front of the belly, two short legs with grey-brown claws, and exactly ONE short pointed plated tail sticking out behind the body on the right.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left (exactly like the front reference): the face looks diagonally down-left with both eyes visible in natural 3/4 perspective (not squished), the cream belly visible. Exactly two arms, two legs, one tail. Crisp 1-2px near-black outline, limited palette, no anti-aliasing.
