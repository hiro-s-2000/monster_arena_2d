# Task (image generation only — do not edit any other files)

Use your image generation tool. Attach BOTH reference images to every generation: `ref_front.png` (pixel-art 3/4 front view, the creature hunches forward and faces lower-left, its short tail sticks out to the right) and `ref_back.png` (pixel-art back 3/4 view, the creature faces right, the tail points to the left). Save the results exactly as:

1. `out/keys_2x2.png` — prompt A below
2. `out/sw_single.png` — prompt B below

Do not write or modify any other file. When both PNGs are saved, reply with their paths and the image sizes.

## Prompt A (2x2 key views)

Create a high-resolution 2x2 grid (1376x768) of authentic 16-bit GBA/DS pixel-art monster sprites on a pure solid white background (#FFFFFF). Turn the creature from the two attached reference sprites into crisp, larger pixel-art sprites, keeping its exact design, proportions and colours. No text, no labels, no grid lines, no shadows on the ground.

CRITICAL ANATOMICAL & DESIGN SPECIFICATIONS (MUST FOLLOW 100%):
1. Body: a stocky, hunched bipedal ground-dwelling pangolin-like creature, standing on two short thick legs and leaning forward, arms held in front ready to slash. Small head with a short pointed cream snout, one small dark eye per side with a white highlight, a cream throat.
2. Colour palette (use only these families):
   - Skin of head, arms, legs and tail: golden yellow (#F6D520) with darker gold shading (#D5AC20) and olive shadow (#947B00, #5A4A00).
   - Belly, chest, throat and snout: pale cream (#DED5B4) with light tan shading (#C5B47B).
   - Back quills: brown (#8B6218) with light tan-brown highlight (#AC8331, #CDA44A) and dark brown shadow (#4A3910).
   - Claws: white (#FFFFFF) with light grey (#8B8B8B) and grey (#525252) shading.
   - Outlines: crisp 1-2px near-black outline (#101010), no anti-aliasing halo.
3. Quills: the whole back, from the top of the head to the rump, is covered by MANY large pointed brown quills (spikes) that point up and backward like a spiky crest. The quills cover only the back; the face, arms, belly and legs stay yellow / cream. Two short yellow spikes stick up on the forehead in front of the brown quills.
4. Limbs (2-Arm / 2-Leg Rule): exactly TWO arms, each hand ending in exactly THREE long curved white claws; exactly TWO short thick legs, each foot with THREE small white claws. No wings, no extra limbs.
5. Tail (1-Tail Rule): exactly ONE short thick yellow tail sticking out backward from the rump, with small yellow spikes along its upper edge. Never two tails.

GRID LAYOUT (4 key poses, large and crisp, same size, feet on the same baseline, all fully inside their quadrant with white margin):
- Top-Left: SOUTH (0°, front view) — facing straight toward the viewer, symmetric: both eyes, the cream snout pointing at the viewer, the cream belly visible, both arms bent forward at the sides with the white claws visible, both legs apart; the brown quills fan out behind the head and shoulders on both sides; the tail is hidden behind the body or peeks out low at one side.
- Top-Right: WEST (270°, left side profile) — facing LEFT in a sharp side profile: one eye, the snout pointing left, both arms reaching forward to the left with claws, the brown quills covering the whole back on the right side, the short spiky tail sticking out to the RIGHT at hip height.
- Bottom-Left: NORTH-EAST (135°, back-right 3/4 view) — seen from behind and to the right: we see the large brown quilled back, the face turned away to the upper right (only the snout tip and one arm peek out on the right); the short tail comes out low from the rump toward the lower left.
- Bottom-Right: NORTH (180°, pure back view) — seen straight from behind, symmetric: a big rounded mass of brown quills covering the whole back and head, the yellow arms and claws only peeking at the sides, the two yellow legs below, the short tail pointing down at the bottom centre.

## Prompt B (south-west single)

Create a single ultra-crisp 16-bit GBA/DS pixel-art sprite (1376x768) on a pure solid white background (#FFFFFF) of the same creature, matching the attached front reference sprite exactly in design, colours and pose: a stocky hunched golden-yellow creature with a cream belly and snout, a back covered with many pointed brown quills, two arms with three long white claws each, two short thick legs with small white claws, and ONE short spiky yellow tail.
View: SOUTH-WEST 3/4 quarter view, halfway between facing the viewer and facing left: the face looks diagonally down-left with the eye visible in natural 3/4 perspective (not squished), the cream belly visible, both clawed arms forward, the brown quills fanning up and back to the RIGHT, the short tail sticking out to the RIGHT at hip height. Exactly two arms, two legs, one tail. Crisp 1-2px near-black outline, limited palette, no anti-aliasing, no text, no ground shadow, the whole creature fully inside the image with white margin.
