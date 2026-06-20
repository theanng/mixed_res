"""
seam_prompts.py — prompts that expose the LR/HR boundary in mixed-resolution generation,
even with Expand-and-Replace + segmentation masks + learned resizers.

Grouped by the FAILURE MECHANISM (not just "texture"), because each class stresses a
different reason the seam survives E&R:

  object_less_texture : nothing for YOLOE to segment -> the mask falls back to a saliency
                        blob and cuts through homogeneous, above-Nyquist texture.
  periodic            : the pattern's phase/period must continue across the seam (a NON-LOCAL
                        constraint E&R can't enforce) and aliases on the LR side -> moire.
  thin_structures     : a sharp line meets its own blurred continuation; sub-LR-pixel-width
                        detail vanishes on the LR side -> the eye tracks the broken edge.
  texture_over_smooth : low frequencies match (E&R is satisfied) but the high-frequency
                        texture stops dead at the boundary -> a pure spectral seam.
  continuous_texture  : a textured subject in a same-frequency surround -> the boundary lands
                        mid-texture, so the resolution drop is obvious.
  detail_field        : fine recurring detail spanning the frame; region-wide HF deficit that
                        a 2-token band cannot hide.
"""

PROMPTS = {
    "object_less_texture": [
        "an overhead macro of dense green moss completely filling the frame",
        "a full-frame field of fine wind-rippled sand, top-down",
        "extreme close-up of dense animal fur filling the entire frame",
        "a flat lawn of fine short grass seen from directly above",
        "a close-up of coarse woven wicker basketry filling the frame",
        "a top-down bed of small wet pebbles covering the whole frame",
        "analog TV static noise filling the entire frame",
    ],
    "periodic": [
        "a full-frame wall of weathered red bricks, straight-on, even lighting",
        "a chain-link fence in front of an out-of-focus parking lot",
        "rows of identical terracotta roof tiles receding to the horizon",
        "the glass window grid of a tall office skyscraper, frontal",
        "a black-and-white houndstooth fabric pattern filling the frame",
        "a tall bookshelf packed with hundreds of colorful book spines",
        "a halftone dot print pattern, macro",
    ],
    "thin_structures": [
        "bare winter tree branches silhouetted against a pale sky",
        "power lines and telephone wires crossing a clear sky",
        "heavy rain streaks against a dark background",
        "a single long strand of hair on a plain surface, macro",
        "an open page of a book filled with small printed text",
        "the rigging ropes of a tall sailing ship against the sky",
    ],
    "texture_over_smooth": [
        "an extreme close-up of human skin showing fine pores",
        "still water surface with very fine ripples, even lighting",
        "heavy film grain over a clear blue sky",
        "a single sheet of finely woven linen on one flat color",
    ],
    "continuous_texture": [
        "a leopard lying in tall dry savanna grass, fur and grass equally sharp",
        "a brown owl perched among dense autumn oak leaves",
        "a sparrow on gravel ground scattered with seeds and twigs",
        "a coral reef packed edge to edge with tiny polyps and small fish",
    ],
    "detail_field": [
        "a dense wheat field in bright sun, countless individual stalks",
        "a forest canopy seen from directly above, countless small leaves",
        "a knitted wool blanket filling the frame, fine stitches",
        "a meadow of thousands of small daisies receding into the distance",
    ],
}


def flat():
    """[(category, prompt), ...] in a stable order."""
    return [(c, p) for c, ps in PROMPTS.items() for p in ps]
