"""
comm_prompts.py — prompts for the CROSS-SCALE COMMUNICATION hunt (Track A), a different job
from seam_prompts.py.

seam_prompts stress the *boundary* with stochastic texture, which FORGIVES disagreement (a
pebble field looks fine even if the two sides differ) -- which is why they kept coming back null.

These stress *communication*: structural content where the LR side and HR side must AGREE across
the cut, and a human notices instantly when they don't. If cross-scale attention transports
information well, a bisected brick wall keeps its courses aligned across the seam; if it doesn't,
the courses step, the letters change height, the face seams. All unimpeachably natural.

Pair with a mask that puts the cut THROUGH the structure. (Default saliency already fixes the
mask across DEC/MEAN/ALL arms, since stage-1 is shared; hand-drawn masks come later for the
boundary-density / distance-from-cut study.)

  periodic_bisected   : a repeating lattice whose period+phase must continue across the cut.
                        scale_shift != 1 here = structural disagreement survives full attention.
  text                : letter height + stroke must match across the cut; OCR gives a hard metric.
  face_bisected       : the strongest human prior for continuity; any seam is obvious.
  object_bisected     : a single rigid object cut in half; silhouette + surface must line up.
  perspective_gradient: element size varies smoothly with depth -> is the gradient continuous
                        across the cut, or does it step?
  repeated_motif      : identical repeated elements whose size+spacing must agree across the cut.
"""

PROMPTS = {
    "periodic_bisected": [
        "a full-frame wall of weathered red bricks, straight-on, even lighting",
        "rows of identical terracotta roof tiles, straight-on",
        "a chain-link fence filling the frame",
        "a grid of identical windows on an office building facade, frontal",
        "a black-and-white checkerboard tile floor, straight-on",
    ],
    "text": [
        "a vintage storefront sign with large bold lettering",
        "an open book page filled with printed text, macro",
        "a green street sign with clear white letters",
        "a newspaper front page photographed straight-on",
    ],
    "face_bisected": [
        "a close-up portrait of a woman's face, even lighting",
        "two people standing side by side, portrait photo",
    ],
    "object_bisected": [
        "a red sports car in profile against a plain wall",
        "a large ripe pineapple on a plain table",
        "a grand piano viewed from the side",
    ],
    "perspective_gradient": [
        "railroad tracks receding into the distance",
        "a long colonnade of identical columns receding",
        "a tiled plaza floor receding to the horizon",
    ],
    "repeated_motif": [
        "ornate floral wallpaper filling the frame",
        "a wall of identical stacked wine bottles",
    ],
}


def flat():
    """[(category, prompt), ...] in a stable order."""
    return [(c, p) for c, ps in PROMPTS.items() for p in ps]
