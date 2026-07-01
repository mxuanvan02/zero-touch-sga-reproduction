"""WeedNet (Sequoia multispectral) real-field loader for the mapping experiment.

Design principles (project contract): NOTHING hard-coded.
  * Dataset dir from WEEDNET_DIR env or --weednet-dir, else repo-relative default.
  * The "field" is a REAL NDVI orthomosaic tile; the detection target is the REAL
    crop/weed annotation mask shipped with the tile -- not a synthetic Gaussian.
  * The detection threshold is derived from the data (the annotation itself
    defines targets), so no magic hotspot parameters are injected.
  * Same author group as WeedMap (Sa et al.); Sequoia RedEdge/NIR/NDVI bands.

Every returned array is a function of the PNG bytes on disk, so a re-run under
the same seed reproduces it exactly.
"""
import os
import glob
import numpy as np
from PIL import Image

_ENV = "WEEDNET_DIR"


def default_weednet_dir():
    env = os.environ.get(_ENV)
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "datasets", "weednet", "data", "Sequoia"),
        os.path.join(here, "..", "datasets", "weednet", "data", "Sequoia"),
    ):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.join(here, "datasets", "weednet", "data", "Sequoia")


def _pairs(seq_dir, band="SequoiaNdvi_30", split="train"):
    """Return sorted (ndvi_png, annot_png) pairs that both exist."""
    img_dir = os.path.join(seq_dir, band, split)
    ann_dir = os.path.join(seq_dir, band, split + "annot")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    pairs = []
    for im in imgs:
        base = os.path.basename(im)
        # image names carry the band tag (e.g. '0000_ndvi_crop.png') while the
        # annotation drops it ('0000_crop.png'); also try a bare-index name
        # ('0000.png') as used in some split folders. Match by trying candidates.
        cands = [
            base,
            base.replace("_ndvi", "").replace("_nir", "").replace("_red", ""),
            base.split("_")[0] + ".png",
        ]
        for c in cands:
            ann = os.path.join(ann_dir, c)
            if os.path.exists(ann):
                pairs.append((im, ann))
                break
    return pairs


def load_field(seq_dir=None, band="SequoiaNdvi_30", split="train", index=0,
               downscale=None):
    """Load ONE real NDVI field + its target mask.

    Returns (field, target) as float/bool arrays in [0,1]. `field` is the
    normalized NDVI intensity; `target` is the (weed/crop) annotation > 0.
    Both come straight from disk -- the planner is scored against a measured
    crop/weed distribution, not injected blobs.
    """
    seq_dir = seq_dir or default_weednet_dir()
    pairs = _pairs(seq_dir, band, split)
    if not pairs:
        raise FileNotFoundError(
            f"No NDVI/annotation pairs under {seq_dir!r} band={band} split={split}. "
            f"Clone github.com/inkyusa/weedNet into datasets/weednet."
        )
    im_path, ann_path = pairs[index % len(pairs)]
    field = np.asarray(Image.open(im_path).convert("L"), dtype=np.float32)
    ann = np.asarray(Image.open(ann_path).convert("L"))
    if downscale and downscale > 1:
        h, w = field.shape
        field = np.asarray(Image.fromarray(field).resize(
            (w // downscale, h // downscale), Image.BILINEAR), dtype=np.float32)
        ann = np.asarray(Image.fromarray(ann).resize(
            (w // downscale, h // downscale), Image.NEAREST))
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    target = ann > 0                              # any annotated crop/weed cell
    return field, target


def list_fields(seq_dir=None, band="SequoiaNdvi_30", split="train"):
    return _pairs(seq_dir or default_weednet_dir(), band, split)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--weednet-dir", default=None)
    ap.add_argument("--band", default="SequoiaNdvi_30")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--downscale", type=int, default=8)
    args = ap.parse_args()
    field, target = load_field(args.weednet_dir, args.band, args.split,
                               args.index, args.downscale)
    print(json.dumps({
        "field_shape": list(field.shape),
        "field_min": float(field.min()), "field_max": float(field.max()),
        "target_frac": float(target.mean()),
        "n_fields": len(list_fields(args.weednet_dir, args.band, args.split)),
    }, indent=2))
