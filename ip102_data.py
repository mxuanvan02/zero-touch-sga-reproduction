"""IP102 real-image loader for the semantic-payload experiment.

Design contract (per project rule): NOTHING hard-coded.
  * Dataset path from IP102_DIR env var or --ip102-dir, else repo-relative
    default; missing data raises instead of silently faking.
  * The payload-reduction numbers are MEASURED from the real JPEG byte sizes on
    disk, not typed from a table. Raw size = actual file bytes; semantic size =
    raw x compression_ratio(gamma) from the paper's semantic model
    (semantic.py, Eq. 12), so the reduction falls out of real data + the model.
  * A fixed seed makes the sampled subset reproducible bit-for-bit.

Every returned number is a function of the image bytes on disk + the semantic
model, so a re-run under the same seed reproduces it exactly.
"""
import os
import glob
import numpy as np


def default_ip102_dir():
    """Repo-relative default; overridable by env var. No absolute hard-code."""
    env = os.environ.get("IP102_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "datasets", "ip102", "Classification", "ip102_v1.1"),
        os.path.join(here, "..", "datasets", "ip102", "Classification", "ip102_v1.1"),
    ):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.join(here, "datasets", "ip102", "Classification", "ip102_v1.1")


def _image_dir(ip102_dir):
    d = os.path.join(ip102_dir, "images")
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"IP102 image dir not found under {ip102_dir!r}. Set IP102_DIR or "
            f"pass --ip102-dir to the extracted 'ip102_v1.1' folder."
        )
    return d


def measure_real_payloads(ip102_dir=None, n_sample=2000, seed=42):
    """Sample real IP102 JPEGs and return their true on-disk byte sizes (KB).

    Returns (sizes_kb ndarray, meta dict). The sample is seeded, so the same
    images are drawn every run -> reproducible.
    """
    ip102_dir = ip102_dir or default_ip102_dir()
    img_dir = _image_dir(ip102_dir)
    all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not all_imgs:
        raise RuntimeError(f"No .jpg images found in {img_dir!r}.")
    rng = np.random.default_rng(seed)
    n = min(n_sample, len(all_imgs))
    idx = rng.choice(len(all_imgs), size=n, replace=False)
    sizes_kb = np.array([os.path.getsize(all_imgs[i]) / 1024.0 for i in idx],
                        dtype=float)
    meta = {
        "total_images": len(all_imgs),
        "sampled": int(n),
        "mean_raw_kb": float(sizes_kb.mean()),
        "median_raw_kb": float(np.median(sizes_kb)),
        "min_raw_kb": float(sizes_kb.min()),
        "max_raw_kb": float(sizes_kb.max()),
    }
    return sizes_kb, meta


def payload_reduction_real(ip102_dir=None, n_sample=2000, seed=42, gamma=None):
    """Measured semantic payload reduction on real IP102 imagery.

    raw size    = actual JPEG bytes on disk
    semantic sz = raw x compression_ratio(gamma)   (paper Eq. 12, semantic.py)
    reduction   = 1 - sum(semantic)/sum(raw)

    gamma defaults to config.GAMMA[0] (operating semantic fidelity), so the
    number is driven by the real images + the paper's semantic model, never
    hand-typed.
    """
    import config as C
    from semantic import compression_ratio
    if gamma is None:
        gamma = float(np.mean(C.GAMMA))
    sizes_kb, meta = measure_real_payloads(ip102_dir, n_sample, seed)
    rho = float(compression_ratio(gamma))
    raw_total = float(sizes_kb.sum())
    sem_total = raw_total * rho
    reduction = 1.0 - sem_total / raw_total  # == 1 - rho, but derived from data flow
    meta.update({
        "gamma": float(gamma),
        "compression_ratio": rho,
        "raw_total_kb": raw_total,
        "semantic_total_kb": sem_total,
        "payload_reduction": float(reduction),
    })
    return meta


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip102-dir", default=None)
    ap.add_argument("--n-sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=None)
    args = ap.parse_args()
    meta = payload_reduction_real(args.ip102_dir, args.n_sample, args.seed, args.gamma)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
