"""DIVINE Pilot-4 real-data loader for the federated-learning experiment.

Design principles (per project contract): NOTHING is hard-coded.
  * Dataset path comes from the DIVINE_DIR env var or --divine-dir, else a
    repo-relative default; missing data raises instead of silently faking.
  * Sensor columns are DISCOVERED by scanning the multi-row DIVINE header
    (node / device / variable rows), never by fixed column index.
  * The classification label (soil-moisture tension class) is derived from the
    DATA via empirical tertiles computed on the pooled training split, not from
    magic thresholds.
  * Federated clients are the natural (node x month) shards, so the non-IID
    structure is real (spatial + seasonal heterogeneity), not injected noise.

Every returned number is a function of the CSV bytes on disk, so a re-run under
the same seed reproduces it exactly.
"""
import os
import csv
import glob
import math
import numpy as np

# Header layout of the DIVINE "EM" export (0-indexed rows):
#   row 2 -> node name ("Gateway" / "NODO 1" / "NODO 2 far")
#   row 3 -> device    ("Temperature Probe" / "Soil Moisture Sensor" / ...)
#   row 5 -> variable   ("Temp - C", "Humedad del suelo  cm - cb", ...)
#   row 6.. -> hourly data, col 0 = "Date & Time"
_ROW_NODE, _ROW_DEV, _ROW_VAR, _ROW_DATA0 = 2, 3, 5, 6
_ENCODING = "latin-1"


def default_divine_dir():
    """Repo-relative default; overridable by env var. No absolute hard-code."""
    env = os.environ.get("DIVINE_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    # search a few conventional spots relative to the repo
    for cand in (
        os.path.join(here, "datasets", "divine_pilot4"),
        os.path.join(here, "..", "datasets", "divine_pilot4"),
        os.path.join(here, "data", "divine_pilot4"),
    ):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.join(here, "datasets", "divine_pilot4")


def _month_files(divine_dir):
    """The 12 monthly exports (exclude the 1-Year roll-up to avoid overlap)."""
    files = sorted(f for f in glob.glob(os.path.join(divine_dir, "*.csv"))
                   if "_Month_" in os.path.basename(f))
    if not files:
        raise FileNotFoundError(
            f"No DIVINE monthly CSVs found in {divine_dir!r}. "
            f"Set DIVINE_DIR or pass --divine-dir to a folder holding the "
            f"'*_Month_*.csv' files from Zenodo record 11432589."
        )
    return files


def _discover_columns(header_rows):
    """Return {node: {'temp':idx,'soil':idx,'leaf':idx,...}} by SCANNING headers.

    Column roles are matched on the variable text, so the loader survives column
    re-ordering or extra sensors -- no fixed indices.
    """
    node_row = header_rows[_ROW_NODE]
    dev_row = header_rows[_ROW_DEV]
    var_row = header_rows[_ROW_VAR]
    ncol = len(var_row)

    def norm(s):
        return (s or "").strip().lower()

    nodes = {}
    for i in range(ncol):
        node = (node_row[i] or "").strip() if i < len(node_row) else ""
        if not node.startswith("NODO"):        # field nodes only (shared schema)
            continue
        dev = norm(dev_row[i]) if i < len(dev_row) else ""
        var = norm(var_row[i])
        slot = nodes.setdefault(node, {})
        # role assignment by semantic content of the variable label
        if "soil moisture" in dev or "humedad del suelo" in var:
            slot["soil"] = i
        elif "temp" in var and "alta" in var:
            slot["temp_hi"] = i
        elif "temp" in var and "baja" in var:
            slot["temp_lo"] = i
        elif var.startswith("temp"):
            slot["temp"] = i
        elif "minutos" in var:                  # leaf-wetness minutes
            slot["leaf_min"] = i
        elif "alta humedad foliar" in var:
            slot["leaf_hi"] = i
        elif "humedad foliar" in var:
            slot["leaf"] = i
    return nodes


def _to_float(x):
    x = (x or "").strip().replace(",", ".")
    if x in ("", "---", "--", "nan", "NaN", "null"):
        return math.nan
    try:
        return float(x)
    except ValueError:
        return math.nan


def _hour_of_day(datestr):
    """Parse 'd/m/yy HH:MM' -> hour int; robust to spacing. Used for cyclic time."""
    try:
        timepart = datestr.strip().split(" ")[-1]
        return int(timepart.split(":")[0])
    except Exception:
        return 0


# Feature roles used to build X (label role 'soil' is excluded from X).
_FEATURE_ROLES = ["temp", "temp_hi", "temp_lo", "leaf", "leaf_hi", "leaf_min"]


def load_shards(divine_dir=None):
    """Parse every monthly CSV into (node, month) shards of real records.

    Returns a list of dicts: {name, node, month, X_raw (n,F), soil (n,)}.
    X_raw holds ambient/canopy features; soil holds the tension reading used to
    derive the label downstream. Rows with any NaN feature or NaN soil dropped.
    """
    divine_dir = divine_dir or default_divine_dir()
    shards = []
    for path in _month_files(divine_dir):
        with open(path, encoding=_ENCODING) as fh:
            rows = list(csv.reader(fh))
        if len(rows) <= _ROW_DATA0:
            continue
        cols_by_node = _discover_columns(rows[:_ROW_DATA0])
        month_tag = os.path.basename(path).split("_1_Month_")[0].split("EM_")[-1]
        for node, roles in cols_by_node.items():
            if "soil" not in roles or "temp" not in roles:
                continue
            feats, soils = [], []
            for r in rows[_ROW_DATA0:]:
                if not r or len(r) <= max(roles.values()):
                    continue
                hour = _hour_of_day(r[0])
                fv = [_to_float(r[roles[k]]) if k in roles else math.nan
                      for k in _FEATURE_ROLES]
                # cyclic time-of-day (data-derived, deterministic)
                fv.append(math.sin(2 * math.pi * hour / 24.0))
                fv.append(math.cos(2 * math.pi * hour / 24.0))
                sv = _to_float(r[roles["soil"]])
                if any(math.isnan(v) for v in fv) or math.isnan(sv):
                    continue
                feats.append(fv)
                soils.append(sv)
            if len(feats) >= 24:                # need at least a day of clean data
                shards.append({
                    "name": f"{node}|{month_tag}",
                    "node": node,
                    "month": month_tag,
                    "X_raw": np.asarray(feats, dtype=np.float32),
                    "soil": np.asarray(soils, dtype=np.float32),
                })
    if not shards:
        raise RuntimeError(
            f"Parsed 0 usable shards from {divine_dir!r}; check the CSV format."
        )
    return shards


def build_federated_task(seed=42, divine_dir=None, test_frac=0.25):
    """Turn DIVINE shards into a reproducible non-IID federated classification task.

    Task: predict the soil-moisture *tension class* (dry / normal / wet) from
    ambient temperature + leaf-wetness + time-of-day. Class boundaries are the
    empirical tertiles of the POOLED TRAINING soil readings -> data-derived, not
    hard-coded. Features are standardized with train-split statistics only.

    Returns (clients, test, meta) where clients=[(Xtr,ytr,name)...],
    test=(Xte,yte). All splits are seeded, so results reproduce exactly.
    """
    rng = np.random.default_rng(seed)
    shards = load_shards(divine_dir)

    # ---- per-shard temporal train/test split (no leakage across time) ----
    tr_parts, te_parts, client_specs = [], [], []
    for sh in shards:
        n = len(sh["soil"])
        cut = int(round(n * (1.0 - test_frac)))
        cut = max(1, min(n - 1, cut))
        client_specs.append({
            "name": sh["name"],
            "Xtr": sh["X_raw"][:cut], "str": sh["soil"][:cut],
            "Xte": sh["X_raw"][cut:], "ste": sh["soil"][cut:],
        })
        tr_parts.append(sh["soil"][:cut])
        te_parts.append((sh["X_raw"][cut:], sh["soil"][cut:]))

    # ---- data-derived label thresholds: tertiles of pooled TRAIN soil ----
    pooled_train_soil = np.concatenate(tr_parts)
    q1, q2 = np.quantile(pooled_train_soil, [1.0 / 3.0, 2.0 / 3.0])

    def label(soil):
        return np.digitize(soil, [q1, q2]).astype(np.int64)  # 0,1,2

    # ---- feature standardization from pooled TRAIN features only ----
    pooled_train_X = np.concatenate([c["Xtr"] for c in client_specs], axis=0)
    mu = pooled_train_X.mean(axis=0)
    sd = pooled_train_X.std(axis=0) + 1e-6

    clients = []
    for c in client_specs:
        Xtr = (c["Xtr"] - mu) / sd
        ytr = label(c["str"])
        # shuffle within client (seeded)
        idx = rng.permutation(len(ytr))
        clients.append((Xtr[idx].astype(np.float32), ytr[idx], c["name"]))

    Xte = np.concatenate([(c["Xte"] - mu) / sd for c in client_specs], axis=0)
    yte = label(np.concatenate([c["ste"] for c in client_specs]))
    Xte = Xte.astype(np.float32)

    meta = {
        "n_clients": len(clients),
        "n_features": int(pooled_train_X.shape[1]),
        "n_classes": 3,
        "class_thresholds_cb": [float(q1), float(q2)],
        "nodes": sorted({c["name"].split("|")[0] for c in client_specs}),
        "train_records": int(sum(len(y) for _, y, _ in clients)),
        "test_records": int(len(yte)),
    }
    return clients, (Xte, yte), meta


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--divine-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    clients, (Xte, yte), meta = build_federated_task(seed=args.seed,
                                                      divine_dir=args.divine_dir)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    # class balance on test (sanity, data-derived)
    uniq, cnt = np.unique(yte, return_counts=True)
    print("test class balance:", dict(zip(uniq.tolist(), cnt.tolist())))
    print("per-client sizes:", [(n, len(y)) for _, y, n in clients][:6], "...")
