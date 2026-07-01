"""Multi-UAV cooperative knowledge discovery via Federated Learning (Sec. VI),
with a real neural network and FedAvg over non-IID clients.

Each MSS (client) trains a small MLP on its own skewed slice of the
agricultural feature space and shares ONLY model weights (never raw data).
FedAvg averages client weights each round. Compared against isolated local
training (no collaboration) on a shared global test set.

Faithful to the paper's claim: privacy-preserving collective intelligence
that is robust under non-IID data.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import torch
import torch.nn as nn
import config as C

DEVICE = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, dim, hidden=64, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _make_noniid_clients(rng, n_clients, dim=10, n_classes=3, per=400, test=1500):
    """Non-IID split: each client sees a different, skewed class mixture and a
    shifted feature distribution (label + covariate shift)."""
    centers = rng.normal(0, 2.5, size=(n_classes, dim))
    clients = []
    for c in range(n_clients):
        # skewed class prior per client (Dirichlet -> non-IID label distribution)
        prior = rng.dirichlet(np.ones(n_classes) * 0.3)
        labels = rng.choice(n_classes, size=per, p=prior)
        shift = rng.normal(0, 0.6, size=dim)             # covariate shift
        X = centers[labels] + shift + rng.normal(0, 1.0, size=(per, dim))
        clients.append((X.astype(np.float32), labels.astype(np.int64)))
    # balanced global test set
    yt = rng.integers(0, n_classes, size=test)
    Xt = centers[yt] + rng.normal(0, 1.0, size=(test, dim))
    return clients, (Xt.astype(np.float32), yt.astype(np.int64))


def _local_train(model, X, y, steps, lr):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(X); yt = torch.as_tensor(y)
    for _ in range(steps):
        opt.zero_grad()
        out = model(Xt)
        loss = lossf(out, yt)
        loss.backward(); opt.step()
    return model


def _acc(model, test):
    X, y = test
    with torch.no_grad():
        pred = model(torch.as_tensor(X)).argmax(1).numpy()
    return float((pred == y).mean())


def _avg_state(states):
    out = {}
    for k in states[0]:
        out[k] = sum(s[k] for s in states) / len(states)
    return out


def _load_task(seed, n_clients, source, divine_dir):
    """Return (clients, test, meta). source='divine' uses real DIVINE data;
    source='synthetic' keeps the legacy Gaussian generator (fallback only)."""
    if source == "divine":
        from divine_data import build_federated_task
        clients_named, test, meta = build_federated_task(seed=seed, divine_dir=divine_dir)
        clients = [(X, y) for (X, y, _name) in clients_named]
        return clients, test, meta
    # ---- synthetic fallback ----
    rng = np.random.default_rng(seed)
    dim, n_classes = 10, 3
    clients, test = _make_noniid_clients(rng, n_clients, dim, n_classes)
    meta = {"n_clients": len(clients), "n_features": dim, "n_classes": n_classes,
            "source": "synthetic"}
    return clients, test, meta


def run_federated_nn(seed=C.SEED, n_clients=C.V, rounds=25, local_steps=10, lr=0.05,
                     source=None, divine_dir=None):
    """FedAvg vs isolated local-only on a non-IID task.

    Data source is selected (in priority order) by the explicit `source` arg, the
    FL_SOURCE env var, else defaults to real DIVINE data. Model shape (input dim,
    #classes) is taken from the task meta -- never hard-coded -- so the same code
    runs on synthetic or real data without edits.
    """
    source = (source or os.environ.get("FL_SOURCE", "divine")).lower()
    torch.manual_seed(seed)

    clients, test, meta = _load_task(seed, n_clients, source, divine_dir)
    dim = meta["n_features"]
    n_classes = meta["n_classes"]

    # ---- isolated local-only: each client trains alone, eval on global test ----
    local_accs = []
    for (X, y) in clients:
        m = MLP(dim, n_classes=n_classes)
        _local_train(m, X, y, steps=rounds * local_steps, lr=lr)
        local_accs.append(_acc(m, test))
    local_acc = float(np.mean(local_accs))            # mean isolated generalisation

    # ---- FedAvg: share weights only ----
    glob = MLP(dim, n_classes=n_classes)
    curve = []
    for _ in range(rounds):
        states = []
        for (X, y) in clients:
            m = MLP(dim, n_classes=n_classes)
            m.load_state_dict(glob.state_dict())
            _local_train(m, X, y, steps=local_steps, lr=lr)
            states.append(m.state_dict())
        glob.load_state_dict(_avg_state(states))
        curve.append(_acc(glob, test))
    fed_acc = curve[-1]

    improvement = (fed_acc - local_acc) / max(local_acc, 1e-9)
    return {
        "local_acc": local_acc,
        "fed_acc": float(fed_acc),
        "improvement": float(improvement),
        "curve": curve,
        "meta": meta,
    }


if __name__ == "__main__":
    import json
    r = run_federated_nn()
    print("task meta:", json.dumps(r["meta"], ensure_ascii=False))
    print(f"Local-only acc (mean): {r['local_acc']:.3f}")
    print(f"FedAvg acc:            {r['fed_acc']:.3f}")
    print(f"Improvement:           {100*r['improvement']:.1f}%")
