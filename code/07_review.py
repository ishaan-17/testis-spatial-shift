"""Two robustness experiments requested in review.

A. GNN depth ablation. The paper's mechanism says the GNN's *later* layers learn a function of the
   mouse neighbourhood distribution. That predicts a 1-layer GraphSAGE (which only ever sees the same
   neighbourhood mean the linear model sees) should transfer better than 2 or 3 layers. Tests depths 1/2/3.

B. Training-only standardisation. The paper z-scores each gene within each puck, including the test puck.
   That is unsupervised but still touches test data and may attenuate the apparent shift. Refits everything
   with mean/SD taken from the training pucks only.

Outputs: results/gnn_depth.csv, results/scaler_check.csv
"""
import os, time, gc
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
P, R = ROOT / "proc", ROOT / "results"
N_PC, K, SEEDS = 50, 15, [0, 1, 2]
CLASSES = ["SPG", "SPC", "RS", "ES"]
PUCKS = ["WT1", "WT2", "WT3", "DB1", "DB2", "DB3", "HU1", "HU2"]

A = ad.read_h5ad(P / "all.h5ad")
puck = A.obs.puck.values.astype(str)
y_raw = A.obs.cell_type.values.astype(str)
nn_idx = A.obsm["nn_idx"]
HVG = list(A.var_names)
cidx = {c: i for i, c in enumerate(CLASSES)}
in_task = np.isin(y_raw, CLASSES)
y_int = np.array([cidx.get(v, -1) for v in y_raw])
X_pp = np.asarray(A.X, dtype=np.float32)          # per-puck z-scored (the paper's version)
del A; gc.collect()

# Settings are ordered so the shift regimes the review asks about run first. This is a prioritised
# subset of the full grid: the species-shift folds (where GraphSAGE fails), plus one in-distribution
# reference per condition and one organization-shift fold.
SETTINGS = [
    ("WT->HU",   ["WT1", "WT2", "WT3"], "HU1"),
    ("WT->HU",   ["WT1", "WT2", "WT3"], "HU2"),
    ("WT->WT",   ["WT1", "WT2"],        "WT3"),
    ("HU->HU",   ["HU1"],               "HU2"),
    ("WT->DB",   ["WT1", "WT2", "WT3"], "DB1"),
    ("WTDB->HU", ["WT1", "WT2", "WT3", "DB1", "DB2", "DB3"], "HU1"),
]


def local_adj(idx, k):
    pos = -np.ones(len(nn_idx), dtype=np.int64); pos[idx] = np.arange(len(idx))
    nb = pos[nn_idx[idx, :k]]; rows, cols = np.nonzero(nb >= 0); cols = nb[rows, cols]
    deg = np.bincount(rows, minlength=len(idx)).astype(np.float32); vals = 1.0 / np.maximum(deg[rows], 1)
    return torch.sparse_coo_tensor(torch.tensor(np.stack([rows, cols])), torch.tensor(vals), (len(idx), len(idx))).coalesce()


class SAGE(nn.Module):
    """GraphSAGE-mean with a configurable number of layers."""
    def __init__(self, d, depth, h=128, c=4, p=0.2):
        super().__init__()
        dims = [d] + [h] * depth
        self.self_lin = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(depth)])
        self.nbr_lin = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(depth)])
        self.out = nn.Linear(dims[-1], c); self.p = p
    def forward(self, x, adj):
        h = x
        for s, n in zip(self.self_lin, self.nbr_lin):
            h = F.relu(s(h) + n(torch.sparse.mm(adj, h)))
            h = F.dropout(h, self.p, self.training)
        return self.out(h)


def cls_w(y, C=4):
    cnt = np.bincount(y, minlength=C).astype(np.float32)
    return torch.tensor(np.where(cnt > 0, cnt.sum() / (C * np.maximum(cnt, 1)), 0.0))


def fit_sage(Ztr, ytr, adj_tr, depth, seed, epochs=150, lr=3e-3, wd=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SAGE(Ztr.shape[1], depth); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Ztr); yt = torch.tensor(ytr); w = cls_w(ytr)
    for _ in range(epochs):
        m.train(); loss = F.cross_entropy(m(Xt, adj_tr), yt, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    def pred(Zte, adj_te):
        with torch.no_grad(): return m(torch.tensor(Zte), adj_te).argmax(1).numpy()
    return pred


def mets(yt, yp):
    return {"macro_f1": f1_score(yt, yp, average="macro", labels=np.unique(yt)),
            "bal_acc": balanced_accuracy_score(yt, yp)}


# ------------------------- Part A: GNN depth -------------------------
outA = R / "gnn_depth.csv"
rowsA = pd.read_csv(outA).to_dict("records") if outA.exists() else []
doneA = {(r["setting"], r["test"], r["depth"], r["seed"]) for r in rowsA}
for setting, trp, tep in SETTINGS:
    tr = np.where(np.isin(puck, trp) & in_task)[0]; te = np.where((puck == tep) & in_task)[0]
    if all((setting, tep, d, s) in doneA for d in (1, 2, 3) for s in SEEDS):
        continue
    t0 = time.time()
    Z = PCA(N_PC, random_state=0).fit(X_pp[tr]).transform(X_pp).astype(np.float32)
    adj_tr, adj_te = local_adj(tr, K), local_adj(te, K)
    for depth in (1, 2, 3):
        for seed in SEEDS:
            if (setting, tep, depth, seed) in doneA: continue
            pr = fit_sage(Z[tr], y_int[tr], adj_tr, depth, seed)
            r = {"setting": setting, "test": tep, "depth": depth, "seed": seed, **mets(y_int[te], pr(Z[te], adj_te))}
            rowsA.append(r)
            print(f"  [depth] {setting:9s} {tep} d={depth} s{seed} macroF1={r['macro_f1']:.3f}", flush=True)
    pd.DataFrame(rowsA).to_csv(outA, index=False)
    print(f"[depth {setting} -> {tep}] {time.time()-t0:.0f}s", flush=True)
    del Z; gc.collect()
print("PART A DONE", flush=True)

# ------------------- Part B: training-only standardisation -------------------
del X_pp; gc.collect()
orth = pd.read_csv(ROOT / "data" / "orthologs_1to1.csv") if (ROOT / "data" / "orthologs_1to1.csv").exists() else pd.read_csv(P / "orthologs_1to1.csv")
m2h = dict(zip(orth.mouse, orth.human))
blocks = []
for name in PUCKS:
    a = ad.read_h5ad(P / f"{name}.h5ad")
    a.obs["n_umi"] = np.asarray(a.X.sum(1)).ravel()
    a = a[a.obs["cell_type"].notna() & (a.obs["n_umi"] >= 100)].copy()
    if a.obs["species"].iloc[0] == "mouse":
        a = a[:, a.var_names.isin(m2h.keys())].copy(); a.var_names = [m2h[g] for g in a.var_names]
    a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    a = a[:, [g for g in HVG]].copy()
    blocks.append(np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float32))
    print(f"  [scaler] loaded {name} {blocks[-1].shape}", flush=True)
    del a; gc.collect()
X_ln = np.vstack(blocks); del blocks; gc.collect()
assert X_ln.shape[0] == len(puck), (X_ln.shape, len(puck))

rowsB = []
for setting, trp, tep in SETTINGS:
    tr = np.where(np.isin(puck, trp) & in_task)[0]; te = np.where((puck == tep) & in_task)[0]
    t0 = time.time()
    mu = X_ln[np.where(np.isin(puck, trp))[0]].mean(0)
    sd = X_ln[np.where(np.isin(puck, trp))[0]].std(0) + 1e-6
    Xs = np.clip((X_ln - mu) / sd, -10, 10).astype(np.float32)     # scaler from TRAINING pucks only
    Z = PCA(N_PC, random_state=0).fit(Xs[tr]).transform(Xs).astype(np.float32)
    del Xs; gc.collect()
    N = Z[nn_idx[:, :K]].mean(1); Fm = np.concatenate([Z, N], 1)
    lr_e = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", n_jobs=2).fit(Z[tr], y_int[tr])
    lr_n = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", n_jobs=2).fit(Fm[tr], y_int[tr])
    Pe = lr_e.predict_proba(Z[te]); adj_te = local_adj(te, K)
    Ps = (0.5 * torch.tensor(Pe) + 0.5 * torch.sparse.mm(adj_te, torch.tensor(Pe, dtype=torch.float32))).numpy()
    preds = {"expr": Pe.argmax(1), "nbr_k15": lr_n.predict(Fm[te]), "smooth_a0.5": Ps.argmax(1)}
    adj_tr = local_adj(tr, K)
    preds["sage_k15"] = fit_sage(Z[tr], y_int[tr], adj_tr, 2, 0)(Z[te], adj_te)
    for mdl, yp in preds.items():
        rowsB.append({"setting": setting, "test": tep, "model": mdl, **mets(y_int[te], yp)})
    pd.DataFrame(rowsB).to_csv(R / "scaler_check.csv", index=False)
    print(f"[scaler {setting} -> {tep}] {time.time()-t0:.0f}s  " +
          " ".join(f"{k}={f1_score(y_int[te], v, average='macro'):.3f}" for k, v in preds.items()), flush=True)
    del Z, N, Fm; gc.collect()
print("PART B DONE", flush=True)
