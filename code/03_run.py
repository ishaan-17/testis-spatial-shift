"""Shift benchmark for bead-level cell typing on Slide-seqV2 testis (Chen et al. 2021).

Label schemes:
  germ4  : spermatogenic stage on germ-cell beads only (SPG, SPC, RS, ES)   [primary]
  coarse6: SPG, SPC, RS, ES, TubSom (Sertoli+Myoid), Interst (Leydig+Endothelial+Macrophage)
  full9  : the nine NMFreg labels as published

Feature families (all on a 50-PC embedding fit on training pucks only):
  expr        : own-bead PCs                     -> logistic regression (LR) / MLP
  nbr_k       : [own PCs, mean of k spatial-neighbour PCs] -> LR / MLP  (BANKSY-style neighbourhood augmentation)
  nbr_only_k  : neighbour-mean PCs only (diagnostic)
  sage_k      : 2-layer GraphSAGE-mean, full batch
  smooth_a    : expr LR posteriors smoothed over the test puck's spatial kNN graph at test time
                p' = (1-a) p + a * mean_nbr(p)   (transductive; no spatial features at training time)

Shift settings:
  WT->WT (leave-one-puck-out), WT->DB, DB->DB, WT->HU, HU->HU, WTDB->HU
"""
import os, time
import numpy as np, pandas as pd, anndata as ad
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repository root


torch.set_num_threads(2)
P = str(ROOT / "proc"); R = str(ROOT / "results"); os.makedirs(R, exist_ok=True)
N_PC = 50
SEEDS = [0, 1, 2]
ALPHAS = [0.25, 0.5, 0.75, 1.0]

A = ad.read_h5ad(f"{P}/all.h5ad")
X_all = np.asarray(A.X, dtype=np.float32)
puck = A.obs.puck.values.astype(str)
y_raw = A.obs.cell_type.values.astype(str)
nn_idx = A.obsm["nn_idx"]

SCHEMES = {
    "germ4": ({"SPG": "SPG", "SPC": "SPC", "RS": "RS", "ES": "ES"}, ["SPG", "SPC", "RS", "ES"], [5, 15, 30], True),
    "coarse6": ({"SPG": "SPG", "SPC": "SPC", "RS": "RS", "ES": "ES", "Sertoli": "TubSom", "Myoid": "TubSom",
                 "Leydig": "Interst", "Endothelial": "Interst", "Macrophage": "Interst"},
                ["SPG", "SPC", "RS", "ES", "TubSom", "Interst"], [15], False),
    "full9": ({c: c for c in ["ES", "RS", "SPC", "SPG", "Sertoli", "Myoid", "Leydig", "Endothelial", "Macrophage"]},
              ["ES", "RS", "SPC", "SPG", "Sertoli", "Myoid", "Leydig", "Endothelial", "Macrophage"], [15], False),
}

SETTINGS = []
for t in ["WT1", "WT2", "WT3"]:
    SETTINGS.append(("WT->WT", [p for p in ["WT1", "WT2", "WT3"] if p != t], t))
for t in ["DB1", "DB2", "DB3"]:
    SETTINGS.append(("WT->DB", ["WT1", "WT2", "WT3"], t))
    SETTINGS.append(("DB->DB", [p for p in ["DB1", "DB2", "DB3"] if p != t], t))
for t in ["HU1", "HU2"]:
    SETTINGS.append(("WT->HU", ["WT1", "WT2", "WT3"], t))
    SETTINGS.append(("HU->HU", [p for p in ["HU1", "HU2"] if p != t], t))
    SETTINGS.append(("WTDB->HU", ["WT1", "WT2", "WT3", "DB1", "DB2", "DB3"], t))


def nbr_mean(Z, k):
    return Z[nn_idx[:, :k]].mean(1)


def fit_lr(Xtr, ytr):
    return LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", n_jobs=2).fit(Xtr, ytr)


class MLP(nn.Module):
    def __init__(self, d, h=256, c=9, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, c))
    def forward(self, x): return self.net(x)


def class_weights(y, C):
    cnt = np.bincount(y, minlength=C).astype(np.float32)
    return torch.tensor(np.where(cnt > 0, cnt.sum() / (C * np.maximum(cnt, 1)), 0.0))


def fit_mlp(Xtr, ytr, C, seed, epochs=40, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1], c=C); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); w = class_weights(ytr, C)
    n = len(yt); bs = 512
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(m(Xt[b]), yt[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    def pred(Xte):
        with torch.no_grad(): return m(torch.tensor(Xte)).argmax(1).numpy()
    return pred


class SAGE(nn.Module):
    def __init__(self, d, h=128, c=9, p=0.2):
        super().__init__()
        self.s1 = nn.Linear(d, h); self.n1 = nn.Linear(d, h)
        self.s2 = nn.Linear(h, h); self.n2 = nn.Linear(h, h)
        self.out = nn.Linear(h, c); self.p = p
    def forward(self, x, adj):
        h = F.relu(self.s1(x) + self.n1(torch.sparse.mm(adj, x))); h = F.dropout(h, self.p, self.training)
        h = F.relu(self.s2(h) + self.n2(torch.sparse.mm(adj, h))); h = F.dropout(h, self.p, self.training)
        return self.out(h)


def local_adj(idx, k):
    """row-normalised sparse adjacency over a local index set. Neighbours outside the set (e.g. non-germ beads
    in germ4) are dropped and rows renormalised, so context comes only from beads in the task subset."""
    pos = -np.ones(len(nn_idx), dtype=np.int64); pos[idx] = np.arange(len(idx))
    nb = pos[nn_idx[idx, :k]]
    rows, cols = np.nonzero(nb >= 0)
    cols = nb[rows, cols]
    deg = np.bincount(rows, minlength=len(idx)).astype(np.float32)
    vals = 1.0 / np.maximum(deg[rows], 1)
    return torch.sparse_coo_tensor(torch.tensor(np.stack([rows, cols])), torch.tensor(vals), (len(idx), len(idx))).coalesce()


def fit_sage(Ztr, ytr, adj_tr, C, seed, epochs=150, lr=3e-3, wd=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SAGE(Ztr.shape[1], c=C); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Ztr); yt = torch.tensor(ytr); w = class_weights(ytr, C)
    for ep in range(epochs):
        m.train(); loss = F.cross_entropy(m(Xt, adj_tr), yt, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    def pred(Zte, adj_te):
        with torch.no_grad(): return m(torch.tensor(Zte), adj_te).argmax(1).numpy()
    return pred


def smooth_probs(Pr, adj, alpha, iters=1):
    Pt = torch.tensor(Pr, dtype=torch.float32)
    for _ in range(iters):
        Pt = (1 - alpha) * Pt + alpha * torch.sparse.mm(adj, Pt)
    return Pt.numpy()


done_path = f"{R}/results_long.csv"
rows = pd.read_csv(done_path).to_dict("records") if os.path.exists(done_path) else []
done = {(r["scheme"], r["setting"], r["test"], r["model"], r["seed"]) for r in rows}

for scheme, (mapping, classes, K_LIST, do_nonlinear) in SCHEMES.items():
    C = len(classes); cidx = {c: i for i, c in enumerate(classes)}
    in_task = np.array([v in mapping for v in y_raw])
    y_int = np.array([cidx[mapping[v]] if v in mapping else -1 for v in y_raw])

    def metrics(y_true, y_pred):
        out = {"macro_f1": f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true)),
               "bal_acc": balanced_accuracy_score(y_true, y_pred), "acc": accuracy_score(y_true, y_pred)}
        f1s = f1_score(y_true, y_pred, average=None, labels=range(C), zero_division=0)
        for c, f in zip(classes, f1s):
            out[f"f1_{c}"] = f if (y_true == cidx[c]).any() else np.nan
        return out

    for setting, train_pucks, test_puck in SETTINGS:
        tr = np.where(np.isin(puck, train_pucks) & in_task)[0]; te = np.where((puck == test_puck) & in_task)[0]
        t0 = time.time()
        pca = PCA(N_PC, random_state=0).fit(X_all[tr])
        Z = pca.transform(X_all).astype(np.float32)
        # neighbour means are computed over ALL beads of the puck (context is label-free, so non-task beads may contribute)
        ytr, yte = y_int[tr], y_int[te]
        feats = {"expr": Z}
        for k in K_LIST:
            Nk = nbr_mean(Z, k); feats[f"nbr_k{k}"] = np.concatenate([Z, Nk], 1); feats[f"nbr_only_k{k}"] = Nk

        def record(model, seed, ypred):
            r = {"scheme": scheme, "setting": setting, "train": "+".join(train_pucks), "test": test_puck, "model": model,
                 "seed": seed, "n_train": len(tr), "n_test": len(te), **metrics(yte, ypred)}
            rows.append(r)
            print(f"  [{scheme}] {setting:9s} {test_puck} {model:14s} s{seed} macroF1={r['macro_f1']:.3f} balacc={r['bal_acc']:.3f}", flush=True)

        # linear models
        for name, Fm in feats.items():
            if (scheme, setting, test_puck, name, 0) in done: continue
            clf = fit_lr(Fm[tr], ytr); record(name, 0, clf.predict(Fm[te]))
            if name == "expr":  # test-time smoothing of the expression-only posterior
                Pr = clf.predict_proba(Fm[te])
                adj_te = local_adj(te, 15)
                for a in ALPHAS:
                    record(f"smooth_a{a}", 0, smooth_probs(Pr, adj_te, a).argmax(1))
        if not do_nonlinear:
            pd.DataFrame(rows).to_csv(done_path, index=False); continue
        for seed in SEEDS:
            for name in ["expr", "nbr_k15"]:
                mname = f"{name}_mlp"
                if (scheme, setting, test_puck, mname, seed) in done: continue
                pred = fit_mlp(feats[name][tr], ytr, C, seed); record(mname, seed, pred(feats[name][te]))
            mname = "sage_k15"
            if (scheme, setting, test_puck, mname, seed) in done: continue
            adj_tr = local_adj(tr, 15); adj_te = local_adj(te, 15)
            pred = fit_sage(Z[tr], ytr, adj_tr, C, seed); record(mname, seed, pred(Z[te], adj_te))
        pd.DataFrame(rows).to_csv(done_path, index=False)
        print(f"[{scheme} {setting} -> {test_puck}] {time.time()-t0:.0f}s", flush=True)

print("ALL DONE")
