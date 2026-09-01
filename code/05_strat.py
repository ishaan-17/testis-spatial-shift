"""Refit the deterministic LR models (expr, expr+nbr k=15, test-time smoothing) for germ4 and save per-bead
predictions, then stratify the spatial-context gain by the test bead's UMI depth and by its local homogeneity.
Uses exactly the same preprocessing/PCA/LR as 03_run.py, so numbers reproduce the grid."""
import os, numpy as np, pandas as pd, anndata as ad, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repository root


P = str(ROOT / "proc"); R = str(ROOT / "results")
A = ad.read_h5ad(f"{P}/all.h5ad")
X_all = np.asarray(A.X, dtype=np.float32); puck = A.obs.puck.values.astype(str)
y_raw = A.obs.cell_type.values.astype(str); nn_idx = A.obsm["nn_idx"]; umi = A.obs.n_umi.values
classes = ["SPG", "SPC", "RS", "ES"]; cidx = {c: i for i, c in enumerate(classes)}
in_task = np.isin(y_raw, classes); y_int = np.array([cidx.get(v, -1) for v in y_raw])
lab = y_raw
homog = (lab[nn_idx[:, :15]] == lab[:, None]).mean(1)

SETTINGS = []
for t in ["WT1", "WT2", "WT3"]: SETTINGS.append(("WT->WT", [p for p in ["WT1", "WT2", "WT3"] if p != t], t))
for t in ["DB1", "DB2", "DB3"]:
    SETTINGS.append(("WT->DB", ["WT1", "WT2", "WT3"], t)); SETTINGS.append(("DB->DB", [p for p in ["DB1", "DB2", "DB3"] if p != t], t))
for t in ["HU1", "HU2"]:
    SETTINGS.append(("WT->HU", ["WT1", "WT2", "WT3"], t)); SETTINGS.append(("HU->HU", [p for p in ["HU1", "HU2"] if p != t], t))
    SETTINGS.append(("WTDB->HU", ["WT1", "WT2", "WT3", "DB1", "DB2", "DB3"], t))


def local_adj(idx, k):
    pos = -np.ones(len(nn_idx), dtype=np.int64); pos[idx] = np.arange(len(idx))
    nb = pos[nn_idx[idx, :k]]; rows, cols = np.nonzero(nb >= 0); cols = nb[rows, cols]
    deg = np.bincount(rows, minlength=len(idx)).astype(np.float32); vals = 1.0 / np.maximum(deg[rows], 1)
    return torch.sparse_coo_tensor(torch.tensor(np.stack([rows, cols])), torch.tensor(vals), (len(idx), len(idx))).coalesce()


preds = []
for setting, train_pucks, test_puck in SETTINGS:
    tr = np.where(np.isin(puck, train_pucks) & in_task)[0]; te = np.where((puck == test_puck) & in_task)[0]
    pca = PCA(50, random_state=0).fit(X_all[tr]); Z = pca.transform(X_all).astype(np.float32)
    N = Z[nn_idx[:, :15]].mean(1); F = np.concatenate([Z, N], 1)
    lr_e = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", n_jobs=2).fit(Z[tr], y_int[tr])
    lr_n = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", n_jobs=2).fit(F[tr], y_int[tr])
    Pe = lr_e.predict_proba(Z[te]); adj = local_adj(te, 15)
    Ps = (0.5 * torch.tensor(Pe) + 0.5 * torch.sparse.mm(adj, torch.tensor(Pe, dtype=torch.float32))).numpy()
    d = pd.DataFrame({"setting": setting, "test": test_puck, "bead": A.obs_names[te], "y": y_int[te], "umi": umi[te], "homog": homog[te],
                      "p_expr": Pe.argmax(1), "p_nbr": lr_n.predict(F[te]), "p_smooth": Ps.argmax(1)})
    preds.append(d)
    print(setting, test_puck, "expr", round(f1_score(d.y, d.p_expr, average="macro"), 3), "nbr", round(f1_score(d.y, d.p_nbr, average="macro"), 3),
          "smooth", round(f1_score(d.y, d.p_smooth, average="macro"), 3), flush=True)
preds = pd.concat(preds); preds.to_csv(f"{R}/germ4_lr_predictions.csv.gz", index=False)

# stratified gains: UMI tertiles (within test puck) and local homogeneity bins
rows = []
for (s, t), d in preds.groupby(["setting", "test"]):
    d = d.copy(); d["umi_bin"] = pd.qcut(d.umi, 3, labels=["low", "mid", "high"])
    d["hom_bin"] = pd.cut(d.homog, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"])
    for col in ["umi_bin", "hom_bin"]:
        for b, dd in d.groupby(col, observed=True):
            rows.append({"setting": s, "test": t, "strat": col, "bin": b, "n": len(dd),
                         "acc_expr": (dd.y == dd.p_expr).mean(), "acc_nbr": (dd.y == dd.p_nbr).mean(), "acc_smooth": (dd.y == dd.p_smooth).mean()})
st = pd.DataFrame(rows); st["gain_nbr"] = st.acc_nbr - st.acc_expr; st["gain_smooth"] = st.acc_smooth - st.acc_expr
st.to_csv(f"{R}/stratified_gains.csv", index=False)
print(st.groupby(["strat", "setting", "bin"], observed=True)[["n", "acc_expr", "gain_nbr", "gain_smooth"]].mean().round(3).to_string())
