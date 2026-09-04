"""Quantify how much the *representation* shifts between train and test, separately for the bead's own
embedding and for the neighbourhood-mean embedding. This tests the paper's mechanistic claim that models
which learn a function of neighbourhood structure are more exposed to species shift than the bead itself is.

Two measures, both in the exact feature spaces the models see (PCA fit on training pucks only):
  1. Domain-classifier AUC: how separable are train-domain and test-domain beads? Higher = larger shift.
  2. Per-class centroid displacement: how far does each germ stage's centroid move, in units of that
     class's own within-class spread in the training data?
Output: results/shift_quant.csv, results/centroid_shift.csv
"""
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
P, R = ROOT / "proc", ROOT / "results"
N_PC, K, CAP, SEED = 50, 15, 15000, 0
CLASSES = ["SPG", "SPC", "RS", "ES"]

A = ad.read_h5ad(P / "all.h5ad")
X = np.asarray(A.X, dtype=np.float32)
puck = A.obs.puck.values.astype(str)
y = A.obs.cell_type.values.astype(str)
nn_idx = A.obsm["nn_idx"]
germ = np.isin(y, CLASSES)

SETTINGS = [
    ("WT->WT",   ["WT1", "WT2"], "WT3"),
    ("WT->ob/ob", ["WT1", "WT2", "WT3"], "DB1"),
    ("WT->ob/ob", ["WT1", "WT2", "WT3"], "DB2"),
    ("WT->ob/ob", ["WT1", "WT2", "WT3"], "DB3"),
    ("WT->human", ["WT1", "WT2", "WT3"], "HU1"),
    ("WT->human", ["WT1", "WT2", "WT3"], "HU2"),
]

rng = np.random.default_rng(SEED)


def domain_auc(F, tr, te):
    """AUC of a logistic regression discriminating train-domain from test-domain beads in feature space F."""
    a = rng.choice(tr, min(CAP, len(tr)), replace=False)
    b = rng.choice(te, min(CAP, len(te)), replace=False)
    Xd = np.vstack([F[a], F[b]])
    yd = np.r_[np.zeros(len(a)), np.ones(len(b))]
    aucs = []
    for itr, ite in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xd, yd):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xd[itr], yd[itr])
        aucs.append(roc_auc_score(yd[ite], m.predict_proba(Xd[ite])[:, 1]))
    return float(np.mean(aucs))


rows, crows = [], []
for setting, train_pucks, test_puck in SETTINGS:
    tr = np.where(np.isin(puck, train_pucks) & germ)[0]
    te = np.where((puck == test_puck) & germ)[0]
    pca = PCA(N_PC, random_state=SEED).fit(X[tr])
    Z = pca.transform(X).astype(np.float32)
    N = Z[nn_idx[:, :K]].mean(1)                      # neighbourhood mean, same construction as the models
    print(f"[{setting} -> {test_puck}] PCA done", flush=True)

    spaces = {"own": Z, "neighbourhood": N}
    r = {"setting": setting, "test": test_puck}
    for name, F in spaces.items():
        r[f"auc_{name}"] = domain_auc(F, tr, te)
    r["auc_ratio"] = r["auc_neighbourhood"] / r["auc_own"]
    rows.append(r)
    print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

    # per-class centroid displacement, normalised by that class's within-class spread in training data
    for c in CLASSES:
        itr = tr[y[tr] == c]; ite = te[y[te] == c]
        if len(itr) < 30 or len(ite) < 30:
            continue
        cr = {"setting": setting, "test": test_puck, "cls": c, "n_train": len(itr), "n_test": len(ite)}
        for name, F in spaces.items():
            mu_tr, mu_te = F[itr].mean(0), F[ite].mean(0)
            radius = np.sqrt(np.mean(np.sum((F[itr] - mu_tr) ** 2, axis=1)))
            cr[f"disp_{name}"] = float(np.linalg.norm(mu_te - mu_tr) / radius)
        cr["disp_ratio"] = cr["disp_neighbourhood"] / cr["disp_own"]
        crows.append(cr)
    del Z, N, pca

pd.DataFrame(rows).to_csv(R / "shift_quant.csv", index=False)
pd.DataFrame(crows).to_csv(R / "centroid_shift.csv", index=False)
print("\n=== domain-classifier AUC ===")
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\n=== per-class centroid displacement (in units of within-class spread) ===")
print(pd.DataFrame(crows).round(3).to_string(index=False))
