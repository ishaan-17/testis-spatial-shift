"""Harmonize pucks: restrict to 1:1 orthologs, filter beads, log-normalize, select HVGs on WT mouse only.
Output: proc/all.h5ad (dense float32 on HVGs, per-puck z-scored) plus per-puck spatial kNN indices.
Memory-conscious: everything stays sparse and per-puck until after HVG subsetting.
"""
import gc
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repository root


P = str(ROOT / "proc")
PUCKS = ["WT1", "WT2", "WT3", "DB1", "DB2", "DB3", "HU1", "HU2"]
MIN_UMI = 100  # bead filter
N_HVG = 2000
KMAX = 50

orth = pd.read_csv(ROOT / "data" / "orthologs_1to1.csv")  # built by code/00_orthologs.py
m2h = dict(zip(orth.mouse, orth.human))


def load(name):
    a = ad.read_h5ad(f"{P}/{name}.h5ad")
    a.obs["n_umi"] = np.asarray(a.X.sum(1)).ravel()
    a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
    a = a[a.obs["cell_type"].notna() & (a.obs["n_umi"] >= MIN_UMI)].copy()
    if a.obs["species"].iloc[0] == "mouse":
        a = a[:, a.var_names.isin(m2h.keys())].copy()
        a.var_names = [m2h[g] for g in a.var_names]
    else:
        a = a[:, a.var_names.isin(set(orth.human))].copy()
    a.var_names_make_unique()
    return a


# pass 1: common gene set
common = None
for name in PUCKS:
    a = load(name)
    common = set(a.var_names) if common is None else common & set(a.var_names)
    print(name, a.shape, "median UMI", np.median(a.obs.n_umi), flush=True)
    del a; gc.collect()
common = sorted(common)
print("common ortholog genes:", len(common), flush=True)

# pass 2: HVGs on WT mouse pucks only (seurat_v3 on counts, per-puck batches)
wt = ad.concat([load(n)[:, common].copy() for n in ["WT1", "WT2", "WT3"]], index_unique="-")
sc.pp.highly_variable_genes(wt, flavor="seurat_v3", n_top_genes=N_HVG, batch_key="puck")
hvg = wt.var_names[wt.var.highly_variable].tolist()
del wt; gc.collect()
print("HVGs:", len(hvg), flush=True)

# pass 3: normalize, log, subset HVG, densify, per-puck z-score
parts = []
for name in PUCKS:
    a = load(name)[:, common].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, hvg].copy()
    X = a.X.toarray().astype(np.float32)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    X = np.clip((X - mu) / sd, -10, 10)
    a.X = X
    parts.append(a)
    print(name, "dense", a.shape, flush=True)
    gc.collect()
A = ad.concat(parts, index_unique="-")
del parts; gc.collect()
A.obs["puck"] = A.obs["puck"].astype(str)

# spatial kNN per puck (KMAX neighbours, excluding self), stored as obsm arrays of global indices
nn_idx = np.zeros((A.n_obs, KMAX), dtype=np.int64)
nn_dist = np.zeros((A.n_obs, KMAX), dtype=np.float32)
for pk in PUCKS:
    idx = np.where(A.obs.puck.values == pk)[0]
    xy = A.obsm["spatial"][idx]
    nbr = NearestNeighbors(n_neighbors=KMAX + 1).fit(xy)
    d, i = nbr.kneighbors(xy)
    nn_idx[idx] = idx[i[:, 1:]]
    nn_dist[idx] = d[:, 1:]
    print(pk, "median 1st-NN dist (px)", np.median(d[:, 1]), "median 15th", np.median(d[:, 15]), flush=True)
A.obsm["nn_idx"] = nn_idx
A.obsm["nn_dist"] = nn_dist

# neighbourhood label homogeneity diagnostic (description only, never a feature)
lab = A.obs.cell_type.values.astype(str)
for k in [5, 15, 30]:
    A.obs[f"nbr_homog_k{k}"] = (lab[nn_idx[:, :k]] == lab[:, None]).mean(1)
print(A.obs.groupby("puck")[["nbr_homog_k5", "nbr_homog_k15", "nbr_homog_k30"]].mean())
print(A.obs.groupby(["puck", "cell_type"]).size().unstack(fill_value=0))
A.write_h5ad(f"{P}/all.h5ad")
print("saved", A.shape)
