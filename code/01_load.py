"""Load the 8 Slide-seqV2 testis pucks (Chen et al. 2021) into sparse AnnData objects.

Dense CSVs are streamed in row chunks and converted to CSR to fit in ~7 GB RAM.
"""
import os, glob, sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repository root


RAW = str(ROOT / "data" / "raw")   # see code/00_download.sh
OUT = str(ROOT / "proc"); os.makedirs(OUT, exist_ok=True)

CT = {1: "ES", 2: "RS", 3: "Myoid", 4: "SPC", 5: "SPG", 6: "Sertoli", 7: "Leydig", 8: "Endothelial", 9: "Macrophage"}

PUCKS = {
    # name: (dge, loc, ct, species, condition)
    "WT1": ("mouse/Data/WT Slide-seq data/WT1/MappedDGEForR_T3_Trimmed.csv",
            "mouse/Data/WT Slide-seq data/WT1/BeadLocationsForR_T3_Trimmed.csv",
            "mouse/Data/WT Slide-seq data/WT1/PuckT3_bead_maxct_df.csv", "mouse", "WT"),
    "WT2": ("mouse/Data/WT Slide-seq data/WT2/MappedDGEForR_Puck24_Trimmed.csv",
            "mouse/Data/WT Slide-seq data/WT2/BeadLocationsForR_Puck24_Trimmed_cleaned.csv",
            "mouse/Data/WT Slide-seq data/WT2/Puck24_bead_maxct_df.csv", "mouse", "WT"),
    "WT3": ("mouse/Data/WT Slide-seq data/WT3/MappedDGEForR_Normal_Puck7_Trimmed.csv",
            "mouse/Data/WT Slide-seq data/WT3/BeadLocationsForR_Normal_Puck7_Trimmed.csv",
            "mouse/Data/WT Slide-seq data/WT3/Normal_Puck7_bead_maxct_df.csv", "mouse", "WT"),
    "DB1": ("mouse/Data/Diabetes Slide-seq data/Diabetes_1/MappedDGEForR_T4_Trimmed.csv",
            "mouse/Data/Diabetes Slide-seq data/Diabetes_1/BeadLocationsForR_T4_Trimmed.csv",
            "mouse/Data/Diabetes Slide-seq data/Diabetes_1/Puck_T4_bead_maxct_df.csv", "mouse", "diabetic"),
    "DB2": ("mouse/Data/Diabetes Slide-seq data/Diabetes_2/MappedDGEForR_Diabetes_Puck10_Trimmed.csv",
            "mouse/Data/Diabetes Slide-seq data/Diabetes_2/BeadLocationsForR_Diabetes_Puck10_Trimmed.csv",
            "mouse/Data/Diabetes Slide-seq data/Diabetes_2/Diabetes_Puck10_bead_maxct_df.csv", "mouse", "diabetic"),
    "DB3": ("mouse/Data/Diabetes Slide-seq data/Diabetes_3/MappedDGEForR_Diabetes_Puck11_Trimmed.csv",
            "mouse/Data/Diabetes Slide-seq data/Diabetes_3/BeadLocationsForR_Diabetes_Puck11_Trimmed.csv",
            "mouse/Data/Diabetes Slide-seq data/Diabetes_3/Diabetes_Puck11_bead_maxct_df.csv", "mouse", "diabetic"),
    "HU1": ("human/Human/MappedDGEForR_Puck5_Human_Trimmed.csv",
            "human/Human/BeadLocationsForR_Puck5_Human_Trimmed.csv",
            "human/Human/Puck5_Human_bead_maxct_df.csv", "human", "normal"),
    "HU2": ("human/Human/MappedDGEForR_Puck6_Human_Trimmed.csv",
            "human/Human/BeadLocationsForR_Puck6_Human_Trimmed.csv",
            "human/Human/Puck6_Human_bead_maxct_df.csv", "human", "normal"),
}


def read_dge_sparse(path, chunk=2000):
    import pyarrow.csv as pc
    opts = pc.ReadOptions(block_size=64 << 20, use_threads=True)
    reader = pc.open_csv(path, read_options=opts)
    genes = None; blocks, barcodes = [], []
    for b in reader:
        names = b.schema.names
        if genes is None:
            genes = np.array([c for c in names if c != "barcode" and c != "" and not c.startswith("Unnamed")])
            gcols = [names.index(g) for g in genes]
        barcodes.extend(b.column("barcode").to_pylist())
        arr = np.column_stack([b.column(i).to_numpy(zero_copy_only=False).astype(np.float32) for i in gcols])
        blocks.append(sp.csr_matrix(arr))
        sys.stdout.write("."); sys.stdout.flush()
    X = sp.vstack(blocks).tocsr()
    return X, np.array([str(x) for x in barcodes]), genes


for name, (dge, loc, ct, species, cond) in PUCKS.items():
    outp = f"{OUT}/{name}.h5ad"
    if os.path.exists(outp):
        print("skip", name); continue
    print("loading", name)
    X, bcs, genes = read_dge_sparse(os.path.join(RAW, dge))
    locs = pd.read_csv(os.path.join(RAW, loc))
    locs = locs[[c for c in locs.columns if not c.startswith("Unnamed")]]
    locs["barcode"] = locs["barcode"].astype(str)
    cts = pd.read_csv(os.path.join(RAW, ct))
    cts = cts[[c for c in cts.columns if not c.startswith("Unnamed")]]
    cts["barcode"] = cts["barcode"].astype(str)
    obs = pd.DataFrame({"barcode": bcs}).merge(locs, on="barcode", how="left").merge(cts, on="barcode", how="left")
    assert len(obs) == X.shape[0]
    obs.index = obs["barcode"].values
    obs["cell_type"] = obs["max_cell_type"].map(CT)
    obs["puck"] = name; obs["species"] = species; obs["condition"] = cond
    a = ad.AnnData(X=X, obs=obs.drop(columns=["barcode"]), var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    a.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    print(f"\n{name}: {a.shape}, missing loc {obs['x'].isna().sum()}, missing ct {obs['cell_type'].isna().sum()}, "
          f"UMI median {np.median(np.asarray(X.sum(1)).ravel()):.0f}")
    print(obs["cell_type"].value_counts().to_dict())
    a.write_h5ad(outp)
    del X, a
