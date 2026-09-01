"""Aggregate results into tables and figures for the paper."""
import numpy as np, pandas as pd, anndata as ad, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' styles)
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 5.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "figure.dpi": 150, "lines.markersize": 3})
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repository root


R = str(ROOT / "results"); FIG = str(ROOT / "paper" / "figs"); os.makedirs(FIG, exist_ok=True)
df = pd.read_csv(f"{R}/results_long.csv")
A = ad.read_h5ad(ROOT / "proc" / "all.h5ad", backed="r")
obs = A.obs.copy()

ORDER = ["WT->WT", "DB->DB", "HU->HU", "WT->DB", "WT->HU", "WTDB->HU"]
NICE = {"WT->WT": "WT→WT", "DB->DB": "ob/ob→ob/ob", "HU->HU": "human→human", "WT->DB": "WT→ob/ob", "WT->HU": "WT→human", "WTDB->HU": "mouse→human"}

# ---- Table 1: main results (mean ± sd over test pucks and seeds), germ4 & coarse6
def agg(scheme, models, metric="macro_f1"):
    d = df[(df.scheme == scheme) & (df.model.isin(models))]
    # average seeds first, then across test pucks
    m = d.groupby(["setting", "test", "model"])[metric].mean().reset_index()
    g = m.groupby(["setting", "model"])[metric].agg(["mean", "std", "count"]).reset_index()
    return g

MODELS = ["expr", "nbr_k15", "nbr_only_k15", "smooth_a0.5", "expr_mlp", "nbr_k15_mlp", "sage_k15"]
MNAME = {"expr": "LR (expr)", "nbr_k15": "LR (expr+nbr)", "nbr_only_k15": "LR (nbr only)", "smooth_a0.5": "LR (expr) + smooth ($\\alpha$=0.5)",
         "expr_mlp": "MLP (expr)", "nbr_k15_mlp": "MLP (expr+nbr)", "sage_k15": "GraphSAGE"}

for scheme in ["germ4", "coarse6", "full9"]:
    g = agg(scheme, MODELS)
    if g.empty: continue
    tab = g.pivot(index="model", columns="setting", values="mean").reindex(MODELS)
    sd = g.pivot(index="model", columns="setting", values="std").reindex(MODELS)
    cols = [c for c in ORDER if c in tab.columns]
    tab = tab[cols]; sd = sd[cols]
    tab.to_csv(f"{R}/table_{scheme}_macro_f1.csv")
    with open(f"{R}/table_{scheme}_macro_f1.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\nModel & " + " & ".join(NICE[c] for c in cols) + " \\\\\n\\midrule\n")
        for m in tab.index:
            if tab.loc[m].isna().all(): continue
            cells = []
            for c in cols:
                v, s = tab.loc[m, c], sd.loc[m, c]
                cells.append("--" if np.isnan(v) else (f"{v:.3f}" if np.isnan(s) else f"{v:.3f}\\,{{\\scriptsize$\\pm${s:.3f}}}"))
            f.write(MNAME[m] + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\n=== {scheme} macro-F1 ===\n", tab.round(3).to_string())
    if scheme == "germ4":  # main-text version without MLP rows
        lines = open(f"{R}/table_{scheme}_macro_f1.tex").read().splitlines()
        open(f"{R}/table_germ4_main.tex", "w").write("\n".join(l for l in lines if not l.startswith("MLP")) + "\n")

# ---- Delta from spatial context, per test puck, vs neighbourhood homogeneity of the test puck
germ = obs[obs.cell_type.isin(["SPG", "SPC", "RS", "ES"])]
hom = germ.groupby("puck")["nbr_homog_k15"].mean()
d = df[(df.scheme == "germ4")]
piv = d.groupby(["setting", "test", "model"])["macro_f1"].mean().unstack("model")
delta = pd.DataFrame({"gain_nbr": piv["nbr_k15"] - piv["expr"], "gain_smooth": piv["smooth_a0.5"] - piv["expr"],
                      "gain_sage": piv["sage_k15"] - piv["expr"], "nbr_only": piv["nbr_only_k15"], "expr": piv["expr"]}).reset_index()
delta["homog"] = delta["test"].map(hom)
delta.to_csv(f"{R}/delta_vs_homogeneity.csv", index=False)
print("\n=== spatial-context gain (germ4 macro-F1) ===\n", delta.round(3).to_string())

# ---- Figure 1: gain vs setting (bar) + gain vs homogeneity (scatter)
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.8))
ax = axes[0]
gm = agg("germ4", ["expr", "nbr_k15", "sage_k15", "smooth_a0.5"])
w = 0.2; xs = np.arange(len(ORDER))
for i, (m, lab) in enumerate([("expr", "expr only"), ("nbr_k15", "expr+nbr (LR)"), ("sage_k15", "GraphSAGE"), ("smooth_a0.5", "expr + test-time smooth")]):
    vals = [gm[(gm.setting == s) & (gm.model == m)]["mean"].values[0] if ((gm.setting == s) & (gm.model == m)).any() else np.nan for s in ORDER]
    errs = [gm[(gm.setting == s) & (gm.model == m)]["std"].values[0] if ((gm.setting == s) & (gm.model == m)).any() else 0 for s in ORDER]
    ax.bar(xs + (i - 1.5) * w, vals, w, yerr=np.nan_to_num(errs), capsize=2, label=lab)
ax.set_xticks(xs); ax.set_xticklabels([NICE[s] for s in ORDER], rotation=35, ha="right", fontsize=5.5)
ax.set_ylim(0.3, 0.93); ax.set_ylabel("macro-F1 (germ-cell stage)"); ax.legend(fontsize=5, frameon=False, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1, bbox_to_anchor=(0.5, 1.02)); ax.set_title("A. Accuracy by shift setting", loc="left")
ax = axes[1]
delta["ratio"] = delta["nbr_only"] / delta["expr"]
for s in ORDER:
    dd = delta[delta.setting == s]
    ax.scatter(dd.ratio, dd.gain_nbr, label=NICE[s], s=12)
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("relative informativeness of context\n(nbr-only / expr-only macro-F1)"); ax.set_ylabel("$\\Delta$ macro-F1, expr+nbr $-$ expr")
ax.legend(fontsize=5, frameon=False, loc="upper left", handletextpad=0.2); ax.set_title("B. Gain from neighbourhood features", loc="left")
ax = axes[2]
sm = df[(df.scheme == "germ4") & (df.model.str.startswith("smooth") | (df.model == "expr"))].copy()
sm["alpha"] = sm.model.map(lambda m: 0.0 if m == "expr" else float(m.split("_a")[1]))
sm = sm.groupby(["setting", "alpha"])["macro_f1"].mean().unstack("alpha")
for s in ORDER:
    if s in sm.index: ax.plot(sm.columns, sm.loc[s].values, marker="o", ms=2, lw=0.9, label=NICE[s])
ax.set_xlabel("smoothing weight $\\alpha$"); ax.set_ylabel("macro-F1"); ax.set_title("C. Test-time smoothing", loc="left"); ax.legend(fontsize=5, frameon=False, loc="center left", handlelength=1.2)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_main.pdf"); plt.close()

# ---- Figure 2: per-class gain (germ4), by setting
cls = ["SPG", "SPC", "RS", "ES"]
pc = d.groupby(["setting", "test", "model"])[[f"f1_{c}" for c in cls]].mean().unstack("model")
fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.7), sharey=True)
w = 0.13
for ax, mdl, ttl in zip(axes, ["nbr_k15", "sage_k15"], ["A. expr+nbr (LR) $-$ expr", "B. GraphSAGE $-$ expr"]):
    for i, s in enumerate(ORDER):
        if s not in pc.index.get_level_values(0): continue
        sub = pc.loc[s]
        gains = [(sub[(f"f1_{c}", mdl)] - sub[(f"f1_{c}", "expr")]).mean() for c in cls]
        ax.bar(np.arange(4) + (i - 2.5) * w, gains, w, label=NICE[s])
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(cls); ax.set_title(ttl, loc="left")
axes[0].set_ylabel("$\\Delta$ F1 per stage"); axes[1].legend(fontsize=5, frameon=False, ncol=2, loc="lower left", columnspacing=0.6, handlelength=1)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_perclass.pdf"); plt.close()

# ---- Figure 3: puck maps (one WT, one ob/ob, one human), germ-cell stage labels
fig, axes = plt.subplots(1, 3, figsize=(5.3, 2.05))
colors = {"SPG": "#d62728", "SPC": "#ff7f0e", "RS": "#2ca02c", "ES": "#1f77b4"}
for ax, pk in zip(axes, ["WT1", "DB1", "HU1"]):
    o = obs[(obs.puck == pk) & obs.cell_type.isin(cls)]
    xy = A[o.index].obsm["spatial"] if False else np.c_[o.x.values, o.y.values]
    for c in cls:
        m = o.cell_type.values == c
        ax.scatter(xy[m, 0], xy[m, 1], s=0.15, c=colors[c], label=c, rasterized=True, linewidths=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{pk}: {o.shape[0]:,} germ beads, homogeneity {hom[pk]:.2f}", fontsize=6.5)
fig.legend(*axes[0].get_legend_handles_labels(), markerscale=25, fontsize=6, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(f"{FIG}/fig_pucks.png", dpi=300); plt.close()

# ---- homogeneity table
ht = obs.groupby(["puck"])[["nbr_homog_k5", "nbr_homog_k15", "nbr_homog_k30"]].mean()
ht["germ_homog_k15"] = hom
ht["n_beads"] = obs.groupby("puck").size()
ht["median_umi"] = obs.groupby("puck")["n_umi"].median()
ht.to_csv(f"{R}/homogeneity.csv"); print("\n", ht.round(3).to_string())

# ---- sweeps table (k and alpha), germ4
d = df[df.scheme == "germ4"]
ksw = d[d.model.isin(["nbr_k5", "nbr_k15", "nbr_k30"])].groupby(["setting", "test", "model"])["macro_f1"].mean().groupby(["setting", "model"]).mean().unstack("model")
asw = d[d.model.str.startswith("smooth") | (d.model == "expr")].groupby(["setting", "test", "model"])["macro_f1"].mean().groupby(["setting", "model"]).mean().unstack("model")
with open(f"{R}/table_sweeps.tex", "w") as f:
    f.write("\\begin{tabular}{lccc|ccccc}\n\\toprule\n & \\multicolumn{3}{c|}{expr+nbr, $k$} & \\multicolumn{5}{c}{smoothing weight $\\alpha$} \\\\\n")
    f.write("Setting & 5 & 15 & 30 & 0 & 0.25 & 0.5 & 0.75 & 1 \\\\\n\\midrule\n")
    for s in ORDER:
        f.write(NICE[s] + " & " + " & ".join(f"{ksw.loc[s, m]:.3f}" for m in ["nbr_k5", "nbr_k15", "nbr_k30"]) + " & " +
                " & ".join(f"{asw.loc[s, m]:.3f}" for m in ["expr", "smooth_a0.25", "smooth_a0.5", "smooth_a0.75", "smooth_a1.0"]) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---- stratified table, if available
sp = f"{R}/stratified_gains.csv"
if os.path.exists(sp):
    st = pd.read_csv(sp)
    g = st.groupby(["strat", "setting", "bin"])[["acc_expr", "gain_nbr", "gain_smooth"]].mean()
    with open(f"{R}/table_strat.tex", "w") as f:
        ub = ["low", "mid", "high"]; hb = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"]
        f.write("\\begin{tabular}{l" + "c" * 3 + "|" + "c" * 5 + "}\n\\toprule\n & \\multicolumn{3}{c|}{UMI tertile} & \\multicolumn{5}{c}{local homogeneity} \\\\\n")
        f.write("Setting & " + " & ".join(ub) + " & " + " & ".join(hb) + " \\\\\n\\midrule\n")
        for s in ORDER:
            cells = []
            for b in ub: cells.append(f"{g.loc[('umi_bin', s, b), 'gain_nbr']:+.3f}" if ("umi_bin", s, b) in g.index else "--")
            for b in hb: cells.append(f"{g.loc[('hom_bin', s, b), 'gain_nbr']:+.3f}" if ("hom_bin", s, b) in g.index else "--")
            f.write(NICE[s] + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
else:
    open(f"{R}/table_strat.tex", "w").write("\\begin{tabular}{l}pending\\end{tabular}\n")
for sch in ["coarse6", "full9"]:
    if not os.path.exists(f"{R}/table_{sch}_macro_f1.tex"):
        open(f"{R}/table_{sch}_macro_f1.tex", "w").write("\\begin{tabular}{l}pending\\end{tabular}\n")
print("tables written")
