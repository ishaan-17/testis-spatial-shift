# Does spatial context transfer?

Code, results, and manuscript for a benchmark of neighbourhood-aware cell typing in seminiferous tubules under tissue-organization shift (wild-type → *ob/ob* mouse) and species shift (mouse → human), built on the public Slide-seqV2 testis pucks of [Chen et al. 2021](https://doi.org/10.1016/j.celrep.2021.109915).

Submitted to the NeurIPS 2026 workshop *Machine Learning for Spatially Resolved High-dimensional Biology* (ml4spatialbio). The manuscript is under double-blind review; please keep this repository private until the decision.

## Question

Methods that annotate spatial transcriptomics spots increasingly use the spot's tissue neighbourhood as input. They are benchmarked within one tissue and condition, so it is unknown whether the spatial signal they learn holds when the tissue's organization changes. We ask how much spatial context improves bead-level annotation of germ-cell stage (SPG, SPC, RS, ES), and whether the improvement survives three levels of shift on one tissue, one platform, and one labelling procedure:

| Setting | Train | Test |
|---|---|---|
| WT→WT | 2 WT mouse pucks | held-out WT puck (3 folds) |
| WT→ob/ob | 3 WT pucks | each *ob/ob* puck |
| WT→human, mouse→human | 3 WT (+3 *ob/ob*) pucks | each human puck |
| ob/ob→ob/ob, human→human | same-condition pucks | in-distribution references |

Models: expression-only logistic regression and MLP; neighbourhood-augmented features (own PCs + mean of the 15 nearest spatial neighbours, BANKSY-style); a two-layer GraphSAGE; and test-time smoothing of the expression-only posterior over the target puck's spatial graph.

## Main result (macro-F1, germ-cell stage)

| Model | WT→WT | ob/ob→ob/ob | human→human | WT→ob/ob | WT→human | mouse→human |
|---|---|---|---|---|---|---|
| LR (expr) | 0.699 | 0.717 | 0.486 | 0.713 | 0.385 | 0.385 |
| LR (expr+nbr) | 0.711 | 0.723 | 0.513 | 0.721 | 0.398 | 0.396 |
| LR (expr) + smooth (α=0.5) | 0.712 | 0.733 | 0.501 | 0.732 | 0.401 | 0.400 |
| GraphSAGE | 0.711 | 0.716 | 0.505 | 0.721 | 0.347 | 0.350 |

Spatial context is worth about one point within a species and up to three on human pucks. The *ob/ob* condition is not a distribution shift for this task. Across species, fixed neighbourhood aggregation and test-time smoothing keep their gains; GraphSAGE matches them in-distribution and becomes the worst model, losing five points overall and fifteen F1 points on round spermatids. Full tables, per-class and stratified results are in `results/` and the appendix of the paper.

## Layout

```
code/      pipeline scripts, run in order (see below)
data/      orthologs_1to1.csv (MGI one-to-one mouse-human orthologs used in the paper); raw pucks go in data/raw/ (git-ignored)
proc/      derived AnnData files (git-ignored; ~2.5 GB after 02_prep.py)
results/   results_long.csv (every run), LaTeX tables, per-puck statistics, stratified gains
paper/     NeurIPS 2026 LaTeX sources, figures, draft_v1.pdf
```

## Reproducing

CPU only. Peak memory about 7 GB in `02_prep.py`; the full grid in `03_run.py` takes about five hours on two cores.

```bash
pip install -r requirements.txt
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

bash   code/00_download.sh      # ~310 MB download, ~14 GB extracted
python code/00_orthologs.py     # optional: rebuilds data/orthologs_1to1.csv from MGI (the committed file is what the paper used)
python code/01_load.py          # dense CSVs -> sparse per-puck AnnData (proc/{WT1..HU2}.h5ad)
python code/02_prep.py          # ortholog gene space, HVGs on WT only, per-puck z-score, spatial kNN -> proc/all.h5ad
python code/03_run.py           # the benchmark grid -> results/results_long.csv (resumable)
python code/05_strat.py         # gain by UMI depth and local homogeneity -> results/stratified_gains.csv
python code/04_analyze.py       # tables (results/*.tex) and figures (paper/figs/)
bash   paper/build.sh           # compile paper/main.pdf
```

`03_run.py` appends to `results/results_long.csv` and skips finished (scheme, setting, test puck, model, seed) combinations, so it can be stopped and restarted. Models are deterministic given the seed; the numbers in the paper were produced with seeds 0, 1, 2 for the MLP and GraphSAGE.

## Data and labels

The pucks are the processed matrices released with Chen et al. 2021 (Dropbox links in `code/00_download.sh`, from the authors' [GitHub README](https://github.com/thechenlab/Testis_Slide-seq)): three wild-type mouse, three *ob/ob* diabetic mouse, and two normal human pucks, each with counts, bead coordinates, and a per-bead NMFreg cell-type label (1 ES, 2 RS, 3 myoid, 4 SPC, 5 SPG, 6 Sertoli, 7 Leydig, 8 endothelial, 9 macrophage). Labels are NMFreg assignments, not ground truth; results are agreement with the published annotation.

## Figures

Figures use the [SciencePlots](https://github.com/garrettj403/SciencePlots) `science` style with `no-latex`.

## License

MIT (see `LICENSE`). The Slide-seq data belong to their authors and are redistributed under their terms.
