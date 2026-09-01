"""Build the one-to-one mouse-human ortholog table from MGI's HOM_MouseHumanSequence.rpt.

Writes data/orthologs_1to1.csv with columns (mouse, human). A HomoloGene/DB-class group is kept only
when it contains exactly one mouse symbol and exactly one human symbol. The version used for the paper
(17,609 pairs, downloaded 2026-09-01) is committed in data/; re-running this script may give slightly
different numbers as MGI updates the file.
"""
import urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
rpt = ROOT / "data" / "HOM_MouseHumanSequence.rpt"
out = ROOT / "data" / "orthologs_1to1.csv"

if not rpt.exists():
    print("downloading", URL)
    urllib.request.urlretrieve(URL, rpt)

d = pd.read_csv(rpt, sep="\t")
key = [c for c in d.columns if "DB Class Key" in c or "HomoloGene" in c][0]
m = d[d["Common Organism Name"].str.contains("mouse")][[key, "Symbol"]].rename(columns={"Symbol": "mouse"})
h = d[d["Common Organism Name"].str.contains("human")][[key, "Symbol"]].rename(columns={"Symbol": "human"})
mc = m.groupby(key).size(); hc = h.groupby(key).size()
ok = mc[mc == 1].index.intersection(hc[hc == 1].index)
o = m[m[key].isin(ok)].merge(h[h[key].isin(ok)], on=key)[["mouse", "human"]]
o = o.drop_duplicates("mouse").drop_duplicates("human")
o.to_csv(out, index=False)
print("1:1 orthologs:", len(o), "->", out)
