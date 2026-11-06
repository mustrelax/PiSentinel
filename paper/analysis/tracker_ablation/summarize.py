"""Aggregate tracker_ablation_raw.json into tables and a figure.
Uncertainty: sequence-level 95 % t-intervals (each sequence is one replicate).
Usage: python -I summarize.py <out_dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "legend.fontsize": 7,
                     "axes.titlesize": 9, "savefig.dpi": 300})

OUT = Path(sys.argv[1])
raw = json.loads((OUT / "tracker_ablation_raw.json").read_text(encoding="utf-8"))
df = pd.DataFrame(raw["results"])
ORDER = ["PiSentinel (deployed thresholds)", "PiSentinel full (unit-corrected)", "fixed Q and R",
         "adaptive Q only", "confidence-scaled R only", "without direction term",
         "SORT", "ByteTrack", "OC-SORT", "BoT-SORT (CMC, no ReID)"]
SCEN = ["crossing", "swarm", "occlusion", "egomotion", "maneuver"]


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
    return float(x.mean()), float(h)


summary = {"verification": raw["verification"], "overall_N1": {}, "per_scenario_N1": {}, "N2": {}}
n1 = df[df.noise == "N1"]
for tr in ORDER:
    g = n1[n1.tracker == tr]
    rec = g.occ_recovered.sum() / max(1, g.occ_events.sum())
    summary["overall_N1"][tr] = {
        **{m: ci(g[m]) for m in ("hota", "idf1", "mota", "idsw_per_1k", "frag_per_1k", "ms_per_frame")},
        "occ_recovery": float(rec), "occ_events": int(g.occ_events.sum()), "n_seq": int(len(g))}
    summary["per_scenario_N1"][tr] = {s: {m: ci(g[g.scenario == s][m]) for m in ("idf1", "hota", "idsw_per_1k")}
                                      for s in SCEN}
n2 = df[df.noise == "N2"]
for tr in ["PiSentinel full (unit-corrected)", "fixed Q and R", "confidence-scaled R only"]:
    g2, g1 = n2[n2.tracker == tr], n1[n1.tracker == tr]
    summary["N2"][tr] = {"N2": {m: ci(g2[m]) for m in ("hota", "idf1", "idsw_per_1k")},
                         "N1": {m: ci(g1[m]) for m in ("hota", "idf1", "idsw_per_1k")}}
# paired differences (same frozen detections) for the ablation questions
def paired(a, b, metric="idf1", noise="N1"):
    d = df[df.noise == noise]
    x = d[d.tracker == a].sort_values(["scenario", "seed"])[metric].values
    y = d[d.tracker == b].sort_values(["scenario", "seed"])[metric].values
    m, h = ci(x - y)
    p = stats.wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    return {"mean_diff": m, "ci95": h, "wilcoxon_p": float(p)}


summary["paired_idf1"] = {
    "full_vs_fixedQR": paired("PiSentinel full (unit-corrected)", "fixed Q and R"),
    "full_vs_noDirection": paired("PiSentinel full (unit-corrected)", "without direction term"),
    "adaptQ_vs_fixed": paired("adaptive Q only", "fixed Q and R"),
    "adaptR_vs_fixed": paired("confidence-scaled R only", "fixed Q and R"),
    "adaptR_vs_fixed_N2": paired("confidence-scaled R only", "fixed Q and R", noise="N2"),
    "corrected_vs_deployed": paired("PiSentinel full (unit-corrected)", "PiSentinel (deployed thresholds)"),
}
summary["paired_idsw"] = {
    "full_vs_fixedQR": paired("PiSentinel full (unit-corrected)", "fixed Q and R", "idsw_per_1k"),
    "full_vs_noDirection": paired("PiSentinel full (unit-corrected)", "without direction term", "idsw_per_1k"),
}
(OUT / "tracker_ablation_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

rows = []
for tr in ORDER:
    o = summary["overall_N1"][tr]
    rows.append([tr] + [f"{o[m][0]:.3f} ± {o[m][1]:.3f}" for m in ("hota", "idf1")] +
                [f"{o[m][0]:.1f} ± {o[m][1]:.1f}" for m in ("idsw_per_1k", "frag_per_1k")] +
                [f"{100 * o['occ_recovery']:.0f}% ({o['occ_events']})", f"{o['ms_per_frame'][0]:.2f}"])
tab = pd.DataFrame(rows, columns=["tracker", "HOTA", "IDF1", "IDSW/1k GT", "Frag/1k GT", "occl. recovery", "ms/frame"])
tab.to_csv(OUT / "tracker_ablation_table.csv", index=False)
print(tab.to_string(index=False))
print(json.dumps(summary["paired_idf1"], indent=1))
print(json.dumps(summary["paired_idsw"], indent=1))
print(json.dumps(summary["N2"], indent=1))
per = pd.DataFrame({tr: {s: summary["per_scenario_N1"][tr][s]["idf1"][0] for s in SCEN} for tr in ORDER}).T
print(per.round(3).to_string())

# Figure: IDF1 per scenario (mean +- 95 % CI over seeds)
show = ["PiSentinel (deployed thresholds)", "PiSentinel full (unit-corrected)", "fixed Q and R",
        "SORT", "ByteTrack", "OC-SORT", "BoT-SORT (CMC, no ReID)"]
labels = ["PiSentinel (as deployed)", "PiSentinel (unit-corrected)", "PiSentinel, fixed Q/R",
          "SORT", "ByteTrack", "OC-SORT", "BoT-SORT (CMC)"]
cols = ["#7f7f7f", "#c0392b", "#e59866", "#9b9b9b", "#2a6fb0", "#3a8f4f", "#8e44ad"]
mk = ["s", "o", "o", "v", "^", "D", "P"]
fig, ax = plt.subplots(figsize=(7.2, 2.9))
x = np.arange(len(SCEN))
off = np.linspace(-0.33, 0.33, len(show))
for i, (tr, lab) in enumerate(zip(show, labels)):
    m = [summary["per_scenario_N1"][tr][s]["idf1"][0] for s in SCEN]
    h = [summary["per_scenario_N1"][tr][s]["idf1"][1] for s in SCEN]
    ax.errorbar(x + off[i], m, yerr=h, fmt=mk[i], color=cols[i], ms=4, capsize=2, lw=0.8, label=lab)
ax.set_xticks(x)
ax.set_xticklabels(["Crossing", "Swarm (5)", "Occlusion", "Camera ego-motion", "Abrupt maneuver"])
ax.set_ylabel("IDF1 (mean ± 95% CI, 10 seq.)")
ax.set_ylim(0, 1.02)
ax.grid(axis="y", alpha=0.3)
for xi in x[:-1]:
    ax.axvline(xi + 0.5, color="#dddddd", lw=0.6)
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
fig.savefig(OUT / "fig_tracker_ablation.png", bbox_inches="tight")
