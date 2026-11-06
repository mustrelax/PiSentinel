"""Measured loop rate: [LATENCY] lines are emitted every 30 processed frames, so
30 / (interval between consecutive lines) is the realised loop rate including
skipped-inference frames. Gaps > 5 s (idle/pauses) are excluded.
Usage: python -I loop_rate.py <pisentinel.log> <out.json>"""
import json, re, sys
import numpy as np
ts = []
pat = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)\].*\[LATENCY\]")
from datetime import datetime
for line in open(sys.argv[1], encoding="utf-8"):
    m = pat.search(line)
    if m:
        ts.append(datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").timestamp() + int(m[2]) / 1000)
d = np.diff(np.array(ts))
d = d[(d > 0.05) & (d < 5)]
fps = 30.0 / d
r = {"n_intervals": int(len(d)), "fps_mean_of_windows": float(fps.mean()),
     "fps_overall": float(30 * len(d) / d.sum()), "fps_median": float(np.median(fps)),
     "fps_p5": float(np.percentile(fps, 5)), "fps_p95": float(np.percentile(fps, 95)),
     "excluded_gaps": int(len(np.diff(np.array(ts))) - len(d))}
print(json.dumps(r, indent=1)); open(sys.argv[2], "w").write(json.dumps(r, indent=2))
