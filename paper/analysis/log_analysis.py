"""Revision R1 -- re-analysis of the recorded 23 May 2026 session logs.

Produces, from a single consistent data set per figure/table:
  * fig_control_error.png    (replaces Figure 4)  + control_stats.json   (Table 2)
  * fig_fusion_sigma.png     (replaces Figure 6)  + fusion_stats.json    (Table 3)
  * fig_latency.png          (replaces Figure 7)  + latency_stats.json   (Table 4)

Usage:  python -I log_analysis.py <log_dir> <out_dir>
"""
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
})

C_BLUE, C_RED, C_GREEN, C_ORANGE, C_GREY = "#2a6fb0", "#c0392b", "#3a8f4f", "#d9822b", "#7f7f7f"


def pct(s, q):
    return float(np.percentile(s, q))


def summary(s):
    s = np.asarray(s, dtype=float)
    return {
        "n": int(len(s)), "mean": float(s.mean()), "median": float(np.median(s)),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "p90": pct(s, 90), "p95": pct(s, 95), "p99": pct(s, 99), "max": float(s.max()),
    }


# --------------------------------------------------------------------------- servo
SERVO = re.compile(
    r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\].*?\[SERVO\] "
    r"id=(?P<id>\d+) err_px=\((?P<dx>[+-]?[\d.]+),(?P<dy>[+-]?[\d.]+)\) r=(?P<r>[\d.]+) "
    r"pid=\((?P<pid_pan>[+-]?[\d.]+),(?P<pid_tilt>[+-]?[\d.]+)\) "
    r"ff=\((?P<ff_pan>[+-]?[\d.]+),(?P<ff_tilt>[+-]?[\d.]+)\) "
    r"cmd=\((?P<cmd_pan>[+-]?[\d.]+),(?P<cmd_tilt>[+-]?[\d.]+)\) "
    r"ang=\((?P<ang_pan>[\d.]+),(?P<ang_tilt>[\d.]+)\) sat=(?P<sat>\d)")
STATUS = re.compile(r"\[(?P<ts>[\d\- :,]+)\].*?\[STATUS\] Track #(?P<id>\d+): (?P<a>\S+) -> (?P<b>\S+)")


def analyse_servo(log: Path, out: Path) -> dict:
    rows, status = [], []
    with open(log, encoding="utf-8") as f:
        for line in f:
            m = SERVO.search(line)
            if m:
                rows.append(m.groupdict())
                continue
            m = STATUS.search(line)
            if m:
                status.append(m.groupdict())
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != "ts":
            df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S,%f")
    df = df.sort_values("ts").reset_index(drop=True)

    span = (df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds()
    dts = df["ts"].diff().dt.total_seconds().dropna()
    pid_mag = np.hypot(df.pid_pan, df.pid_tilt)
    ff_mag = np.hypot(df.ff_pan, df.ff_tilt)
    cmd_mag = np.hypot(df.cmd_pan, df.cmd_tilt)

    # Per-track statistics: frame-level samples within one session are correlated,
    # so we also report the spread of per-track medians (track = sequence unit).
    per_track = df.groupby("id")["r"].agg(["count", "median", "mean"])
    per_track_long = per_track[per_track["count"] >= 50]

    st = pd.DataFrame(status)
    lost_events = int(((st.a == "LOCKED") & (st.b.str.startswith("LOST"))).sum()) if len(st) else 0
    lock_events = int((st.b == "LOCKED").sum()) if len(st) else 0

    res = {
        "n_samples": int(len(df)),
        "session_span_s": span,
        "logging_rate_hz": len(df) / span,
        "median_log_interval_s": float(dts.median()),
        "n_tracks": int(df["id"].nunique()),
        "n_tracks_ge50": int(len(per_track_long)),
        "error_px": summary(df["r"]),
        "error_deg_median": float(df["r"].median() / 16.755),
        "error_deg_p95": pct(df["r"], 95) / 16.755,
        "frac_within_40px": float((df["r"] <= 40).mean()),
        "frac_within_8px": float((df["r"] <= 8).mean()),
        "pid_mag_mean_deg": float(pid_mag.mean()),
        "ff_mag_mean_deg": float(ff_mag.mean()),
        "cmd_mag_mean_deg": float(cmd_mag.mean()),
        "cmd_mag_p95_deg": pct(cmd_mag, 95),
        "ff_active_frac": float((ff_mag > 0).mean()),
        "sat_frac": float(df["sat"].mean()),
        "per_track_median_px": {
            "median_of_medians": float(per_track_long["median"].median()),
            "iqr_low": float(per_track_long["median"].quantile(0.25)),
            "iqr_high": float(per_track_long["median"].quantile(0.75)),
        },
        "lock_events": lock_events,
        "locked_to_lost_events": lost_events,
    }

    # Figure: representative longest track; time axis from real timestamps.
    best = int(per_track["count"].idxmax())
    t = df[df["id"] == best].copy()
    t["t"] = (t["ts"] - t["ts"].iloc[0]).dt.total_seconds()
    res["fig_track_id"] = best
    res["fig_track_samples"] = int(len(t))
    res["fig_track_duration_s"] = float(t["t"].iloc[-1])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [2.1, 1]})
    a1.plot(t["t"], t["r"], lw=0.5, color=C_BLUE, alpha=0.55, label="logged error $r$")
    a1.plot(t["t"], t["r"].rolling(20, center=True, min_periods=1).median(), lw=1.2,
            color=C_RED, label="rolling median (20 samples)")
    a1.axhline(40, color=C_GREY, ls="--", lw=0.8, label="dead zone (40 px)")
    a1.set_xlabel("Time since lock of track %d [s]" % best)
    a1.set_ylabel("Error magnitude $r$ [px]")
    a1.set_title("(a) Representative track (n = %d logged samples)" % len(t))
    a1.legend(loc="upper right", frameon=False)
    a1.grid(alpha=0.3)
    a1.set_xlim(0, t["t"].iloc[-1])

    r_sorted = np.sort(df["r"].values)
    a2.plot(r_sorted, np.arange(1, len(r_sorted) + 1) / len(r_sorted), color=C_BLUE, lw=1.2)
    for q, ls in ((50, ":"), (95, "--")):
        v = pct(df["r"], q)
        a2.axvline(v, color=C_GREY, ls=ls, lw=0.8)
        a2.text(v, 0.08 if q == 50 else 0.25, f" p{q}={v:.1f}", fontsize=7.5, color="k")
    a2.set_xscale("log")
    a2.set_xlabel("Error magnitude $r$ [px] (log)")
    a2.set_ylabel("Empirical CDF")
    a2.set_title("(b) All tracks (n = %d)" % len(df))
    a2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_control_error.png")
    plt.close(fig)
    return res


# --------------------------------------------------------------------------- latency
LAT = re.compile(
    r"\[LATENCY\] Capture: (?P<capture>[\d.]+)ms \| Inference: (?P<inference>[\d.]+ms|\(skip\)) \| "
    r"Tracker: (?P<tracker>[\d.]+)ms \| Stream: (?P<stream>[\d.]+)ms \| Total: (?P<total>[\d.]+)ms "
    r"\((?P<fps>[\d.]+) FPS\)")


def analyse_latency(log: Path, out: Path) -> dict:
    rows = []
    with open(log, encoding="utf-8") as f:
        for line in f:
            m = LAT.search(line)
            if m:
                d = m.groupdict()
                d["skip"] = d["inference"] == "(skip)"
                d["inference"] = np.nan if d["skip"] else float(d["inference"][:-2])
                for k in ("capture", "tracker", "stream", "total", "fps"):
                    d[k] = float(d[k])
                rows.append(d)
    df = pd.DataFrame(rows)
    full = df[~df.skip]
    skip = df[df.skip]
    comps = ["capture", "inference", "tracker", "stream"]
    res = {
        "n_lines": int(len(df)), "n_full": int(len(full)), "n_skip": int(len(skip)),
        "skip_fraction_of_logged": float(len(skip) / len(df)),
        "full": {c: summary(full[c]) for c in comps + ["total"]},
        "skip_total": summary(skip["total"]),
        "full_total_equiv_fps": 1000.0 / full["total"].mean(),
        "measured_cycle_fps_ema": summary(df["fps"]),
    }
    res["shares_pct"] = {c: 100 * full[c].mean() / full["total"].mean() for c in comps}

    labels = ["Capture", "Inference", "Tracking +\npriority scoring", "Overlay +\nstream hand-off"]
    means = [full[c].mean() for c in comps]
    p95 = [pct(full[c], 95) for c in comps]
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_RED]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(comps))[::-1]
    a1.barh(y, means, color=colors, edgecolor="k", lw=0.4, height=0.6, label="mean")
    a1.errorbar(means, y, xerr=[np.zeros(4), np.array(p95) - np.array(means)], fmt="none",
                ecolor="k", capsize=3, lw=0.8, label="mean to p95")
    for yi, m, s in zip(y, means, res["shares_pct"].values()):
        a1.text(m + 1.2, yi + 0.18, f"{m:.2f} ms ({s:.1f}%)", va="bottom", fontsize=7.5)
    a1.set_yticks(y)
    a1.set_yticklabels(labels)
    a1.set_xlabel("Per-frame time [ms]")
    a1.set_xlim(0, max(p95) * 1.25)
    a1.set_title("(a) Components, inference frames (n = %d)" % len(full))
    a1.grid(axis="x", alpha=0.3)
    a1.legend(loc="lower right", frameon=False)

    bins = np.linspace(0, max(pct(full["total"], 99.5), 80), 45)
    a2.hist(full["total"], bins=bins, color=C_ORANGE, alpha=0.75, label="inference frames")
    a2.hist(skip["total"], bins=bins, color=C_BLUE, alpha=0.75, label="skipped-inference frames")
    a2.axvline(full["total"].mean(), color="k", ls="--", lw=0.8)
    a2.text(full["total"].mean(), a2.get_ylim()[1] * 0.55, f" mean {full['total'].mean():.1f} ms",
            fontsize=7.5)
    a2.set_xlabel("Local processing time per frame [ms]")
    a2.set_ylabel("Count")
    a2.set_title("(b) Per-frame totals")
    a2.legend(frameon=False, loc="upper right")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_latency.png")
    plt.close(fig)
    return res


# --------------------------------------------------------------------------- fusion
GTM = re.compile(
    r"^(?P<ts>\S+) \[GTM\] ingest (?P<key>\S+) -> G(?P<gid>\d+) (?:RESURRECTED \(orphan_age=[\d.]+s\) )?"
    r"\(method=(?P<method>\w+), contributors=(?P<n>\d+) \[(?P<contrib>[^\]]+)\], "
    r"σ=(?P<sigma>[\d.]+)m, d_norm=(?P<dnorm>[\d.]+)")
FLAT = re.compile(r"\[FUSION_LAT\] (?P<node>\S+) pi→browser=(?P<pb>-?\d+)ms srv→browser=(?P<sb>-?\d+)ms")


def analyse_fusion(log: Path, out: Path) -> dict:
    rows, lat = [], []
    with open(log, encoding="utf-8") as f:
        for line in f:
            m = GTM.search(line)
            if m:
                rows.append(m.groupdict())
                continue
            m = FLAT.search(line)
            if m:
                lat.append(float(m["sb"]))
    df = pd.DataFrame(rows)
    df["sigma"] = df["sigma"].astype(float)
    df["ts"] = pd.to_datetime(df["ts"])
    res = {"n_events": int(len(df)), "methods": df["method"].value_counts().to_dict(),
           "session_start": str(df["ts"].min()), "session_end": str(df["ts"].max())}
    for m, g in df.groupby("method"):
        res[m] = summary(g["sigma"])
    # Same quantity, but only where both nodes were tracking the same global track at
    # (almost) the same time: compare single-source events that occur within 1 s of a
    # bearing3D event of the same global track -> a coarse time-paired comparison.
    df = df.sort_values("ts")
    paired = []
    for gid, g in df.groupby("gid"):
        b = g[g.method == "bearing3D"]
        s = g[g.method == "single"]
        if b.empty or s.empty:
            continue
        idx = np.searchsorted(b["ts"].values, s["ts"].values)
        for i, (ts_s, sig_s) in enumerate(zip(s["ts"].values, s["sigma"].values)):
            cands = [j for j in (idx[i] - 1, idx[i]) if 0 <= j < len(b)]
            if not cands:
                continue
            j = min(cands, key=lambda j: abs(b["ts"].values[j] - ts_s))
            if abs((b["ts"].values[j] - ts_s) / np.timedelta64(1, "s")) <= 1.0:
                paired.append((sig_s, b["sigma"].values[j]))
    paired = np.array(paired)
    if len(paired):
        res["time_paired_within_1s"] = {
            "n_pairs": int(len(paired)),
            "single": summary(paired[:, 0]), "bearing3D": summary(paired[:, 1]),
            "frac_bearing_lower": float((paired[:, 1] < paired[:, 0]).mean()),
        }
    res["srv_to_browser_ms"] = summary(lat) if lat else None

    # Figure from exactly the same samples as the table (no clipping of statistics;
    # only the x-axis is log-scaled so the tail stays visible).
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    bins = np.logspace(np.log10(max(0.4, df["sigma"].min())), np.log10(df["sigma"].max() * 1.05), 50)
    for m, c, lab in (("single", C_BLUE, "single-source"), ("bearing3D", C_RED, "dual-source (bearing3D)")):
        g = df[df.method == m]["sigma"]
        a1.hist(g, bins=bins, color=c, alpha=0.55,
                label=f"{lab}\nn={len(g)}, mean={g.mean():.2f} m, median={g.median():.2f} m")
    a1.set_xscale("log")
    a1.set_xlabel(r"Reported 1$\sigma$ uncertainty [m] (log)")
    a1.set_ylabel("Ingest events")
    a1.set_title("(a) Predicted uncertainty, all ingest events")
    a1.legend(frameon=False, loc="upper right", fontsize=6.5)
    a1.grid(alpha=0.3)
    for m, c, lab in (("single", C_BLUE, "single-source"), ("bearing3D", C_RED, "dual-source")):
        g = np.sort(df[df.method == m]["sigma"].values)
        a2.plot(g, np.arange(1, len(g) + 1) / len(g), color=c, lw=1.2, label=lab)
    a2.axhline(0.9, color=C_GREY, ls="--", lw=0.8)
    a2.text(120, 0.84, "p90", fontsize=7.5)
    a2.set_xscale("log")
    a2.set_xlabel(r"Reported 1$\sigma$ uncertainty [m] (log)")
    a2.set_ylabel("Empirical CDF")
    a2.set_title("(b) CDF: lower median, heavier upper tail")
    a2.legend(frameon=False, loc="lower right")
    a2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_fusion_sigma.png")
    plt.close(fig)
    return res


def main():
    log_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name, fn, src in (("control_stats", analyse_servo, "pisentinel.log"),
                          ("latency_stats", analyse_latency, "pisentinel.log"),
                          ("fusion_stats", analyse_fusion, "fusion.log")):
        r = fn(log_dir / src, out)
        (out / f"{name}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        print(f"=== {name} ===")
        print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
