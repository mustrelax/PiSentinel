"""Revision R1 -- simulation-based (Monte Carlo) evaluation of the geolocation estimators.

No ground-truth trajectories were logged during the recorded UE5 session, so the
paired estimator comparison requested by Reviewer 1 is carried out here in a
controlled simulation that reproduces the deployed geometry and the exact fusion
logic of the server (central_server/globalTrackManager.js) and the Pi
(tracking/geolocation.py).  Truth is used ONLY for scoring.

Per synchronised event the same pair of noisy observations is passed to:
  A      node A alone (spherical projection with monocular range)
  B      node B alone
  AVG    Cartesian average of A and B
  IVW    inverse-variance weighting with the reported sigmas (Eq. 10)
  TRI    two-bearing triangulation exactly as in the GTM (ray midpoint,
         reject if sin(angle)<0.05 or t<0 -> fallback IVW)
  TRI-G  TRI with an additional geometry gate: triangulate only if the ray
         intersection angle is within [15, 165] deg, else IVW

Scenarios (noise actually applied; the estimators always use the deployed
sigma model sigma_b = 1 deg, sigma_r = 0.10 r):
  S1 model-consistent : bearing noise 1 deg (az, el), range noise 10 % Gaussian
  S2 mismatched       : range from box width z = f W / (w + n_w), n_w ~ N(0, 1 px),
                        0.5 deg joint-angle noise + 1 deg mount-yaw bias on node B
  S3 asynchronous     : S1 + 600 ms observation skew between nodes (the GTM stale
                        limit), target speed 15 m/s
  S4 near-parallel    : S1 noise, ranges extended to 1500 m so that small ray
                        intersection angles occur (stress test, beyond detection range)

Also bounds the additive camera-to-world angle approximation used on the Pi.

Usage: python -I fusion_mc.py <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "legend.fontsize": 7.5,
                     "axes.titlesize": 9.5, "savefig.dpi": 300})

RNG = np.random.default_rng(2026)
N_EVENTS = 40000

# Deployed geometry (config.py of Sentinel-01 / Sentinel-02)
PA = np.array([0.0, 0.0, 2.5]);      FACE_A = 310.0
PB = np.array([-235.1, 55.4, -11.1]); FACE_B = 50.0
F_PX = 960.0          # focal length [px]
W_M = 2.60            # calibrated apparent drone width used for monocular range [m]
SIG_B_DEG = 1.0       # GTM / Pi bearing sigma
K_RANGE = 0.10        # Pi range sigma fraction
MIN_UNC = 2.0
PARALLEL_SIN = 0.05
SIN_FLOOR = 0.1
MIN_SIGMA = 0.5
CHI2_3_95 = 7.814727903251178
CHI2_3_68 = 3.5267403802617303   # 68.27 % quantile of chi2(3)


def unit(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], -1)


def angles(p, origin):
    d = p - origin
    rng = np.linalg.norm(d, axis=-1)
    az = np.degrees(np.arctan2(d[..., 0], d[..., 1])) % 360
    el = np.degrees(np.arcsin(d[..., 2] / rng))
    return az, el, rng


def wrap180(a):
    return (a + 180) % 360 - 180


def sample_targets(n, rmax=500.0):
    out = []
    while sum(len(o) for o in out) < n:
        m = 4 * n
        span = rmax + 100
        p = np.stack([RNG.uniform(-span, span, m), RNG.uniform(-100, span, m), RNG.uniform(15, 180, m)], -1)
        aza, ela, ra = angles(p, PA)
        azb, elb, rb = angles(p, PB)
        ok = ((np.abs(wrap180(aza - FACE_A)) <= 90) & (np.abs(wrap180(azb - FACE_B)) <= 90) &
              (ra >= 40) & (ra <= rmax) & (rb >= 40) & (rb <= rmax) & (ela < 75) & (elb < 75))
        out.append(p[ok])
    return np.concatenate(out)[:n]


def reported_sigma(r):
    sr = np.maximum(MIN_UNC, K_RANGE * r)
    sb = np.sin(np.radians(SIG_B_DEG)) * r
    return np.sqrt(sr ** 2 + sb ** 2)


def observe(p, origin, scen, is_b):
    az, el, r = angles(p, origin)
    n = len(p)
    if scen == "S2":
        az = az + RNG.normal(0, 0.5, n) + (1.0 if is_b else 0.0)
        el = el + RNG.normal(0, 0.5, n)
        w_true = F_PX * W_M / r
        w_meas = np.maximum(w_true + RNG.normal(0, 1.0, n), 1.0)
        r_meas = F_PX * W_M / w_meas
    else:
        az = az + RNG.normal(0, SIG_B_DEG, n)
        el = el + RNG.normal(0, SIG_B_DEG, n)
        r_meas = np.maximum(r * (1 + RNG.normal(0, K_RANGE, n)), 1.0)
    pos = origin + r_meas[:, None] * unit(az, el)
    return {"az": az, "el": el, "r": r_meas, "pos": pos, "sig": reported_sigma(r_meas)}


def triangulate(oa, ob):
    d1, d2 = unit(oa["az"], oa["el"]), unit(ob["az"], ob["el"])
    c = np.sum(d1 * d2, -1)
    sin2 = np.maximum(0, 1 - c ** 2)
    w0 = PA - PB
    a = d1 @ w0
    b = d2 @ w0
    den = np.where(sin2 > 0, 1 - c ** 2, np.nan)
    t1 = (c * b - a) / den
    t2 = (b - c * a) / den
    p1 = PA + t1[:, None] * d1
    p2 = PB + t2[:, None] * d2
    miss = np.linalg.norm(p1 - p2, axis=-1)
    sin_int = np.maximum(np.sqrt(sin2), SIN_FLOOR)
    sb = np.radians(SIG_B_DEG)
    sig = np.maximum(MIN_SIGMA, np.hypot(sb * np.abs(t1), sb * np.abs(t2)) / sin_int + 0.5 * miss)
    valid = (sin2 >= PARALLEL_SIN ** 2) & (t1 >= 0) & (t2 >= 0)
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return 0.5 * (p1 + p2), sig, valid, ang


def run_scenario(scen):
    truth = sample_targets(N_EVENTS, 1500.0 if scen == "S4" else 500.0)
    truth_b = truth.copy()
    if scen == "S3":
        v = RNG.normal(size=truth.shape)
        v[:, 2] *= 0.3
        v = 15.0 * v / np.linalg.norm(v, axis=1, keepdims=True)
        truth_b = truth - 0.600 * v           # node B observation 600 ms old (GTM stale limit)
    oa = observe(truth, PA, scen, False)
    ob = observe(truth_b, PB, scen, True)

    est = {"A": (oa["pos"], oa["sig"]), "B": (ob["pos"], ob["sig"])}
    est["AVG"] = (0.5 * (oa["pos"] + ob["pos"]), 0.5 * np.hypot(oa["sig"], ob["sig"]))
    wa, wb = 1 / oa["sig"] ** 2, 1 / ob["sig"] ** 2
    ivw = (wa[:, None] * oa["pos"] + wb[:, None] * ob["pos"]) / (wa + wb)[:, None]
    ivw_s = 1 / np.sqrt(wa + wb)
    est["IVW"] = (ivw, ivw_s)
    tri, tri_s, valid, ang = triangulate(oa, ob)
    tri_f = np.where(valid[:, None], tri, ivw)
    tri_fs = np.where(valid, tri_s, ivw_s)
    est["TRI"] = (tri_f, tri_fs)
    gate = valid & (ang >= 15) & (ang <= 165)
    est["TRI-G"] = (np.where(gate[:, None], tri, ivw), np.where(gate, tri_s, ivw_s))

    _, _, ra = angles(truth, PA)
    _, _, rb = angles(truth, PB)
    rmax = np.maximum(ra, rb)
    res = {"n_events": int(len(truth)), "tri_available": float(valid.mean()),
           "tri_gate_pass": float(gate.mean()), "estimators": {}, "by_angle": {}, "by_range": {}}
    for k, (p, s) in est.items():
        e = np.linalg.norm(p - truth, axis=-1)
        z2 = (e / s) ** 2      # isotropic per-axis interpretation of the scalar sigma
        res["estimators"][k] = {
            "median_err": float(np.median(e)), "mean_err": float(e.mean()),
            "p90_err": float(np.percentile(e, 90)), "p99_err": float(np.percentile(e, 99)),
            "outlier_frac_gt50m": float((e > 50).mean()),
            "median_reported_sigma": float(np.median(s)),
            "coverage_68": float((z2 <= CHI2_3_68).mean()),
            "coverage_95": float((z2 <= CHI2_3_95).mean()),
        }
    angle_bins = [(0, 3), (3, 8), (8, 15), (15, 30), (30, 60), (60, 90), (90, 120), (120, 177)]
    for lo, hi in angle_bins:
        m = (ang >= lo) & (ang < hi)
        res["by_angle"][f"{lo}-{hi}"] = {
            "n": int(m.sum()),
            **{k: float(np.median(np.linalg.norm(est[k][0][m] - truth[m], axis=-1))) for k in est}}
    for lo, hi in [(40, 150), (150, 300), (300, 500), (500, 1000), (1000, 1500)]:
        m = (rmax >= lo) & (rmax < hi)
        res["by_range"][f"{lo}-{hi}"] = {
            "n": int(m.sum()),
            **{k: float(np.median(np.linalg.norm(est[k][0][m] - truth[m], axis=-1))) for k in est}}
    return res, truth, est, ang


def additive_angle_bound():
    """Max azimuth/elevation error of the additive mapping (Eq. 5) vs exact pan-tilt
    kinematics, for optical offset |(u, v)| and camera elevation up to eps_max."""
    out = {}
    for eps_max in (30, 60):
        for off_px in (6, 24, 105):      # median, p95, p99 logged tracking error
            worst_az, worst_el = 0.0, 0.0
            for eps_c in np.linspace(0, eps_max, eps_max + 1):
                for th in np.linspace(0, 2 * np.pi, 72, endpoint=False):
                    u, v = off_px * np.cos(th), off_px * np.sin(th)
                    ray = np.array([u / F_PX, 1.0, v / F_PX])  # x right, y forward, z up
                    ray /= np.linalg.norm(ray)
                    e = np.radians(eps_c)
                    R = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
                    w = R @ ray
                    az_true = np.degrees(np.arctan2(w[0], w[1]))
                    el_true = np.degrees(np.arcsin(w[2]))
                    az_add = np.degrees(np.arctan(u / F_PX))
                    el_add = eps_c + np.degrees(np.arctan(v / F_PX))
                    worst_az = max(worst_az, abs(wrap180(az_true - az_add)))
                    worst_el = max(worst_el, abs(el_true - el_add))
            out[f"eps<={eps_max}deg_{off_px}px"] = {"max_az_err_deg": worst_az, "max_el_err_deg": worst_el}
    return out


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    allres = {}
    keep = {}
    for scen in ("S1", "S2", "S3", "S4"):
        r, truth, est, ang = run_scenario(scen)
        allres[scen] = r
        keep[scen] = (truth, est, ang)
    allres["additive_angle_bound_eps0_60deg"] = additive_angle_bound()
    (out / "fusion_mc.json").write_text(json.dumps(allres, indent=2), encoding="utf-8")
    print(json.dumps(allres, indent=1))

    # Figure: median 3-D error vs intersection angle (S1 and S2), and reported sigma vs error (S1)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    styles = {"A": ("#7f7f7f", ":"), "IVW": ("#2a6fb0", "-"), "TRI": ("#c0392b", "-"),
              "TRI-G": ("#3a8f4f", "--")}
    edges = np.arange(5, 180, 10)
    for ax, scen in zip(axs[:2], ("S1", "S4")):
        truth, est, ang = keep[scen]
        mids = 0.5 * (edges[1:] + edges[:-1])
        for k, (c, ls) in styles.items():
            e = np.linalg.norm(est[k][0] - truth, axis=-1)
            med = [np.median(e[(ang >= a) & (ang < b)]) if ((ang >= a) & (ang < b)).sum() > 50 else np.nan
                   for a, b in zip(edges[:-1], edges[1:])]
            ax.plot(mids, med, color=c, ls=ls, lw=1.3, label=k)
        ax.set_yscale("log")
        ax.set_xlabel("Ray intersection angle [deg]")
        ax.set_ylabel("Median 3-D error [m]")
        ax.set_title({"S1": "(a) S1 model-consistent noise", "S4": "(b) S4 incl. near-parallel rays"}[scen])
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(frameon=False, loc="upper right")
    truth, est, ang = keep["S1"]
    for k, c in (("IVW", "#2a6fb0"), ("TRI", "#c0392b")):
        p, s = est[k]
        z = np.linalg.norm(p - truth, axis=-1) / s
        zs = np.sort(z)
        axs[2].plot(zs, np.arange(1, len(zs) + 1) / len(zs), color=c, lw=1.2, label=k)
    from math import erf
    xs = np.linspace(0, 6, 200)
    # chi(3) CDF for |e|/sigma under the isotropic per-axis assumption
    chi3 = [erf(x / np.sqrt(2)) - np.sqrt(2 / np.pi) * x * np.exp(-x ** 2 / 2) for x in xs]
    axs[2].plot(xs, chi3, color="k", ls="--", lw=1, label=r"ideal $\chi_3$")
    axs[2].set_xlim(0, 6)
    axs[2].set_xlabel(r"Normalised error $|e|/\sigma$")
    axs[2].set_ylabel("Empirical CDF")
    axs[2].set_title("(c) Reported $\\sigma$ calibration, S1")
    axs[2].legend(frameon=False, loc="lower right")
    axs[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_fusion_mc.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
