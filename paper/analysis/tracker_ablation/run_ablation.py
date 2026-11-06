"""Tracker ablation and baseline comparison on frozen synthetic detections.

Usage (run from a scratch working directory; needs filterpy, motmetrics, boxmot):
  python -I run_ablation.py <analysis/tracker_ablation dir> <PiSentinel source root> <out_dir> [n_seeds]
"""
import contextlib
import io
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import motmetrics as mm

HERE = Path(sys.argv[1]).resolve()
PIS_ROOT = Path(sys.argv[2]).resolve()
OUT = Path(sys.argv[3]).resolve()
N_SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

import scenarios as S                     # noqa: E402
from pisentinel_tracker.params import PARAMS  # noqa: E402
from pisentinel_tracker.multi_tracker import MultiObjectTracker  # noqa: E402
from pisentinel_tracker.track import Track    # noqa: E402

import logging
logging.disable(logging.CRITICAL)


class SimClock:
    t = 0.0

    def __call__(self):
        return self.t


CLOCK = SimClock()
PARAMS["clock"] = CLOCK


# ----------------------------------------------------------------- tracker wrappers
class PiSentinelTracker:
    def __init__(self, **params):
        self.params = params
        self.mot = MultiObjectTracker()

    def update(self, dets, frame, dt):
        PARAMS.update({"adapt_q": True, "adapt_r": True, "direction": True, "unit_scale": 1.0})
        PARAMS.update(self.params)
        CLOCK.t += dt
        boxes = [tuple(d[:4]) for d in dets]
        confs = [float(d[4]) for d in dets]
        with contextlib.redirect_stdout(io.StringIO()):
            self.mot.update(boxes, confs)
        out = []
        for t in self.mot.tracks:
            if t.state == Track.CONFIRMED and t.time_since_update == 0:
                out.append((t.track_id, *t.get_predicted_box()))
        return out


def iou_batch(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class _SortTrack:
    """Kalman box tracker of the reference SORT implementation (Bewley et al. 2016)."""
    count = 0

    def __init__(self, bbox):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]], float)
        kf.H = np.eye(4, 7)
        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        kf.x[:4, 0] = self.z(bbox)
        self.kf = kf
        _SortTrack.count += 1
        self.id = _SortTrack.count
        self.time_since_update = 0
        self.hit_streak = 0

    @staticmethod
    def z(b):
        w, h = b[2] - b[0], b[3] - b[1]
        return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])

    def box(self):
        x = self.kf.x[:, 0]
        w = np.sqrt(max(x[2] * x[3], 1e-6))
        h = x[2] / w
        return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])

    def predict(self):
        if self.kf.x[6, 0] + self.kf.x[2, 0] <= 0:
            self.kf.x[6, 0] = 0
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.box()

    def update(self, b):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.z(b))


class SortTracker:
    """Reference SORT: max_age=1, min_hits=3, IoU threshold 0.3."""

    def __init__(self, max_age=1, min_hits=3, iou_thr=0.3):
        self.max_age, self.min_hits, self.iou_thr = max_age, min_hits, iou_thr
        self.trks = []
        self.frame = 0

    def update(self, dets, frame, dt):
        self.frame += 1
        preds = np.array([t.predict() for t in self.trks]).reshape(-1, 4)
        keep = [i for i, p in enumerate(preds) if np.all(np.isfinite(p))]
        self.trks = [self.trks[i] for i in keep]
        preds = preds[keep]
        dboxes = dets[:, :4] if len(dets) else np.zeros((0, 4))
        matched, um_d = [], set(range(len(dboxes)))
        if len(preds) and len(dboxes):
            iou = iou_batch(dboxes, preds)
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= self.iou_thr:
                    matched.append((i, j))
                    um_d.discard(i)
        for i, j in matched:
            self.trks[j].update(dboxes[i])
        for i in sorted(um_d):
            self.trks.append(_SortTrack(dboxes[i]))
        out = []
        for t in self.trks:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame <= self.min_hits):
                out.append((t.id, *t.box()))
        self.trks = [t for t in self.trks if t.time_since_update <= self.max_age]
        return out


class BoxmotTracker:
    def __init__(self, name):
        from boxmot import ByteTrack, OcSort, BotSort
        from boxmot.trackers.botsort.config import BotSortConfig
        if name == "ByteTrack":
            self.t = ByteTrack()
        elif name == "OC-SORT":
            self.t = OcSort()
        elif name == "BoT-SORT":
            self.t = BotSort(config=BotSortConfig(use_embeddings=False, use_cmc=True))
        self.name = name

    def update(self, dets, frame, dt):
        d = np.zeros((len(dets), 6))
        if len(dets):
            d[:, :5] = dets
        res = self.t.update(d, frame)
        res = np.asarray(res).reshape(-1, 8) if np.size(res) else np.zeros((0, 8))
        return [(int(r[4]), *r[:4]) for r in res]


# ----------------------------------------------------------------- metrics
ALPHAS = np.arange(0.05, 0.99, 0.05)


def hota(gt_frames, tr_frames):
    """HOTA following the TrackEval reference algorithm (Luiten et al., IJCV 2021)."""
    gt_ids = sorted({int(g[0]) for f in gt_frames for g in f})
    tr_ids = sorted({int(t[0]) for f in tr_frames for t in f})
    gmap = {g: i for i, g in enumerate(gt_ids)}
    tmap = {t: i for i, t in enumerate(tr_ids)}
    ng, nt = len(gt_ids), len(tr_ids)
    pot = np.zeros((ng, nt))
    gcnt = np.zeros((ng, 1))
    tcnt = np.zeros((1, nt))
    sims = []
    for g, t in zip(gt_frames, tr_frames):
        gi = np.array([gmap[int(x[0])] for x in g], int)
        ti = np.array([tmap[int(x[0])] for x in t], int)
        sim = iou_batch([x[1:5] for x in g], [x[1:5] for x in t]) if len(gi) and len(ti) \
            else np.zeros((len(gi), len(ti)))
        sims.append((gi, ti, sim))
        if len(gi) and len(ti):
            den = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            s = np.where(den > np.finfo(float).eps, sim / np.maximum(den, 1e-12), 0)
            pot[gi[:, None], ti[None, :]] += s
        gcnt[gi] += 1
        tcnt[0, ti] += 1
    gas = pot / np.maximum(gcnt + tcnt - pot, 1e-12)
    TP, FN, FP = np.zeros(len(ALPHAS)), np.zeros(len(ALPHAS)), np.zeros(len(ALPHAS))
    mcount = [np.zeros((ng, nt)) for _ in ALPHAS]
    for gi, ti, sim in sims:
        if len(gi) == 0 or len(ti) == 0:
            FN += len(gi)
            FP += len(ti)
            continue
        score = gas[gi[:, None], ti[None, :]] * sim
        r, c = linear_sum_assignment(-score)
        for a, al in enumerate(ALPHAS):
            ok = sim[r, c] >= al - np.finfo(float).eps
            n = int(ok.sum())
            TP[a] += n
            FN[a] += len(gi) - n
            FP[a] += len(ti) - n
            if n:
                mcount[a][gi[r[ok]], ti[c[ok]]] += 1
    AssA = np.zeros(len(ALPHAS))
    for a in range(len(ALPHAS)):
        m = mcount[a]
        ass = m / np.maximum(1, gcnt + tcnt - m)
        AssA[a] = (m * ass).sum() / max(1, TP[a])
    DetA = TP / np.maximum(1, TP + FN + FP)
    return float(np.mean(np.sqrt(DetA * AssA))), float(DetA.mean()), float(AssA.mean())


def evaluate(seq, tracks):
    acc = mm.MOTAccumulator(auto_id=True)
    gt_frames = []
    for g, t in zip(seq["gt"], tracks):
        gt_frames.append([(int(x[0]), *x[1:]) for x in g])
        gb = g[:, 1:5] if len(g) else np.zeros((0, 4))
        tb = np.array([x[1:5] for x in t], float).reshape(-1, 4)
        d = 1 - iou_batch(gb, tb) if len(gb) and len(tb) else np.zeros((len(gb), len(tb)))
        d[d > 0.5] = np.nan                     # IoU >= 0.5 to count as a match
        acc.update([int(x[0]) for x in g], [int(x[0]) for x in t], d)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "mota", "num_switches", "num_fragmentations",
                                 "mostly_tracked", "num_unique_objects", "num_objects"], name="s")
    r = {k: float(s[k].iloc[0]) for k in s.columns}
    h, deta, assa = hota(gt_frames, tracks)
    r.update({"hota": h, "deta": deta, "assa": assa})
    # occlusion recovery: same hypothesis id before and after each injected gap
    ev = acc.mot_events
    match = ev[ev.Type.isin(["MATCH", "SWITCH"])]
    by_obj = {o: dict(zip(g.index.get_level_values(0), g.HId)) for o, g in match.groupby("OId")}
    occ = seq["occluded"]
    rec, tot = 0, 0
    for k in range(occ.shape[1]):
        f = 0
        while f < occ.shape[0]:
            if occ[f, k]:
                s0 = f
                while f < occ.shape[0] and occ[f, k]:
                    f += 1
                before = [by_obj.get(k, {}).get(i) for i in range(max(0, s0 - 10), s0)]
                before = [b for b in before if b is not None]
                after = [by_obj.get(k, {}).get(i) for i in range(f, min(occ.shape[0], f + 30))]
                after = [a for a in after if a is not None]
                if before and after:
                    tot += 1
                    rec += int(before[-1] == after[0])
            f += 1
    r["occ_recovered"], r["occ_events"] = rec, tot
    r["idsw_per_1k"] = 1000 * r["num_switches"] / max(1, r["num_objects"])
    r["frag_per_1k"] = 1000 * r["num_fragmentations"] / max(1, r["num_objects"])
    return r


# ----------------------------------------------------------------- runs
VARIANTS = {
    "PiSentinel (deployed thresholds)": lambda: PiSentinelTracker(unit_scale=1.0),
    "PiSentinel full (unit-corrected)": lambda: PiSentinelTracker(unit_scale=30.0),
    "  fixed Q and R": lambda: PiSentinelTracker(unit_scale=30.0, adapt_q=False, adapt_r=False),
    "  adaptive Q only": lambda: PiSentinelTracker(unit_scale=30.0, adapt_r=False),
    "  confidence-scaled R only": lambda: PiSentinelTracker(unit_scale=30.0, adapt_q=False),
    "  without direction term": lambda: PiSentinelTracker(unit_scale=30.0, direction=False),
    "SORT": lambda: SortTracker(),
    "ByteTrack": lambda: BoxmotTracker("ByteTrack"),
    "OC-SORT": lambda: BoxmotTracker("OC-SORT"),
    "BoT-SORT (CMC, no ReID)": lambda: BoxmotTracker("BoT-SORT"),
}


def run(seq, make, frames):
    trk = make()
    out, t_ms = [], []
    for i, d in enumerate(seq["dets"]):
        t0 = time.perf_counter()
        out.append(trk.update(d, frames[i], seq["dt"][i]))
        t_ms.append(1000 * (time.perf_counter() - t0))
    return out, float(np.mean(t_ms))


def render(seq):
    rr = S.Renderer(np.random.default_rng(seq["renderer_seed"] + 99))
    return [rr.frame(seq["cam"][i], [x[:4] for x in seq["dets"][i]]) for i in range(len(seq["dets"]))]


def verify_against_original():
    """The ablation copy with default PARAMS must reproduce the deployed tracker exactly."""
    sys.path.insert(0, str(PIS_ROOT))
    import importlib
    # load tracking.track / tracking.multi_tracker without executing tracking/__init__.py
    # (which pulls in the detector and camera stack)
    pkg = types.ModuleType("tracking")
    pkg.__path__ = [str(PIS_ROOT / "tracking")]
    sys.modules["tracking"] = pkg
    orig_track = importlib.import_module("tracking.track")
    orig_mt = importlib.import_module("tracking.multi_tracker")
    fake = types.SimpleNamespace(monotonic=CLOCK)
    orig_track.time = fake
    mismatches, frames = 0, 0
    for scen in S.SCENARIOS:
        seq = S.make_sequence(scen, 1)
        CLOCK.t = 0.0
        a = orig_mt.MultiObjectTracker()
        b = PiSentinelTracker()
        t_a = 0.0
        for d, dt in zip(seq["dets"], seq["dt"]):
            t_a += dt
            CLOCK.t = t_a
            with contextlib.redirect_stdout(io.StringIO()):
                a.update([tuple(x[:4]) for x in d], [float(x[4]) for x in d])
            oa = sorted((t.track_id, *t.get_predicted_box()) for t in a.tracks
                        if t.state == orig_track.Track.CONFIRMED and t.time_since_update == 0)
            CLOCK.t = t_a - dt
            ob = sorted(b.update(d, None, dt))
            frames += 1
            mismatches += int(oa != ob)
    return {"frames_compared": frames, "frames_mismatched": mismatches}


def main():
    verification = verify_against_original()
    print("verification:", verification, flush=True)
    results = []
    for noise in ("N1", "N2"):
        for scen in S.SCENARIOS:
            for seed in range(N_SEEDS):
                seq = S.make_sequence(scen, 1000 * (1 + S.SCENARIOS.index(scen)) + seed, noise)
                frames = render(seq)
                for name, make in VARIANTS.items():
                    if noise == "N2" and not name.startswith(("PiSentinel full", "  fixed", "  confidence")):
                        continue
                    CLOCK.t = 0.0
                    tracks, ms = run(seq, make, frames)
                    r = evaluate(seq, tracks)
                    r.update({"tracker": name.strip(), "scenario": scen, "seed": seed,
                              "noise": noise, "ms_per_frame": ms})
                    results.append(r)
            print(noise, scen, "done", flush=True)
    (OUT / "tracker_ablation_raw.json").write_text(
        json.dumps({"verification": verification, "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
