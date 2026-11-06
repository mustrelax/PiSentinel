"""Check that the patched release tracker reproduces the 'unit-corrected' ablation variant.
Usage: python -I verify_release.py <tracker_ablation dir> <release PiSentinel dir>"""
import contextlib
import io
import importlib
import sys
import types
from pathlib import Path

HERE, REL = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(HERE))
import scenarios as S                                  # noqa: E402
from pisentinel_tracker.params import PARAMS           # noqa: E402
from pisentinel_tracker.multi_tracker import MultiObjectTracker as VarMOT  # noqa: E402
from pisentinel_tracker.track import Track as VarTrack  # noqa: E402

sys.path.insert(0, str(REL))
pkg = types.ModuleType("tracking")
pkg.__path__ = [str(REL / "tracking")]
sys.modules["tracking"] = pkg
rel_track = importlib.import_module("tracking.track")
rel_mt = importlib.import_module("tracking.multi_tracker")

clock = types.SimpleNamespace(t=0.0)
rel_track.time = types.SimpleNamespace(monotonic=lambda: clock.t)
PARAMS.update({"clock": lambda: clock.t, "unit_scale": 30.0})

frames = mism = 0
for scen in S.SCENARIOS:
    seq = S.make_sequence(scen, 7)
    a, b = rel_mt.MultiObjectTracker(), VarMOT()
    clock.t = 0.0
    for d, dt in zip(seq["dets"], seq["dt"]):
        clock.t += dt
        boxes, confs = [tuple(x[:4]) for x in d], [float(x[4]) for x in d]
        with contextlib.redirect_stdout(io.StringIO()):
            a.update(boxes, confs)
            b.update(boxes, confs)
        oa = sorted((t.track_id, *t.get_predicted_box()) for t in a.tracks
                    if t.state == rel_track.Track.CONFIRMED and t.time_since_update == 0)
        ob = sorted((t.track_id, *t.get_predicted_box()) for t in b.tracks
                    if t.state == VarTrack.CONFIRMED and t.time_since_update == 0)
        frames += 1
        mism += int(oa != ob)
print({"frames_compared": frames, "frames_mismatched": mism})
