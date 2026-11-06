"""Build an ablation-ready copy of the deployed PiSentinel tracker.

Copies tracking/track.py, matcher.py and multi_tracker.py from the PiSentinel source
tree into ./pisentinel_tracker/ and applies a small set of textual substitutions
(each asserted to apply exactly once) so that the following can be switched at run
time through pisentinel_tracker.params.PARAMS:

  adapt_q     confidence/motion-adaptive process noise Q   (deployed: True)
  adapt_r     confidence-scaled measurement noise R        (deployed: True)
  direction   velocity-direction bonus in the cost matrix  (deployed: True)
  unit_scale  multiplier on the motion thresholds that were tuned in px/frame
              (12, 18, 150, 120, 3, 2) while velocity is now in px/s.
              1 = deployed code, 30 = thresholds converted to px/s at 30 fps.
  clock       time source used by Track.predict (simulation clock)

With PARAMS at their defaults the copy is behaviourally identical to the deployed
code; run_ablation.py verifies this frame by frame.

Usage: python -I make_variant.py <PiSentinel source root>
"""
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) / "tracking"
DST = Path(__file__).resolve().parent / "pisentinel_tracker"
DST.mkdir(exist_ok=True)

HDR = "from .params import PARAMS\nfrom . import config\n"

SUBS = {
    "track.py": [
        ("import config\n", HDR),
        ("now = time.monotonic()", "now = PARAMS['clock']()"),
        ("return self.get_average_speed() > self.FAST_MOTION_THRESHOLD",
         "return self.get_average_speed() > self.FAST_MOTION_THRESHOLD * PARAMS['unit_scale']"),
        ("return self.get_average_speed() > self.AGGRESSIVE_MODE_THRESHOLD",
         "return self.get_average_speed() > self.AGGRESSIVE_MODE_THRESHOLD * PARAMS['unit_scale']"),
        ("if avg_speed < 3.0:", "if avg_speed < 3.0 * PARAMS['unit_scale']:"),
        ("if self.get_current_speed() > 150:", "if self.get_current_speed() > 150 * PARAMS['unit_scale']:"),
        ("        if self.is_aggressive_motion():\n            self.kf.Q = self.base_Q * 3.0 * dt_scale",
         "        if not PARAMS['adapt_q']:\n            self.kf.Q = self.base_Q * dt_scale\n"
         "        elif self.is_aggressive_motion():\n            self.kf.Q = self.base_Q * 3.0 * dt_scale"),
        ("confidence_factor = max(0.5, 2.0 - confidence * 1.5)",
         "confidence_factor = max(0.5, 2.0 - confidence * 1.5) if PARAMS['adapt_r'] else 1.0"),
        ("        if self.is_fast_moving():\n            adaptive_R *= 0.8",
         "        if PARAMS['adapt_r'] and self.is_fast_moving():\n            adaptive_R *= 0.8"),
    ],
    "matcher.py": [
        ("import config\n", HDR),
        ("if speed > 2.0:", "if PARAMS['direction'] and speed > 2.0 * PARAMS['unit_scale']:"),
    ],
    "multi_tracker.py": [
        ("import config\n", HDR),
        ("from tracking.track import Track", "from .track import Track"),
        ("from tracking.matcher import HungarianMatcher", "from .matcher import HungarianMatcher"),
        ("if speed > 120:", "if speed > 120 * PARAMS['unit_scale']:"),
    ],
}

for name, subs in SUBS.items():
    text = (SRC / name).read_text(encoding="utf-8")
    for old, new in subs:
        n = text.count(old)
        assert n == 1, f"{name}: expected 1 occurrence of {old!r}, found {n}"
        text = text.replace(old, new)
    (DST / name).write_text(text, encoding="utf-8")

(DST / "params.py").write_text(
    "import time\nPARAMS = {'adapt_q': True, 'adapt_r': True, 'direction': True,\n"
    "          'unit_scale': 1.0, 'clock': time.monotonic}\n", encoding="utf-8")
# Only the config values the three tracker modules read (copied from PiSentinel/config.py).
(DST / "config.py").write_text("FRAME_WIDTH = 640\nFRAME_HEIGHT = 480\nMIN_TRACK_SIZE = 3\n",
                               encoding="utf-8")
(DST / "__init__.py").write_text("", encoding="utf-8")
print("variant tracker written to", DST)
