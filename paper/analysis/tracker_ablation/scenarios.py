"""Synthetic image-plane multi-UAV sequences with frozen (pre-generated) detections.

Every sequence is generated once per seed and the same detection set is fed to all
trackers ("frozen detections").  Frames are also rendered (textured sky background
shifted by the camera ego-motion, UAVs as dark blobs) so that trackers with camera
motion compensation (BoT-SORT CMC) can use image content.

Scenarios (640 x 480, nominal 30 fps, frame interval jittered +-15 %):
  crossing  two UAVs crossing repeatedly at 4-8 px/frame with small vertical separation
  swarm     five similar-looking UAVs in a loose formation, members exchange places
  occlusion two UAVs; each is undetected for 5-25 frames roughly every 80 frames
  egomotion three slow UAVs, image motion dominated by camera slews (4-12 px/frame) + jitter
  maneuver  two agile UAVs with abrupt 90-180 deg turns every 20-60 frames, 3-9 px/frame

Detection model ("N1"): p_det 0.93 (0.80 below 12 px), centre noise 1.2 px, size noise 6 %,
confidence ~ Beta(8, 2.5) independent of the box error, false positives Poisson(0.08)/frame
with confidence U(0.23, 0.5).  "N2" makes the confidence informative: centre noise
sigma = 0.5 + 4 (1 - conf) px.
"""
import numpy as np
import cv2

W, H = 640, 480
N_FRAMES = 600
SCENARIOS = ("crossing", "swarm", "occlusion", "egomotion", "maneuver")


def _smooth_walk(rng, n, sigma, alpha=0.97):
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = alpha * v[t - 1] + rng.normal(0, sigma)
    return v


def trajectories(scen, rng):
    """Return pos (T, K, 2) image-plane centres before camera motion, sizes (K,),
    cam (T, 2) camera offset, occluded (T, K) bool."""
    T = N_FRAMES
    t = np.arange(T)
    cam = np.zeros((T, 2))
    if scen == "crossing":
        K = 2
        A = 230.0
        w = rng.uniform(4, 8) / A
        y0 = 240 + rng.uniform(-40, 40)
        dy = rng.uniform(0, 14)
        pos = np.zeros((T, K, 2))
        for k, ph in enumerate((0.0, np.pi)):
            pos[:, k, 0] = 320 + A * np.sin(w * t + ph)
            pos[:, k, 1] = y0 + (dy / 2 if k else -dy / 2) + _smooth_walk(rng, T, 0.15)
        sizes = rng.uniform(14, 24, K)
    elif scen == "swarm":
        K = 5
        cx = 320 + 150 * np.sin(2 * np.pi * t / 400 + rng.uniform(0, 6.28))
        cy = 240 + 90 * np.sin(2 * np.pi * t / 290 + rng.uniform(0, 6.28))
        base = rng.uniform(-45, 45, (K, 2))
        pos = np.zeros((T, K, 2))
        for k in range(K):
            pos[:, k, 0] = cx + base[k, 0] + np.cumsum(_smooth_walk(rng, T, 0.04)) * 0.6
            pos[:, k, 1] = cy + base[k, 1] + np.cumsum(_smooth_walk(rng, T, 0.04)) * 0.6
        pos[:, :, 0] = np.clip(pos[:, :, 0], 30, W - 30)
        pos[:, :, 1] = np.clip(pos[:, :, 1], 30, H - 30)
        sizes = rng.uniform(12, 16, K)
    elif scen == "occlusion":
        K = 2
        pos = np.zeros((T, K, 2))
        for k in range(K):
            pos[:, k, 0] = 320 + 200 * np.sin(2 * np.pi * t / rng.uniform(250, 400) + rng.uniform(0, 6.28))
            pos[:, k, 1] = 240 + 140 * np.sin(2 * np.pi * t / rng.uniform(200, 350) + rng.uniform(0, 6.28))
        sizes = rng.uniform(14, 24, K)
    elif scen == "egomotion":
        K = 3
        pos = np.zeros((T, K, 2))
        for k in range(K):
            pos[:, k, 0] = rng.uniform(150, 490) + np.cumsum(_smooth_walk(rng, T, 0.03))
            pos[:, k, 1] = rng.uniform(120, 360) + np.cumsum(_smooth_walk(rng, T, 0.03))
        # bounded back-and-forth slews (hold, then slew towards a new random
        # offset within +-140 px) so that the targets stay in view
        cam_s = np.zeros((T, 2))
        cur = np.zeros(2)
        f = 0
        while f < T:
            hold = int(rng.integers(15, 45))
            cam_s[f:f + hold] = cur
            f += hold
            goal = rng.uniform(-140, 140, 2)
            speed = rng.uniform(4, 12)
            n = max(1, int(np.linalg.norm(goal - cur) / speed))
            for i in range(1, n + 1):
                if f >= T:
                    break
                cam_s[f] = cur + (goal - cur) * i / n
                f += 1
            cur = goal
        cam = cam_s + rng.normal(0, 1.5, (T, 2))
        sizes = rng.uniform(14, 22, K)
    elif scen == "maneuver":
        K = 2
        pos = np.zeros((T, K, 2))
        for k in range(K):
            p = np.array([rng.uniform(200, 440), rng.uniform(150, 330)])
            ang = rng.uniform(0, 2 * np.pi)
            sp = rng.uniform(3, 9)
            nxt = int(rng.integers(20, 60))
            for i in range(T):
                if i == nxt:
                    ang += rng.choice([-1, 1]) * rng.uniform(np.pi / 2, np.pi)
                    sp = rng.uniform(3, 9)
                    nxt = i + int(rng.integers(20, 60))
                p = p + sp * np.array([np.cos(ang), np.sin(ang)])
                # reflect at borders
                if p[0] < 40 or p[0] > W - 40:
                    ang = np.pi - ang
                    p[0] = np.clip(p[0], 40, W - 40)
                if p[1] < 40 or p[1] > H - 40:
                    ang = -ang
                    p[1] = np.clip(p[1], 40, H - 40)
                pos[i, k] = p
        sizes = rng.uniform(14, 24, K)
    else:
        raise ValueError(scen)

    occluded = np.zeros((T, K), bool)
    if scen == "occlusion":
        for k in range(K):
            f = int(rng.integers(30, 90))
            while f < T - 30:
                L = int(rng.integers(5, 26))
                occluded[f:f + L, k] = True
                f += L + int(rng.integers(60, 100))
    # image position = world position - camera offset
    img = pos - (cam - cam[0])[:, None, :]
    return img, sizes, cam, occluded


def detections(img, sizes, occluded, rng, noise="N1"):
    """Frozen detection list per frame: list of (N, 5) arrays [x1, y1, x2, y2, conf]."""
    T, K, _ = img.shape
    out, gt = [], []
    for t in range(T):
        rows, g = [], []
        for k in range(K):
            cx, cy = img[t, k]
            s = sizes[k]
            if not (s / 2 <= cx <= W - s / 2 and s / 2 <= cy <= H - s / 2):
                continue
            g.append((k, cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2))
            if occluded[t, k]:
                continue
            if rng.random() > (0.80 if s < 12 else 0.93):
                continue
            conf = float(np.clip(rng.beta(8, 2.5), 0.23, 0.99))
            sig = 1.2 if noise == "N1" else 0.5 + 4.0 * (1 - conf)
            ex, ey = rng.normal(0, sig, 2)
            ww, hh = s * (1 + rng.normal(0, 0.06)), s * (1 + rng.normal(0, 0.06))
            rows.append((cx + ex - ww / 2, cy + ey - hh / 2, cx + ex + ww / 2, cy + ey + hh / 2, conf))
        for _ in range(rng.poisson(0.08)):
            s = rng.uniform(8, 20)
            x, y = rng.uniform(s, W - s), rng.uniform(s, H - s)
            rows.append((x - s / 2, y - s / 2, x + s / 2, y + s / 2, rng.uniform(0.23, 0.5)))
        out.append(np.array(rows, float).reshape(-1, 5))
        gt.append(np.array(g, float).reshape(-1, 5))
    return out, gt


class Renderer:
    """Textured sky crop following the camera offset, with dark UAV blobs."""

    def __init__(self, rng):
        big = rng.normal(0, 1, (900, 1200)).astype(np.float32)
        big = cv2.GaussianBlur(big, (0, 0), 9)
        big = (big - big.min()) / (big.max() - big.min())
        grad = np.linspace(0.55, 0.85, 900)[:, None]
        self.bg = np.clip(255 * (0.6 * grad + 0.4 * big), 0, 255).astype(np.uint8)
        self.bg = cv2.cvtColor(self.bg, cv2.COLOR_GRAY2BGR)
        self.origin = np.array([280.0, 210.0])

    def frame(self, cam_off, boxes):
        ox, oy = np.clip(self.origin + cam_off, 0, [1200 - W - 1, 900 - H - 1]).astype(int)
        img = self.bg[oy:oy + H, ox:ox + W].copy()
        for x1, y1, x2, y2 in boxes:
            c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            ax = (max(2, int((x2 - x1) / 2)), max(2, int((y2 - y1) / 3)))
            cv2.ellipse(img, c, ax, 0, 0, 360, (40, 40, 40), -1)
        return img


def make_sequence(scen, seed, noise="N1"):
    rng = np.random.default_rng(seed)
    img, sizes, cam, occ = trajectories(scen, rng)
    dets, gt = detections(img, sizes, occ, rng, noise)
    dt = (1 / 30) * (1 + rng.uniform(-0.15, 0.15, N_FRAMES))
    return {"dets": dets, "gt": gt, "cam": cam - cam[0], "dt": dt, "occluded": occ,
            "renderer_seed": seed}
