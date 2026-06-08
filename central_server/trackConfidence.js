// Per-track confidence scorer.
//
// Each global track gets a rolling [0, 100] confidence number reflecting
// "how much should the operator trust this contact is a real, well-tracked
// drone vs. a transient ghost?" The score is a weighted average of five
// independent components, each in [0, 100]:
//
//   contributors  — single observer 50, dual 100, both-LOCKED 110 (clamped)
//   sigma         — small fused σ → high confidence; ramps 100→0 over [1, 25] m
//   innovation    — small obs-vs-prediction residual → high confidence;
//                   ramps 100→0 over normalized residual [0, 5]; EMA-smoothed
//   age           — fresh tracks (< AGE_RAMP_S) score lower so brief ghosts
//                   don't pop as 100/100
//   motion        — velocity-direction stability over the last N samples;
//                   zigzagging drops the score
//
// Final = weighted sum (weights below). The score itself is EMA-smoothed so it
// doesn't flicker each tick.

const W = {
    contributors: 0.30,
    sigma:        0.20,
    innovation:   0.20,
    age:          0.10,
    motion:       0.20,
};

const AGE_RAMP_S    = 3.0;     // tracks younger than this get penalised
const SIGMA_GOOD_M  = 1.0;     // σ at/below this → 100
const SIGMA_BAD_M   = 25.0;    // σ at/above this → 0
const INNOV_GOOD    = 0.5;     // normalised innovation at/below this → 100
const INNOV_BAD     = 5.0;     // at/above this → 0
const INNOV_EMA_A   = 0.30;    // EMA for the innovation channel
const SCORE_EMA_A   = 0.40;    // EMA for the final published confidence
const MOTION_WIN    = 5;       // number of velocity samples for direction stability
const MOTION_EPS    = 0.5;     // m/s; velocities below this don't contribute to stability

function linRamp(value, hi, lo) {
    // Returns 100 when value ≤ hi, 0 when value ≥ lo, linear in between.
    if (value <= hi) return 100;
    if (value >= lo) return 0;
    return 100 * (1 - (value - hi) / (lo - hi));
}

class ConfidenceScorer {
    constructor(firstSeenMs) {
        this.firstSeenMs = firstSeenMs || Date.now();
        this._innovEMA = 0;        // smoothed normalised innovation
        this._scoreEMA = null;     // smoothed published confidence
        this._velHistory = [];     // last MOTION_WIN velocity vectors for stability check
    }

    // step():
    //   contribInfo: { count: int, allLocked: bool }
    //   fusedSigma:  number, m
    //   innovation_norm: number (|observation - prediction| / σ_combined). May
    //                    be null on the very first observation.
    //   velocity: {x,y,z} or null
    //   nowMs: timestamp for age computation
    // Returns { confidence, components }.
    step(contribInfo, fusedSigma, innovation_norm, velocity, nowMs) {
        // -- contributors --
        let cContrib;
        if ((contribInfo.count || 0) <= 0) cContrib = 0;
        else if (contribInfo.count === 1) cContrib = 50;
        else cContrib = contribInfo.allLocked ? 100 : 90;

        // -- sigma --
        const cSigma = linRamp(Math.max(0, fusedSigma || 0), SIGMA_GOOD_M, SIGMA_BAD_M);

        // -- innovation (EMA) --
        if (typeof innovation_norm === 'number' && isFinite(innovation_norm)) {
            this._innovEMA = INNOV_EMA_A * innovation_norm + (1 - INNOV_EMA_A) * this._innovEMA;
        }
        const cInnov = linRamp(this._innovEMA, INNOV_GOOD, INNOV_BAD);

        // -- age ramp --
        const ageS = Math.max(0, ((nowMs || Date.now()) - this.firstSeenMs) / 1000);
        const cAge = Math.min(100, (ageS / AGE_RAMP_S) * 100);

        // -- motion stability (direction-of-velocity consistency) --
        if (velocity) {
            const sp = Math.hypot(velocity.x, velocity.y, velocity.z);
            if (sp > MOTION_EPS) {
                this._velHistory.push({ x: velocity.x / sp, y: velocity.y / sp, z: velocity.z / sp });
                if (this._velHistory.length > MOTION_WIN) this._velHistory.shift();
            }
        }
        let cMotion = 100;
        if (this._velHistory.length >= 2) {
            // Average pairwise cosine similarity → 1 means perfectly consistent direction.
            let dotSum = 0, n = 0;
            for (let i = 1; i < this._velHistory.length; i++) {
                const a = this._velHistory[i - 1], b = this._velHistory[i];
                dotSum += a.x * b.x + a.y * b.y + a.z * b.z;
                n++;
            }
            const cos = n > 0 ? dotSum / n : 1;
            cMotion = Math.max(0, Math.min(100, (cos + 1) * 50));
        }

        const components = {
            contributors: Math.round(cContrib),
            sigma:        Math.round(cSigma),
            innovation:   Math.round(cInnov),
            age:          Math.round(cAge),
            motion:       Math.round(cMotion),
        };
        const raw = W.contributors * cContrib + W.sigma * cSigma +
                    W.innovation * cInnov + W.age * cAge + W.motion * cMotion;

        if (this._scoreEMA === null) {
            this._scoreEMA = raw;
        } else {
            this._scoreEMA = SCORE_EMA_A * raw + (1 - SCORE_EMA_A) * this._scoreEMA;
        }
        return {
            confidence: Math.max(0, Math.min(100, Math.round(this._scoreEMA))),
            components,
        };
    }
}

module.exports = { ConfidenceScorer };
