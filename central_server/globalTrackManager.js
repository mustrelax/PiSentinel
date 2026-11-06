// Cross-node track association + fusion.
//
// Each Pi reports geolocationUpdate independently with its own track ID.
// When two Pis see the same drone, this manager joins their reports under a
// single global track. Position fusion:
//   - 2 contributors with valid azimuth + elevation + unit_pos:
//     3D closest-point-of-approach triangulation on the two ENU rays
//     (Becker 1999; Pages-Zamora 2002). Elevation is treated as a first-class
//     observable so altitude is recovered from angle geometry, NOT from the
//     monocular range channel — drone climbing no longer drags the ground
//     projection.
//   - Single contributor: re-project (x,y,z) from azimuth/elevation/range_m
//     anchored at the unit position. This decouples horizontal scaling and
//     vertical scaling — monocular-depth noise affects horiz and z
//     proportionally instead of swapping between them.
//   - 3+ or degenerate geometry: inverse-variance weighted fallback.
//
// The published track.fused is the raw geometric solution above with a
// light EMA for sub-meter jitter only. NO motion-mode Kalman filter sits
// between the observation and the output — real-time tracking needs
// observation latency under 200 ms, and any multi-model filter (IMM,
// CV-CA, etc.) lags multi-second manoeuvres badly enough that the next
// observation falls outside its own association gate and spawns ghost
// tracks. immFilter.js is retained as an offline-analysis module but
// is NOT wired into the live path.
//
// motion_mode (STATIONARY / HORIZONTAL / VERTICAL / BOTH) is derived
// post-hoc from the trail-window velocity — pure classifier, no
// filter, no latency on the published position.
//
// Engagement state (LOCKED vs PASSIVE) is carried per contributor and
// derived per global track: LOCKED if any contributor reports LOCKED,
// else PASSIVE. LOST is the lifecycle state for orphaned tracks.
//
// Re-merge / re-acquisition handling:
//   - When a track loses all contributors it is moved to `orphanedTracks` for
//     ORPHAN_TTL_MS, not hard-deleted. New incoming reports first try to match
//     against active tracks, then orphans, with a TIME-AWARE gate that inflates
//     σ by a max-velocity motion uncertainty term and uses constant-velocity
//     prediction. A successful orphan match RESURRECTS the global track with
//     the same globalId — UI sees a continuous trail.
//
// Public API:
//   ingest(report) -> {globalId, fused, contributors[], method, timestamp, ...}
//   pruneStale(maxAgeMs) -> [removedGlobalIds]
//   removeUnit(deviceName) -> [removedGlobalIds]
//   snapshot() -> [...]

// NOTE: IMMFilter is retained on disk (immFilter.js) for offline trajectory
// analysis but is intentionally NOT required here — see header for rationale.

const { ConfidenceScorer } = require('./trackConfidence');

const ASSOC_GATE = 3.0;             // 3-sigma normalized distance for association
const AMBIGUITY_RATIO = 1.5;        // best dNorm must beat second-best by this factor to win;
                                    // otherwise the new observation is treated as ambiguous and
                                    // spawns a fresh global (mergeConverged later consolidates).
                                    // This is what kills swap-into-wrong-drone during fast
                                    // multi-target manoeuvres.
const VEL_EMA_ALPHA = 0.4;          // EMA on trail-derived velocity. Smoother velocity → tighter
                                    // _gateMatch prediction → fewer mis-associations under jitter.
// Triangulate only if the ray intersection angle is >= 15 deg (simulation study,
// manuscript Sec. 4.4); otherwise fall back to inverse-variance weighting.
const PARALLEL_SIN_THRESH = Math.sin(15 * Math.PI / 180);
const SIN_FLOOR = 0.1;              // min sin(intersection_angle) for GDOP scaling
const MIN_SIGMA = 0.5;              // floor on per-contributor σ to avoid div-by-zero
const ORPHAN_TTL_MS = 5000;         // orphan track lifetime before hard delete
const MAX_TARGET_VELOCITY_M_S = 35; // m/s — drone upper bound for motion-uncertainty gating
const TRAIL_MAX = 8;                // last N fused samples kept for velocity estimation
const VEL_WINDOW = 3;               // # of trail points used to estimate velocity
const OUTPUT_EMA_ALPHA = 0.6;       // EMA on emitted fused position (high → responsive)
const MOTION_MODE_VTHRESH = 0.6;    // m/s — below this on an axis, treat as still on that axis
const STATIONARY_DEADBAND_FRAC = 0.5; // fraction of σ; STATIONARY tracks ignore moves smaller than
                                    // this (eliminates "back and forth" on the map from residual
                                    // bbox/detector jitter that survives Pi-side smoothing)
const STATIONARY_DEADBAND_MIN_M = 0.5; // absolute floor on the dead-band so very-low-σ tracks still
                                    // get protected
const STALE_CONTRIB_MS = 600;       // ms — if a device's existing contribution is older than this,
                                    // a fresh report from the same device is allowed to REPLACE it
                                    // in the same global track (Pi's MOT lost+re-detected the drone
                                    // with a new internal trackId). Prevents ghost-twin globals on
                                    // brief detection drops.
const BEARING_SIGMA_DEG = 1.0;      // EO bearing precision (matches Pi GEOLOC_BEARING_UNCERTAINTY_DEG).
                                    // Used by 3D triangulation only — range error cancels there, so
                                    // contributor.uncertainty_m (which is dominated by range error
                                    // at long range) over-states σ by 5-10×.
const BEARING_SIGMA_RAD = BEARING_SIGMA_DEG * Math.PI / 180;

class GlobalTrackManager {
    constructor() {
        this.globalTracks = new Map();      // globalId -> {id, contributors:Map, fused, _trail, _velocity, _lastUpdateTs, ...}
        this.orphanedTracks = new Map();    // globalId -> {lastFused, lastVelocity, orphanedAt, lastMethod}
        this.contributorIndex = new Map();  // "device::trackId" -> globalId
        this._nextId = 1;
    }

    ingest(report) {
        if (!report || !report.deviceName || report.trackId == null || !report.world) return null;

        const now = report.ts || Date.now();
        const sigma = Math.max(report.uncertainty_m || 0, MIN_SIGMA);
        const contribKey = `${report.deviceName}::${report.trackId}`;

        const contrib = {
            deviceName: report.deviceName,
            trackId: report.trackId,
            world: { x: report.world.x, y: report.world.y, z: report.world.z },
            uncertainty_m: sigma,
            azimuth_world: report.azimuth_world || 0,
            elevation: typeof report.elevation === 'number' ? report.elevation : 0,
            range_m: typeof report.range_m === 'number' ? report.range_m : 0,
            track_state: report.track_state === 'PASSIVE' ? 'PASSIVE' : 'LOCKED',
            unit_pos: report.unit_position
                ? { x: report.unit_position.x, y: report.unit_position.y, z: report.unit_position.z || 0 }
                : null,
            lastUpdate: now,
        };

        // Step 1: re-key — already a contributor of some active global track?
        let globalId = this.contributorIndex.get(contribKey);
        let track = globalId != null ? this.globalTracks.get(globalId) : null;
        if (track) {
            track.contributors.set(report.deviceName, contrib);
            this._recomputeFusion(track);
            this._appendTrail(track, now);
            track._lastUpdateTs = now;
            return this._snapshot(track, /*dNorm=*/0, /*resurrected=*/false);
        }

        // Step 2A: time-aware gating against active global tracks.
        // We normally skip globals where this device already contributes — a Pi
        // shouldn't double-count. EXCEPTION: if that device's existing
        // contribution is older than STALE_CONTRIB_MS, this is almost certainly
        // the Pi's MOT having lost+re-detected the same drone with a fresh
        // internal trackId. Allow the new report to compete for that global so
        // it can REPLACE the stale contributor in place. Without this fix you
        // get a ghost twin global on every detection drop during fast motion.
        //
        // Ambiguity handling: track the SECOND-best match too. If best and
        // second-best are within AMBIGUITY_RATIO of each other, the
        // observation could plausibly belong to two existing globals — picking
        // one greedily would corrupt that track with another drone's data. In
        // the ambiguous case we spawn a fresh global instead, and let
        // mergeConverged consolidate later once the drones move enough to
        // disambiguate.
        let bestId = null, bestDNorm = Infinity, bestSource = null;
        let secondDNorm = Infinity;
        for (const [gid, gtrack] of this.globalTracks) {
            const existing = gtrack.contributors.get(report.deviceName);
            if (existing && (now - existing.lastUpdate) < STALE_CONTRIB_MS) continue;
            const match = this._gateMatch(report, sigma, gtrack.fused, gtrack._velocity, gtrack._lastUpdateTs, now);
            if (match.dNorm >= ASSOC_GATE) continue;
            if (match.dNorm < bestDNorm) {
                secondDNorm = bestDNorm;
                bestDNorm = match.dNorm;
                bestId = gid;
                bestSource = 'active';
            } else if (match.dNorm < secondDNorm) {
                secondDNorm = match.dNorm;
            }
        }

        // Step 2B: same gating against orphaned tracks (resurrection candidates).
        // Orphans don't participate in the ambiguity check vs. active globals —
        // resurrection of the SAME globalId is always preferable to spawning a
        // new one, even if some active global is technically a closer match.
        let orphanBestId = null, orphanBestDNorm = Infinity;
        for (const [gid, orph] of this.orphanedTracks) {
            const ageMs = now - orph.orphanedAt;
            if (ageMs > ORPHAN_TTL_MS) continue;
            const match = this._gateMatch(report, sigma, orph.lastFused, orph.lastVelocity, orph.orphanedAt, now);
            if (match.dNorm < ASSOC_GATE && match.dNorm < orphanBestDNorm) {
                orphanBestDNorm = match.dNorm;
                orphanBestId = gid;
            }
        }

        // Reconcile: orphan match wins outright if it exists; otherwise apply
        // the active-side ambiguity check.
        if (orphanBestId != null) {
            bestId = orphanBestId;
            bestDNorm = orphanBestDNorm;
            bestSource = 'orphan';
        } else if (bestSource === 'active' && secondDNorm < AMBIGUITY_RATIO * Math.max(bestDNorm, 1e-3)) {
            console.log(
                `[GTM] ambiguous match for ${report.deviceName}::t${report.trackId} ` +
                `(best dNorm=${bestDNorm.toFixed(2)} vs 2nd=${secondDNorm.toFixed(2)}) — ` +
                `spawning new global, mergeConverged will consolidate if appropriate`);
            bestId = null;
            bestSource = null;
        }

        if (bestId != null && bestSource === 'active') {
            track = this.globalTracks.get(bestId);
            // If this device already had a (stale) contributor here under a
            // different trackId, evict it first so the contributor index
            // doesn't keep pointing at a dead trackId.
            const old = track.contributors.get(report.deviceName);
            if (old && old.trackId !== report.trackId) {
                this.contributorIndex.delete(`${report.deviceName}::${old.trackId}`);
                console.log(
                    `[GTM] re-key ${report.deviceName} t${old.trackId} -> t${report.trackId} ` +
                    `on G${bestId} (existing stale by ${(now - old.lastUpdate)|0}ms)`);
            }
            track.contributors.set(report.deviceName, contrib);
            this.contributorIndex.set(contribKey, bestId);
            this._recomputeFusion(track);
            this._appendTrail(track, now);
            track._lastUpdateTs = now;
            return this._snapshot(track, bestDNorm, /*resurrected=*/false);
        }

        if (bestId != null && bestSource === 'orphan') {
            // Resurrect: pull orphan back into active globalTracks with the SAME id.
            const orph = this.orphanedTracks.get(bestId);
            this.orphanedTracks.delete(bestId);
            track = {
                id: bestId,
                contributors: new Map([[report.deviceName, contrib]]),
                fused: { ...orph.lastFused },
                firstSeen: now,
                _method: orph.lastMethod || 'single',
                _trail: [{ x: orph.lastFused.x, y: orph.lastFused.y, z: orph.lastFused.z, t: orph.orphanedAt }],
                _velocity: orph.lastVelocity || null,
                _lastUpdateTs: now,
                _resurrectedAt: now,
                _orphanAgeMs: now - orph.orphanedAt,
            };
            this.globalTracks.set(bestId, track);
            this.contributorIndex.set(contribKey, bestId);
            this._recomputeFusion(track);
            this._appendTrail(track, now);
            return this._snapshot(track, bestDNorm, /*resurrected=*/true);
        }

        // Step 3: brand-new global track.
        // Initialize fused via _recomputeFusion so the single-observer branch
        // re-projects from azimuth/elevation/range — the Pi's raw (x,y,z) may
        // be noisy and we don't want to seed downstream consumers with that.
        globalId = this._nextId++;
        track = {
            id: globalId,
            contributors: new Map([[report.deviceName, contrib]]),
            // fused + _method intentionally null — _recomputeFusion's seed
            // branch will populate both from the projected solution. Pre-
            // seeding with the Pi's raw (x,y,z) would defeat that guard and
            // cause EMA to blend noisy raw data into the first few samples,
            // manifesting as phantom velocity even on a stationary drone.
            fused: null,
            firstSeen: now,
            _method: null,
            _trail: [],
            _velocity: null,
            _lastUpdateTs: now,
        };
        this.globalTracks.set(globalId, track);
        this.contributorIndex.set(contribKey, globalId);
        this._recomputeFusion(track);
        this._appendTrail(track, now);
        // Eagerly consolidate: if this new global happens to coincide with an
        // existing one (e.g. ambiguous Step-2A rejection that turned out to be
        // the same drone), mergeConverged on the next periodic tick handles it.
        // We do NOT call mergeConverged here because the new track only has one
        // sample — its velocity is null and sigma is large; consolidation is
        // safer once a couple of updates have refined both tracks' positions.
        return this._snapshot(track, 0, /*resurrected=*/false);
    }

    // Track-to-track merge. Two active globals that have CONVERGED (now sit
    // within 3σ of each other and don't share a contributor device) get fused
    // into one — the lower globalId wins; the loser's contributors are
    // re-homed and contributorIndex updated. Fixes "two tracks for one drone"
    // when their initial estimates were too far apart for ingest-time gating
    // but later positions converge.
    //
    // Returns array of removed (loser) globalIds for the caller to broadcast.
    mergeConverged() {
        const ids = Array.from(this.globalTracks.keys()).sort((a, b) => a - b);
        const removed = [];
        for (let i = 0; i < ids.length; i++) {
            const a = this.globalTracks.get(ids[i]);
            if (!a) continue;
            for (let j = i + 1; j < ids.length; j++) {
                const b = this.globalTracks.get(ids[j]);
                if (!b) continue;
                // Skip if any device contributes to both (would imply same Pi
                // is reporting two separate tracks of the same drone — keep
                // them separate, that's a Pi-side ambiguity).
                let overlap = false;
                for (const dev of a.contributors.keys()) {
                    if (b.contributors.has(dev)) { overlap = true; break; }
                }
                if (overlap) continue;

                // 3σ Mahalanobis-lite (no time prediction here — both tracks
                // are active so their fused positions are current).
                const d = Math.hypot(
                    a.fused.x - b.fused.x,
                    a.fused.y - b.fused.y,
                    a.fused.z - b.fused.z);
                const sigmaCombined = Math.hypot(a.fused.sigma_m, b.fused.sigma_m);
                const dNorm = sigmaCombined > 0 ? d / sigmaCombined : Infinity;
                if (dNorm >= ASSOC_GATE) continue;

                // Merge: a (lower id) wins, b's contributors move to a.
                for (const [dev, c] of b.contributors) {
                    a.contributors.set(dev, c);
                    this.contributorIndex.set(`${dev}::${c.trackId}`, a.id);
                }
                this.globalTracks.delete(b.id);
                this._recomputeFusion(a);
                this._appendTrail(a, Date.now());
                a._lastUpdateTs = Date.now();
                removed.push(b.id);
                console.log(
                    `[GTM] merge G${b.id} -> G${a.id} (d=${d.toFixed(1)}m, ` +
                    `d_norm=${dNorm.toFixed(2)}, contributors_after=${a.contributors.size})`);
                // Restart inner loop because a's fused has changed; cascade merges possible.
                break;
            }
        }
        return removed;
    }

    pruneStale(maxAgeMs = 2000) {
        const now = Date.now();
        const cutoff = now - maxAgeMs;
        const removedIds = [];

        // 1) Stale contributor pruning. A track that loses its last contributor
        //    becomes an orphan (soft-delete) instead of being hard-removed.
        for (const [gid, track] of this.globalTracks) {
            const stale = [];
            for (const [dev, c] of track.contributors) {
                if (c.lastUpdate < cutoff) stale.push([dev, c.trackId]);
            }
            for (const [dev, tid] of stale) {
                track.contributors.delete(dev);
                this.contributorIndex.delete(`${dev}::${tid}`);
            }
            if (track.contributors.size === 0) {
                // Soft-delete: snapshot state into orphan map for re-acquisition.
                this.orphanedTracks.set(gid, {
                    lastFused: { ...track.fused },
                    lastVelocity: track._velocity || null,
                    orphanedAt: now,
                    lastMethod: track._method || 'single',
                });
                this.globalTracks.delete(gid);
                // No removedIds yet — UI keeps the marker through orphan TTL.
            } else if (stale.length) {
                this._recomputeFusion(track);
            }
        }

        // 2) Hard-delete orphans whose TTL has expired.
        for (const [gid, orph] of this.orphanedTracks) {
            if (now - orph.orphanedAt > ORPHAN_TTL_MS) {
                this.orphanedTracks.delete(gid);
                removedIds.push(gid);
            }
        }
        return removedIds;
    }

    // Drop a single (device, trackId) contributor. Called when a Pi explicitly
    // reports LOST for that trackId — beats waiting STALE_AGE_MS for pruneStale
    // to evict it, which is what causes ghost-twin globals on brief detection
    // drops with fast re-acquisition.
    // Returns the globalId removed (if the track had no other contributors), or null.
    removeContributor(deviceName, trackId) {
        if (deviceName == null || trackId == null) return null;
        const contribKey = `${deviceName}::${trackId}`;
        const gid = this.contributorIndex.get(contribKey);
        if (gid == null) return null;
        this.contributorIndex.delete(contribKey);
        const track = this.globalTracks.get(gid);
        if (!track) return null;
        const c = track.contributors.get(deviceName);
        if (!c || c.trackId !== trackId) return null;
        track.contributors.delete(deviceName);
        if (track.contributors.size === 0) {
            // Soft-delete to orphan map so the same drone re-acquired with a
            // new Pi-side trackId can resurrect the same globalId.
            this.orphanedTracks.set(gid, {
                lastFused: { ...track.fused },
                lastVelocity: track._velocity || null,
                orphanedAt: Date.now(),
                lastMethod: track._method || 'single',
            });
            this.globalTracks.delete(gid);
            return null; // UI keeps marker; orphan TTL applies.
        }
        this._recomputeFusion(track);
        return null;
    }

    removeUnit(deviceName) {
        const removedIds = [];
        const now = Date.now();
        for (const [gid, track] of this.globalTracks) {
            const c = track.contributors.get(deviceName);
            if (!c) continue;
            track.contributors.delete(deviceName);
            this.contributorIndex.delete(`${deviceName}::${c.trackId}`);
            if (track.contributors.size === 0) {
                // Soft-delete on disconnect too — peer may reconnect briefly.
                this.orphanedTracks.set(gid, {
                    lastFused: { ...track.fused },
                    lastVelocity: track._velocity || null,
                    orphanedAt: now,
                    lastMethod: track._method || 'single',
                });
                this.globalTracks.delete(gid);
            } else {
                this._recomputeFusion(track);
            }
        }
        return removedIds;
    }

    snapshot() {
        return Array.from(this.globalTracks.values()).map(t => this._snapshot(t, null, false));
    }

    _snapshot(track, dNorm, resurrected) {
        // Published fused = raw triangulation + light EMA (see _recomputeFusion).
        // motion_mode is a post-hoc classification of the trail-derived velocity,
        // not a filter — so it adds zero latency to the position itself.
        return {
            globalId: track.id,
            fused: { ...track.fused },
            velocity: track._velocity ? { ...track._velocity } : null,
            contributors: Array.from(track.contributors.values()).map(c => ({
                deviceName: c.deviceName,
                trackId: c.trackId,
                world: { ...c.world },
                uncertainty_m: c.uncertainty_m,
                azimuth_world: c.azimuth_world,
                elevation: c.elevation,
                range_m: c.range_m,
                track_state: c.track_state,
                unit_pos: c.unit_pos ? { ...c.unit_pos } : null,
            })),
            method: track._method || 'single',
            motion_mode: track._motionMode || 'BOTH',
            engagement_state: track._engagementState || 'PASSIVE',
            confidence: typeof track._confidence === 'number' ? track._confidence : null,
            confidence_components: track._confidenceComponents || null,
            d_norm: dNorm,
            resurrected: !!resurrected,
            orphan_age_ms: track._orphanAgeMs || 0,
            timestamp: Date.now(),
        };
    }

    // Time-aware σ-weighted gating. Predicts where the global track would be
    // NOW given its last fused state and velocity, then compares against the
    // new report. σ is inflated by a motion-uncertainty term that grows with
    // dt — the longer it has been since the last update, the wider the gate.
    _gateMatch(report, sigmaNew, fused, velocity, lastTs, nowTs) {
        const dt_s = Math.max(0, (nowTs - (lastTs || nowTs)) / 1000);
        // Predict global position with constant-velocity model.
        const vx = velocity ? velocity.x : 0;
        const vy = velocity ? velocity.y : 0;
        const vz = velocity ? velocity.z : 0;
        const px = fused.x + vx * dt_s;
        const py = fused.y + vy * dt_s;
        const pz = fused.z + vz * dt_s;

        const d = Math.hypot(report.world.x - px, report.world.y - py, report.world.z - pz);
        // Motion uncertainty: assume target may have moved by up to half of
        // MAX_TARGET_VELOCITY * dt orthogonal to the velocity prediction.
        const motionSigma = MAX_TARGET_VELOCITY_M_S * dt_s * 0.5;
        const sigmaCombined = Math.sqrt(sigmaNew * sigmaNew + fused.sigma_m * fused.sigma_m + motionSigma * motionSigma);
        const dNorm = sigmaCombined > 0 ? d / sigmaCombined : Infinity;
        return { d, dNorm, sigmaCombined, dt_s };
    }

    _appendTrail(track, now) {
        if (!track._trail) track._trail = [];
        track._trail.push({ x: track.fused.x, y: track.fused.y, z: track.fused.z, t: now });
        if (track._trail.length > TRAIL_MAX) track._trail.shift();
        track._velocity = this._estimateVelocity(track);
        track._motionMode = this._classifyMotionMode(track._velocity);
    }

    // Constant-velocity estimate from the last VEL_WINDOW trail samples, with
    // EMA smoothing on top. The raw finite-difference is jittery on a moving
    // drone (triangulation σ × 1/dt amplifies frame-to-frame noise); the
    // smoothed value is what _gateMatch uses to predict where the track will
    // be on the next ingest, so stability here directly cuts gate-failure
    // ghost-spawn events.
    _estimateVelocity(track) {
        if (!track._trail || track._trail.length < 2) return track._smoothedVel || null;
        const trail = track._trail;
        const newest = trail[trail.length - 1];
        const oldest = trail[Math.max(0, trail.length - VEL_WINDOW)];
        const dt_s = (newest.t - oldest.t) / 1000;
        if (dt_s <= 0) return track._smoothedVel || null;
        const raw = {
            x: (newest.x - oldest.x) / dt_s,
            y: (newest.y - oldest.y) / dt_s,
            z: (newest.z - oldest.z) / dt_s,
        };
        if (!track._smoothedVel) {
            track._smoothedVel = raw;
        } else {
            const a = VEL_EMA_ALPHA;
            track._smoothedVel = {
                x: a * raw.x + (1 - a) * track._smoothedVel.x,
                y: a * raw.y + (1 - a) * track._smoothedVel.y,
                z: a * raw.z + (1 - a) * track._smoothedVel.z,
            };
        }
        return track._smoothedVel;
    }

    _recomputeFusion(track) {
        const contribs = Array.from(track.contributors.values());
        if (contribs.length === 0) return;

        let rawFused = null;
        let method = null;

        if (contribs.length === 1) {
            // Single observer — re-project from azimuth/elevation/range anchored at
            // the unit position. Independent of the Pi's monocular (x,y,z) so a
            // noisy slant range scales horiz and z proportionally rather than
            // swapping between them.
            rawFused = this._projectFromAngles(contribs[0]);
            method = 'single';
        } else if (contribs.length === 2 &&
                   contribs[0].unit_pos && contribs[1].unit_pos &&
                   this._hasFiniteElevation(contribs[0]) &&
                   this._hasFiniteElevation(contribs[1])) {
            const tri = this._triangulate3D(contribs[0], contribs[1]);
            if (tri) {
                rawFused = tri;
                method = 'bearing3D';
            }
        }

        // Fallback for 3+ contributors or degenerate 2-contributor geometry:
        // inverse-variance weighted position fusion. The per-contributor
        // (x,y,z) at this point is already the Pi's own angle-derived
        // estimate (compute_world_position), so this is a sane fallback.
        if (!rawFused) {
            let sumW = 0, sumWX = 0, sumWY = 0, sumWZ = 0;
            for (const c of contribs) {
                const w = 1 / (c.uncertainty_m ** 2);
                sumW += w;
                sumWX += w * c.world.x;
                sumWY += w * c.world.y;
                sumWZ += w * c.world.z;
            }
            rawFused = {
                x: sumWX / sumW,
                y: sumWY / sumW,
                z: sumWZ / sumW,
                sigma_m: 1 / Math.sqrt(sumW),
            };
            method = 'weighted';
        }

        // Engagement state: LOCKED dominates PASSIVE (any contributor LOCKED
        // means at least one Pi is actively servoing onto this drone).
        let engagementState = 'PASSIVE';
        let allLocked = contribs.length > 0;
        for (const c of contribs) {
            if (c.track_state === 'LOCKED') engagementState = 'LOCKED';
            else allLocked = false;
        }

        // Innovation (observation vs prior prediction) for confidence scoring.
        // Computed BEFORE the EMA so we measure raw disagreement, not the
        // smoothed disagreement (which would saturate at ~0 by construction).
        let innovationNorm = null;
        if (track.fused && track._lastUpdateTs) {
            const dt = Math.max(0, (Date.now() - track._lastUpdateTs) / 1000);
            const vx = track._velocity ? track._velocity.x : 0;
            const vy = track._velocity ? track._velocity.y : 0;
            const vz = track._velocity ? track._velocity.z : 0;
            const px = track.fused.x + vx * dt;
            const py = track.fused.y + vy * dt;
            const pz = track.fused.z + vz * dt;
            const d = Math.hypot(rawFused.x - px, rawFused.y - py, rawFused.z - pz);
            const sCombined = Math.hypot(rawFused.sigma_m || MIN_SIGMA, track.fused.sigma_m || MIN_SIGMA);
            innovationNorm = sCombined > 0 ? d / sCombined : null;
        }

        // Stationary dead-band: when the track's previous classification is
        // STATIONARY, snap rawFused onto the current fused position if the
        // delta is within DEADBAND_FRAC × σ (with absolute floor). This kills
        // the residual back-and-forth on the map that comes from detector-
        // level bbox jitter surviving Pi-side smoothing — a hovering drone
        // SHOULD render as a hovering marker. Defence-grade stationary-
        // target surveillance practice (Bar-Shalom & Li 1993, §7.4).
        if (track.fused && track._motionMode === 'STATIONARY') {
            const dx = rawFused.x - track.fused.x;
            const dy = rawFused.y - track.fused.y;
            const dz = rawFused.z - track.fused.z;
            const d = Math.hypot(dx, dy, dz);
            const band = Math.max(
                STATIONARY_DEADBAND_MIN_M,
                STATIONARY_DEADBAND_FRAC * (track.fused.sigma_m || STATIONARY_DEADBAND_MIN_M)
            );
            if (d < band) {
                rawFused.x = track.fused.x;
                rawFused.y = track.fused.y;
                rawFused.z = track.fused.z;
                // sigma_m may still update — it's a quality metric, not a position move.
            }
        }

        // Published fused state = raw geometric solution + LIGHT EMA only.
        // Real-time tracking of a moving drone requires the next observation
        // gate against a position close to truth; any heavier filter (IMM, CV
        // Kalman) lags manoeuvres enough to break association on the next
        // ingest and spawn ghost tracks. The trail-based velocity estimator
        // below (used by _gateMatch's predict-forward) is what compensates
        // for the inter-sample interval — NOT this smoother.
        if (!track.fused || track._method !== method) {
            // First sample, or fusion method changed (e.g. second Pi just
            // joined → bearing3D supersedes single). Seed directly; no EMA.
            track.fused = { ...rawFused };
        } else {
            const a = OUTPUT_EMA_ALPHA;
            track.fused = {
                x: a * rawFused.x + (1 - a) * track.fused.x,
                y: a * rawFused.y + (1 - a) * track.fused.y,
                z: a * rawFused.z + (1 - a) * track.fused.z,
                sigma_m: rawFused.sigma_m,
            };
        }
        track._rawFused = rawFused;
        track._method = method;
        track._engagementState = engagementState;

        // Confidence scoring — must run AFTER track.fused is updated so the
        // σ-based component uses the freshly-published value.
        if (!track._confidenceScorer) {
            track._confidenceScorer = new ConfidenceScorer(track.firstSeen || Date.now());
        }
        const conf = track._confidenceScorer.step(
            { count: contribs.length, allLocked },
            track.fused.sigma_m,
            innovationNorm,
            track._velocity,
            Date.now(),
        );
        track._confidence = conf.confidence;
        track._confidenceComponents = conf.components;
    }

    // Classify motion mode from the trail-derived velocity. Pure post-hoc
    // classifier — does NOT smooth or alter the published position.
    _classifyMotionMode(velocity) {
        if (!velocity) return 'BOTH';
        const horiz = Math.hypot(velocity.x, velocity.y);
        const vert = Math.abs(velocity.z);
        const horizStill = horiz < MOTION_MODE_VTHRESH;
        const vertStill = vert < MOTION_MODE_VTHRESH;
        if (horizStill && vertStill) return 'STATIONARY';
        if (horizStill) return 'VERTICAL';
        if (vertStill) return 'HORIZONTAL';
        return 'BOTH';
    }

    _hasFiniteElevation(c) {
        return typeof c.elevation === 'number' && Math.abs(c.elevation) < 89.0;
    }

    // Project a single observation to ENU using unit position + spherical
    // bearing. Bearing convention: 0=N (+y), 90=E (+x). Elevation is
    // signed degrees, +up.
    _projectFromAngles(c) {
        const az = (c.azimuth_world || 0) * Math.PI / 180;
        const el = (c.elevation || 0) * Math.PI / 180;
        const range = c.range_m > 0 ? c.range_m : Math.hypot(
            c.world.x - (c.unit_pos ? c.unit_pos.x : 0),
            c.world.y - (c.unit_pos ? c.unit_pos.y : 0),
            c.world.z - (c.unit_pos ? c.unit_pos.z : 0));
        if (!c.unit_pos || range <= 0) {
            return { x: c.world.x, y: c.world.y, z: c.world.z, sigma_m: c.uncertainty_m };
        }
        const horiz = range * Math.cos(el);
        return {
            x: c.unit_pos.x + horiz * Math.sin(az),
            y: c.unit_pos.y + horiz * Math.cos(az),
            z: c.unit_pos.z + range * Math.sin(el),
            sigma_m: c.uncertainty_m,
        };
    }

    // 3D ray closest-point-of-approach.
    // Two ENU rays r_i(t) = O_i + t · d_i, d_i = (cos(el)·sin(az),
    // cos(el)·cos(az), sin(el)). Solve min |r1 - r2|² → 2×2 normal equations
    // on (t1, t2). Fused point = midpoint of r1(t1*) and r2(t2*).
    //
    // Uncertainty model (Foy 1976; Becker 1999): in bearings-only
    // triangulation, RANGE error CANCELS. The position uncertainty is set by
    // bearing precision and geometry, not by the contributors' reported σ
    // (which is dominated by their range estimates). Per observer i, the
    // cross-bearing σ ≈ σ_bearing × |t_i|. Combined and scaled by 1/sin(angle)
    // is the GDOP-conditioned positional σ. Miss distance gets folded in as a
    // residual consistency penalty.
    _triangulate3D(c1, c2) {
        const az1 = c1.azimuth_world * Math.PI / 180;
        const az2 = c2.azimuth_world * Math.PI / 180;
        const el1 = c1.elevation * Math.PI / 180;
        const el2 = c2.elevation * Math.PI / 180;
        const d1 = {
            x: Math.cos(el1) * Math.sin(az1),
            y: Math.cos(el1) * Math.cos(az1),
            z: Math.sin(el1),
        };
        const d2 = {
            x: Math.cos(el2) * Math.sin(az2),
            y: Math.cos(el2) * Math.cos(az2),
            z: Math.sin(el2),
        };
        const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
        const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
        const p1 = c1.unit_pos, p2 = c2.unit_pos;

        const d1d2 = dot(d1, d2);
        // |sin(angle between rays)|² = 1 - cos² (both rays are unit vectors)
        const sin2 = Math.max(0, 1 - d1d2 * d1d2);
        if (sin2 < PARALLEL_SIN_THRESH * PARALLEL_SIN_THRESH) return null;

        const w0 = sub(p1, p2);
        const denom = 1 - d1d2 * d1d2;
        const a = dot(d1, w0);
        const b = dot(d2, w0);
        // t1 = (d1d2·b - a) / denom ;  t2 = (b - d1d2·a) / denom
        const t1 = (d1d2 * b - a) / denom;
        const t2 = (b - d1d2 * a) / denom;
        if (t1 < 0 || t2 < 0) return null;  // intersection behind one of the cameras

        const pt1 = { x: p1.x + t1 * d1.x, y: p1.y + t1 * d1.y, z: p1.z + t1 * d1.z };
        const pt2 = { x: p2.x + t2 * d2.x, y: p2.y + t2 * d2.y, z: p2.z + t2 * d2.z };
        const miss = Math.hypot(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z);

        // Bearings-only positional σ — independent of monocular range error.
        const sin_int = Math.max(Math.sqrt(sin2), SIN_FLOOR);
        const sigma1 = BEARING_SIGMA_RAD * Math.abs(t1);
        const sigma2 = BEARING_SIGMA_RAD * Math.abs(t2);
        const sigmaBearing = Math.hypot(sigma1, sigma2);
        const sigma_m = Math.max(MIN_SIGMA, sigmaBearing / sin_int + 0.5 * miss);

        return {
            x: 0.5 * (pt1.x + pt2.x),
            y: 0.5 * (pt1.y + pt2.y),
            z: 0.5 * (pt1.z + pt2.z),
            sigma_m,
            miss,
        };
    }
}

module.exports = { GlobalTrackManager };
