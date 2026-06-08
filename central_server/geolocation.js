// Server-side geolocation relay + cross-node track fusion.
//
// Each Pi reports geolocationUpdate with its own track ID. We:
//   1) fan out the raw event to every connected socket (single-Pi UI legacy)
//   2) feed it into a GlobalTrackManager that associates reports across nodes
//      via 3-sigma position gating + bearing-only triangulation; emits
//      `globalTrackUpdate` and `globalTrackRemoved` events to all clients.
//
// The manager is a process-singleton; the prune tick is started lazily on the
// first registerGeolocationHandlers call so we have an `io` to emit on.

const { GlobalTrackManager } = require('./globalTrackManager');
const fs = require('fs');
const path = require('path');

const PRUNE_INTERVAL_MS = 250;
const STALE_AGE_MS = 2000;
const AAR_LOG_INTERVAL_MS = 1000;
const AAR_LOG_PATH = path.join(__dirname, 'logs', 'global_tracks.jsonl');
const AAR_LOG_MAX_BYTES = 50 * 1024 * 1024;  // 50 MB rotation cap

const manager = new GlobalTrackManager();
let pruneTick = null;
let aarTick = null;

function startPruneTick(io) {
    if (pruneTick) return;
    pruneTick = setInterval(() => {
        // 1) Merge two active globals that have converged onto the same drone
        //    (e.g. when initial tracks opened too far apart for ingest gating).
        const merged = manager.mergeConverged();
        for (const loserId of merged) {
            io.emit('globalTrackRemoved', { globalId: loserId });
        }
        // After merge, re-broadcast surviving tracks so UI picks up the new
        // contributor counts / fused σ.
        if (merged.length) {
            for (const snap of manager.snapshot()) {
                io.emit('globalTrackUpdate', snap);
            }
        }

        // 2) Stale-contributor prune + orphan TTL hard-delete.
        const removed = manager.pruneStale(STALE_AGE_MS);
        for (const globalId of removed) {
            io.emit('globalTrackRemoved', { globalId });
        }
    }, PRUNE_INTERVAL_MS);
}

// After-action review (AAR) logger. Dumps every live global track once per
// second to a JSONL file. One line per track, includes fused position + σ,
// engagement state, confidence + components, contributor count. Useful for
// post-mission analysis and confidence-tuning. Rotates when file exceeds
// AAR_LOG_MAX_BYTES (single rotation: .jsonl → .jsonl.1).
function startAarLogger() {
    if (aarTick) return;
    try {
        fs.mkdirSync(path.dirname(AAR_LOG_PATH), { recursive: true });
    } catch (_) { /* directory may already exist */ }
    aarTick = setInterval(() => {
        const snaps = manager.snapshot();
        if (snaps.length === 0) return;
        try {
            // Cheap size check + single-rotation when oversized.
            try {
                const st = fs.statSync(AAR_LOG_PATH);
                if (st.size > AAR_LOG_MAX_BYTES) {
                    fs.renameSync(AAR_LOG_PATH, AAR_LOG_PATH + '.1');
                }
            } catch (_) { /* file does not exist yet */ }

            const lines = snaps.map(s => JSON.stringify({
                ts: s.timestamp,
                globalId: s.globalId,
                fused: s.fused,
                velocity: s.velocity,
                engagement_state: s.engagement_state,
                motion_mode: s.motion_mode,
                confidence: s.confidence,
                confidence_components: s.confidence_components,
                method: s.method,
                contributors: s.contributors.map(c => `${c.deviceName}:t${c.trackId}:${c.track_state}`),
            }));
            fs.appendFile(AAR_LOG_PATH, lines.join('\n') + '\n', () => {});
        } catch (e) {
            // Logging must never crash the server.
            console.warn('[AAR] log write failed:', e.message);
        }
    }, AAR_LOG_INTERVAL_MS);
}

function registerGeolocationHandlers(io, socket) {
    startPruneTick(io);
    startAarLogger();

    socket.on('geolocationUpdate', (data) => {
        if (!data || !data.deviceName || !data.world) return;
        data.serverTs = Date.now();
        // Raw fan-out preserved for sidebar readout / debug consumers.
        io.emit('geolocationUpdate', data);

        const result = manager.ingest({
            deviceName: data.deviceName,
            trackId: data.trackId,
            world: data.world,
            uncertainty_m: data.uncertainty_m,
            azimuth_world: data.azimuth_world,
            elevation: data.elevation,         // signed degrees, +up. Required for 3D triangulation.
            range_m: data.range_m,             // slant range (consistency / single-observer reprojection)
            track_state: data.track_state,     // 'LOCKED' or 'PASSIVE' — drives minimap rendering
            unit_position: data.unit && data.unit.position,
            ts: data.serverTs,
        });
        if (result) {
            io.emit('globalTrackUpdate', result);
            // Compact diagnostic line — surfaces config alignment problems
            // and re-merge events.
            const contribStr = result.contributors
                .map(c => `${c.deviceName}:t${c.trackId}`).join(',');
            const dNorm = result.d_norm != null ? result.d_norm.toFixed(2) : '-';
            const tag = result.resurrected
                ? ` RESURRECTED (orphan_age=${(result.orphan_age_ms / 1000).toFixed(1)}s)`
                : '';
            console.log(
                `[GTM] ingest ${data.deviceName}:t${data.trackId} -> G${result.globalId}${tag} ` +
                `(method=${result.method}, contributors=${result.contributors.length} [${contribStr}], ` +
                `σ=${result.fused.sigma_m.toFixed(1)}m, d_norm=${dNorm})`
            );
        }
    });
}

function removeUnitFromGlobalTracks(io, deviceName) {
    const removed = manager.removeUnit(deviceName);
    for (const globalId of removed) {
        io.emit('globalTrackRemoved', { globalId });
    }
    // Surviving tracks may have had their fusion recomputed; broadcast fresh
    // snapshots so the UI updates σ/method (cheap — only runs on disconnect).
    for (const snap of manager.snapshot()) {
        io.emit('globalTrackUpdate', snap);
    }
}

// Called from server.js when a Pi reports trackingStatus=LOST for a specific
// trackId. Immediately drops that contributor instead of waiting for pruneStale.
function dropTrackFromGlobalTracks(io, deviceName, trackId) {
    manager.removeContributor(deviceName, trackId);
    // Re-broadcast surviving snapshots so the UI sees the updated contributor
    // list / σ promptly. (The orphan path is handled inside removeContributor.)
    for (const snap of manager.snapshot()) {
        io.emit('globalTrackUpdate', snap);
    }
}

module.exports = { registerGeolocationHandlers, removeUnitFromGlobalTracks, dropTrackFromGlobalTracks };
