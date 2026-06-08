// Top-down ENU minimap for PiSentinel.
//
// Consumes:
//   - `globalTrackUpdate` / `globalTrackRemoved` for fused multi-node tracks
//   - `geolocationUpdate` for per-unit position + live look bearing (the
//     spinning "where the camera is aiming" wedge)
//
// One drone seen by two Pis renders as a single fused circle (smaller σ from
// triangulation) plus dashed bearing rays from each contributing unit.

const TRAIL_MAX = 60;
const STALE_FADE_S = 3.0;
const STALE_DROP_S = 6.0;
const PRUNE_INTERVAL_MS = 500;
const LOOK_LERP_FACTOR = 0.22;   // per-frame lerp closure for live bearing wedge
const POS_LERP_FACTOR = 0.35;    // per-frame lerp closure for fused track position
const REMOVED_FADE_MS = 2200;    // how long a removed/merged track lingers as a "?" ghost
const TRAIL_SMOOTH_ALPHA = 0.25; // heavier EMA on the trail source — breadcrumb may lag ~500 ms
                                 // behind the marker, but kills sub-meter wiggle on cruise tracks

function lerpAngle(from, to, t) {
    // Shortest-arc lerp on a 0-360 circle.
    const d = ((to - from + 540) % 360) - 180;
    return (from + d * t + 360) % 360;
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

// Engagement-state palette. State (LOCKED / PASSIVE / LOST) is the only
// semantic axis on the minimap — random per-globalId colours were causing
// false visual identity (same drone re-acquired under a new globalId got a
// new colour, looked like a different drone).
//   LOCKED  → red (operator is engaging this target)
//   PASSIVE → green (detected, not engaged)
//   GHOST   → white-ish (kept simple; fading "?" already conveys state)
const STATE_COLOR_LOCKED  = '#ff3b30';
const STATE_COLOR_PASSIVE = '#34c759';
const STATE_COLOR_GHOST   = 'rgba(220,220,220,0.85)';

function colorFor(engagementState, isGhost) {
    if (isGhost) return STATE_COLOR_GHOST;
    return engagementState === 'LOCKED' ? STATE_COLOR_LOCKED : STATE_COLOR_PASSIVE;
}

export class Minimap {
    constructor({ canvas, expandButton, modal, modalCanvas, modalCloseBtn, readout }) {
        this.canvas = canvas;
        this.modalCanvas = modalCanvas;
        this.modal = modal;
        this.readout = readout;

        this.globalTracks = new Map();   // globalId -> {fused, contributors[], method, color, trail, lastTs}
        this.units = new Map();          // deviceName -> {position, camera_facing, look_azimuth, look_ts, lastTs}

        this.viewCenter = { x: 0, y: 0 };
        this.userPanned = false;
        this.metersPerPixel = 2.0;
        this.expanded = false;

        if (expandButton) expandButton.addEventListener('click', () => this.toggleExpanded());
        canvas.addEventListener('click', () => this.toggleExpanded());

        if (modalCloseBtn) modalCloseBtn.addEventListener('click', () => this.toggleExpanded());
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) this.toggleExpanded();
            });
        }

        if (modalCanvas) this._attachPanZoom(modalCanvas);

        this._lastPruneTs = 0;
        this._startRenderLoop();

        window.addEventListener('resize', () => this._syncCanvasSize());
        this._syncCanvasSize();
    }

    _startRenderLoop() {
        const tick = () => {
            const now = performance.now();
            if (now - this._lastPruneTs > PRUNE_INTERVAL_MS) {
                this._prune();
                this._lastPruneTs = now;
            }
            this._tickInterpolation();
            this.render();
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }

    _tickInterpolation() {
        // Smooth the live look bearing on each unit toward its latest target.
        for (const u of this.units.values()) {
            if (typeof u.look_azimuth_target !== 'number') continue;
            if (typeof u.look_azimuth_displayed !== 'number') {
                u.look_azimuth_displayed = u.look_azimuth_target;
                continue;
            }
            u.look_azimuth_displayed = lerpAngle(
                u.look_azimuth_displayed, u.look_azimuth_target, LOOK_LERP_FACTOR);
        }
        // Smooth global track fused position (so trail looks fluid between samples).
        for (const g of this.globalTracks.values()) {
            if (!g.fused_target) continue;
            if (!g.fused_displayed) {
                g.fused_displayed = { ...g.fused_target };
                continue;
            }
            g.fused_displayed.x = lerp(g.fused_displayed.x, g.fused_target.x, POS_LERP_FACTOR);
            g.fused_displayed.y = lerp(g.fused_displayed.y, g.fused_target.y, POS_LERP_FACTOR);
            g.fused_displayed.z = lerp(g.fused_displayed.z, g.fused_target.z, POS_LERP_FACTOR);
            g.fused_displayed.sigma_m = lerp(g.fused_displayed.sigma_m, g.fused_target.sigma_m, POS_LERP_FACTOR);
        }
    }

    _activeCanvas() {
        return this.expanded ? this.modalCanvas : this.canvas;
    }

    _syncCanvasSize() {
        for (const c of [this.canvas, this.modalCanvas]) {
            if (!c) continue;
            const rect = c.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            const w = Math.max(1, Math.round(rect.width * dpr));
            const h = Math.max(1, Math.round(rect.height * dpr));
            if (c.width !== w || c.height !== h) {
                c.width = w;
                c.height = h;
            }
        }
    }

    _attachPanZoom(canvas) {
        let dragging = false;
        let lastX = 0, lastY = 0;
        canvas.addEventListener('mousedown', (e) => {
            dragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            canvas.style.cursor = 'grabbing';
        });
        window.addEventListener('mouseup', () => {
            dragging = false;
            if (canvas) canvas.style.cursor = 'grab';
        });
        window.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            this.userPanned = true;
            this.viewCenter.x -= dx * this.metersPerPixel;
            this.viewCenter.y += dy * this.metersPerPixel;
            this.render();
        });
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
            this.metersPerPixel = Math.max(0.1, Math.min(2000, this.metersPerPixel * factor));
            this.render();
        }, { passive: false });
    }

    toggleExpanded() {
        if (!this.modal) return;
        this.expanded = !this.expanded;
        this.modal.classList.toggle('open', this.expanded);
        this.userPanned = false;
        this._syncCanvasSize();
        this.render();
    }

    // --- Inputs ---

    onGeolocationUpdate(data) {
        // Per-track payload; carries unit metadata + bearing TO the reported
        // drone. Track geometry itself arrives via onGlobalTrackUpdate.
        //
        // The look-direction wedge follows where the camera is physically
        // aimed — that's the LOCKED target. PASSIVE tracks are drones the
        // servo is NOT pointing at, so their azimuth_world must NOT drive
        // the wedge or it'll oscillate between the LOCKED and PASSIVE
        // bearings as their telemetry interleaves on the wire.
        if (!data || !data.deviceName) return;
        const now = performance.now() / 1000;
        if (data.unit && data.unit.position) {
            const prev = this.units.get(data.deviceName) || {};
            const isLocked = data.track_state !== 'PASSIVE';
            const lookTarget = (isLocked && typeof data.azimuth_world === 'number')
                ? data.azimuth_world
                : prev.look_azimuth_target;
            this.units.set(data.deviceName, {
                ...prev,
                position: data.unit.position,
                camera_facing: data.unit.camera_facing || 0,
                look_azimuth_target: lookTarget,
                // Only refresh the "look fresh" timer on LOCKED reports so
                // the wedge fades back to camera_facing when no engagement.
                look_ts: isLocked ? now : (prev.look_ts || 0),
                lastTs: now,
            });
        }
    }

    onUnitTelemetry(data) {
        // Always-on (10 Hz): refresh unit position + live look bearing even
        // when no drone is being tracked. Drives the smooth wedge in sentry.
        if (!data || !data.deviceName) return;
        const now = performance.now() / 1000;
        const prev = this.units.get(data.deviceName) || {};
        this.units.set(data.deviceName, {
            ...prev,
            position: data.position || prev.position,
            camera_facing: typeof data.camera_facing === 'number'
                ? data.camera_facing : (prev.camera_facing || 0),
            look_azimuth_target: typeof data.look_bearing === 'number'
                ? data.look_bearing : prev.look_azimuth_target,
            pan_angle: data.pan_angle,
            tilt_angle: data.tilt_angle,
            state: data.state,
            look_ts: now,
            lastTs: now,
        });
    }

    onGlobalTrackUpdate(payload) {
        if (!payload || payload.globalId == null || !payload.fused) return;
        const now = performance.now() / 1000;
        let g = this.globalTracks.get(payload.globalId);
        if (!g) {
            // color is derived per-frame from engagement_state in _drawGlobalTrack,
            // so we no longer store a per-track random colour at creation.
            g = { trail: [] };
            this.globalTracks.set(payload.globalId, g);
        }
        // Resurrection: clear removed-fade marker if the same id comes back alive.
        if (g._removedAt) g._removedAt = null;
        g.globalId = payload.globalId;
        g.fused_target = {
            x: payload.fused.x,
            y: payload.fused.y,
            z: payload.fused.z,
            sigma_m: payload.fused.sigma_m,
        };
        if (!g.fused_displayed) g.fused_displayed = { ...g.fused_target };
        g.contributors = payload.contributors || [];
        g.method = payload.method || 'single';
        g.engagement_state = payload.engagement_state || 'PASSIVE';
        g.motion_mode = payload.motion_mode || 'BOTH';
        g.velocity = payload.velocity || null;
        g.confidence = typeof payload.confidence === 'number' ? payload.confidence : null;
        g.confidence_components = payload.confidence_components || null;

        // Trail source uses a heavier EMA than the marker. The marker can
        // stay responsive (low display lag is mission-critical) while the
        // trail breadcrumb is allowed to lag ~500 ms — kills the wiggle.
        if (!g._trailSmoothed) {
            g._trailSmoothed = { x: payload.fused.x, y: payload.fused.y };
        } else {
            const a = TRAIL_SMOOTH_ALPHA;
            g._trailSmoothed.x = a * payload.fused.x + (1 - a) * g._trailSmoothed.x;
            g._trailSmoothed.y = a * payload.fused.y + (1 - a) * g._trailSmoothed.y;
        }
        g.lastTs = now;
        g.trail.push({ x: g._trailSmoothed.x, y: g._trailSmoothed.y, t: now });
        if (g.trail.length > TRAIL_MAX) g.trail.shift();
    }

    onGlobalTrackRemoved({ globalId } = {}) {
        // Don't delete immediately — let it fade out as a "?" ghost so users
        // see that something briefly was here (often a merge artifact: same
        // drone got re-grouped under another globalId).
        if (globalId == null) return;
        const g = this.globalTracks.get(globalId);
        if (!g) return;
        g._removedAt = performance.now();
    }

    onTrackingStatus(_data) {
        // Track lifecycle (PASSIVE/LOCKED/LOST) is authoritative in
        // GlobalTrackManager and arrives via globalTrackUpdate.engagement_state.
        // The minimap reads it from there directly — this handler exists only
        // so the producer's per-track status broadcast doesn't blow up the
        // client and is otherwise a no-op.
    }

    onUnitDisconnected(deviceName) {
        if (!deviceName) return;
        this.units.delete(deviceName);
        // Server emits globalTrackRemoved/Update on disconnect; nothing to
        // strip from this.globalTracks locally.
    }

    _prune() {
        const now = performance.now() / 1000;
        const nowMs = performance.now();
        for (const [k, g] of this.globalTracks) {
            // Hard-delete tracks that have been faded out long enough.
            if (g._removedAt && (nowMs - g._removedAt) > REMOVED_FADE_MS) {
                this.globalTracks.delete(k);
                continue;
            }
            // Stale safety net: if no update in STALE_DROP_S and not already
            // marked-removed, delete.
            if (!g._removedAt && now - g.lastTs > STALE_DROP_S) {
                this.globalTracks.delete(k);
            }
        }
    }

    _autoFit(canvas) {
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        let hasPoint = false;

        for (const u of this.units.values()) {
            minX = Math.min(minX, u.position.x);
            maxX = Math.max(maxX, u.position.x);
            minY = Math.min(minY, u.position.y);
            maxY = Math.max(maxY, u.position.y);
            hasPoint = true;
        }
        for (const g of this.globalTracks.values()) {
            const f = g.fused_displayed || g.fused_target;
            if (!f) continue;
            const r = Math.max(f.sigma_m || 0, 5);
            minX = Math.min(minX, f.x - r);
            maxX = Math.max(maxX, f.x + r);
            minY = Math.min(minY, f.y - r);
            maxY = Math.max(maxY, f.y + r);
            hasPoint = true;
        }

        if (!hasPoint) {
            this.viewCenter = { x: 0, y: 0 };
            this.metersPerPixel = 2.0;
            return;
        }

        const padding = 20;
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        const spanX = Math.max(20, maxX - minX);
        const spanY = Math.max(20, maxY - minY);
        const mppX = spanX / Math.max(1, canvas.width - padding * 2);
        const mppY = spanY / Math.max(1, canvas.height - padding * 2);
        this.metersPerPixel = Math.max(mppX, mppY) * 1.2;
        this.viewCenter = { x: cx, y: cy };
    }

    _worldToScreen(canvas, worldX, worldY) {
        const sx = canvas.width / 2 + (worldX - this.viewCenter.x) / this.metersPerPixel;
        const sy = canvas.height / 2 - (worldY - this.viewCenter.y) / this.metersPerPixel;
        return [sx, sy];
    }

    render() {
        const canvas = this._activeCanvas();
        if (!canvas) return;
        this._syncCanvasSize();

        if (!this.userPanned) this._autoFit(canvas);

        const ctx = canvas.getContext('2d');
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        this._drawGrid(ctx, canvas);
        this._drawRangeRings(ctx, canvas);
        this._drawNorthIndicator(ctx, canvas);

        for (const [name, u] of this.units) {
            this._drawUnit(ctx, canvas, name, u);
        }

        const now = performance.now() / 1000;
        const nowMs = performance.now();
        for (const g of this.globalTracks.values()) {
            let alpha;
            let isGhost = false;
            if (g._removedAt) {
                // Server told us this track is gone (merged or hard-deleted).
                // Linear fade over REMOVED_FADE_MS, draw "?" while ghosting.
                alpha = Math.max(0, 1 - (nowMs - g._removedAt) / REMOVED_FADE_MS);
                isGhost = true;
            } else {
                const age = now - g.lastTs;
                alpha = age > STALE_FADE_S
                    ? Math.max(0, 1 - (age - STALE_FADE_S) / (STALE_DROP_S - STALE_FADE_S))
                    : 1;
                isGhost = age > STALE_FADE_S;
            }
            // Confidence-driven alpha modulation: low-confidence tracks (likely
            // ghosts) get visually de-emphasised without being hidden entirely.
            // Floor at 0.4 so the operator can still see a low-conf contact.
            if (!isGhost && typeof g.confidence === 'number') {
                alpha *= Math.max(0.4, g.confidence / 100);
            }
            this._drawGlobalTrack(ctx, canvas, g, alpha, isGhost);
        }

        if (this.readout && this.expanded) this._updateReadout();
    }

    _drawGrid(ctx, canvas) {
        const W = canvas.width, H = canvas.height;
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        const rawStep = this.metersPerPixel * 80;
        const pow = Math.pow(10, Math.floor(Math.log10(Math.max(1, rawStep))));
        const stepCandidates = [1, 2, 5, 10].map(n => n * pow);
        const step = stepCandidates.reduce((a, b) =>
            Math.abs(b - rawStep) < Math.abs(a - rawStep) ? b : a);

        const left = this.viewCenter.x - (W / 2) * this.metersPerPixel;
        const right = this.viewCenter.x + (W / 2) * this.metersPerPixel;
        const bottom = this.viewCenter.y - (H / 2) * this.metersPerPixel;
        const top = this.viewCenter.y + (H / 2) * this.metersPerPixel;

        for (let x = Math.ceil(left / step) * step; x <= right; x += step) {
            const [sx] = this._worldToScreen(canvas, x, 0);
            ctx.beginPath();
            ctx.moveTo(sx, 0);
            ctx.lineTo(sx, H);
            ctx.stroke();
        }
        for (let y = Math.ceil(bottom / step) * step; y <= top; y += step) {
            const [, sy] = this._worldToScreen(canvas, 0, y);
            ctx.beginPath();
            ctx.moveTo(0, sy);
            ctx.lineTo(W, sy);
            ctx.stroke();
        }
        if (this.expanded) {
            ctx.fillStyle = 'rgba(255,255,255,0.35)';
            ctx.font = `${Math.round(11 * (window.devicePixelRatio || 1))}px monospace`;
            ctx.fillText(`grid: ${step}m`, 8 * (window.devicePixelRatio || 1), H - 8 * (window.devicePixelRatio || 1));
        }
        ctx.restore();
    }

    _drawRangeRings(ctx, canvas) {
        if (this.units.size === 0) return;
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.setLineDash([4, 6]);
        for (const u of this.units.values()) {
            const [cx, cy] = this._worldToScreen(canvas, u.position.x, u.position.y);
            for (const r of [50, 150, 300, 600]) {
                const radPx = r / this.metersPerPixel;
                if (radPx < 4 || radPx > Math.max(canvas.width, canvas.height) * 2) continue;
                ctx.beginPath();
                ctx.arc(cx, cy, radPx, 0, Math.PI * 2);
                ctx.stroke();
            }
        }
        ctx.restore();
    }

    _drawNorthIndicator(ctx, canvas) {
        const dpr = window.devicePixelRatio || 1;
        const pad = 10 * dpr;
        ctx.save();
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = `bold ${Math.round(12 * dpr)}px monospace`;
        ctx.textAlign = 'center';
        ctx.fillText('N', canvas.width - pad - 8 * dpr, pad + 12 * dpr);
        ctx.beginPath();
        ctx.moveTo(canvas.width - pad - 8 * dpr, pad + 16 * dpr);
        ctx.lineTo(canvas.width - pad - 13 * dpr, pad + 26 * dpr);
        ctx.lineTo(canvas.width - pad - 3 * dpr, pad + 26 * dpr);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    _drawUnit(ctx, canvas, name, u) {
        const [sx, sy] = this._worldToScreen(canvas, u.position.x, u.position.y);
        const dpr = window.devicePixelRatio || 1;
        const size = 8 * dpr;
        const now = performance.now() / 1000;

        const baseBearing = u.camera_facing || 0;
        const lookFresh = (typeof u.look_azimuth_displayed === 'number') &&
            (now - (u.look_ts || 0) < STALE_FADE_S);
        const liveBearing = lookFresh ? u.look_azimuth_displayed : baseBearing;

        this._drawWedge(ctx, sx, sy, baseBearing, 40, 70 * dpr,
            'rgba(120,160,200,0.08)', 'rgba(120,160,200,0.25)');

        if (lookFresh) {
            this._drawWedge(ctx, sx, sy, liveBearing, 6, 120 * dpr,
                'rgba(255,210,90,0.22)', 'rgba(255,210,90,0.85)');
        }

        const rad = liveBearing * Math.PI / 180;
        const hx = Math.sin(rad);
        const hy = -Math.cos(rad);

        ctx.save();
        ctx.translate(sx, sy);
        ctx.rotate(Math.atan2(hx, -hy));
        ctx.fillStyle = lookFresh ? '#ffd25a' : '#5fdfff';
        ctx.strokeStyle = lookFresh ? '#b88a20' : '#2b8db0';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, -size);
        ctx.lineTo(size * 0.7, size * 0.8);
        ctx.lineTo(-size * 0.7, size * 0.8);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();

        if (this.expanded) {
            ctx.save();
            ctx.fillStyle = 'rgba(200,230,255,0.85)';
            ctx.font = `${Math.round(11 * dpr)}px monospace`;
            ctx.fillText(name, sx + 10 * dpr, sy - 10 * dpr);
            if (lookFresh) {
                ctx.fillStyle = 'rgba(255,210,90,0.9)';
                ctx.fillText(`→ ${liveBearing.toFixed(0)}°`, sx + 10 * dpr, sy + 4 * dpr);
            }
            ctx.restore();
        }
    }

    _drawWedge(ctx, cx, cy, bearingDeg, halfAngleDeg, radiusPx, fill, stroke) {
        const a0 = (bearingDeg - halfAngleDeg - 90) * Math.PI / 180;
        const a1 = (bearingDeg + halfAngleDeg - 90) * Math.PI / 180;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, radiusPx, a0, a1);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
    }

    _drawGlobalTrack(ctx, canvas, g, alpha, isGhost = false) {
        const f = g.fused_displayed || g.fused_target;
        if (!f) return;
        const [sx, sy] = this._worldToScreen(canvas, f.x, f.y);
        const dpr = window.devicePixelRatio || 1;
        const radPx = Math.max(3 * dpr, (f.sigma_m || 0) / this.metersPerPixel);

        const isLocked = g.engagement_state === 'LOCKED';
        const isTriangulated = (g.method === 'bearing3D' || g.method === 'bearing');
        const color = colorFor(g.engagement_state, isGhost);

        // Contributor bearing rays — dashed line from each unit_pos to fused
        // point. EACH ray is coloured by ITS OWN contributor's track_state,
        // not by the global engagement state:
        //   LOCKED contributor  → solid-ish red ray   (that Pi is engaging)
        //   PASSIVE contributor → dashed green ray    (that Pi is only watching)
        // So the operator instantly sees which Pi has the gun on the target
        // and which is providing free situational awareness.
        if (!isGhost && g.contributors && g.contributors.length > 0) {
            ctx.save();
            const baseAlpha = (isTriangulated ? 0.65 : 0.35) * alpha;
            ctx.lineWidth = 1;
            for (const c of g.contributors) {
                if (!c.unit_pos) continue;
                const [ux, uy] = this._worldToScreen(canvas, c.unit_pos.x, c.unit_pos.y);
                const contribLocked = c.track_state === 'LOCKED';
                ctx.strokeStyle = contribLocked ? STATE_COLOR_LOCKED : STATE_COLOR_PASSIVE;
                ctx.globalAlpha = baseAlpha * (contribLocked ? 1.0 : 0.85);
                ctx.setLineDash(contribLocked ? [6, 3] : [3, 4]);
                ctx.beginPath();
                ctx.moveTo(ux, uy);
                ctx.lineTo(sx, sy);
                ctx.stroke();
            }
            ctx.restore();
        }

        // Trail of fused position — Catmull-Rom curve through points for smooth path.
        if (g.trail.length > 1) {
            ctx.save();
            ctx.strokeStyle = color;
            ctx.globalAlpha = (isLocked ? 0.4 : 0.25) * alpha;
            ctx.lineWidth = isLocked ? 1.5 : 1;
            const pts = g.trail.map(p => this._worldToScreen(canvas, p.x, p.y));
            this._strokeCatmullRom(ctx, pts);
            ctx.restore();
        }

        // Uncertainty ring.
        //  LOCKED  -> solid filled red,    bold stroke (full engagement)
        //  PASSIVE -> dashed hollow green, dimmer       (detected but not engaged)
        //  GHOST   -> short-dash gray,    ? glyph
        ctx.save();
        if (isGhost) {
            ctx.setLineDash([3, 4]);
        } else if (!isLocked) {
            ctx.setLineDash([3, 3]);
        }
        if (isLocked && !isGhost) {
            ctx.globalAlpha = 0.2 * alpha;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(sx, sy, radPx, 0, Math.PI * 2);
            ctx.fill();
        } else if (!isGhost) {
            // Passive: very faint fill so the user sees its footprint but
            // doesn't confuse it with the engaged target.
            ctx.globalAlpha = 0.08 * alpha;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(sx, sy, radPx, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = (isLocked ? 0.85 : 0.55) * alpha;
        ctx.strokeStyle = color;
        ctx.lineWidth = isLocked && isTriangulated ? 2 : 1;
        ctx.beginPath();
        ctx.arc(sx, sy, radPx, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        // Center marker or "?" overlay
        ctx.save();
        ctx.globalAlpha = alpha;
        if (isGhost) {
            ctx.fillStyle = 'rgba(255,255,255,0.85)';
            ctx.font = `bold ${Math.round(14 * dpr)}px monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('?', sx, sy);
        } else {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(sx, sy, (isLocked ? 3 : 2) * dpr, 0, Math.PI * 2);
            ctx.fill();
        }

        // Motion-mode glyph next to the centre dot. ⇅ vertical-only, ⇄
        // horizontal-only, ✛ stationary, no glyph for BOTH.
        if (!isGhost && g.motion_mode) {
            const glyph = g.motion_mode === 'VERTICAL'   ? '⇳'
                        : g.motion_mode === 'HORIZONTAL' ? '⇄'
                        : g.motion_mode === 'STATIONARY' ? '✚'
                        : null;
            if (glyph) {
                ctx.fillStyle = isLocked ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.65)';
                ctx.font = `bold ${Math.round(12 * dpr)}px monospace`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(glyph, sx, sy - 12 * dpr);
            }
        }

        // Label: G-id, engagement tag, fusion tag, motion mode, alt + σ.
        if (this.expanded && !isGhost) {
            ctx.fillStyle = 'rgba(255,255,255,0.9)';
            ctx.font = `${Math.round(11 * dpr)}px monospace`;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'alphabetic';
            const alt = (f.z || 0).toFixed(0);
            const sigma = (f.sigma_m || 0).toFixed(1);
            const fusionTag = g.method === 'bearing3D' ? 'TRI3D'
                            : g.method === 'bearing' ? 'TRI'
                            : g.method === 'weighted' ? 'WGT'
                            : 'SOLO';
            const engTag = isLocked ? 'LOCK' : 'PASV';
            const contribCount = g.contributors ? g.contributors.length : 0;
            const conf = typeof g.confidence === 'number' ? ` [${g.confidence}%]` : '';
            ctx.fillText(`#G${g.globalId} [${engTag}|${fusionTag} ×${contribCount}]${conf}`, sx + 8 * dpr, sy - 4 * dpr);
            ctx.fillText(`alt ${alt}m | σ ${sigma}m | ${g.motion_mode || 'BOTH'}`, sx + 8 * dpr, sy + 10 * dpr);
        } else if (!isGhost && typeof g.confidence === 'number' && g.confidence < 60) {
            // Non-expanded view: surface a tiny numeric tag only when confidence
            // is low — so the operator's eye is drawn to suspect contacts
            // without cluttering the map with badges on every healthy track.
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.fillStyle = 'rgba(255,200,80,0.95)';
            ctx.font = `bold ${Math.round(9 * dpr)}px monospace`;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(`${g.confidence}%`, sx + 6 * dpr, sy + 4 * dpr);
            ctx.restore();
        }
        ctx.restore();
    }

    // Smooth a polyline using Catmull-Rom spline → cubic Bézier conversion.
    // Each interior segment uses neighbours for tangent control points.
    _strokeCatmullRom(ctx, pts) {
        if (pts.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        if (pts.length === 2) {
            ctx.lineTo(pts[1][0], pts[1][1]);
            ctx.stroke();
            return;
        }
        for (let i = 0; i < pts.length - 1; i++) {
            const p0 = pts[Math.max(0, i - 1)];
            const p1 = pts[i];
            const p2 = pts[i + 1];
            const p3 = pts[Math.min(pts.length - 1, i + 2)];
            // Catmull-Rom with tension 0.5 → cubic Bézier control points.
            const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
            const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
            const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
            const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2[0], p2[1]);
        }
        ctx.stroke();
    }

    _updateReadout() {
        const rows = [];
        for (const g of this.globalTracks.values()) {
            const f = g.fused_target || g.fused_displayed;
            if (!f) continue;
            const fusionTag = g.method === 'bearing3D' ? 'TRI3D'
                            : g.method === 'bearing' ? 'TRI'
                            : g.method === 'weighted' ? 'WGT'
                            : 'SOLO';
            const eng = g.engagement_state || 'PASSIVE';
            const mode = g.motion_mode || 'BOTH';
            const contribStr = (g.contributors || [])
                .map(c => `${c.deviceName}:t${c.trackId}/${c.track_state || 'L'}`).join(', ');
            rows.push(
                `#G${g.globalId} [${eng}|${fusionTag}|${mode}]  ` +
                `world=(${f.x.toFixed(1)}, ${f.y.toFixed(1)}, ${f.z.toFixed(1)})  ` +
                `σ ${f.sigma_m.toFixed(1)}m  ` +
                `contributors: ${contribStr}`
            );
        }
        this.readout.textContent = rows.length ? rows.join('\n') : 'no active tracks';
    }
}
