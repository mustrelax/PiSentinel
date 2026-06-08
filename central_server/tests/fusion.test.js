// Smoke tests for the new GTM fusion path.
//
// No test framework — run with `node tests/fusion.test.js` from
// central_server/. Each test prints PASS/FAIL and the suite exits 0
// only if every case passes.

const path = require('path');
const { GlobalTrackManager } = require(path.join(__dirname, '..', 'globalTrackManager'));
const { IMMFilter } = require(path.join(__dirname, '..', 'immFilter'));

let pass = 0, fail = 0;

function expect(label, cond, detail) {
    if (cond) {
        pass++;
        console.log(`  PASS  ${label}`);
    } else {
        fail++;
        console.error(`  FAIL  ${label}${detail ? '  ' + detail : ''}`);
    }
}

function nearly(a, b, eps) { return Math.abs(a - b) <= eps; }

// Geometry helper — what bearing + elevation should a unit at `u` see
// for a target at `t`?
function aim(u, t) {
    const dx = t.x - u.x, dy = t.y - u.y, dz = t.z - u.z;
    const range = Math.hypot(dx, dy, dz);
    const az = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
    const el = Math.atan2(dz, Math.hypot(dx, dy)) * 180 / Math.PI;
    return { az, el, range };
}

function report(gtm, dev, tid, u, target, state = 'LOCKED', ts) {
    const a = aim(u, target);
    return gtm.ingest({
        deviceName: dev,
        trackId: tid,
        world: { x: target.x, y: target.y, z: target.z },  // Pi's own est (may be noisy)
        uncertainty_m: 3,
        azimuth_world: a.az,
        elevation: a.el,
        range_m: a.range,
        track_state: state,
        unit_position: u,
        ts: ts || Date.now(),
    });
}

// ---- Test 1: 3D triangulation recovers exact position ----
console.log('Test 1: 3D ray-pair triangulation recovers target');
{
    const gtm = new GlobalTrackManager();
    const u1 = { x: 0, y: 0, z: 0 };
    const u2 = { x: 100, y: 0, z: 0 };
    const target = { x: 50, y: 50, z: 30 };

    report(gtm, 'A', 1, u1, target);
    const r = report(gtm, 'B', 1, u2, target);

    expect('method=bearing3D for 2 contributors with elevation', r.method === 'bearing3D', `got ${r.method}`);
    expect('fused.x within 0.5m', nearly(r.fused.x, 50, 0.5), `got ${r.fused.x}`);
    expect('fused.y within 0.5m', nearly(r.fused.y, 50, 0.5), `got ${r.fused.y}`);
    expect('fused.z within 0.5m', nearly(r.fused.z, 30, 0.5), `got ${r.fused.z}`);
    expect('engagement=LOCKED', r.engagement_state === 'LOCKED');
}

// ---- Test 2: altitude-only motion does NOT drift horizontal ----
console.log('Test 2: vertical-only motion keeps (x,y) fixed');
{
    const gtm = new GlobalTrackManager();
    const u1 = { x: 0, y: 0, z: 0 };
    const u2 = { x: 100, y: 0, z: 0 };
    let r;
    for (let i = 0; i < 40; i++) {
        const t = { x: 50, y: 50, z: 10 + i * 1.0 };  // climb 1 m per tick
        report(gtm, 'A', 1, u1, t, 'LOCKED', Date.now() + i * 100);
        r = report(gtm, 'B', 1, u2, t, 'LOCKED', Date.now() + i * 100 + 1);
    }
    expect('x drift < 1m after vertical sweep', nearly(r.fused.x, 50, 1.0), `got x=${r.fused.x}`);
    expect('y drift < 1m after vertical sweep', nearly(r.fused.y, 50, 1.0), `got y=${r.fused.y}`);
    expect('z tracks climb (>40)', r.fused.z > 40, `got z=${r.fused.z}`);
    expect('motion_mode is VERTICAL', r.motion_mode === 'VERTICAL', `got ${r.motion_mode}`);
}

// ---- Test 3: horizontal-only cruise tagged HORIZONTAL ----
console.log('Test 3: horizontal cruise tagged HORIZONTAL');
{
    const gtm = new GlobalTrackManager();
    const u1 = { x: 0, y: 0, z: 0 };
    const u2 = { x: 100, y: 0, z: 0 };
    let r;
    for (let i = 0; i < 40; i++) {
        const t = { x: 50, y: 80 - i * 1.0, z: 25 };  // approach at 10 m/s @ 10 Hz
        report(gtm, 'A', 1, u1, t, 'LOCKED', Date.now() + i * 100);
        r = report(gtm, 'B', 1, u2, t, 'LOCKED', Date.now() + i * 100 + 1);
    }
    expect('z stays near 25', nearly(r.fused.z, 25, 1.0), `got z=${r.fused.z}`);
    expect('motion_mode is HORIZONTAL', r.motion_mode === 'HORIZONTAL', `got ${r.motion_mode}`);
}

// ---- Test 4: PASSIVE engagement when no LOCKED contributor ----
console.log('Test 4: engagement_state reflects contributors');
{
    const gtm = new GlobalTrackManager();
    const u1 = { x: 0, y: 0, z: 0 };
    const target = { x: 30, y: 40, z: 20 };
    let r = report(gtm, 'A', 7, u1, target, 'PASSIVE');
    expect('single PASSIVE contributor -> PASSIVE', r.engagement_state === 'PASSIVE', `got ${r.engagement_state}`);
    expect('method=single', r.method === 'single', `got ${r.method}`);

    const u2 = { x: 100, y: 0, z: 0 };
    r = report(gtm, 'B', 7, u2, target, 'LOCKED');
    expect('one LOCKED contributor -> LOCKED', r.engagement_state === 'LOCKED', `got ${r.engagement_state}`);
}

// ---- Test 5: single-observer projection uses elevation, not Pi xyz ----
console.log('Test 5: single-observer re-projection');
{
    const gtm = new GlobalTrackManager();
    const u1 = { x: 0, y: 0, z: 0 };
    const target = { x: 30, y: 40, z: 20 };
    // Pi's reported world is deliberately WRONG; the server should
    // recompute from azimuth/elevation/range anchored at unit_pos.
    const a = aim(u1, target);
    const r = gtm.ingest({
        deviceName: 'A', trackId: 99,
        world: { x: 999, y: 999, z: 999 },  // garbage
        uncertainty_m: 3,
        azimuth_world: a.az, elevation: a.el, range_m: a.range,
        track_state: 'LOCKED',
        unit_position: u1, ts: Date.now(),
    });
    expect('single-observer ignores Pi xyz', nearly(r.fused.x, 30, 0.5) && nearly(r.fused.y, 40, 0.5) && nearly(r.fused.z, 20, 0.5),
           `got (${r.fused.x.toFixed(2)}, ${r.fused.y.toFixed(2)}, ${r.fused.z.toFixed(2)})`);
}

// ---- Test 6: parallel rays -> falls back to weighted ----
console.log('Test 6: parallel ray degeneracy');
{
    const gtm = new GlobalTrackManager();
    // Place u2 colinearly behind u1 on the line through the target so both
    // see the drone along the same 3D direction → rays parallel → fall back.
    const target = { x: 0, y: 100, z: 20 };
    const u1 = { x: 0, y: 0, z: 0 };
    const u2 = { x: 0, y: -100, z: -20 };  // (target - u1) and (target - u2) are colinear
    report(gtm, 'A', 1, u1, target);
    const r = report(gtm, 'B', 1, u2, target);
    expect('parallel rays fall back from bearing3D', r.method !== 'bearing3D', `got ${r.method}`);
}

// ---- Test 7: IMM transitions H -> V mid-track ----
console.log('Test 7: IMM mode switching');
{
    const imm = new IMMFilter({ x: 0, y: 80, z: 20 });
    // Phase 1: horizontal cruise 20 steps
    let out;
    for (let k = 0; k < 30; k++) out = imm.step({ x: 0, y: 80 - k * 1.0, z: 20 }, 1.0, 0.1);
    expect('IMM detects HORIZONTAL after cruise', out.motion_mode === 'HORIZONTAL', `got ${out.motion_mode}`);
    // Phase 2: pure climb, holding y constant
    const yStop = out.y;
    for (let k = 0; k < 30; k++) out = imm.step({ x: 0, y: yStop, z: 20 + k * 1.0 }, 1.0, 0.1);
    expect('IMM switches to VERTICAL after climb', out.motion_mode === 'VERTICAL', `got ${out.motion_mode}`);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
