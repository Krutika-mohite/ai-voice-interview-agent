// Run:  node tests/test_proctor_rules.js
const assert = require("assert");
const P = require("../proctor_component/rules.js");

let passed = 0;
function test(name, fn) { fn(); passed++; console.log("PASS", name); }

const OK = { faceCount: 1, yaw: 5, pitch: 3 };
const NOFACE = { faceCount: 0 };
const TWO = { faceCount: 2, yaw: 0, pitch: 0 };
const AWAY = { faceCount: 1, yaw: 50, pitch: 0 };
const DOWN = { faceCount: 1, yaw: 0, pitch: -40 };
const COVER = { faceCount: 1, handOverFace: true };

function run(rules, frames, fps = 5, start = 0) {
  // frames: array of [obs, seconds]; returns violations with time
  const out = []; let t = start;
  for (const [obs, secs] of frames) {
    const n = Math.round(secs * fps);
    for (let i = 0; i < n; i++) { t += 1 / fps; const v = rules.update(obs, t); if (v) out.push({ ...v, t }); }
  }
  return out;
}

test("normal behaviour never raises a violation", () => {
  assert.deepStrictEqual(run(new P.ProctorRules(), [[OK, 120]]), []);
});

test("no face for 3 s raises exactly one violation", () => {
  const v = run(new P.ProctorRules(), [[OK, 2], [NOFACE, 4], [OK, 5]]);
  assert.strictEqual(v.length, 1); assert.strictEqual(v[0].type, "no_face");
});

test("short absence below threshold is ignored", () => {
  assert.strictEqual(run(new P.ProctorRules(), [[OK, 2], [NOFACE, 2.4], [OK, 5]]).length, 0);
});

test("detection flicker does not reset the timer", () => {
  // no face, with a 0.2 s false face detection in the middle
  const v = run(new P.ProctorRules(), [[NOFACE, 1.8], [OK, 0.2], [NOFACE, 1.8]]);
  assert.strictEqual(v.length, 1);
});

test("multiple faces after 2 s, priority over looking away", () => {
  const v = run(new P.ProctorRules(), [[TWO, 2.6]]);
  assert.strictEqual(v.length, 1); assert.strictEqual(v[0].type, "multiple_faces");
});

test("looking away (yaw) needs 4 s; looking down (pitch) also counts", () => {
  assert.strictEqual(run(new P.ProctorRules(), [[AWAY, 3.5], [OK, 3]]).length, 0);
  const v1 = run(new P.ProctorRules(), [[AWAY, 4.6]]);
  assert.strictEqual(v1[0].type, "looking_away");
  const v2 = run(new P.ProctorRules(), [[DOWN, 4.6]]);
  assert.strictEqual(v2[0].type, "looking_away");
});

test("face covered by hand after 2.5 s", () => {
  const v = run(new P.ProctorRules(), [[COVER, 3]]);
  assert.strictEqual(v.length, 1); assert.strictEqual(v[0].type, "face_covered");
});

test("persistent absence repeats every 15 s so it adds up to termination", () => {
  const v = run(new P.ProctorRules(), [[NOFACE, 40]]);
  assert.strictEqual(v.length, 3);       // at ~3 s, ~18 s, ~33 s
  assert.ok(v[1].t - v[0].t >= 15);
});

test("cooldown: two different problems back to back count once within 8 s", () => {
  const v = run(new P.ProctorRules(), [[NOFACE, 3.4], [TWO, 3]]);
  assert.strictEqual(v.length, 1);
});

test("custom thresholds are respected", () => {
  const v = run(new P.ProctorRules({ noFaceSeconds: 1 }), [[NOFACE, 1.4]]);
  assert.strictEqual(v.length, 1);
});

test("angles from a rotation matrix (both layouts)", () => {
  const a = 40 * Math.PI / 180;
  // pure yaw rotation about Y, column-major
  const m = [Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, 0, 1];
  const r = P.anglesFromMatrix(m);
  assert.ok(Math.abs(Math.abs(r.yaw) - 40) < 0.01 && Math.abs(r.pitch) < 0.01);
  // transposed (row-major) gives the same magnitude
  const t = [m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14], m[3], m[7], m[11], m[15]];
  assert.ok(Math.abs(Math.abs(P.anglesFromMatrix(t).yaw) - 40) < 0.01);
  // pure pitch
  const b = 30 * Math.PI / 180;
  const pm = [1, 0, 0, 0, 0, Math.cos(b), Math.sin(b), 0, 0, -Math.sin(b), Math.cos(b), 0, 0, 0, 0, 1];
  assert.ok(Math.abs(Math.abs(P.anglesFromMatrix(pm).pitch) - 30) < 0.01);
});

test("hand over face detection and landmark yaw fallback", () => {
  const face = P.bbox([{ x: 0.4, y: 0.3 }, { x: 0.6, y: 0.6 }], 0.1);
  const handOn = Array.from({ length: 21 }, (_, i) => ({ x: 0.45 + (i % 5) * 0.02, y: 0.35 + Math.floor(i / 5) * 0.04 }));
  const handOff = handOn.map(p => ({ x: p.x + 0.4, y: p.y }));
  assert.strictEqual(P.handCoversFace(face, [handOn]), true);
  assert.strictEqual(P.handCoversFace(face, [handOff]), false);
  const lm = Array.from({ length: 468 }, () => ({ x: 0.5, y: 0.5 }));
  lm[234] = { x: 0.3 }; lm[454] = { x: 0.7 }; lm[1] = { x: 0.5 };
  assert.ok(Math.abs(P.yawFromLandmarks(lm)) < 1);
  lm[1] = { x: 0.66 };
  assert.ok(P.yawFromLandmarks(lm) > 35);
});

test("a hand resting on the chin is not 'face covered'; a hand over the face is", () => {
  // face landmarks spanning x 0.40-0.60, y 0.30-0.60; inner area after -15%: x 0.43-0.57, y 0.345-0.555
  const faceLm = [{ x: 0.4, y: 0.3 }, { x: 0.6, y: 0.6 }];
  const inner = P.bbox(faceLm, -0.15);
  const onChin = Array.from({ length: 21 }, (_, i) => ({ x: 0.44 + (i % 5) * 0.03, y: 0.58 + Math.floor(i / 5) * 0.03 }));
  const overFace = Array.from({ length: 21 }, (_, i) => ({ x: 0.44 + (i % 5) * 0.025, y: 0.36 + Math.floor(i / 5) * 0.04 }));
  assert.strictEqual(P.handCoversFace(inner, [onChin], 10), false);
  assert.strictEqual(P.handCoversFace(inner, [overFace], 10), true);
});

console.log(`${passed} passed`);
