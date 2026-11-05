/*
 * Proctoring rules - pure logic, no browser APIs, so it can be unit tested in Node.
 *
 * Each video frame the component passes an observation:
 *   { faceCount, yaw, pitch, handOverFace }      (yaw/pitch in degrees, may be null)
 * and the current time in seconds. update() returns a violation object when a
 * condition has lasted long enough, otherwise null.
 *
 * Design choices that keep false alarms down:
 *  - a condition must persist for its threshold (e.g. 3 s without a face)
 *  - short detection flickers (< gapTolerance) do not reset the timer
 *  - one episode raises one violation; it repeats only if it continues for
 *    `repeatAfter` more seconds (so permanently leaving the frame still adds up)
 *  - at least `cooldown` seconds between any two violations
 */
(function (root) {
  "use strict";

  var LABELS = {
    multiple_faces: "More than one person in view",
    face_covered: "Face covered",
    no_face: "Face not visible",
    looking_away: "Looking away from the screen",
  };

  var DEFAULTS = {
    noFaceSeconds: 3,
    multiFaceSeconds: 2,
    lookAwaySeconds: 4,
    faceCoverSeconds: 2.5,
    yawDegrees: 35,
    pitchDegrees: 25,
    gapTolerance: 0.6,
    cooldown: 8,
    repeatAfter: 15,
  };

  function classify(obs, cfg) {
    if (!obs) return null;
    var faces = obs.faceCount || 0;
    if (faces >= 2) return "multiple_faces";
    if (obs.handOverFace) return "face_covered";
    if (faces === 0) return "no_face";
    var yaw = typeof obs.yaw === "number" ? Math.abs(obs.yaw) : 0;
    var pitch = typeof obs.pitch === "number" ? Math.abs(obs.pitch) : 0;
    if (yaw > cfg.yawDegrees || pitch > cfg.pitchDegrees) return "looking_away";
    return null;
  }

  function threshold(cond, cfg) {
    return {
      multiple_faces: cfg.multiFaceSeconds,
      face_covered: cfg.faceCoverSeconds,
      no_face: cfg.noFaceSeconds,
      looking_away: cfg.lookAwaySeconds,
    }[cond];
  }

  function ProctorRules(options) {
    this.cfg = {};
    for (var k in DEFAULTS) this.cfg[k] = DEFAULTS[k];
    for (var key in options || {}) {
      if (options[key] !== undefined && options[key] !== null) this.cfg[key] = Number(options[key]);
    }
    this.reset();
  }

  ProctorRules.prototype.reset = function () {
    this.cond = null;       // current condition (string) or null
    this.since = 0;         // when the current episode started
    this.lastSeen = 0;      // last time the condition was observed
    this.firedAt = null;    // when this episode last raised a violation
    this.lastViolation = -Infinity;
  };

  /** Current live status for the UI: condition label or null when all is well. */
  ProctorRules.prototype.status = function () {
    return this.cond ? LABELS[this.cond] : null;
  };

  ProctorRules.prototype.update = function (obs, t) {
    var cfg = this.cfg;
    var c = classify(obs, cfg);

    if (c === this.cond) {
      if (c) this.lastSeen = t;
    } else if (this.cond && t - this.lastSeen < cfg.gapTolerance) {
      // brief flicker: keep the current episode running, don't extend lastSeen
    } else {
      this.cond = c;
      this.since = t;
      this.lastSeen = t;
      this.firedAt = null;
    }

    if (!this.cond) return null;

    var duration = this.lastSeen - this.since;
    if (duration < threshold(this.cond, cfg)) return null;
    if (t - this.lastViolation < cfg.cooldown) return null;
    if (this.firedAt !== null && t - this.firedAt < cfg.repeatAfter) return null;

    this.firedAt = t;
    this.lastViolation = t;
    return { type: this.cond, label: LABELS[this.cond], duration: Math.round(duration * 10) / 10 };
  };

  /* ---------- Geometry helpers (also pure, also tested) ---------- */

  /** Yaw/pitch in degrees from a 4x4 facial transformation matrix (16 numbers).
   *  Uses absolute-value-safe formulas, so row- or column-major layout both work. */
  function anglesFromMatrix(data) {
    if (!data || data.length < 11) return null;
    // column-major: R[i][j] = data[j*4 + i]
    var r02 = data[8], r22 = data[10], r12 = data[9];
    var yaw = Math.atan2(r02, r22) * 180 / Math.PI;
    var s = Math.max(-1, Math.min(1, -r12));
    var pitch = Math.asin(s) * 180 / Math.PI;
    return { yaw: yaw, pitch: pitch };
  }

  /** Fallback yaw from face landmarks: nose tip (1) vs cheek edges (234, 454). */
  function yawFromLandmarks(lm) {
    if (!lm || lm.length < 455) return null;
    var left = lm[234], right = lm[454], nose = lm[1];
    var w = right.x - left.x;
    if (Math.abs(w) < 1e-6) return null;
    var ratio = (nose.x - left.x) / w;     // 0.5 = facing camera
    return (ratio - 0.5) * 180;
  }

  function bbox(points, pad) {
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    var padX = (maxX - minX) * (pad || 0), padY = (maxY - minY) * (pad || 0);
    return { minX: minX - padX, minY: minY - padY, maxX: maxX + padX, maxY: maxY + padY };
  }

  /** True when at least `minPoints` hand landmarks fall inside the face box. */
  function handCoversFace(faceBox, hands, minPoints) {
    if (!faceBox || !hands || !hands.length) return false;
    var need = minPoints || 6;
    for (var h = 0; h < hands.length; h++) {
      var inside = 0;
      for (var i = 0; i < hands[h].length; i++) {
        var p = hands[h][i];
        if (p.x >= faceBox.minX && p.x <= faceBox.maxX && p.y >= faceBox.minY && p.y <= faceBox.maxY) inside++;
      }
      if (inside >= need) return true;
    }
    return false;
  }

  var api = {
    ProctorRules: ProctorRules,
    LABELS: LABELS,
    DEFAULTS: DEFAULTS,
    classify: classify,
    anglesFromMatrix: anglesFromMatrix,
    yawFromLandmarks: yawFromLandmarks,
    bbox: bbox,
    handCoversFace: handCoversFace,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Proctor = api;
})(typeof window !== "undefined" ? window : this);
