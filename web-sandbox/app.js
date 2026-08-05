/* =====================================================================
 * SoundCue — Web Calibration Sandbox (browser side)
 * ---------------------------------------------------------------------
 * Collects (trueAzimuth, userPerceivedAzimuth) trial pairs using
 * WebAudio HRTF binaural spatialisation. Zero network, zero server,
 * zero external assets (sound samples are synthesized in-browser).
 *
 * Output: calibration_dataset.json — imported by the SoundCue desktop
 * app to fit a PerceptualCorrectionTable (Option A, post-pipelines
 * azimuth correction for overlay rendering).
 *
 * Coordinate system (matches spec):
 *   listener = (0,0,0), forward = -Z, right = +X, up = +Y
 *   azimuth 0° = (0,0,-3) · +90° = (3,0,0) · ±180° = (0,0,+3)
 * ===================================================================== */
'use strict';

/* ---------------- state ---------------- */
const S = {
  phase: 'setup',                 // setup | running | results
  cfg: {
    trials: 60,
    clipMode: 'all',
    volume: 0.8,
    deviceName: '',
    showDot: false,
  },
  clipDefs: [],                   // { id, name, make() -> AudioBuffer }
  queue: [],                      // [{ az, el, clipId, practice }]
  idx: -1,
  trial: null,                    // runtime state for current trial
  data: [],                       // recorded trials (export payload)
  sessionId: null,
  practiceTotal: 3,
  startMs: 0,
};

let ctx = null, master = null;
let renderer = null, scene = null, camera = null;
let dotMesh = null, dotGlow = null;
let locked = false;
const KEY = { sens: 0.0023 };
const _fwd = new THREE.Vector3();   // scratch vector for listener orientation

/* ---------------- tiny helpers ---------------- */
const $ = (id) => document.getElementById(id);
const wrap180 = (d) => { d = ((d + 180) % 360 + 360) % 360 - 180; return d; };
const rad2deg = (r) => r * 180 / Math.PI;
const deg2rad = (d) => d * Math.PI / 180;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const circMean = (degArr) => {
  let x = 0, y = 0;
  for (const d of degArr) { x += Math.cos(deg2rad(d)); y += Math.sin(deg2rad(d)); }
  return rad2deg(Math.atan2(y / degArr.length, x / degArr.length));
};
function toast(msg, ms = 2200) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove('show'), ms);
}
function uid() {
  return 'sc-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
}

/* =====================================================================
 * AUDIO — synthesized samples + HRTF panner
 * ===================================================================== */
function synthNoise(ctx, dur) {
  const buf = ctx.createBuffer(1, Math.max(1, Math.floor(ctx.sampleRate * dur)), ctx.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  return buf;
}
function envGain(ctx, peak, attack, decay) {
  const g = ctx.createGain();
  const t = ctx.currentTime;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(peak, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + attack + decay);
  return g;
}

function makeFootstep(ctx) {
  /* low-passed noise thump + 70 Hz body — reads as a footstep on concrete */
  const dur = 0.22;
  const src = ctx.createBufferSource();
  src.buffer = synthNoise(ctx, dur);
  const lp = ctx.createBiquadFilter(); lp.type = 'lowpass';
  lp.frequency.setValueAtTime(1400, ctx.currentTime);
  lp.frequency.exponentialRampToValueAtTime(180, ctx.currentTime + dur);
  lp.Q.value = 0.8;
  const body = ctx.createBufferSource();
  const bb = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.09), ctx.sampleRate);
  const bd = bb.getChannelData(0);
  for (let i = 0; i < bd.length; i++) bd[i] = Math.sin(2 * Math.PI * 65 * i / ctx.sampleRate) * Math.exp(-i / (0.028 * ctx.sampleRate));
  body.buffer = bb;
  const g1 = envGain(ctx, 0.9, 0.004, 0.16);
  const g2 = envGain(ctx, 0.7, 0.004, 0.07);
  // IMPORTANT: the synth chain ends in ONE local 'out' node — the caller
  // routes it through the HRTF panner. NEVER connect straight to master
  // here: that would play an unspatialised dead-centre copy on top of the
  // panned one and mask every directional cue.
  const out = ctx.createGain(); out.gain.value = 1;
  src.connect(lp); lp.connect(g1); g1.connect(out);
  body.connect(g2); g2.connect(out);
  return { sources: [src, body], out, start(t) { src.start(t); body.start(t); }, duration: dur };
}

function makeGunshot(ctx) {
  /* sharp crack (high-pass noise) + 300 Hz report — gunshot-like transient */
  const dur = 0.34;
  const crack = ctx.createBufferSource();
  crack.buffer = synthNoise(ctx, 0.05);
  const hp = ctx.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 2200;
  const gC = envGain(ctx, 0.95, 0.0015, 0.045);
  const report = ctx.createBufferSource();
  report.buffer = synthNoise(ctx, dur);
  const bp = ctx.createBiquadFilter(); bp.type = 'bandpass';
  bp.frequency.setValueAtTime(900, ctx.currentTime);
  bp.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 0.2);
  bp.Q.value = 1.2;
  const gR = envGain(ctx, 1.0, 0.002, 0.3);
  const out = ctx.createGain(); out.gain.value = 1;
  crack.connect(hp); hp.connect(gC); gC.connect(out);
  report.connect(bp); bp.connect(gR); gR.connect(out);
  return { sources: [crack, report], out, start(t) { crack.start(t); report.start(t); }, duration: dur };
}

function makeClap(ctx) {
  /* short band-passed noise burst — a sharp, easy-to-localise clap/snap */
  const dur = 0.08;
  const src = ctx.createBufferSource();
  src.buffer = synthNoise(ctx, dur);
  const bp = ctx.createBiquadFilter(); bp.type = 'bandpass';
  bp.frequency.value = 1600; bp.Q.value = 1.6;
  const g = envGain(ctx, 0.95, 0.002, 0.07);
  const out = ctx.createGain(); out.gain.value = 1;
  src.connect(bp); bp.connect(g); g.connect(out);
  return { sources: [src], out, start(t) { src.start(t); }, duration: dur };
}

function makeSweep(ctx) {
  /* upward tone sweep — strong ITD cue, used to check tonal localisation */
  const dur = 0.4;
  const src = ctx.createBufferSource();
  const buf = ctx.createBuffer(1, Math.floor(ctx.sampleRate * dur), ctx.sampleRate);
  const d = buf.getChannelData(0);
  let ph = 0;
  for (let i = 0; i < d.length; i++) {
    const f = 500 + 1400 * (i / d.length);           // 500 -> 1900 Hz
    ph += 2 * Math.PI * f / ctx.sampleRate;
    d[i] = Math.sin(ph) * Math.exp(-3.2 * i / d.length);
  }
  src.buffer = buf;
  const g = envGain(ctx, 0.8, 0.02, 0.3);
  const out = ctx.createGain(); out.gain.value = 1;
  src.connect(g); g.connect(out);
  return { sources: [src], out, start(t) { src.start(t); }, duration: dur };
}

function initAudio() {
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  master = ctx.createGain();
  master.gain.value = S.cfg.volume;
  master.connect(ctx.destination);
  // listener: origin, facing -Z, up +Y
  const L = ctx.listener;
  if (L.positionX) {
    L.positionX.value = 0; L.positionY.value = 0; L.positionZ.value = 0;
    L.forwardX.value = 0; L.forwardY.value = 0; L.forwardZ.value = -1;
    L.upX.value = 0; L.upY.value = 1; L.upZ.value = 0;
  } else {
    L.setPosition(0, 0, 0); L.setOrientation(0, 0, -1, 0, 1, 0);
  }
  S.clipDefs = [
    { id: 'footstep', name: 'footstep_concrete.wav', make: () => makeFootstep(ctx) },
    { id: 'gunshot',  name: 'gunshot_rifle.wav',      make: () => makeGunshot(ctx) },
    { id: 'clap',     name: 'clap_snap.wav',          make: () => makeClap(ctx) },
    { id: 'sweep',    name: 'tone_sweep.wav',         make: () => makeSweep(ctx) },
  ];
}

/* =====================================================================
 * SCENE — pure black void, one glowing green dot, fixed camera origin
 * ===================================================================== */
function initScene() {
  const canvas = $('scene');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(0x000000, 1);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 0);
  // FPS mouse-look convention: yaw around WORLD up, pitch around the camera's
  // LOCAL right axis. The default order 'XYZ' pitches around world X instead —
  // facing ±90° yaw makes pitch dead and facing 180° inverts it ("can't move
  // up/down, sometimes well, sometimes not").
  camera.rotation.order = 'YXZ';

  const dotGeo = new THREE.SphereGeometry(0.055, 20, 20);
  const dotMat = new THREE.MeshBasicMaterial({ color: 0x39ff88 });
  dotMesh = new THREE.Mesh(dotGeo, dotMat);
  dotMesh.visible = false;
  scene.add(dotMesh);

  // soft glow sprite (additive) so the dot reads against pure black
  const cv = document.createElement('canvas'); cv.width = cv.height = 128;
  const g = cv.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 2, 64, 64, 64);
  grad.addColorStop(0, 'rgba(57,255,136,0.85)');
  grad.addColorStop(0.35, 'rgba(57,255,136,0.22)');
  grad.addColorStop(1, 'rgba(57,255,136,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(cv);
  dotGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, blending: THREE.AdditiveBlending, depthWrite: false }));
  dotGlow.scale.set(0.9, 0.9, 1);
  dotGlow.visible = false;
  scene.add(dotGlow);

  resize();
  window.addEventListener('resize', resize);
  renderer.setAnimationLoop(render);
}
function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight, false);
}
function render() {
  // The listener turns with the camera (like a game: turning re-pans the
  // world audio relative to your head). Without this, rotating the view
  // changes nothing audible and the task becomes impossible.
  if (ctx && ctx.listener) {
    camera.getWorldDirection(_fwd);
    const L = ctx.listener;
    if (L.forwardX) {
      L.forwardX.value = _fwd.x;
      L.forwardY.value = _fwd.y;
      L.forwardZ.value = _fwd.z;
      L.upX.value = 0; L.upY.value = 1; L.upZ.value = 0;
    } else {
      L.setOrientation(_fwd.x, _fwd.y, _fwd.z, 0, 1, 0);
    }
  }
  updateViewSphere();
  renderer.render(scene, camera);
}

/** Top-right view sphere: view azimuth on the fixed ring, pitch pulls the
 * marker toward the centre. Front (F) is always at the top. */
function updateViewSphere() {
  if (S.phase !== 'running') return;
  const az = wrap180(-rad2deg(camera.rotation.y));
  const pitch = rad2deg(camera.rotation.x);              // + = looking up
  const r = 58 * Math.cos(deg2rad(Math.min(Math.abs(pitch), 88)));
  const x = 70 + r * Math.sin(deg2rad(az));
  const y = 70 - r * Math.cos(deg2rad(az));
  const m = $('vs-marker');
  m.setAttribute('transform',
    `translate(${x.toFixed(2)},${y.toFixed(2)}) rotate(${az.toFixed(2)})`);
  $('viewsphere-pitch').textContent =
    `${pitch >= 0 ? '+' : ''}${Math.round(pitch)}°`;
}

function placeDot(pos) {
  dotMesh.position.set(pos.x, pos.y, pos.z);
  dotGlow.position.set(pos.x, pos.y, pos.z);
}
function setDotVisible(v) {
  dotMesh.visible = v && S.cfg.showDot;
  dotGlow.visible = v && S.cfg.showDot;
}

/* =====================================================================
 * TRIAL ENGINE
 * ===================================================================== */
function pickClipId(i, n) {
  if (S.cfg.clipMode !== 'all') return S.cfg.clipMode;
  return S.clipDefs[Math.floor(i * S.clipDefs.length / n) % S.clipDefs.length].id;
}
function buildQueue() {
  const n = S.cfg.trials;
  const q = [];
  const step = 360 / n;
  const phase = Math.random() * step;                 // session-random grid phase
  for (let i = 0; i < n; i++) {
    const base = phase + i * step + step * 0.5 - 180;
    const az = wrap180(base + (Math.random() - 0.5) * step * 0.7);
    const el = (Math.random() - 0.5) * 20;            // ±10° elevation
    q.push({ az, el, clipId: pickClipId(i, n), practice: false });
  }
  // Fisher-Yates shuffle (stratification survives: coverage stays uniform)
  for (let i = q.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [q[i], q[j]] = [q[j], q[i]];
  }
  // practice trials (fixed, familiarisation only, not recorded)
  const practice = [
    { az: 0,   el: 0, clipId: 'sweep', practice: true },
    { az: 90,  el: 0, clipId: 'sweep', practice: true },
    { az: -90, el: 0, clipId: 'sweep', practice: true },
  ];
  return practice.concat(q);
}
function azToPos(azDeg, elDeg, dist) {
  const az = deg2rad(azDeg), el = deg2rad(elDeg);
  const hz = dist * Math.cos(el);
  return { x: hz * Math.sin(az), y: dist * Math.sin(el), z: -hz * Math.cos(az) };
}
function posToAz(pos) {
  /* azimuth of a point in the XZ plane (forward = -Z) */
  return wrap180(rad2deg(Math.atan2(pos.x, -pos.z)));
}

function beginTrial() {
  const t = S.queue[S.idx];
  const pos = azToPos(t.az, t.el, 3);
  S.trial = {
    ...t, pos,
    playCount: 0,
    recorded: false,
    soundStart: 0, soundEnd: 0,
    clickedDuring: false,
  };
  placeDot(pos);
  setDotVisible(t.practice || S.cfg.showDot);
  updateHud();
  const badge = $('hud-badge');
  badge.style.display = t.practice ? 'inline-block' : 'none';
  if (t.practice) {
    // familiarisation: show the answer during practice so ears can learn the render
    $('hud-err').textContent = `practice — true azimuth ${t.az.toFixed(0)}°`;
  }
  // play after a short settle delay (guard against stale fires if user confirms fast)
  const tid = S.idx;
  setTimeout(() => { if (S.idx === tid) playTrialSound(); }, 350);
}

function playTrialSound() {
  if (!S.trial) return;
  const def = S.clipDefs.find((c) => c.id === S.trial.clipId) || S.clipDefs[0];
  const inst = def.make();
  const panner = ctx.createPanner();
  panner.panningModel = 'HRTF';
  panner.distanceModel = 'linear';
  panner.refDistance = 1;
  panner.maxDistance = 50;
  panner.rolloffFactor = 1;
  panner.positionX.value = S.trial.pos.x;
  panner.positionY.value = S.trial.pos.y;
  panner.positionZ.value = S.trial.pos.z;
  inst.out.connect(panner);
  panner.connect(master);
  const t0 = ctx.currentTime + 0.02;
  inst.start(t0);
  S.trial.playCount++;
  S.trial.soundStart = performance.now() + 20;
  S.trial.soundEnd = S.trial.soundStart + inst.duration * 1000 + 40;
}

/** Plays a clip at an absolute azimuth (for the pre-trial audio test). */
function playAt(azDeg, clipId, whenSec = 0, showDot = false) {
  const def = S.clipDefs.find((c) => c.id === clipId) || S.clipDefs[0];
  const inst = def.make();
  const panner = ctx.createPanner();
  panner.panningModel = 'HRTF';
  panner.distanceModel = 'linear';
  panner.refDistance = 1;
  panner.maxDistance = 50;
  panner.rolloffFactor = 1;
  const pos = azToPos(azDeg, 0, 3);
  panner.positionX.value = pos.x;
  panner.positionY.value = pos.y;
  panner.positionZ.value = pos.z;
  inst.out.connect(panner);
  panner.connect(master);
  const t0 = ctx.currentTime + whenSec;
  inst.start(t0);
  if (showDot) {
    setTimeout(() => { placeDot(pos); dotMesh.visible = true; dotGlow.visible = true; }, whenSec * 1000);
    setTimeout(() => { dotMesh.visible = false; dotGlow.visible = false; }, (whenSec + inst.duration) * 1000 + 250);
  }
}

/** Pre-trial spatial-audio verification: clap LEFT → CENTRE → RIGHT. */
function runAudioTest() {
  if (!ctx) initAudio();
  if (ctx.state === 'suspended') ctx.resume();
  S.cfg.showDot = true;                  // force dot visible so you can SEE where it should sound
  playAt(-90, 'clap', 0.0, true);
  playAt(0, 'clap', 0.8, true);
  playAt(90, 'clap', 1.6, true);
  toast('Audio test: you should hear claps left → front → right. Turn with the mouse — the sound should pan with your head like in a game.');
}

function recordAim() {
  if (!S.trial || S.trial.recorded) return;
  S.trial.recorded = true;
  const now = performance.now();
  const perceivedAz = wrap180(-rad2deg(camera.rotation.y));
  const pitchDeg = -rad2deg(camera.rotation.x);
  const aimPos = azToPos(perceivedAz, 0, 3);           // horizontal projection, 3 m
  const clickedDuring = now < S.trial.soundEnd;
  const resp = Math.max(0, now - (clickedDuring ? S.trial.soundStart : S.trial.soundEnd));

  const rec = {
    trialId: S.data.length + 1,
    soundClip: (S.clipDefs.find((c) => c.id === S.trial.clipId) || {}).name || S.trial.clipId,
    trueAzimuthDeg: round1(S.trial.az),
    trueX: round2(S.trial.pos.x),
    trueZ: round2(S.trial.pos.z),
    trueElevationDeg: round1(S.trial.el),
    trueY: round2(S.trial.pos.y),
    userPerceivedAzimuthDeg: round1(perceivedAz),
    userAimX: round2(aimPos.x),
    userAimZ: round2(aimPos.z),
    userPitchDeg: round1(pitchDeg),
    responseTimeMs: Math.round(resp),
    clickedDuringPlayback: clickedDuring,
    dotVisible: S.cfg.showDot,
    timeStampMs: Date.now(),
  };
  if (!S.trial.practice) S.data.push(rec);
  flashCrosshair();

  const err = Math.abs(wrap180(perceivedAz - S.trial.az));
  $('hud-err').textContent = `error ${err.toFixed(0)}°` + (S.trial.practice ? ' (practice)' : '');

  setTimeout(advance, 550);
}
function flashCrosshair() {
  const c = $('crosshair');
  c.classList.remove('flash');
  void c.offsetWidth;                                  // restart animation
  c.classList.add('flash');
}
function advance() {
  S.idx++;
  if (S.idx >= S.queue.length) { finish(); return; }
  beginTrial();
}
function finish() {
  S.phase = 'results';
  document.exitPointerLock?.();
  setHudVisible(false);
  showResults();
}

function round1(v) { return Math.round(v * 10) / 10; }
function round2(v) { return Math.round(v * 100) / 100; }

/* =====================================================================
 * INPUT
 * ===================================================================== */
function onCanvasClick() {
  if (S.phase !== 'running') return;
  if (!locked) {
    $('scene').requestPointerLock?.();
    return;
  }
  recordAim();
}
function onMouseMove(e) {
  if (!locked || S.phase !== 'running') return;
  camera.rotation.y -= e.movementX * KEY.sens;
  camera.rotation.x = clamp(camera.rotation.x - e.movementY * KEY.sens, -deg2rad(88), deg2rad(88));
}
function onKey(e) {
  if (S.phase !== 'running') return;
  const k = e.key.toLowerCase();
  if (k === ' ') { e.preventDefault(); playTrialSound(); }   // replay sound
  else if (k === 'p' && locked) { S.cfg.showDot = !S.cfg.showDot; toast('Dot ' + (S.cfg.showDot ? 'visible' : 'hidden') + ' (preview)'); }
  else if (k === 'e') { toast('Current aim: ' + wrap180(-rad2deg(camera.rotation.y)).toFixed(0) + '°'); }
}
function onLockChange() {
  locked = document.pointerLockElement === $('scene');
  $('pause').style.display = locked ? 'none' : (S.phase === 'running' ? 'flex' : 'none');
}

/* =====================================================================
 * UI / HUD
 * ===================================================================== */
function setHudVisible(v) {
  $('hud').style.display = v ? 'block' : 'none';
  $('crosshair').style.display = v ? 'block' : 'none';
  $('viewsphere').style.display = v ? 'block' : 'none';
  if (!v) $('pause').style.display = 'none';
}
function updateHud() {
  $('hud-n').textContent = S.idx + 1;
  $('hud-total').textContent = S.queue.length;
  const def = S.clipDefs.find((c) => c.id === S.trial.clipId);
  $('hud-clip').textContent = def ? def.name : S.trial.clipId;
  $('hud-err').textContent = '';
  $('hud-hints').innerHTML =
    'turn to face the sound (it pans with your head) · <kbd>click</kbd> to confirm · <kbd>Space</kbd> replay · ' +
    '<kbd>P</kbd> toggle dot · <kbd>Esc</kbd> pause';
}

function startCalibration() {
  S.cfg.trials = parseInt($('cfg-trials').value, 10);
  S.cfg.clipMode = $('cfg-clips').value;
  S.cfg.volume = parseFloat($('cfg-vol').value);
  S.cfg.deviceName = $('cfg-device').value.trim();
  S.cfg.showDot = $('cfg-dot').checked;

  initAudio();                                          // resume inside the user gesture
  if (ctx.state === 'suspended') ctx.resume();
  master.gain.value = S.cfg.volume;

  S.sessionId = uid();
  S.data = [];
  S.queue = buildQueue();
  S.idx = 0;
  S.phase = 'running';
  $('panel-setup').style.display = 'none';
  setHudVisible(true);
  camera.rotation.set(0, 0, 0);
  beginTrial();
  toast('Practice trials first — then ' + S.cfg.trials + ' recorded trials. Head still, eyes on the crosshair.');
}

/* =====================================================================
 * RESULTS + EXPORT
 * ===================================================================== */
function quadStats() {
  const quad = (name, minA, maxA) => {
    const rows = S.data.filter((r) => {
      const a = r.trueAzimuthDeg;
      return (minA <= maxA) ? (a >= minA && a < maxA) : (a >= minA || a < maxA);
    });
    return {
      name,
      n: rows.length,
      err: rows.length ? circMean(rows.map((r) => Math.abs(wrap180(r.userPerceivedAzimuthDeg - r.trueAzimuthDeg)))) : 0,
      conf: rows.filter((r) => Math.abs(wrap180(r.userPerceivedAzimuthDeg - r.trueAzimuthDeg)) > 90).length,
    };
  };
  return [
    quad('Front', -45, 45),
    quad('Right', 45, 135),
    quad('Rear', 135, 180 + 1e-9),                       // rear spans the seam
    quad('Left', -135, -45),
  ];
}
function showResults() {
  const n = S.data.length;
  $('res-sub').textContent =
    `${n} recorded trials · ${S.queue.length - S.practiceTotal} total · ${S.cfg.deviceName || 'no device name'} · ${new Date().toLocaleString()}`;

  const errs = S.data.map((r) => wrap180(r.userPerceivedAzimuthDeg - r.trueAzimuthDeg));
  const absErr = errs.map(Math.abs);
  const meanAbs = absErr.reduce((a, b) => a + b, 0) / n;
  const bias = circMean(errs);
  const conf = absErr.filter((e) => e > 90).length;
  const good = absErr.filter((e) => e <= 15).length;
  const maxErr = Math.max(...absErr);
  const median = [...absErr].sort((a, b) => a - b)[Math.floor(n / 2)];
  const quads = quadStats();

  const stats = [
    { k: 'Mean |error|', v: meanAbs.toFixed(1) + '°', cls: meanAbs <= 15 ? 'good' : meanAbs <= 30 ? 'warn' : 'bad' },
    { k: 'Median error', v: median.toFixed(1) + '°', cls: '' },
    { k: 'Mean signed bias', v: (bias >= 0 ? '+' : '') + bias.toFixed(1) + '°', cls: Math.abs(bias) > 10 ? 'warn' : '' },
    { k: 'Front/back confusions', v: conf + ' <small>/ ' + n + '</small>', cls: conf / n > 0.15 ? 'bad' : conf / n > 0.05 ? 'warn' : 'good' },
    { k: 'Within 15° of true', v: Math.round(100 * good / n) + '%', cls: good / n > 0.7 ? 'good' : 'warn' },
    { k: 'Worst single error', v: maxErr.toFixed(0) + '°', cls: maxErr > 90 ? 'bad' : '' },
  ];
  $('res-stats').innerHTML = stats.map((s) =>
    `<div class="stat ${s.cls}"><div class="k">${s.k}</div><div class="v">${s.v}</div></div>`).join('');

  const tb = $('res-quad').querySelector('tbody');
  tb.innerHTML = quads.map((q) =>
    `<tr><td>${q.name}</td><td class="num">${q.n}</td><td class="num">${q.err.toFixed(1)}°</td><td class="num">${q.conf}</td></tr>`).join('');

  const interp = [];
  if (conf / n > 0.15) interp.push('⚠️ <b>High front/back confusion rate</b> — rear sounds are often perceived as front (or vice versa). The correction table will flag a strong rear-region remap. If you game competitively, this is the single biggest perceptual issue.');
  else if (conf / n > 0.05) interp.push('Moderate front/back confusion. Rear cues are partially swapped — the correction table will apply a mild rear remap.');
  else interp.push('✅ Low front/back confusion — your HRTF + headset resolve the front/rear ambiguity well.');
  if (Math.abs(bias) > 10) interp.push(`Consistent ${bias > 0 ? 'rightward' : 'leftward'} bias of ${Math.abs(bias).toFixed(1)}° — likely headphone asymmetry or head tilt during trials; this is exactly what the correction offsets.`);
  if (meanAbs <= 15) interp.push('Overall localisation is tight. Expect the correction table to be near-identity — that is a good outcome.');
  $('res-interpret').innerHTML = interp.join('<br>');

  $('panel-results').style.display = 'flex';
}
function buildDatasetJson() {
  return JSON.stringify({
    schemaVersion: 1,
    app: 'SoundCue Calibration Sandbox',
    generatedAt: new Date().toISOString(),
    sessionId: S.sessionId,
    deviceName: S.cfg.deviceName,
    trialCount: S.data.length,
    distanceMeters: 3,
    soundClips: S.clipDefs.map((c) => c.name),
    trials: S.data,
  }, null, 2);
}
function downloadDataset() {
  const json = buildDatasetJson();
  const blob = new Blob([json], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'calibration_dataset.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  try { localStorage.setItem('soundcue_calibration_last', json); } catch (_) { /* ignore */ }
  toast('Downloaded calibration_dataset.json — import it in the SoundCue app');
}

/* =====================================================================
 * WIRING
 * ===================================================================== */
function init() {
  try {
    initScene();
  } catch (err) {
    console.error('WebGL init failed:', err);
    toast('WebGL unavailable in this browser — calibration needs WebGL + WebAudio');
  }
  $('cfg-vol').addEventListener('input', () => {
    $('cfg-vol-val').textContent = Math.round($('cfg-vol').value * 100) + '%';
  });
  $('btn-start').addEventListener('click', startCalibration);
  $('btn-download').addEventListener('click', downloadDataset);
  $('btn-redo').addEventListener('click', () => {
    $('panel-results').style.display = 'none';
    startCalibration();
  });
  $('btn-new').addEventListener('click', () => {
    $('panel-results').style.display = 'none';
    $('panel-setup').style.display = 'flex';
    S.phase = 'setup';
  });
  $('btn-audiotest').addEventListener('click', runAudioTest);
  $('scene').addEventListener('click', onCanvasClick);
  $('pause').addEventListener('click', () => {
    if (S.phase === 'running' && !locked) $('scene').requestPointerLock?.();
  });
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('keydown', onKey);
  document.addEventListener('pointerlockchange', onLockChange);
  document.addEventListener('contextmenu', (e) => e.preventDefault());
}
document.addEventListener('DOMContentLoaded', init);

/* =====================================================================
 * SELF-TEST MODE — open index.html?selftest=1 (no audio device needed)
 * Deterministic seeded run through every trial; writes a <pre id=selftest>
 * with the exported dataset so headless verification can read it.
 * ===================================================================== */
if (new URLSearchParams(location.search).has('selftest')) {
  document.addEventListener('DOMContentLoaded', runSelfTest);
}
function runSelfTest() {
  // deterministic RNG
  let seed = 12345;
  Math.random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };

  try {
    initAudio();
  } catch (err) {
    reportSelfTest('AUDIO_INIT_FAILED ' + err.message);
    return;
  }
  // audio-graph integrity: every clip must expose ONE 'out' node so the
  // caller can route it through the HRTF panner (a missing out or a direct
  // connection to master = dead-centre sound that masks all directional cues).
  for (const c of S.clipDefs) {
    let inst;
    try { inst = c.make(); } catch (err) { reportSelfTest('CLIP_MAKE_FAIL ' + c.id + ' ' + err.message); return; }
    if (!inst || !(inst.out instanceof GainNode)) { reportSelfTest('CLIP_OUT_FAIL ' + c.id); return; }
    if (inst.out === master) { reportSelfTest('CLIP_BYPASSES_PANNER ' + c.id); return; }
  }
  // camera-math regression: YXZ mouse-look must keep pitch alive at ANY yaw
  // (the old XYZ order made pitch dead at ±90° yaw and inverted at 180°).
  camera.rotation.order = 'YXZ';
  camera.rotation.set(deg2rad(20), deg2rad(90), 0);      // pitched up, facing right
  camera.getWorldDirection(_fwd);
  if (_fwd.y <= 0.05) { reportSelfTest('CAMERA_PITCH_DEAD_AT_YAW90'); return; }
  camera.rotation.set(deg2rad(25), deg2rad(40), 0);      // pitched, yawed
  const azCheck = wrap180(-rad2deg(camera.rotation.y));
  if (Math.abs(azCheck - (-40)) > 1e-9) { reportSelfTest('CAMERA_AZIMUTH_PITCH_CORRUPT ' + azCheck); return; }
  camera.getWorldDirection(_fwd);
  const hzAz = wrap180(rad2deg(Math.atan2(_fwd.x, -_fwd.z)));
  if (Math.abs(hzAz - (-40)) > 1e-6) { reportSelfTest('CAMERA_HORIZONTAL_AZIMUTH ' + hzAz); return; }
  camera.rotation.set(0, 0, 0);
  S.cfg.trials = 30;
  S.cfg.clipMode = 'all';
  S.cfg.deviceName = 'selftest';
  S.cfg.showDot = false;
  S.sessionId = 'selftest-session';
  S.data = [];
  S.queue = buildQueue();
  S.idx = 0;

  // sanity: queue length = practice(3) + trials(30), all azimuths within range
  if (S.queue.length !== 33) { reportSelfTest('QUEUE_LEN_FAIL ' + S.queue.length); return; }
  for (const t of S.queue) {
    if (t.az < -180 || t.az > 180 || t.el < -10 || t.el > 10) { reportSelfTest('RANGE_FAIL ' + JSON.stringify(t)); return; }
  }
  const recorded = S.queue.filter((t) => !t.practice);
  if (recorded.length !== 30) { reportSelfTest('RECORDED_COUNT_FAIL ' + recorded.length); return; }

  // simulate every trial: perceived = true - 7° (known bias) — export math must reproduce it
  while (S.idx < S.queue.length) {
    const t = S.queue[S.idx];
    S.trial = {
      ...t, pos: azToPos(t.az, t.el, 3),
      playCount: 0, recorded: false, soundStart: 0, soundEnd: 0, clickedDuring: false,
    };
    camera.rotation.set(0, deg2rad(-wrap180(t.az - 7)), 0);
    recordAim();                       // pushes to S.data (skips practice), no setTimeout dependency
    S.idx++;
  }

  const json = buildDatasetJson();
  const parsed = JSON.parse(json);
  let ok = true;
  const checks = [];
  if (parsed.trials.length !== 30) { ok = false; checks.push('export trials != 30'); }
  for (const r of parsed.trials) {
    const err = wrap180(r.userPerceivedAzimuthDeg - r.trueAzimuthDeg);
    if (Math.abs(err - (-7)) > 0.51) { ok = false; checks.push('bias mismatch trial ' + r.trialId + ': ' + err); break; }
    if (typeof r.trueX !== 'number' || typeof r.userAimZ !== 'number') { ok = false; checks.push('missing vec fields trial ' + r.trialId); break; }
  }
  // spot-check coordinate math: az 0 -> (0,-3), az 90 -> (3,0)
  const p0 = azToPos(0, 0, 3), p90 = azToPos(90, 0, 3);
  if (Math.abs(p0.x) > 1e-9 || Math.abs(p0.z + 3) > 1e-9) { ok = false; checks.push('az0 position wrong'); }
  if (Math.abs(p90.x - 3) > 1e-9 || Math.abs(p90.z) > 1e-9) { ok = false; checks.push('az90 position wrong'); }

  reportSelfTest((ok ? 'SELFTEST_OK' : 'SELFTEST_FAIL') + ' trials=' + parsed.trials.length +
    ' queuelen=' + S.queue.length + ' ' + checks.join(' | ') +
    ' sample=' + JSON.stringify(parsed.trials[0]));
}
function reportSelfTest(msg) {
  const pre = document.createElement('pre');
  pre.id = 'selftest';
  pre.textContent = msg;
  document.body.appendChild(pre);
  document.title = 'SELFTEST_DONE';
}
