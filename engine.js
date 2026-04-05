// ═══════════════════════════════════════════════════════════════
//   JumpEngine — pure jump-rope detection logic (no DOM, no Web Audio)
//
//   Usage:
//     import { JumpEngine } from './engine.js';
//     const engine = new JumpEngine();
//     engine.setCalibration(noiseFloor, baselineFlatness, baselineCentroid);
//     // in your audio loop:
//     const events = engine.processFrame(timeData, freqData, now, sampleRate, fftSize);
// ═══════════════════════════════════════════════════════════════

// ─── State enum ───
export const S = Object.freeze({
  IDLE: 'IDLE',
  LEARNING: 'LEARNING',
  TRACKING: 'TRACKING',
  PAUSED: 'PAUSED',
});

// ─── Default constants (all overridable via constructor) ───
const DEFAULTS = Object.freeze({
  N_LEARN: 4,
  W_FACTOR: 0.28,
  THR_WIN_MULT: 0.58,
  DU_RATIO: 0.44,
  TEMPO_ALPHA: 0.22,
  PAUSE_MULT: 2.6,
  PAUSE_ABS: 2800,
  RESUME_GRACE: 9000,
  REFRACTORY: 110,

  // Spectral gate
  SIG_WEIGHT_FLAT: 0.45,
  SIG_WEIGHT_CREST: 0.30,
  SIG_WEIGHT_CENT: 0.25,
  SIG_MIN_SCORE: 0.30,
  CREST_ROPE_MIN: 3.0,
  CREST_NORM_MAX: 12.0,
  CENT_ROPE_MIN_HZ: 1200,
  CENT_ROPE_MAX_HZ: 10000,
  FLAT_ROPE_MIN: 0.15,
  TEMPLATE_BLEND: 0.35,

  // Visual (consumers may use these for UI)
  RIBBON_SCALE: 1.6,
  HIST_MAX: 48,
});

export { DEFAULTS };

// ═══════════════════════════════════════════════════════════════
//   Pure audio helper functions
// ═══════════════════════════════════════════════════════════════

export function computeRMS(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = (arr[i] - 128) / 128;
    s += v * v;
  }
  return Math.sqrt(s / arr.length);
}

export function computePeak(arr) {
  let m = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = Math.abs(arr[i] - 128) / 128;
    if (v > m) m = v;
  }
  return m;
}

export function computeSpectralFlatness(freq) {
  let logSum = 0, linSum = 0, n = 0;
  const eps = 1e-10;
  for (let i = 1; i < freq.length; i++) {
    const v = freq[i] / 255 + eps;
    logSum += Math.log(v);
    linSum += v;
    n++;
  }
  if (n === 0) return 0;
  const geoMean = Math.exp(logSum / n);
  const ariMean = linSum / n;
  return ariMean > eps ? geoMean / ariMean : 0;
}

export function computeCrestFactor(timeArr) {
  let peak = 0, sumSq = 0;
  for (let i = 0; i < timeArr.length; i++) {
    const v = Math.abs(timeArr[i] - 128) / 128;
    if (v > peak) peak = v;
    sumSq += v * v;
  }
  const rms = Math.sqrt(sumSq / timeArr.length);
  return rms > 1e-8 ? peak / rms : 0;
}

export function computeSpectralCentroid(freq, sampleRate, fftSize) {
  let weightedSum = 0, totalEnergy = 0;
  const binWidth = sampleRate / fftSize;
  for (let i = 1; i < freq.length; i++) {
    const energy = freq[i] / 255;
    const hz = i * binWidth;
    weightedSum += energy * hz;
    totalEnergy += energy;
  }
  return totalEnergy > 1e-8 ? weightedSum / totalEnergy : 0;
}

// ═══════════════════════════════════════════════════════════════
//   JumpEngine class
// ═══════════════════════════════════════════════════════════════

export class JumpEngine {

  constructor(opts = {}) {
    // Merge defaults with overrides — store as mutable so tuning panel can adjust
    this.cfg = { ...DEFAULTS, ...opts };
    this._initState();
  }

  _initState() {
    this.state = S.IDLE;
    this.estimatedT = 0;
    this.lastPeakTime = 0;
    this.learnBuf = [];
    this.peakArmed = false;
    this.savedT = 0;

    // Counters
    this.totalJumps = 0;
    this.duJumps = 0;
    this.strikesDone = 0;
    this.bestRun = 0;
    this.currentRun = 0;

    // Calibration baseline
    this.noiseFloor = 0;
    this.sensitivity = 5;
    this.baselineFlatness = 0;
    this.baselineCentroid = 0;

    // Spectral template (learned from user's rope)
    this.ropeTemplate = { flatness: 0, crest: 0, centroid: 0, count: 0 };
  }

  // ─── Public API ───

  setCalibration(noiseFloor, baselineFlatness, baselineCentroid) {
    this.noiseFloor = noiseFloor;
    this.baselineFlatness = baselineFlatness;
    this.baselineCentroid = baselineCentroid;
  }

  setSensitivity(val) {
    this.sensitivity = val;
  }

  /** Update a single engine constant at runtime (for tuning panel). */
  setCfg(key, value) {
    if (key in this.cfg) this.cfg[key] = value;
  }

  getState() {
    return this.state;
  }

  getStats() {
    return {
      total: this.totalJumps,
      du: this.duJumps,
      singles: this.totalJumps - this.duJumps,
      currentRun: this.currentRun,
      bestRun: this.bestRun,
      strikesDone: this.strikesDone,
      estimatedT: this.estimatedT,
      bpm: this.estimatedT > 0 ? Math.round(60000 / this.estimatedT) : 0,
    };
  }

  reset() {
    this._initState();
  }

  /**
   * Process one frame of audio data.
   *
   * @param {Uint8Array} timeData  – time-domain samples (from getByteTimeDomainData)
   * @param {Uint8Array} freqData  – frequency-domain bins (from getByteFrequencyData)
   * @param {number}     now       – current timestamp in ms (e.g. performance.now())
   * @param {number}     sampleRate – audio context sample rate
   * @param {number}     fftSize   – analyser fftSize
   * @returns {Array<Object>} array of events (may be empty, usually 0 or 1 event)
   */
  processFrame(timeData, freqData, now, sampleRate, fftSize) {
    const events = [];
    const C = this.cfg;

    const peak = computePeak(timeData);
    const lvl = Math.min(peak * C.RIBBON_SCALE, 1);

    const thr = this._baseThreshold();
    const inWin = this._isInWindow(now);
    const effThr = inWin ? thr * C.THR_WIN_MULT : thr;

    // Pause watchdog
    const pauseEvt = this._checkPauseTimeout(now);
    if (pauseEvt) events.push(pauseEvt);

    // Hysteresis: must fall below 52% of effThr before re-arming
    if (lvl < effThr * 0.52) this.peakArmed = true;

    if (this.peakArmed && lvl >= effThr) {
      const gap = now - this.lastPeakTime;
      if (gap >= C.REFRACTORY) {
        // Spectral gate
        const flatness = computeSpectralFlatness(freqData);
        const crest = computeCrestFactor(timeData);
        const centroid = computeSpectralCentroid(freqData, sampleRate, fftSize);
        const sigScore = this._computeSignatureScore(flatness, crest, centroid);

        const gateActive = this.state === S.TRACKING || this.state === S.PAUSED;
        const passesGate = !gateActive || sigScore >= C.SIG_MIN_SCORE;

        if (passesGate) {
          this.peakArmed = false;
          // Update rope template during learning and early tracking
          if (this.state === S.LEARNING || (this.state === S.TRACKING && this.ropeTemplate.count < 20)) {
            this._updateRopeTemplate(flatness, crest, centroid);
          }
          const peakEvents = this._onPeak(now);
          // Annotate sigScore on each event
          for (const e of peakEvents) {
            e.sigScore = sigScore;
            e.flatness = flatness;
            e.crest = crest;
            e.centroid = centroid;
          }
          events.push(...peakEvents);
        } else {
          events.push({
            type: 'rejected',
            sigScore,
            flatness,
            crest,
            centroid,
          });
        }
      }
    }

    // Attach frame info for UI consumers
    const frameInfo = {
      peak,
      lvl,
      thr,
      effThr,
      inWindow: inWin,
      windowProgress: this._windowProgress(now),
    };

    return { events, frame: frameInfo };
  }

  // ─── Internal: threshold ───

  _baseThreshold() {
    const base = this.noiseFloor || 0.012;
    const mult = 1 + (10 - this.sensitivity) * 0.58;
    return Math.max(base * mult * 2.6, 0.016);
  }

  // ─── Internal: prediction window ───

  _windowProgress(now) {
    if (this.state !== S.TRACKING || this.estimatedT === 0 || this.lastPeakTime === 0) return -1;
    return (now - this.lastPeakTime) / this.estimatedT;
  }

  _isInWindow(now) {
    const C = this.cfg;
    const progress = this._windowProgress(now);
    if (progress < 0) return false;
    return progress >= (1 - C.W_FACTOR) && progress <= (1 + C.W_FACTOR);
  }

  // ─── Internal: pause watchdog ───

  _checkPauseTimeout(now) {
    const C = this.cfg;
    if (this.state !== S.TRACKING && this.state !== S.LEARNING) return null;
    if (this.lastPeakTime === 0) return null;
    const elapsed = now - this.lastPeakTime;
    const limit = this.state === S.TRACKING
      ? Math.min(Math.max(this.estimatedT * C.PAUSE_MULT, C.PAUSE_ABS), 5000)
      : C.PAUSE_ABS;
    if (elapsed > limit) {
      return this._enterPaused(now);
    }
    return null;
  }

  // ─── Internal: state transitions ───

  _enterLearning() {
    this.state = S.LEARNING;
    this.learnBuf = [];
    return { type: 'state', state: S.LEARNING };
  }

  _enterTracking() {
    this.state = S.TRACKING;
    return { type: 'state', state: S.TRACKING };
  }

  _enterPaused(now) {
    this.savedT = this.estimatedT;
    let runJumps = this.currentRun;
    if (this.currentRun > 0) {
      this.strikesDone++;
      if (this.currentRun > this.bestRun) this.bestRun = this.currentRun;
    }
    this.currentRun = 0;
    this.state = S.PAUSED;
    return { type: 'pause', closedRun: runJumps };
  }

  _resumeFromPause(now) {
    const C = this.cfg;
    const pauseDuration = now - this.lastPeakTime;
    const events = [];
    if (pauseDuration < C.RESUME_GRACE && this.savedT > 0) {
      this.estimatedT = this.savedT;
      events.push(this._enterTracking());
    } else {
      this.estimatedT = 0;
      events.push(this._enterLearning());
      this.learnBuf.push(now);
    }
    this.currentRun = 0;
    return events;
  }

  _lockTempo() {
    const gaps = [];
    for (let i = 1; i < this.learnBuf.length; i++) {
      gaps.push(this.learnBuf[i] - this.learnBuf[i - 1]);
    }
    gaps.sort((a, b) => a - b);
    const median = gaps[Math.floor(gaps.length / 2)];
    const validGaps = gaps.filter(g => g > median * 0.52);
    this.estimatedT = validGaps.reduce((a, b) => a + b, 0) / validGaps.length;
    return this._enterTracking();
  }

  _updateTempo(gap) {
    const C = this.cfg;
    if (this.estimatedT === 0) this.estimatedT = gap;
    else this.estimatedT = this.estimatedT * (1 - C.TEMPO_ALPHA) + gap * C.TEMPO_ALPHA;
  }

  // ─── Internal: peak handler ───

  _onPeak(now) {
    const C = this.cfg;
    const gap = now - this.lastPeakTime;
    const events = [];

    switch (this.state) {
      case S.IDLE:
        events.push(this._enterLearning());
        this.learnBuf.push(now);
        break;

      case S.LEARNING:
        this.learnBuf.push(now);
        if (this.learnBuf.length >= C.N_LEARN + 1) {
          events.push(this._lockTempo());
        }
        break;

      case S.TRACKING:
        if (gap < this.estimatedT * C.DU_RATIO && gap >= C.REFRACTORY) {
          // Double under
          this.duJumps++;
          this.lastPeakTime = now;
          events.push({ type: 'du' });
          return events;
        } else {
          this._updateTempo(gap);
        }
        break;

      case S.PAUSED:
        events.push(...this._resumeFromPause(now));
        break;
    }

    // Confirmed single jump
    this.totalJumps++;
    this.currentRun++;
    this.lastPeakTime = now;
    events.push({ type: 'jump' });
    return events;
  }

  // ─── Internal: spectral scoring ───

  _computeSignatureScore(flatness, crest, centroid) {
    const C = this.cfg;

    const adjFlat = Math.max(0, flatness - this.baselineFlatness * 0.5);
    let flatScore = Math.min(1, adjFlat / 0.6);

    let crestScore = Math.min(1, Math.max(0, (crest - C.CREST_ROPE_MIN) / (C.CREST_NORM_MAX - C.CREST_ROPE_MIN)));

    let centScore = Math.min(1, Math.max(0, (centroid - C.CENT_ROPE_MIN_HZ) / (C.CENT_ROPE_MAX_HZ - C.CENT_ROPE_MIN_HZ)));

    if (this.ropeTemplate.count >= 3) {
      const tFlat = 1 - Math.min(1, Math.abs(flatness - this.ropeTemplate.flatness) / 0.4);
      const tCrest = 1 - Math.min(1, Math.abs(crest - this.ropeTemplate.crest) / 8);
      const tCent = 1 - Math.min(1, Math.abs(centroid - this.ropeTemplate.centroid) / 4000);
      flatScore = flatScore * (1 - C.TEMPLATE_BLEND) + tFlat * C.TEMPLATE_BLEND;
      crestScore = crestScore * (1 - C.TEMPLATE_BLEND) + tCrest * C.TEMPLATE_BLEND;
      centScore = centScore * (1 - C.TEMPLATE_BLEND) + tCent * C.TEMPLATE_BLEND;
    }

    return flatScore * C.SIG_WEIGHT_FLAT + crestScore * C.SIG_WEIGHT_CREST + centScore * C.SIG_WEIGHT_CENT;
  }

  _updateRopeTemplate(flatness, crest, centroid) {
    const c = this.ropeTemplate.count;
    this.ropeTemplate.flatness = (this.ropeTemplate.flatness * c + flatness) / (c + 1);
    this.ropeTemplate.crest = (this.ropeTemplate.crest * c + crest) / (c + 1);
    this.ropeTemplate.centroid = (this.ropeTemplate.centroid * c + centroid) / (c + 1);
    this.ropeTemplate.count++;
  }
}
