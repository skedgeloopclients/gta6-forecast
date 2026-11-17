// GTA VI launch-day Twitch forecaster.
//
// The only anchor is the case: hour 1's print p1. Every other hour is inferred from the shape of past
// big console launches in the case's `comparables` (each one's own release-day hourly viewers). For each
// comparable c and hour h we learn the log-ratio L_c(h) = ln(viewers_c(h) / viewers_c(1)); the forecast
// is p1 * exp(a central estimate of L(h)) with quantiles from the spread of comparable ratios, shrunk
// toward persistence because GTA VI's UTC-hour shape (progressive unlock) differs from any one analog.
// No network, no model, deterministic, standard library only.

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.674490, 0, 0.674490, 1.644854];

interface HourPoint { from: string; viewers: number; channels?: number | null; reads?: number }
interface Comparable { name: string; day_start: string; unlock: string; platforms: string; hourly: HourPoint[]; source: string }
export interface Case {
  id: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  history: Array<{ at: string; viewers: number }>;
  history_hourly: Array<{ from: string; viewers: number; reads: number }>;
  comparables: Comparable[];
}

/** Timestamp in ms for an ISO UTC string. */
function t(s: string): number { return Date.parse(s); }

/** The comparable's release-day hourly curve, indexed by UTC hour 0..23 (null where the hour is absent). */
function dayCurve(c: Comparable): Array<number | null> {
  const start = t(c.day_start);
  if (!Number.isFinite(start)) return Array(24).fill(null);
  const day0 = new Date(start).toISOString().slice(0, 10);
  const out: Array<number | null> = Array(24).fill(null);
  for (const p of c.hourly) {
    if (!p.from.startsWith(day0)) continue;
    const h = Math.floor((t(p.from) - start) / 3600000);
    if (h >= 0 && h < 24 && p.viewers > 0 && out[h] === null) out[h] = p.viewers;
  }
  return out;
}

/** Weighted quantile of a list of [value, weight] at level tau. */
function wquantile(pairs: Array<[number, number]>, tau: number): number {
  const s = pairs.slice().sort((a, b) => a[0] - b[0]);
  const total = s.reduce((m, p) => m + p[1], 0);
  if (total <= 0) return s.length ? s[Math.floor(tau * (s.length - 1))][0] : 0;
  const target = tau * total;
  let acc = 0;
  for (const [v, w] of s) { acc += w; if (acc >= target) return v; }
  return s[s.length - 1][0];
}

/** Across-comparable dispersion of the log-ratio at hour h, in nats. */
function logSpread(ratios: number[]): number {
  if (ratios.length < 2) return 0.18;
  const m = ratios.reduce((a, b) => a + b, 0) / ratios.length;
  const v = ratios.reduce((a, b) => a + (b - m) * (b - m), 0) / (ratios.length - 1);
  return Math.sqrt(Math.max(v, 0.04 * 0.04));
}

export interface Options { shrink: number; minSpread: number; weightPower: number; }

/** Forecast one case. `shrink` blends the comparable log-shape toward persistence (0 = pure comparables). */
export function forecast(c: Case, opt: Options): { hour: number; q: number[] }[] {
  const first = c.known.find((k) => k.hour === 1);
  const p1 = Math.max(1, first ? first.viewers : 1);
  // Learn each comparable's release-day shape relative to its own hour 1.
  const shapes: Array<{ curve: Array<number | null>; base: number; weight: number; name: string }> = [];
  for (const comp of c.comparables) {
    const curve = dayCurve(comp);
    const base = curve[1] ?? null;
    if (base === null || base <= 0) continue;
    let peak = 1;
    for (const v of curve) if (v !== null && v > peak) peak = v;
    // Weight big launches more: their audience dynamics resemble GTA VI's better than a small title's.
    shapes.push({ curve, base, weight: Math.pow(Math.max(1, peak), opt.weightPower), name: comp.name });
  }

  const hours = c.forecast.map((f) => f.hour).sort((a, b) => a - b);
  return hours.map((h) => {
    const pairs: Array<[number, number]> = [];
    const logRatios: number[] = [];
    for (const s of shapes) {
      // Some analogs unlock late, so their UTC-hour curve has no print at early hours; interpolate
      // across the hours they do have, then extrapolate flat beyond the last one.
      let v = s.curve[h];
      if (v === null || v <= 0) {
        const known = s.curve.map((val, hh) => ({ val, hh })).filter((p) => p.val !== null && p.val! > 0);
        if (known.length === 0) continue;
        let lo = known[0]!, hi = known[known.length - 1]!;
        for (const p of known) { if (p.hh <= h) lo = p; if (p.hh >= h) { hi = p; break; } }
        if (lo.hh === hi.hh) v = lo.val;
        else if (h >= hi.hh) v = hi.val; // flat past the last print
        else {
          const f = (h - lo.hh) / (hi.hh - lo.hh);
          v = Math.exp(Math.log(lo.val!) * (1 - f) + Math.log(hi.val!) * f);
        }
      }
      if (v === null || !Number.isFinite(v) || v <= 0) continue;
      const lr = Math.log(v / s.base);
      const blended = (1 - opt.shrink) * lr; // shrink toward 0 (persistence)
      logRatios.push(blended);
      pairs.push([blended, s.weight]);
    }
    if (pairs.length === 0) {
      const lr = 0;
      const sd = 0.15 * Math.sqrt(Math.max(1, h - 1));
      return { hour: h, q: Z.map((z) => Math.max(0.01, Math.min(1e9, round2(p1 * Math.exp(lr + z * sd))))) };
    }
    const spread = Math.max(opt.minSpread, Number.isFinite(logSpread(logRatios)) ? logSpread(logRatios) : opt.minSpread, 0.12 * Math.sqrt(Math.max(1, h - 1)));
    // Quantile of the comparable log-ratio distribution, plus a normal-shaped widening for tail risk.
    const med = wquantile(pairs, 0.5);
    const q = LEVELS.map((tau, i) => {
      const empirical = wquantile(pairs, tau);
      const normal = med + Z[i] * spread;
      const lr = 0.5 * empirical + 0.5 * normal;
      return Math.max(0.01, Math.min(1e9, round2(p1 * Math.exp(lr))));
    });
    // Enforce non-decreasing.
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
    return { hour: h, q };
  });
}

function round2(x: number): number { return Math.round(x * 100) / 100; }
