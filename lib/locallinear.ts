// Local-linear comparable forecaster: instead of one global OLS line, fit a weighted line around the
// target's own log(hour-1) so comparables with a similar checkpoint audience dominate. Comparable
// quality tiers (parsed from the case's unlock notes) down-weight early-access-contaminated and weak
// analogues. Quantiles come from the fit's residual spread.
import type { Case } from './model.ts';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.674490, 0, 0.674490, 1.644854];
interface HP { from: string; viewers: number }

function curve(comp: { day_start: string; hourly: HP[]; unlock?: string }): Array<number | null> {
  const start = Date.parse(comp.day_start);
  const day = comp.day_start.slice(0, 10);
  const out: Array<number | null> = Array(24).fill(null);
  for (const p of comp.hourly) {
    if (!p.from.startsWith(day)) continue;
    const h = Math.floor((Date.parse(p.from) - start) / 3600000);
    if (h >= 0 && h < 24 && p.viewers > 0 && out[h] === null) out[h] = p.viewers;
  }
  return out;
}
function valueAt(c: Array<number | null>, h: number): number | null {
  if (c[h] !== null && c[h]! > 0) return c[h]!;
  const known = c.map((v, hh) => ({ v, hh })).filter((p) => p.v !== null && p.v! > 0);
  if (!known.length) return null;
  let lo = known[0]!, hi = known[known.length - 1]!;
  for (const p of known) { if (p.hh <= h) lo = p; if (p.hh >= h) { hi = p; break; } }
  if (lo.hh === hi.hh) return lo.v;
  if (h >= hi.hh) return hi.v;
  const f = (h - lo.hh) / (hi.hh - lo.hh);
  return Math.exp(Math.log(lo.v!) * (1 - f) + Math.log(hi.v!) * f);
}
function tierWeight(note: string): number {
  const s = (note || '').toLowerCase();
  if (s.includes('early-access-contaminated') || s.includes('contaminated')) return 0.25;
  if (s.includes('weak analogue') || s.includes('weak gta') || /tier:\s*weak/.test(s)) return 0.35;
  if (s.includes('target-history')) return 0.6;
  return 1;
}
export interface Options { h: number; spreadFloor: number; spreadScale: number; ridge: number }

export function forecastLocal(c: Case, opt: Options): { hour: number; q: number[] }[] {
  const first = c.known.find((k) => k.hour === 1);
  const p1 = Math.max(1, first ? first.viewers : 1);
  const xt = Math.log(p1);
  const rows: Array<{ x: number; w: number; h: Record<number, number> }> = [];
  for (const comp of c.comparables) {
    const cv = curve(comp);
    const base = valueAt(cv, 1);
    if (base === null || base <= 0) continue;
    const x = Math.log(base);
    const hw: Record<number, number> = {};
    for (const f of c.forecast) {
      const v = valueAt(cv, f.hour);
      if (v !== null && v > 0) hw[f.hour] = Math.log(v / base);
    }
    if (Object.keys(hw).length < 12) continue;
    const dist = Math.abs(x - xt);
    const kw = Math.exp(-(dist * dist) / (2 * opt.h * opt.h));
    rows.push({ x, w: kw * tierWeight(comp.unlock || ''), h: hw });
  }
  if (rows.length < 3) rows.push({ x: xt, w: 1, h: {} });

  return c.forecast.map((f) => {
    const hr = f.hour;
    const pts = rows.filter((r) => r.h[hr] !== undefined);
    if (pts.length < 3) {
      const sd = opt.spreadFloor + opt.spreadScale * Math.sqrt(Math.max(1, hr - 1));
      return { hour: hr, q: Z.map((z) => clampQ(p1 * Math.exp(z * sd))) };
    }
    // Weighted local-linear (fit y = a + b*(x-xt) around xt).
    let Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for (const r of pts) {
      const dx = r.x - xt, w = r.w;
      Sw += w; Sx += w * dx; Sy += w * r.h[hr]!; Sxx += w * dx * dx; Sxy += w * dx * r.h[hr]!;
    }
    const den = Sw * Sxx - Sx * Sx + opt.ridge;
    const a = (Sy * Sxx - Sx * Sxy) / den;   // prediction at dx=0 (the target)
    const b = (Sw * Sxy - Sx * Sy) / den;
    const mu = a; // evaluated at the target's xt
    let res = 0;
    for (const r of pts) { const e = r.h[hr]! - (a + b * (r.x - xt)); res += r.w * e * e; }
    const sd0 = Math.sqrt(res / Math.max(1, Sw - 2));
    const sd = Math.max(opt.spreadFloor, sd0 * opt.spreadScale);
    const q = LEVELS.map((tau, i) => clampQ(p1 * Math.exp(mu + Z[i] * sd)));
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
    return { hour: hr, q };
  });
}
function clampQ(x: number): number { return Math.max(0.01, Math.min(1e9, Math.round(x * 100) / 100)); }
