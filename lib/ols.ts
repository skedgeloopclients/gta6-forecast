// OLS-on-log-hour-1 forecaster: the shape of a launch day depends on how much of the audience is
// already awake at the checkpoint. For each comparable we learn its own hour-1 print and the log ratio
// of every later hour to it. Across comparables we fit, per hour, log(ratio) = a + b * (log(v1) - xbar),
// then apply that line at the target's log(p1). Quantiles come from the fit's residual spread.
import type { Case } from './model.ts';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.674490, 0, 0.674490, 1.644854];
interface HP { from: string; viewers: number }

function curve(caseObj: Case, comp: { day_start: string; hourly: HP[] }): Array<number | null> {
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

export interface Options { ridge: number; spreadFloor: number; spreadScale: number; shrink: number }

export function forecastOLS(c: Case, opt: Options): { hour: number; q: number[] }[] {
  const first = c.known.find((k) => k.hour === 1);
  const p1 = Math.max(1, first ? first.viewers : 1);
  const rows: Array<{ x: number; h: Record<number, number> }> = [];
  for (const comp of c.comparables) {
    const cv = curve(c, comp);
    const base = valueAt(cv, 1);
    if (base === null || base <= 0) continue;
    const x = Math.log(base);
    const h: Record<number, number> = {};
    for (const f of c.forecast) {
      const v = valueAt(cv, f.hour);
      if (v !== null && v > 0) h[f.hour] = Math.log(v / base);
    }
    if (Object.keys(h).length >= 12) rows.push({ x, h });
  }
  const xbar = rows.length ? rows.reduce((s, r) => s + r.x, 0) / rows.length : 0;
  const xTarget = Math.log(p1);
  const dx = xTarget - xbar;
  const xx = rows.reduce((s, r) => s + (r.x - xbar) ** 2, 0) + opt.ridge;

  return c.forecast.map((f) => {
    const h = f.hour;
    const pts = rows.filter((r) => r.h[h] !== undefined);
    if (pts.length < 3) {
      const sd = opt.spreadFloor + opt.spreadScale * Math.sqrt(Math.max(1, h - 1));
      return { hour: h, q: Z.map((z) => clampQ(p1 * Math.exp(z * sd))) };
    }
    const ybar = pts.reduce((s, r) => s + r.h[h]!, 0) / pts.length;
    const sxy = pts.reduce((s, r) => s + (r.x - xbar) * (r.h[h]! - ybar), 0);
    const b = sxy / xx;
    const a = ybar; // centered regression; prediction at dx
    const mu = a + b * dx;
    const resid = pts.map((r) => r.h[h]! - (ybar + b * (r.x - xbar)));
    const sd0 = Math.sqrt(resid.reduce((s, e) => s + e * e, 0) / Math.max(1, pts.length - 2));
    const sd = Math.max(opt.spreadFloor, sd0 * opt.spreadScale);
    const q = LEVELS.map((tau, i) => clampQ(p1 * Math.exp(mu + Z[i] * sd)));
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
    return { hour: h, q };
  });
}
function clampQ(x: number): number { return Math.max(0.01, Math.min(1e9, Math.round(x * 100) / 100)); }
