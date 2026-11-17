// GTA VI launch-day Twitch forecast — entry point (Node 22, standard library only, no network).
// Reads every *.in in /input, writes the matching *.out forecast. The model is a comparable-launch
// log-ratio forecaster in lib/model.ts. Variants below differ only by the options, so a single build
// can be tuned by the MAKEFUN_VARIANT environment variable during local backtests.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { forecast, type Case, type Options } from './lib/model.ts';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const VARIANTS: Record<string, Options> = {
  // Blend the comparable shape with persistence; tighten the spread a little.
  default: { shrink: 0.35, minSpread: 0.16, weightPower: 0.25 },
  // Trust the comparables more, wider tails.
  bold: { shrink: 0.15, minSpread: 0.20, weightPower: 0.35 },
  // Stay close to persistence with a modest comparable tilt.
  safe: { shrink: 0.55, minSpread: 0.14, weightPower: 0.18 },
};
const opt = VARIANTS[process.env.MAKEFUN_VARIANT ?? 'default'] ?? VARIANTS.default!;

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const hours = forecast(c, opt);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
process.stdout.write(`forecast: ${inputs.length} case(s), variant=${process.env.MAKEFUN_VARIANT ?? 'default'}\n`);
