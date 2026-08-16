/**
 * Deterministic, dependency-free PRNG for the robustness layer.
 *
 * Everything the layer randomizes flows through one of these helpers so that a
 * (seed, config, input) triple always yields the identical output — the
 * interface shared with the mpmify project (explicit seeded rng,
 * never Math.random).
 */

/** sfc32 PRNG. Returns a function yielding floats in [0, 1). */
export function makeRng(seed) {
  // String seeds are hashed (FNV-1a); numeric seeds are splittmix-scrambled.
  let a, b, c, d;
  if (typeof seed === 'string') {
    let h = 0x811c9dc5;
    for (let i = 0; i < seed.length; i++) {
      h ^= seed.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    a = h >>> 0; b = (h ^ 0x9e3779b9) >>> 0; c = (h ^ 0x85ebca6b) >>> 0; d = (h ^ 0xc2b2ae35) >>> 0;
  } else {
    const s = (seed >>> 0) || 1;
    a = s; b = (s ^ 0x9e3779b9) >>> 0; c = (s ^ 0x85ebca6b) >>> 0; d = (s ^ 0xc2b2ae35) >>> 0;
  }
  // Warm up so nearby seeds decorrelate.
  const next = () => {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    const t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    const r = (t + d) | 0;
    c = (c + r) | 0;
    return (r >>> 0) / 4294967296;
  };
  for (let i = 0; i < 12; i++) next();
  return next;
}

/** Uniform float in [lo, hi). */
export const uniform = (rng, lo, hi) => lo + (hi - lo) * rng();

/** Uniform integer in [lo, hi] inclusive. */
export const randint = (rng, lo, hi) => lo + Math.floor(rng() * (hi - lo + 1));

/** Standard normal via Box–Muller (one draw per call, no caching → stateless). */
export function normal(rng, mean = 0, std = 1) {
  let u = 0;
  while (u === 0) u = rng(); // avoid log(0)
  const v = rng();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Bernoulli trial. */
export const chance = (rng, p) => rng() < p;

/** Pick one element; with `weights` (same length, non-negative), weighted pick. */
export function pick(rng, items, weights) {
  if (!weights) return items[Math.floor(rng() * items.length)];
  let total = 0;
  for (const w of weights) total += w;
  let r = rng() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r < 0) return items[i];
  }
  return items[items.length - 1];
}

/** Poisson-distributed count (Knuth; fine for small lambda). */
export function poisson(rng, lambda) {
  if (lambda <= 0) return 0;
  const limit = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rng();
  } while (p > limit);
  return k - 1;
}
