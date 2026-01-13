/**
 * Physical constants and parameters for the Ising model.
 */

/** Critical temperature for 2D Ising model: T_c = 2/ln(1+√2) ≈ 2.269 */
export const T_CRITICAL = 2.269;

/** Coupling constant (ferromagnetic) */
export const DEFAULT_J = 1.0;

/** External field (default zero) */
export const DEFAULT_H = 0.0;

/** Default lattice size */
export const DEFAULT_LATTICE_SIZE = 32;

/** Minimum lattice size */
export const MIN_LATTICE_SIZE = 8;

/** Maximum lattice size */
export const MAX_LATTICE_SIZE = 128;

/** Temperature range */
export const MIN_TEMPERATURE = 0.1;
export const MAX_TEMPERATURE = 10.0;

/** Temperature presets for interesting physics */
export const TEMPERATURE_PRESETS = {
  /** Low temperature - ordered phase */
  LOW: 1.5,
  /** Critical temperature - phase transition */
  CRITICAL: T_CRITICAL,
  /** High temperature - disordered phase */
  HIGH: 3.5,
} as const;

/** Default MCMC parameters */
export const DEFAULT_MCMC = {
  sweeps: 10,
  burnIn: 100,
  samples: 10,
} as const;

/** Default diffusion parameters */
export const DEFAULT_DIFFUSION = {
  steps: 100,
  samples: 1,
} as const;

/** Animation frame rate (ms between frames) */
export const ANIMATION_FRAME_DELAY = 50;

/** WebSocket reconnection delay (ms) */
export const WS_RECONNECT_DELAY = 1000;
