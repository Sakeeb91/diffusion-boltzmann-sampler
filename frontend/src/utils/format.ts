/**
 * Formatting utilities for display values.
 */

import { T_CRITICAL } from '../constants/physics';

/**
 * Format a number to a fixed number of decimal places.
 */
export function formatNumber(value: number, decimals = 3): string {
  return value.toFixed(decimals);
}

/**
 * Format temperature with optional critical temperature indicator.
 */
export function formatTemperature(
  temperature: number,
  options: { showCritical?: boolean; decimals?: number } = {}
): string {
  const { showCritical = true, decimals = 2 } = options;
  const formatted = temperature.toFixed(decimals);

  if (showCritical && Math.abs(temperature - T_CRITICAL) < 0.1) {
    return `${formatted} (T_c)`;
  }

  return formatted;
}

/**
 * Format energy per spin value.
 */
export function formatEnergy(energy: number, decimals = 3): string {
  const sign = energy >= 0 ? '+' : '';
  return `${sign}${energy.toFixed(decimals)}`;
}

/**
 * Format magnetization value with sign.
 */
export function formatMagnetization(magnetization: number, decimals = 3): string {
  const sign = magnetization >= 0 ? '+' : '';
  return `${sign}${magnetization.toFixed(decimals)}`;
}

/**
 * Format lattice size as NxN.
 */
export function formatLatticeSize(size: number): string {
  return `${size} \u00D7 ${size}`;
}

/**
 * Format percentage value.
 */
export function formatPercentage(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format frame count.
 */
export function formatFrameCount(current: number, total: number): string {
  return `${current + 1} / ${total}`;
}

/**
 * Format time in milliseconds to human readable.
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms.toFixed(0)}ms`;
  }
  if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  }
  const minutes = Math.floor(ms / 60000);
  const seconds = ((ms % 60000) / 1000).toFixed(0);
  return `${minutes}m ${seconds}s`;
}

/**
 * Format scientific notation for very small or large numbers.
 */
export function formatScientific(value: number, significantDigits = 3): string {
  if (value === 0) return '0';

  const absValue = Math.abs(value);
  if (absValue >= 0.001 && absValue < 10000) {
    return value.toPrecision(significantDigits);
  }

  return value.toExponential(significantDigits - 1);
}

/**
 * Format autocorrelation time with speedup indicator.
 */
export function formatAutocorrelationTime(
  tau: number,
  reference?: number
): string {
  const formatted = tau.toFixed(1);

  if (reference && tau < reference) {
    const speedup = (reference / tau).toFixed(0);
    return `${formatted} (${speedup}x faster)`;
  }

  return formatted;
}

/**
 * Get phase description based on temperature.
 */
export function getPhaseDescription(temperature: number): string {
  if (temperature < T_CRITICAL - 0.2) {
    return 'Ordered (Ferromagnetic)';
  }
  if (temperature > T_CRITICAL + 0.2) {
    return 'Disordered (Paramagnetic)';
  }
  return 'Critical (Phase Transition)';
}

/**
 * Get color class for temperature indicator.
 */
export function getTemperatureColor(temperature: number): string {
  if (temperature < T_CRITICAL - 0.2) {
    return 'text-blue-400';
  }
  if (temperature > T_CRITICAL + 0.2) {
    return 'text-red-400';
  }
  return 'text-yellow-400';
}
