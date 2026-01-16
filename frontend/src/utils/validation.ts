/**
 * Validation utilities for simulation parameters.
 */

import {
  MIN_LATTICE_SIZE,
  MAX_LATTICE_SIZE,
  MIN_TEMPERATURE,
  MAX_TEMPERATURE,
} from '../constants/physics';

/**
 * Result of a validation check.
 */
export interface ValidationResult {
  isValid: boolean;
  error?: string;
}

/**
 * Validate temperature value.
 */
export function validateTemperature(temperature: number): ValidationResult {
  if (typeof temperature !== 'number' || isNaN(temperature)) {
    return { isValid: false, error: 'Temperature must be a number' };
  }

  if (temperature < MIN_TEMPERATURE) {
    return {
      isValid: false,
      error: `Temperature must be at least ${MIN_TEMPERATURE}`,
    };
  }

  if (temperature > MAX_TEMPERATURE) {
    return {
      isValid: false,
      error: `Temperature must be at most ${MAX_TEMPERATURE}`,
    };
  }

  return { isValid: true };
}

/**
 * Validate lattice size value.
 */
export function validateLatticeSize(size: number): ValidationResult {
  if (typeof size !== 'number' || isNaN(size)) {
    return { isValid: false, error: 'Lattice size must be a number' };
  }

  if (!Number.isInteger(size)) {
    return { isValid: false, error: 'Lattice size must be an integer' };
  }

  if (size < MIN_LATTICE_SIZE) {
    return {
      isValid: false,
      error: `Lattice size must be at least ${MIN_LATTICE_SIZE}`,
    };
  }

  if (size > MAX_LATTICE_SIZE) {
    return {
      isValid: false,
      error: `Lattice size must be at most ${MAX_LATTICE_SIZE}`,
    };
  }

  return { isValid: true };
}

/**
 * Validate number of steps.
 */
export function validateNumSteps(steps: number): ValidationResult {
  if (typeof steps !== 'number' || isNaN(steps)) {
    return { isValid: false, error: 'Number of steps must be a number' };
  }

  if (!Number.isInteger(steps)) {
    return { isValid: false, error: 'Number of steps must be an integer' };
  }

  if (steps < 1) {
    return { isValid: false, error: 'Number of steps must be at least 1' };
  }

  if (steps > 1000) {
    return { isValid: false, error: 'Number of steps must be at most 1000' };
  }

  return { isValid: true };
}

/**
 * Validate number of samples.
 */
export function validateNumSamples(samples: number): ValidationResult {
  if (typeof samples !== 'number' || isNaN(samples)) {
    return { isValid: false, error: 'Number of samples must be a number' };
  }

  if (!Number.isInteger(samples)) {
    return { isValid: false, error: 'Number of samples must be an integer' };
  }

  if (samples < 1) {
    return { isValid: false, error: 'Number of samples must be at least 1' };
  }

  if (samples > 10000) {
    return { isValid: false, error: 'Number of samples must be at most 10000' };
  }

  return { isValid: true };
}

/**
 * Validate spin configuration.
 */
export function validateSpinConfiguration(
  spins: number[][] | null
): ValidationResult {
  if (spins === null) {
    return { isValid: true }; // null is valid (empty state)
  }

  if (!Array.isArray(spins)) {
    return { isValid: false, error: 'Spin configuration must be an array' };
  }

  if (spins.length === 0) {
    return { isValid: false, error: 'Spin configuration cannot be empty' };
  }

  const size = spins.length;

  for (let i = 0; i < size; i++) {
    if (!Array.isArray(spins[i])) {
      return { isValid: false, error: `Row ${i} must be an array` };
    }

    if (spins[i].length !== size) {
      return {
        isValid: false,
        error: 'Spin configuration must be square (NxN)',
      };
    }

    for (let j = 0; j < size; j++) {
      const spin = spins[i][j];
      if (spin !== 1 && spin !== -1) {
        return {
          isValid: false,
          error: `Invalid spin value at (${i}, ${j}): expected 1 or -1`,
        };
      }
    }
  }

  return { isValid: true };
}

/**
 * Validate all sampling parameters at once.
 */
export function validateSamplingParams(params: {
  temperature: number;
  latticeSize: number;
  numSteps: number;
}): ValidationResult {
  const tempResult = validateTemperature(params.temperature);
  if (!tempResult.isValid) return tempResult;

  const sizeResult = validateLatticeSize(params.latticeSize);
  if (!sizeResult.isValid) return sizeResult;

  const stepsResult = validateNumSteps(params.numSteps);
  if (!stepsResult.isValid) return stepsResult;

  return { isValid: true };
}

/**
 * Clamp a value to a valid range.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Clamp temperature to valid range.
 */
export function clampTemperature(temperature: number): number {
  return clamp(temperature, MIN_TEMPERATURE, MAX_TEMPERATURE);
}

/**
 * Clamp lattice size to valid range.
 */
export function clampLatticeSize(size: number): number {
  return Math.round(clamp(size, MIN_LATTICE_SIZE, MAX_LATTICE_SIZE));
}

/**
 * Check if a value is within a valid range.
 */
export function isInRange(
  value: number,
  min: number,
  max: number
): boolean {
  return value >= min && value <= max;
}
