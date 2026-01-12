/**
 * Environment configuration for the frontend.
 *
 * All configuration values can be overridden via environment variables
 * prefixed with VITE_.
 */

interface EnvConfig {
  /** Backend API base URL */
  apiBaseUrl: string;

  /** WebSocket URL for streaming */
  wsBaseUrl: string;

  /** Development mode flag */
  isDevelopment: boolean;

  /** Production mode flag */
  isProduction: boolean;

  /** Health check interval in milliseconds */
  healthCheckInterval: number;

  /** Default lattice size */
  defaultLatticeSize: number;

  /** Default temperature (critical temperature) */
  defaultTemperature: number;

  /** Maximum lattice size allowed */
  maxLatticeSize: number;
}

function getEnvConfig(): EnvConfig {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
  const wsProtocol = apiBaseUrl.startsWith('https') ? 'wss' : 'ws';
  const wsBaseUrl = import.meta.env.VITE_WS_BASE_URL ||
    apiBaseUrl.replace(/^http/, wsProtocol);

  return {
    apiBaseUrl,
    wsBaseUrl,
    isDevelopment: import.meta.env.DEV,
    isProduction: import.meta.env.PROD,
    healthCheckInterval: Number(import.meta.env.VITE_HEALTH_CHECK_INTERVAL) || 10000,
    defaultLatticeSize: Number(import.meta.env.VITE_DEFAULT_LATTICE_SIZE) || 32,
    defaultTemperature: Number(import.meta.env.VITE_DEFAULT_TEMPERATURE) || 2.27,
    maxLatticeSize: Number(import.meta.env.VITE_MAX_LATTICE_SIZE) || 128,
  };
}

export const env = getEnvConfig();

export default env;
