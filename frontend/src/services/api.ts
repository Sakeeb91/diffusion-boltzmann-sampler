import { APIError, TimeoutError, NetworkError, WebSocketError, ErrorCode } from '../utils/errors';
import { env } from '../config/env';

const API_BASE = env.apiBaseUrl;
const WS_BASE = env.wsBaseUrl;

/** Default timeout for API requests in milliseconds */
const DEFAULT_TIMEOUT = 30000;

/** Timeout for long-running operations like sampling */
const SAMPLING_TIMEOUT = 120000;

export interface SampleResponse {
  samples: number[][][];
  energies: number[];
  magnetizations: number[];
  temperature: number;
  lattice_size: number;
}

export interface AnalysisResponse {
  mcmc: {
    magnetization: { M: number[]; P_M: number[] };
    energy: { E: number[]; P_E: number[] };
    correlation: { r: number[]; C_r: number[] };
    autocorrelation_time: number;
    mean_mag: number;
    var_mag: number;
  };
  diffusion: {
    magnetization: { M: number[]; P_M: number[] };
    energy: { E: number[]; P_E: number[] };
    correlation: { r: number[]; C_r: number[] };
    mean_mag: number;
    var_mag: number;
  };
  comparison_metrics: Record<string, number>;
  temperature: number;
  lattice_size: number;
}

export interface PhaseDiagramResponse {
  temperatures: number[];
  mean_magnetization: number[];
  std_magnetization: number[];
  T_critical: number;
  lattice_size: number;
}

export interface HealthResponse {
  status: string;
  version?: string;
  title?: string;
  features?: Record<string, boolean>;
}

/**
 * Fetch with timeout support.
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new TimeoutError(`Request timed out after ${timeoutMs}ms`, timeoutMs);
    }
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new NetworkError('Network request failed. Is the backend running?');
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Handle API response and throw appropriate errors.
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errorData = await response.json();
      message = errorData.detail || errorData.message || message;
    } catch {
      // Use default message if JSON parsing fails
    }
    throw new APIError(message, response.status);
  }
  return response.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetchWithTimeout(`${API_BASE}/health`, {}, 5000);
  return handleResponse<HealthResponse>(response);
}

export async function getConfig(): Promise<Record<string, unknown>> {
  const response = await fetchWithTimeout(`${API_BASE}/config`);
  return handleResponse<Record<string, unknown>>(response);
}

export async function sampleMCMC(params: {
  temperature: number;
  lattice_size: number;
  n_samples: number;
  n_sweeps?: number;
  burn_in?: number;
}): Promise<SampleResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE}/sample/mcmc`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    },
    SAMPLING_TIMEOUT
  );
  return handleResponse<SampleResponse>(response);
}

export async function sampleDiffusion(params: {
  temperature: number;
  lattice_size: number;
  n_samples: number;
  num_steps?: number;
}): Promise<SampleResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE}/sample/diffusion`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    },
    SAMPLING_TIMEOUT
  );
  return handleResponse<SampleResponse>(response);
}

export async function getRandomConfiguration(
  latticeSize: number
): Promise<{
  spins: number[][];
  energy: number;
  magnetization: number;
}> {
  const response = await fetchWithTimeout(
    `${API_BASE}/sample/random?lattice_size=${latticeSize}`
  );
  return handleResponse<{
    spins: number[][];
    energy: number;
    magnetization: number;
  }>(response);
}

export async function compareSamplers(params: {
  temperature: number;
  lattice_size: number;
  n_samples: number;
}): Promise<AnalysisResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE}/analysis/compare`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    },
    SAMPLING_TIMEOUT
  );
  return handleResponse<AnalysisResponse>(response);
}

export async function getPhaseDiagram(params: {
  lattice_size?: number;
  n_temps?: number;
  n_samples_per_temp?: number;
}): Promise<PhaseDiagramResponse> {
  const queryParams = new URLSearchParams();
  if (params.lattice_size) queryParams.set('lattice_size', params.lattice_size.toString());
  if (params.n_temps) queryParams.set('n_temps', params.n_temps.toString());
  if (params.n_samples_per_temp)
    queryParams.set('n_samples_per_temp', params.n_samples_per_temp.toString());

  const response = await fetchWithTimeout(
    `${API_BASE}/analysis/phase_diagram?${queryParams}`,
    {},
    SAMPLING_TIMEOUT
  );
  return handleResponse<PhaseDiagramResponse>(response);
}

export function createSamplingWebSocket(
  onFrame: (data: { spins: number[][]; energy: number; magnetization: number; t?: number }) => void,
  onDone: () => void,
  onError: (error: string) => void
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/ws/sample`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'frame') {
        onFrame(data);
      } else if (data.type === 'done') {
        onDone();
      } else if (data.type === 'error') {
        const wsError = new WebSocketError(
          data.message || 'WebSocket error',
          ErrorCode.WS_MESSAGE_ERROR
        );
        onError(wsError.getUserMessage());
      }
    } catch {
      const parseError = new WebSocketError(
        'Failed to parse WebSocket message',
        ErrorCode.WS_MESSAGE_ERROR
      );
      onError(parseError.getUserMessage());
    }
  };

  ws.onerror = () => {
    const connError = new WebSocketError(
      'WebSocket connection failed',
      ErrorCode.WS_CONNECTION_ERROR
    );
    onError(connError.getUserMessage());
  };

  ws.onclose = (event) => {
    if (!event.wasClean) {
      const closeError = new WebSocketError(
        `Connection closed unexpectedly (code: ${event.code})`,
        ErrorCode.WS_CLOSED
      );
      onError(closeError.getUserMessage());
    }
  };

  return ws;
}
