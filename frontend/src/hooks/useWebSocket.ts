/**
 * WebSocket hook for real-time sampling communication.
 */

import { useCallback, useRef, useState } from 'react';
import { env } from '../config/env';

const WS_BASE = env.wsBaseUrl;

/** WebSocket connection states */
export type WebSocketState = 'disconnected' | 'connecting' | 'connected' | 'error';

/** Frame data received from WebSocket */
export interface FrameData {
  type: 'frame';
  spins: number[][];
  energy: number;
  magnetization: number;
  step: number;
  total_steps: number;
  progress: number;
  sampler?: 'mcmc' | 'diffusion';
  temperature?: number;
  t?: number;
  sigma?: number;
}

/** WebSocket message types */
export type WebSocketMessage =
  | FrameData
  | { type: 'done' }
  | { type: 'error'; message: string };

/** Parameters for sampling request */
export interface SamplingParams {
  temperature: number;
  lattice_size: number;
  sampler: 'mcmc' | 'diffusion';
  num_steps: number;
}

/** Hook configuration options */
export interface UseWebSocketOptions {
  onFrame?: (data: FrameData) => void;
  onDone?: () => void;
  onError?: (message: string) => void;
  onStateChange?: (state: WebSocketState) => void;
}

/** Return type for useWebSocket hook */
export interface UseWebSocketReturn {
  /** Current connection state */
  state: WebSocketState;
  /** Start sampling with given parameters */
  startSampling: (params: SamplingParams) => void;
  /** Stop the current sampling session */
  stopSampling: () => void;
  /** Whether sampling is currently in progress */
  isStreaming: boolean;
  /** Last error message if any */
  error: string | null;
}

/**
 * Hook for managing WebSocket connection to sampling endpoint.
 *
 * @param options - Configuration options for callbacks
 * @returns WebSocket state and control functions
 */
export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const { onFrame, onDone, onError, onStateChange } = options;

  const [state, setState] = useState<WebSocketState>('disconnected');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  const updateState = useCallback(
    (newState: WebSocketState) => {
      setState(newState);
      onStateChange?.(newState);
    },
    [onStateChange]
  );

  const startSampling = useCallback(
    (params: SamplingParams) => {
      // Clean up existing connection
      if (wsRef.current) {
        wsRef.current.close();
      }

      setError(null);
      updateState('connecting');

      const ws = new WebSocket(`${WS_BASE}/ws/sample`);
      wsRef.current = ws;

      ws.onopen = () => {
        updateState('connected');
        setIsStreaming(true);
        // Send sampling parameters
        ws.send(JSON.stringify(params));
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);

          if (data.type === 'frame') {
            onFrame?.(data);
          } else if (data.type === 'done') {
            setIsStreaming(false);
            onDone?.();
          } else if (data.type === 'error') {
            setError(data.message);
            setIsStreaming(false);
            updateState('error');
            onError?.(data.message);
          }
        } catch {
          const parseError = 'Failed to parse WebSocket message';
          setError(parseError);
          onError?.(parseError);
        }
      };

      ws.onerror = () => {
        const errMsg = 'WebSocket connection failed';
        setError(errMsg);
        setIsStreaming(false);
        updateState('error');
        onError?.(errMsg);
      };

      ws.onclose = (event) => {
        setIsStreaming(false);
        if (!event.wasClean && state !== 'error') {
          updateState('disconnected');
        }
      };
    },
    [onFrame, onDone, onError, updateState, state]
  );

  const stopSampling = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsStreaming(false);
    updateState('disconnected');
  }, [updateState]);

  return {
    state,
    startSampling,
    stopSampling,
    isStreaming,
    error,
  };
}

export default useWebSocket;
