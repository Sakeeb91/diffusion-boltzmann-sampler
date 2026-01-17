/**
 * Unit tests for useWebSocket hook.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useWebSocket } from './useWebSocket';
import type { SamplingParams } from './useWebSocket';

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event: { wasClean: boolean }) => void) | null = null;
  readyState = 0;
  sentMessages: string[] = [];

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close() {
    this.readyState = 3;
    this.onclose?.({ wasClean: true });
  }

  // Helper to simulate connection open
  simulateOpen() {
    this.readyState = 1;
    this.onopen?.();
  }

  // Helper to simulate message
  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }

  // Helper to simulate error
  simulateError() {
    this.onerror?.();
  }

  // Helper to simulate close
  simulateClose(wasClean = true) {
    this.readyState = 3;
    this.onclose?.({ wasClean });
  }

  static clear() {
    MockWebSocket.instances = [];
  }

  static get lastInstance(): MockWebSocket | undefined {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1];
  }
}

// Replace global WebSocket
const originalWebSocket = global.WebSocket;
beforeEach(() => {
  MockWebSocket.clear();
  (global as unknown as { WebSocket: typeof MockWebSocket }).WebSocket = MockWebSocket;
});

afterEach(() => {
  global.WebSocket = originalWebSocket;
});

const defaultParams: SamplingParams = {
  temperature: 2.27,
  lattice_size: 32,
  sampler: 'mcmc',
  num_steps: 100,
};

describe('useWebSocket', () => {
  describe('initial state', () => {
    it('should have correct initial state', () => {
      const { result } = renderHook(() => useWebSocket());

      expect(result.current.state).toBe('disconnected');
      expect(result.current.isStreaming).toBe(false);
      expect(result.current.error).toBeNull();
      expect(result.current.reconnectAttempt).toBe(0);
    });
  });

  describe('startSampling', () => {
    it('should create WebSocket connection', () => {
      const { result } = renderHook(() => useWebSocket());

      act(() => {
        result.current.startSampling(defaultParams);
      });

      expect(MockWebSocket.instances).toHaveLength(1);
      expect(MockWebSocket.lastInstance?.url).toContain('/ws/sample');
    });

    it('should set state to connecting initially', () => {
      const { result } = renderHook(() => useWebSocket());

      act(() => {
        result.current.startSampling(defaultParams);
      });

      expect(result.current.state).toBe('connecting');
    });

    it('should set state to connected when WebSocket opens', () => {
      const { result } = renderHook(() => useWebSocket());

      act(() => {
        result.current.startSampling(defaultParams);
      });

      act(() => {
        MockWebSocket.lastInstance?.simulateOpen();
      });

      expect(result.current.state).toBe('connected');
      expect(result.current.isStreaming).toBe(true);
    });

    it('should send params after connection opens', () => {
      const { result } = renderHook(() => useWebSocket());

      act(() => {
        result.current.startSampling(defaultParams);
      });

      act(() => {
        MockWebSocket.lastInstance?.simulateOpen();
      });

      expect(MockWebSocket.lastInstance?.sentMessages).toHaveLength(1);
      expect(JSON.parse(MockWebSocket.lastInstance!.sentMessages[0])).toEqual(
        defaultParams
      );
    });
  });

  describe('frame handling', () => {
    it('should call onFrame callback for frame messages', () => {
      const onFrame = vi.fn();
      const { result } = renderHook(() => useWebSocket({ onFrame }));

      act(() => {
        result.current.startSampling(defaultParams);
        MockWebSocket.lastInstance?.simulateOpen();
      });

      const frameData = {
        type: 'frame',
        spins: [[1, -1], [-1, 1]],
        energy: -0.5,
        magnetization: 0.1,
        step: 5,
        total_steps: 100,
        progress: 0.05,
        sampler: 'mcmc',
      };

      act(() => {
        MockWebSocket.lastInstance?.simulateMessage(frameData);
      });

      expect(onFrame).toHaveBeenCalledWith(frameData);
    });

    it('should call onDone callback for done message', () => {
      const onDone = vi.fn();
      const { result } = renderHook(() => useWebSocket({ onDone }));

      act(() => {
        result.current.startSampling(defaultParams);
        MockWebSocket.lastInstance?.simulateOpen();
      });

      act(() => {
        MockWebSocket.lastInstance?.simulateMessage({ type: 'done' });
      });

      expect(onDone).toHaveBeenCalled();
      expect(result.current.isStreaming).toBe(false);
    });

    it('should handle error messages', () => {
      const onError = vi.fn();
      const { result } = renderHook(() => useWebSocket({ onError }));

      act(() => {
        result.current.startSampling(defaultParams);
        MockWebSocket.lastInstance?.simulateOpen();
      });

      act(() => {
        MockWebSocket.lastInstance?.simulateMessage({
          type: 'error',
          message: 'Test error',
        });
      });

      expect(onError).toHaveBeenCalledWith('Test error');
      expect(result.current.error).toBe('Test error');
      expect(result.current.state).toBe('error');
    });
  });

  describe('connection errors', () => {
    it('should set error state on WebSocket error', () => {
      const onError = vi.fn();
      const { result } = renderHook(() => useWebSocket({ onError }));

      act(() => {
        result.current.startSampling(defaultParams);
      });

      act(() => {
        MockWebSocket.lastInstance?.simulateError();
      });

      expect(result.current.state).toBe('error');
      expect(result.current.error).toBe('WebSocket connection failed');
      expect(onError).toHaveBeenCalledWith('WebSocket connection failed');
    });
  });

  describe('stopSampling', () => {
    it('should close WebSocket connection', () => {
      const { result } = renderHook(() => useWebSocket());

      act(() => {
        result.current.startSampling(defaultParams);
        MockWebSocket.lastInstance?.simulateOpen();
      });

      const ws = MockWebSocket.lastInstance;

      act(() => {
        result.current.stopSampling();
      });

      expect(ws?.readyState).toBe(3);
      expect(result.current.state).toBe('disconnected');
      expect(result.current.isStreaming).toBe(false);
    });
  });

  describe('state change callback', () => {
    it('should call onStateChange when state changes', () => {
      const onStateChange = vi.fn();
      const { result } = renderHook(() => useWebSocket({ onStateChange }));

      act(() => {
        result.current.startSampling(defaultParams);
      });

      expect(onStateChange).toHaveBeenCalledWith('connecting');

      act(() => {
        MockWebSocket.lastInstance?.simulateOpen();
      });

      expect(onStateChange).toHaveBeenCalledWith('connected');
    });
  });

  describe('cleanup', () => {
    it('should close WebSocket on unmount', () => {
      const { result, unmount } = renderHook(() => useWebSocket());

      act(() => {
        result.current.startSampling(defaultParams);
        MockWebSocket.lastInstance?.simulateOpen();
      });

      const ws = MockWebSocket.lastInstance;
      expect(ws?.readyState).toBe(1);

      unmount();

      expect(ws?.readyState).toBe(3);
    });
  });

  describe('reconnection', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should not reconnect when autoReconnect is false', () => {
      const { result } = renderHook(() =>
        useWebSocket({ autoReconnect: false })
      );

      act(() => {
        result.current.startSampling(defaultParams);
        MockWebSocket.lastInstance?.simulateOpen();
      });

      const initialCount = MockWebSocket.instances.length;

      act(() => {
        MockWebSocket.lastInstance?.simulateClose(false);
      });

      // Advance time
      act(() => {
        vi.advanceTimersByTime(5000);
      });

      expect(MockWebSocket.instances.length).toBe(initialCount);
    });

    it('should attempt reconnection when autoReconnect is true', async () => {
      const { result } = renderHook(() =>
        useWebSocket({
          autoReconnect: true,
          maxReconnectAttempts: 3,
          reconnectDelay: 1000,
        })
      );

      act(() => {
        result.current.startSampling(defaultParams);
        MockWebSocket.lastInstance?.simulateOpen();
      });

      const initialCount = MockWebSocket.instances.length;

      act(() => {
        MockWebSocket.lastInstance?.simulateClose(false);
      });

      expect(result.current.reconnectAttempt).toBe(1);

      // Advance time to trigger reconnect
      await act(async () => {
        vi.advanceTimersByTime(1000);
      });

      expect(MockWebSocket.instances.length).toBe(initialCount + 1);
    });

    it('should stop reconnecting after max attempts', async () => {
      const { result } = renderHook(() =>
        useWebSocket({
          autoReconnect: true,
          maxReconnectAttempts: 2,
          reconnectDelay: 100,
        })
      );

      act(() => {
        result.current.startSampling(defaultParams);
        MockWebSocket.lastInstance?.simulateOpen();
      });

      // First disconnect
      act(() => {
        MockWebSocket.lastInstance?.simulateClose(false);
      });

      // First reconnect attempt
      await act(async () => {
        vi.advanceTimersByTime(100);
      });

      // Second disconnect
      act(() => {
        MockWebSocket.lastInstance?.simulateClose(false);
      });

      // Second reconnect attempt
      await act(async () => {
        vi.advanceTimersByTime(200);
      });

      // Third disconnect - should not attempt again
      act(() => {
        MockWebSocket.lastInstance?.simulateClose(false);
      });

      const countAfterMaxAttempts = MockWebSocket.instances.length;

      // Advance time more
      await act(async () => {
        vi.advanceTimersByTime(10000);
      });

      // No more connections should be attempted
      expect(MockWebSocket.instances.length).toBe(countAfterMaxAttempts);
    });

    it('should reset reconnect attempts on successful connection', async () => {
      const { result } = renderHook(() =>
        useWebSocket({
          autoReconnect: true,
          maxReconnectAttempts: 3,
          reconnectDelay: 100,
        })
      );

      act(() => {
        result.current.startSampling(defaultParams);
        MockWebSocket.lastInstance?.simulateOpen();
      });

      // Disconnect
      act(() => {
        MockWebSocket.lastInstance?.simulateClose(false);
      });

      expect(result.current.reconnectAttempt).toBe(1);

      // Reconnect
      await act(async () => {
        vi.advanceTimersByTime(100);
      });

      // Connection succeeds
      act(() => {
        MockWebSocket.lastInstance?.simulateOpen();
      });

      expect(result.current.reconnectAttempt).toBe(0);
    });
  });
});
