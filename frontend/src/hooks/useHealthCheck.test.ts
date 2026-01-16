/**
 * Unit tests for useHealthCheck hook.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useHealthCheck } from './useHealthCheck';
import * as api from '../services/api';

// Mock the API module
vi.mock('../services/api', () => ({
  checkHealth: vi.fn(),
}));

const mockCheckHealth = vi.mocked(api.checkHealth);

// Helper to flush promises
const flushPromises = () => new Promise(resolve => setTimeout(resolve, 0));

describe('useHealthCheck', () => {
  beforeEach(() => {
    mockCheckHealth.mockClear();
  });

  describe('initial state', () => {
    it('should have correct initial state', () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });
      const { result } = renderHook(() => useHealthCheck({ immediate: false }));

      expect(result.current.isConnected).toBe(false);
      expect(result.current.isChecking).toBe(false);
      expect(result.current.error).toBeNull();
      expect(result.current.version).toBeNull();
      expect(result.current.features).toBeNull();
      expect(result.current.lastChecked).toBeNull();
    });
  });

  describe('immediate check', () => {
    it('should check immediately by default', async () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });
      renderHook(() => useHealthCheck());

      // Wait for the effect to run and the promise to resolve
      await act(async () => {
        await flushPromises();
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(1);
    });

    it('should not check immediately when immediate is false', () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });
      renderHook(() => useHealthCheck({ immediate: false }));

      expect(mockCheckHealth).not.toHaveBeenCalled();
    });
  });

  describe('successful health check', () => {
    it('should set isConnected to true on healthy response', async () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy', version: '1.0.0' });
      const { result } = renderHook(() => useHealthCheck());

      await act(async () => {
        await flushPromises();
      });

      expect(result.current.isConnected).toBe(true);
    });

    it('should store version from response', async () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy', version: '2.0.0' });
      const { result } = renderHook(() => useHealthCheck());

      await act(async () => {
        await flushPromises();
      });

      expect(result.current.version).toBe('2.0.0');
    });

    it('should store features from response', async () => {
      const features = { diffusion: true, mcmc: true };
      mockCheckHealth.mockResolvedValue({ status: 'healthy', features });
      const { result } = renderHook(() => useHealthCheck());

      await act(async () => {
        await flushPromises();
      });

      expect(result.current.features).toEqual(features);
    });

    it('should update lastChecked timestamp', async () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });
      const { result } = renderHook(() => useHealthCheck());

      await act(async () => {
        await flushPromises();
      });

      expect(result.current.lastChecked).toBeInstanceOf(Date);
    });

    it('should clear error on success', async () => {
      mockCheckHealth
        .mockRejectedValueOnce(new Error('Connection failed'))
        .mockResolvedValueOnce({ status: 'healthy' });

      const { result } = renderHook(() => useHealthCheck());

      // First call fails
      await act(async () => {
        await flushPromises();
      });

      expect(result.current.error).not.toBeNull();

      // Trigger next check
      await act(async () => {
        await result.current.checkNow();
      });

      expect(result.current.error).toBeNull();
    });
  });

  describe('failed health check', () => {
    it('should set isConnected to false on error', async () => {
      mockCheckHealth.mockRejectedValue(new Error('Network error'));
      const { result } = renderHook(() => useHealthCheck());

      await act(async () => {
        await flushPromises();
      });

      expect(result.current.isConnected).toBe(false);
    });

    it('should store error message', async () => {
      mockCheckHealth.mockRejectedValue(new Error('Connection refused'));
      const { result } = renderHook(() => useHealthCheck());

      await act(async () => {
        await flushPromises();
      });

      expect(result.current.error).toBe('Connection refused');
    });

    it('should use default message for non-Error exceptions', async () => {
      mockCheckHealth.mockRejectedValue('Unknown error');
      const { result } = renderHook(() => useHealthCheck());

      await act(async () => {
        await flushPromises();
      });

      expect(result.current.error).toBe('Connection failed');
    });

    it('should still update lastChecked on error', async () => {
      mockCheckHealth.mockRejectedValue(new Error('Error'));
      const { result } = renderHook(() => useHealthCheck());

      await act(async () => {
        await flushPromises();
      });

      expect(result.current.lastChecked).toBeInstanceOf(Date);
    });
  });

  describe('isChecking state', () => {
    it('should set isChecking to true during check', async () => {
      let resolveCheck: (value: { status: string }) => void;
      mockCheckHealth.mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveCheck = resolve;
          })
      );

      const { result } = renderHook(() => useHealthCheck());

      // Should be checking
      expect(result.current.isChecking).toBe(true);

      // Resolve the check
      await act(async () => {
        resolveCheck!({ status: 'healthy' });
        await flushPromises();
      });

      expect(result.current.isChecking).toBe(false);
    });
  });

  describe('periodic checks', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should check periodically at specified interval', async () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });
      renderHook(() => useHealthCheck({ interval: 5000 }));

      // Initial check (called immediately)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(1);

      // Advance timer by interval
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000);
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(2);

      // Advance again
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000);
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(3);
    });

    it('should use default interval of 10000ms', async () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });
      renderHook(() => useHealthCheck());

      // Initial check
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(1);

      // Advance by less than default interval
      await act(async () => {
        await vi.advanceTimersByTimeAsync(9000);
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(1);

      // Advance to complete interval
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(2);
    });
  });

  describe('onStatusChange callback', () => {
    it('should call onStatusChange when connection established', async () => {
      const onStatusChange = vi.fn();
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });

      renderHook(() => useHealthCheck({ onStatusChange }));

      await act(async () => {
        await flushPromises();
      });

      expect(onStatusChange).toHaveBeenCalledWith(true);
    });

    it('should call onStatusChange when connection lost', async () => {
      const onStatusChange = vi.fn();
      mockCheckHealth.mockRejectedValue(new Error('Connection lost'));

      renderHook(() => useHealthCheck({ onStatusChange }));

      await act(async () => {
        await flushPromises();
      });

      expect(onStatusChange).toHaveBeenCalledWith(false);
    });

    it('should only call onStatusChange when status changes', async () => {
      vi.useFakeTimers();
      const onStatusChange = vi.fn();
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });

      renderHook(() => useHealthCheck({ onStatusChange, interval: 1000 }));

      // Initial check
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(onStatusChange).toHaveBeenCalledTimes(1);

      // Trigger another check with same status
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(2);

      // Should not call again since status didn't change
      expect(onStatusChange).toHaveBeenCalledTimes(1);

      vi.useRealTimers();
    });
  });

  describe('checkNow function', () => {
    it('should provide checkNow function', () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });
      const { result } = renderHook(() => useHealthCheck({ immediate: false }));

      expect(typeof result.current.checkNow).toBe('function');
    });

    it('should trigger immediate check when called', async () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });
      const { result } = renderHook(() => useHealthCheck({ immediate: false }));

      expect(mockCheckHealth).not.toHaveBeenCalled();

      await act(async () => {
        await result.current.checkNow();
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(1);
    });
  });

  describe('cleanup', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should clear interval on unmount', async () => {
      mockCheckHealth.mockResolvedValue({ status: 'healthy' });
      const { unmount } = renderHook(() => useHealthCheck({ interval: 5000 }));

      // Initial check
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(1);

      unmount();

      // Advance timer - should not trigger more checks
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10000);
      });

      expect(mockCheckHealth).toHaveBeenCalledTimes(1);
    });
  });
});
