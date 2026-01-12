import { useState, useEffect, useCallback, useRef } from 'react';
import { checkHealth } from '../services/api';

interface HealthStatus {
  /** Whether the backend is connected */
  isConnected: boolean;
  /** Whether a health check is in progress */
  isChecking: boolean;
  /** Last error message if connection failed */
  error: string | null;
  /** API version from health response */
  version: string | null;
  /** Available features from health response */
  features: Record<string, boolean> | null;
  /** Timestamp of last successful check */
  lastChecked: Date | null;
}

interface UseHealthCheckOptions {
  /** Interval in ms between checks (default: 10000) */
  interval?: number;
  /** Whether to start checking immediately (default: true) */
  immediate?: boolean;
  /** Callback when connection status changes */
  onStatusChange?: (isConnected: boolean) => void;
}

/**
 * Custom hook for monitoring backend health.
 *
 * Periodically checks the backend health endpoint and tracks connection status.
 */
export function useHealthCheck(options: UseHealthCheckOptions = {}): HealthStatus & {
  /** Manually trigger a health check */
  checkNow: () => Promise<void>;
} {
  const {
    interval = 10000,
    immediate = true,
    onStatusChange,
  } = options;

  const [status, setStatus] = useState<HealthStatus>({
    isConnected: false,
    isChecking: false,
    error: null,
    version: null,
    features: null,
    lastChecked: null,
  });

  const previousConnected = useRef<boolean | null>(null);

  const performCheck = useCallback(async () => {
    setStatus((prev) => ({ ...prev, isChecking: true }));

    try {
      const response = await checkHealth();
      const isConnected = response.status === 'healthy';

      setStatus({
        isConnected,
        isChecking: false,
        error: null,
        version: response.version || null,
        features: response.features || null,
        lastChecked: new Date(),
      });

      // Notify on status change
      if (previousConnected.current !== isConnected) {
        previousConnected.current = isConnected;
        onStatusChange?.(isConnected);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Connection failed';

      setStatus((prev) => ({
        ...prev,
        isConnected: false,
        isChecking: false,
        error: errorMessage,
        lastChecked: new Date(),
      }));

      // Notify on status change
      if (previousConnected.current !== false) {
        previousConnected.current = false;
        onStatusChange?.(false);
      }
    }
  }, [onStatusChange]);

  // Initial check
  useEffect(() => {
    if (immediate) {
      performCheck();
    }
  }, [immediate, performCheck]);

  // Periodic checks
  useEffect(() => {
    const intervalId = setInterval(performCheck, interval);
    return () => clearInterval(intervalId);
  }, [interval, performCheck]);

  return {
    ...status,
    checkNow: performCheck,
  };
}

export default useHealthCheck;
