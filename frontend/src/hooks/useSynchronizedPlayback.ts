/**
 * Hook for synchronized playback of multiple frame sequences.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

/** State for a single sequence in synchronized playback */
export interface SequenceState {
  frames: number[][][];
  currentFrame: number;
  totalFrames: number;
}

/** Configuration for synchronized playback */
export interface SynchronizedPlaybackConfig {
  /** Whether playback is synchronized across sequences */
  synchronized: boolean;
  /** Base frame delay in milliseconds */
  frameDelay: number;
  /** Playback speed multiplier */
  playbackSpeed: number;
  /** Whether to loop at end */
  loop: boolean;
}

/** Return type for the useSynchronizedPlayback hook */
export interface UseSynchronizedPlaybackReturn {
  /** Whether playback is currently active */
  isPlaying: boolean;
  /** Start playback */
  play: () => void;
  /** Pause playback */
  pause: () => void;
  /** Toggle play/pause */
  toggle: () => void;
  /** Go to specific frame (normalized 0-1 for sync mode) */
  seek: (normalizedPosition: number) => void;
  /** Step forward one frame */
  stepForward: () => void;
  /** Step backward one frame */
  stepBackward: () => void;
  /** Go to start */
  goToStart: () => void;
  /** Go to end */
  goToEnd: () => void;
  /** Current normalized position (0-1) */
  normalizedPosition: number;
  /** Update sequence frame count (call when data changes) */
  updateSequence: (id: string, totalFrames: number) => void;
  /** Get current frame for a sequence */
  getCurrentFrame: (id: string) => number;
}

const DEFAULT_CONFIG: SynchronizedPlaybackConfig = {
  synchronized: true,
  frameDelay: 50,
  playbackSpeed: 1,
  loop: false,
};

/**
 * Hook for coordinating playback across multiple frame sequences.
 * In synchronized mode, all sequences progress at the same normalized rate.
 *
 * @param config - Playback configuration
 * @returns Playback controls and state
 */
export function useSynchronizedPlayback(
  config: Partial<SynchronizedPlaybackConfig> = {}
): UseSynchronizedPlaybackReturn {
  const { synchronized, frameDelay, playbackSpeed, loop } = {
    ...DEFAULT_CONFIG,
    ...config,
  };

  const [isPlaying, setIsPlaying] = useState(false);
  const [normalizedPosition, setNormalizedPosition] = useState(0);

  // Track frame counts for each sequence
  const sequenceFrameCounts = useRef<Map<string, number>>(new Map());
  const animationRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  // Calculate actual delay based on speed
  const actualDelay = Math.round(frameDelay / playbackSpeed);

  // Get max frame count across all sequences
  const getMaxFrameCount = useCallback(() => {
    let max = 0;
    sequenceFrameCounts.current.forEach((count) => {
      max = Math.max(max, count);
    });
    return max;
  }, []);

  // Animation tick
  const tick = useCallback(() => {
    const maxFrames = getMaxFrameCount();
    if (maxFrames <= 1) {
      setIsPlaying(false);
      return;
    }

    setNormalizedPosition((prev) => {
      const step = 1 / (maxFrames - 1);
      const next = prev + step;

      if (next >= 1) {
        if (loop) {
          return 0;
        } else {
          setIsPlaying(false);
          return 1;
        }
      }

      return next;
    });
  }, [getMaxFrameCount, loop]);

  // Animation loop
  useEffect(() => {
    if (!isPlaying) {
      if (animationRef.current !== null) {
        clearTimeout(animationRef.current);
        animationRef.current = null;
      }
      return;
    }

    const scheduleNext = () => {
      animationRef.current = window.setTimeout(() => {
        tick();
        if (isPlaying) {
          scheduleNext();
        }
      }, actualDelay);
    };

    scheduleNext();

    return () => {
      if (animationRef.current !== null) {
        clearTimeout(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [isPlaying, actualDelay, tick]);

  // Playback controls
  const play = useCallback(() => {
    const maxFrames = getMaxFrameCount();
    if (maxFrames > 1) {
      // If at end, restart from beginning
      if (normalizedPosition >= 1) {
        setNormalizedPosition(0);
      }
      setIsPlaying(true);
    }
  }, [getMaxFrameCount, normalizedPosition]);

  const pause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const toggle = useCallback(() => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }, [isPlaying, play, pause]);

  const seek = useCallback((position: number) => {
    setNormalizedPosition(Math.max(0, Math.min(1, position)));
  }, []);

  const stepForward = useCallback(() => {
    const maxFrames = getMaxFrameCount();
    if (maxFrames <= 1) return;

    setNormalizedPosition((prev) => {
      const step = 1 / (maxFrames - 1);
      return Math.min(1, prev + step);
    });
  }, [getMaxFrameCount]);

  const stepBackward = useCallback(() => {
    const maxFrames = getMaxFrameCount();
    if (maxFrames <= 1) return;

    setNormalizedPosition((prev) => {
      const step = 1 / (maxFrames - 1);
      return Math.max(0, prev - step);
    });
  }, [getMaxFrameCount]);

  const goToStart = useCallback(() => {
    setNormalizedPosition(0);
    setIsPlaying(false);
  }, []);

  const goToEnd = useCallback(() => {
    setNormalizedPosition(1);
    setIsPlaying(false);
  }, []);

  const updateSequence = useCallback((id: string, totalFrames: number) => {
    sequenceFrameCounts.current.set(id, totalFrames);
  }, []);

  const getCurrentFrame = useCallback(
    (id: string) => {
      const totalFrames = sequenceFrameCounts.current.get(id) || 0;
      if (totalFrames <= 1) return 0;

      if (synchronized) {
        // In sync mode, map normalized position to frame index
        return Math.round(normalizedPosition * (totalFrames - 1));
      } else {
        // In independent mode, each sequence maintains its own position
        // For now, still use normalized position
        return Math.round(normalizedPosition * (totalFrames - 1));
      }
    },
    [synchronized, normalizedPosition]
  );

  return {
    isPlaying,
    play,
    pause,
    toggle,
    seek,
    stepForward,
    stepBackward,
    goToStart,
    goToEnd,
    normalizedPosition,
    updateSequence,
    getCurrentFrame,
  };
}

export default useSynchronizedPlayback;
