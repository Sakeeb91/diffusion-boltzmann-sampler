/**
 * Unit tests for simulationStore.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { useSimulationStore, DEFAULT_CONFIG, T_CRITICAL } from './simulationStore';

describe('simulationStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useSimulationStore.getState().reset();
  });

  describe('initial state', () => {
    it('should have default configuration values', () => {
      const state = useSimulationStore.getState();

      expect(state.temperature).toBe(DEFAULT_CONFIG.temperature);
      expect(state.latticeSize).toBe(DEFAULT_CONFIG.latticeSize);
      expect(state.samplerType).toBe(DEFAULT_CONFIG.samplerType);
      expect(state.numSteps).toBe(DEFAULT_CONFIG.numSteps);
      expect(state.playbackSpeed).toBe(DEFAULT_CONFIG.playbackSpeed);
    });

    it('should have null state values', () => {
      const state = useSimulationStore.getState();

      expect(state.spins).toBeNull();
      expect(state.energy).toBeNull();
      expect(state.magnetization).toBeNull();
    });

    it('should have empty animation state', () => {
      const state = useSimulationStore.getState();

      expect(state.animationFrames).toEqual([]);
      expect(state.currentFrame).toBe(0);
      expect(state.isPlaying).toBe(false);
    });

    it('should have correct connection state', () => {
      const state = useSimulationStore.getState();

      expect(state.isConnected).toBe(false);
      expect(state.error).toBeNull();
      expect(state.isRunning).toBe(false);
    });
  });

  describe('configuration setters', () => {
    it('should set temperature', () => {
      useSimulationStore.getState().setTemperature(3.5);
      expect(useSimulationStore.getState().temperature).toBe(3.5);
    });

    it('should set lattice size', () => {
      useSimulationStore.getState().setLatticeSize(64);
      expect(useSimulationStore.getState().latticeSize).toBe(64);
    });

    it('should set sampler type', () => {
      useSimulationStore.getState().setSamplerType('diffusion');
      expect(useSimulationStore.getState().samplerType).toBe('diffusion');
    });

    it('should set number of steps', () => {
      useSimulationStore.getState().setNumSteps(200);
      expect(useSimulationStore.getState().numSteps).toBe(200);
    });

    it('should set playback speed', () => {
      useSimulationStore.getState().setPlaybackSpeed(2.0);
      expect(useSimulationStore.getState().playbackSpeed).toBe(2.0);
    });
  });

  describe('state setters', () => {
    it('should set spins', () => {
      const spins = [
        [1, -1],
        [-1, 1],
      ];
      useSimulationStore.getState().setSpins(spins);
      expect(useSimulationStore.getState().spins).toEqual(spins);
    });

    it('should set energy', () => {
      useSimulationStore.getState().setEnergy(-128);
      expect(useSimulationStore.getState().energy).toBe(-128);
    });

    it('should set magnetization', () => {
      useSimulationStore.getState().setMagnetization(0.75);
      expect(useSimulationStore.getState().magnetization).toBe(0.75);
    });

    it('should set isRunning', () => {
      useSimulationStore.getState().setIsRunning(true);
      expect(useSimulationStore.getState().isRunning).toBe(true);
    });

    it('should set isConnected', () => {
      useSimulationStore.getState().setIsConnected(true);
      expect(useSimulationStore.getState().isConnected).toBe(true);
    });

    it('should set error', () => {
      useSimulationStore.getState().setError('Connection failed');
      expect(useSimulationStore.getState().error).toBe('Connection failed');
    });

    it('should clear error', () => {
      useSimulationStore.getState().setError('Error');
      useSimulationStore.getState().setError(null);
      expect(useSimulationStore.getState().error).toBeNull();
    });
  });

  describe('animation actions', () => {
    const frame1 = [
      [1, 1],
      [1, 1],
    ];
    const frame2 = [
      [1, -1],
      [-1, 1],
    ];

    it('should add animation frame and update spins', () => {
      useSimulationStore.getState().addAnimationFrame(frame1);

      const state = useSimulationStore.getState();
      expect(state.animationFrames).toHaveLength(1);
      expect(state.animationFrames[0]).toEqual(frame1);
      expect(state.spins).toEqual(frame1);
    });

    it('should accumulate animation frames', () => {
      useSimulationStore.getState().addAnimationFrame(frame1);
      useSimulationStore.getState().addAnimationFrame(frame2);

      const state = useSimulationStore.getState();
      expect(state.animationFrames).toHaveLength(2);
      expect(state.animationFrames[1]).toEqual(frame2);
      expect(state.spins).toEqual(frame2);
    });

    it('should clear animation frames', () => {
      useSimulationStore.getState().addAnimationFrame(frame1);
      useSimulationStore.getState().addAnimationFrame(frame2);
      useSimulationStore.getState().setCurrentFrame(1);
      useSimulationStore.getState().setIsPlaying(true);

      useSimulationStore.getState().clearAnimationFrames();

      const state = useSimulationStore.getState();
      expect(state.animationFrames).toEqual([]);
      expect(state.currentFrame).toBe(0);
      expect(state.isPlaying).toBe(false);
    });

    it('should set current frame and update spins', () => {
      useSimulationStore.getState().addAnimationFrame(frame1);
      useSimulationStore.getState().addAnimationFrame(frame2);

      useSimulationStore.getState().setCurrentFrame(0);
      expect(useSimulationStore.getState().spins).toEqual(frame1);

      useSimulationStore.getState().setCurrentFrame(1);
      expect(useSimulationStore.getState().spins).toEqual(frame2);
    });

    it('should preserve spins when frame index is out of bounds', () => {
      const initialSpins = [
        [-1, -1],
        [-1, -1],
      ];
      useSimulationStore.getState().setSpins(initialSpins);

      useSimulationStore.getState().setCurrentFrame(5);
      expect(useSimulationStore.getState().spins).toEqual(initialSpins);
    });

    it('should set isPlaying', () => {
      useSimulationStore.getState().setIsPlaying(true);
      expect(useSimulationStore.getState().isPlaying).toBe(true);

      useSimulationStore.getState().setIsPlaying(false);
      expect(useSimulationStore.getState().isPlaying).toBe(false);
    });
  });

  describe('reset actions', () => {
    it('should reset entire store to initial state', () => {
      // Modify multiple state values
      useSimulationStore.getState().setTemperature(5.0);
      useSimulationStore.getState().setLatticeSize(64);
      useSimulationStore.getState().setSamplerType('diffusion');
      useSimulationStore.getState().setSpins([[1, -1]]);
      useSimulationStore.getState().setEnergy(-50);
      useSimulationStore.getState().setIsConnected(true);
      useSimulationStore.getState().setError('Test error');

      useSimulationStore.getState().reset();

      const state = useSimulationStore.getState();
      expect(state.temperature).toBe(DEFAULT_CONFIG.temperature);
      expect(state.latticeSize).toBe(DEFAULT_CONFIG.latticeSize);
      expect(state.samplerType).toBe(DEFAULT_CONFIG.samplerType);
      expect(state.spins).toBeNull();
      expect(state.energy).toBeNull();
      expect(state.isConnected).toBe(false);
      expect(state.error).toBeNull();
    });

    it('should reset only config to defaults', () => {
      // Modify config and state
      useSimulationStore.getState().setTemperature(5.0);
      useSimulationStore.getState().setLatticeSize(64);
      useSimulationStore.getState().setSpins([[1, -1]]);
      useSimulationStore.getState().setEnergy(-50);

      useSimulationStore.getState().resetConfig();

      const state = useSimulationStore.getState();
      // Config should be reset
      expect(state.temperature).toBe(DEFAULT_CONFIG.temperature);
      expect(state.latticeSize).toBe(DEFAULT_CONFIG.latticeSize);
      // State should be preserved
      expect(state.spins).toEqual([[1, -1]]);
      expect(state.energy).toBe(-50);
    });

    it('should clear simulation state but preserve config', () => {
      // Modify config and state
      useSimulationStore.getState().setTemperature(5.0);
      useSimulationStore.getState().setLatticeSize(64);
      useSimulationStore.getState().setSpins([[1, -1]]);
      useSimulationStore.getState().setEnergy(-50);
      useSimulationStore.getState().addAnimationFrame([[1, 1]]);
      useSimulationStore.getState().setIsRunning(true);
      useSimulationStore.getState().setError('Test error');

      useSimulationStore.getState().clearState();

      const state = useSimulationStore.getState();
      // Config should be preserved
      expect(state.temperature).toBe(5.0);
      expect(state.latticeSize).toBe(64);
      // State should be cleared
      expect(state.spins).toBeNull();
      expect(state.energy).toBeNull();
      expect(state.magnetization).toBeNull();
      expect(state.animationFrames).toEqual([]);
      expect(state.currentFrame).toBe(0);
      expect(state.isPlaying).toBe(false);
      expect(state.isRunning).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('constants', () => {
    it('should export critical temperature constant', () => {
      expect(T_CRITICAL).toBeCloseTo(2.269, 3);
    });

    it('should have critical temperature in default config', () => {
      expect(DEFAULT_CONFIG.temperature).toBeCloseTo(T_CRITICAL, 2);
    });
  });
});
