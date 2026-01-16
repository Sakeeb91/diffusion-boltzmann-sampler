/**
 * Component tests for DiffusionAnimation.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DiffusionAnimation } from './DiffusionAnimation';
import { useSimulationStore } from '../store/simulationStore';

// Reset store before each test
beforeEach(() => {
  useSimulationStore.getState().reset();
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('DiffusionAnimation', () => {
  const mockFrames = [
    [[1, 1], [1, 1]],
    [[1, -1], [1, 1]],
    [[-1, -1], [1, 1]],
  ];

  describe('empty state', () => {
    it('should show empty state when no frames', () => {
      render(<DiffusionAnimation />);
      expect(screen.getByText('Animation')).toBeInTheDocument();
      expect(screen.getByText('Generate a sample to see the animation')).toBeInTheDocument();
    });

    it('should not show controls when no frames', () => {
      render(<DiffusionAnimation />);
      expect(screen.queryByRole('slider')).not.toBeInTheDocument();
    });
  });

  describe('with frames', () => {
    beforeEach(() => {
      mockFrames.forEach((frame) => {
        useSimulationStore.getState().addAnimationFrame(frame);
      });
    });

    it('should show frame counter', () => {
      render(<DiffusionAnimation />);
      expect(screen.getByText('Frame 3 / 3')).toBeInTheDocument();
    });

    it('should show scrubber slider', () => {
      render(<DiffusionAnimation />);
      const slider = screen.getByRole('slider');
      expect(slider).toBeInTheDocument();
      expect(slider).toHaveAttribute('max', '2');
    });

    it('should show playback controls', () => {
      render(<DiffusionAnimation />);
      expect(screen.getByTitle('Play')).toBeInTheDocument();
      expect(screen.getByTitle('Go to start')).toBeInTheDocument();
      expect(screen.getByTitle('Go to end')).toBeInTheDocument();
    });

    it('should show speed controls', () => {
      render(<DiffusionAnimation />);
      expect(screen.getByText('Speed:')).toBeInTheDocument();
      expect(screen.getByText('0.5x')).toBeInTheDocument();
      expect(screen.getByText('1x')).toBeInTheDocument();
      expect(screen.getByText('2x')).toBeInTheDocument();
      expect(screen.getByText('4x')).toBeInTheDocument();
    });
  });

  describe('scrubber interaction', () => {
    beforeEach(() => {
      mockFrames.forEach((frame) => {
        useSimulationStore.getState().addAnimationFrame(frame);
      });
      useSimulationStore.getState().setCurrentFrame(0);
    });

    it('should update current frame when scrubbing', () => {
      const onFrameChange = vi.fn();
      render(<DiffusionAnimation onFrameChange={onFrameChange} />);

      const slider = screen.getByRole('slider');
      fireEvent.change(slider, { target: { value: '1' } });

      expect(useSimulationStore.getState().currentFrame).toBe(1);
      expect(onFrameChange).toHaveBeenCalledWith(1);
    });
  });

  describe('navigation buttons', () => {
    beforeEach(() => {
      mockFrames.forEach((frame) => {
        useSimulationStore.getState().addAnimationFrame(frame);
      });
      useSimulationStore.getState().setCurrentFrame(1);
    });

    it('should go to start when clicking go to start button', () => {
      const onFrameChange = vi.fn();
      render(<DiffusionAnimation onFrameChange={onFrameChange} />);

      fireEvent.click(screen.getByTitle('Go to start'));

      expect(useSimulationStore.getState().currentFrame).toBe(0);
      expect(onFrameChange).toHaveBeenCalledWith(0);
    });

    it('should go to end when clicking go to end button', () => {
      const onFrameChange = vi.fn();
      render(<DiffusionAnimation onFrameChange={onFrameChange} />);

      fireEvent.click(screen.getByTitle('Go to end'));

      expect(useSimulationStore.getState().currentFrame).toBe(2);
      expect(onFrameChange).toHaveBeenCalledWith(2);
    });

    it('should go to previous frame when clicking previous button', () => {
      const onFrameChange = vi.fn();
      render(<DiffusionAnimation onFrameChange={onFrameChange} />);

      fireEvent.click(screen.getByTitle('Previous frame'));

      expect(useSimulationStore.getState().currentFrame).toBe(0);
      expect(onFrameChange).toHaveBeenCalledWith(0);
    });

    it('should go to next frame when clicking next button', () => {
      const onFrameChange = vi.fn();
      render(<DiffusionAnimation onFrameChange={onFrameChange} />);

      fireEvent.click(screen.getByTitle('Next frame'));

      expect(useSimulationStore.getState().currentFrame).toBe(2);
      expect(onFrameChange).toHaveBeenCalledWith(2);
    });

    it('should not go below frame 0', () => {
      useSimulationStore.getState().setCurrentFrame(0);
      const onFrameChange = vi.fn();
      render(<DiffusionAnimation onFrameChange={onFrameChange} />);

      fireEvent.click(screen.getByTitle('Previous frame'));

      expect(useSimulationStore.getState().currentFrame).toBe(0);
      expect(onFrameChange).not.toHaveBeenCalled();
    });

    it('should not go above last frame', () => {
      useSimulationStore.getState().setCurrentFrame(2);
      const onFrameChange = vi.fn();
      render(<DiffusionAnimation onFrameChange={onFrameChange} />);

      fireEvent.click(screen.getByTitle('Next frame'));

      expect(useSimulationStore.getState().currentFrame).toBe(2);
      expect(onFrameChange).not.toHaveBeenCalled();
    });
  });

  describe('play/pause', () => {
    beforeEach(() => {
      mockFrames.forEach((frame) => {
        useSimulationStore.getState().addAnimationFrame(frame);
      });
      useSimulationStore.getState().setCurrentFrame(0);
    });

    it('should start playing when play button is clicked', () => {
      render(<DiffusionAnimation />);

      fireEvent.click(screen.getByTitle('Play'));

      expect(useSimulationStore.getState().isPlaying).toBe(true);
    });

    it('should show pause button when playing', () => {
      useSimulationStore.getState().setIsPlaying(true);
      render(<DiffusionAnimation />);

      expect(screen.getByTitle('Pause')).toBeInTheDocument();
    });

    it('should stop playing when pause button is clicked', () => {
      useSimulationStore.getState().setIsPlaying(true);
      render(<DiffusionAnimation />);

      fireEvent.click(screen.getByTitle('Pause'));

      expect(useSimulationStore.getState().isPlaying).toBe(false);
    });
  });

  describe('playback speed', () => {
    beforeEach(() => {
      mockFrames.forEach((frame) => {
        useSimulationStore.getState().addAnimationFrame(frame);
      });
    });

    it('should default to 1x speed', () => {
      render(<DiffusionAnimation />);
      const speedButton = screen.getByText('1x');
      expect(speedButton.className).toContain('bg-blue-600');
    });

    it('should change speed when speed button is clicked', () => {
      render(<DiffusionAnimation />);

      fireEvent.click(screen.getByText('2x'));

      expect(useSimulationStore.getState().playbackSpeed).toBe(2);
    });

    it('should highlight selected speed', () => {
      useSimulationStore.getState().setPlaybackSpeed(4);
      render(<DiffusionAnimation />);

      const speedButton = screen.getByText('4x');
      expect(speedButton.className).toContain('bg-blue-600');
    });
  });

  describe('progress bar', () => {
    beforeEach(() => {
      mockFrames.forEach((frame) => {
        useSimulationStore.getState().addAnimationFrame(frame);
      });
    });

    it('should show progress at 100% when on last frame', () => {
      useSimulationStore.getState().setCurrentFrame(2);
      const { container } = render(<DiffusionAnimation />);

      const progressBar = container.querySelector('.bg-gradient-to-r');
      expect(progressBar).toHaveStyle({ width: '100%' });
    });

    it('should show progress at ~33% when on first frame', () => {
      useSimulationStore.getState().setCurrentFrame(0);
      const { container } = render(<DiffusionAnimation />);

      const progressBar = container.querySelector('.bg-gradient-to-r');
      // Frame 1 of 3 = 33.33%
      expect(progressBar).toBeInTheDocument();
    });
  });

  describe('animation loop', () => {
    beforeEach(() => {
      mockFrames.forEach((frame) => {
        useSimulationStore.getState().addAnimationFrame(frame);
      });
      useSimulationStore.getState().setCurrentFrame(0);
    });

    it('should advance frames when playing', async () => {
      const onFrameChange = vi.fn();
      useSimulationStore.getState().setIsPlaying(true);

      render(<DiffusionAnimation onFrameChange={onFrameChange} />);

      // Fast-forward timers
      await vi.advanceTimersByTimeAsync(500);

      // Should have advanced at least one frame
      expect(useSimulationStore.getState().currentFrame).toBeGreaterThan(0);
    });

    it('should stop at last frame', async () => {
      useSimulationStore.getState().setCurrentFrame(2);
      useSimulationStore.getState().setIsPlaying(true);

      render(<DiffusionAnimation />);

      // Fast-forward timers
      await vi.advanceTimersByTimeAsync(1000);

      // Should stop playing at end
      expect(useSimulationStore.getState().isPlaying).toBe(false);
    });
  });
});
