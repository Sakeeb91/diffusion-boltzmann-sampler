import React, { useState, useCallback } from 'react';
import { FrameMetadata } from '../store/simulationStore';

/** Comparison view configuration */
export interface ComparisonConfig {
  /** Temperature for MCMC sampling */
  mcmcTemperature: number;
  /** Lattice size for both samplers */
  latticeSize: number;
  /** Number of steps for MCMC */
  mcmcSteps: number;
  /** Number of steps for diffusion */
  diffusionSteps: number;
}

/** State for a single sampler in comparison */
export interface SamplerState {
  frames: number[][][];
  metadata: FrameMetadata[];
  currentFrame: number;
  isLoading: boolean;
  error: string | null;
}

interface MCMCvsDiffusionComparisonProps {
  className?: string;
}

/** Default configuration values */
const DEFAULT_CONFIG: ComparisonConfig = {
  mcmcTemperature: 2.27,
  latticeSize: 32,
  mcmcSteps: 100,
  diffusionSteps: 100,
};

/**
 * Component for side-by-side comparison of MCMC and diffusion sampling.
 * Allows users to run both samplers with the same configuration and
 * compare their outputs visually.
 */
export const MCMCvsDiffusionComparison: React.FC<
  MCMCvsDiffusionComparisonProps
> = ({ className = '' }) => {
  // Configuration state
  const [config, setConfig] = useState<ComparisonConfig>(DEFAULT_CONFIG);

  // Sampler states
  const [mcmcState, setMcmcState] = useState<SamplerState>({
    frames: [],
    metadata: [],
    currentFrame: 0,
    isLoading: false,
    error: null,
  });

  const [diffusionState, setDiffusionState] = useState<SamplerState>({
    frames: [],
    metadata: [],
    currentFrame: 0,
    isLoading: false,
    error: null,
  });

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSynchronized, setIsSynchronized] = useState(true);

  // Configuration handlers
  const updateConfig = useCallback(
    (updates: Partial<ComparisonConfig>) => {
      setConfig((prev) => ({ ...prev, ...updates }));
    },
    []
  );

  // Check if both samplers have data
  const hasData = mcmcState.frames.length > 0 || diffusionState.frames.length > 0;
  const isLoading = mcmcState.isLoading || diffusionState.isLoading;

  return (
    <div className={`bg-slate-800 rounded-lg shadow-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">
            MCMC vs Diffusion Comparison
          </h2>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 text-sm text-slate-400">
              <input
                type="checkbox"
                checked={isSynchronized}
                onChange={(e) => setIsSynchronized(e.target.checked)}
                className="rounded border-slate-600 bg-slate-700 text-blue-500"
              />
              Synchronized
            </label>
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="p-4 border-b border-slate-700">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs text-slate-500 mb-1">
              Lattice Size
            </label>
            <input
              type="number"
              value={config.latticeSize}
              onChange={(e) =>
                updateConfig({ latticeSize: parseInt(e.target.value) || 32 })
              }
              min={8}
              max={64}
              className="w-full px-2 py-1 text-sm bg-slate-700 border border-slate-600 rounded text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">
              MCMC Temperature
            </label>
            <input
              type="number"
              value={config.mcmcTemperature}
              onChange={(e) =>
                updateConfig({
                  mcmcTemperature: parseFloat(e.target.value) || 2.27,
                })
              }
              min={0.1}
              max={10}
              step={0.1}
              className="w-full px-2 py-1 text-sm bg-slate-700 border border-slate-600 rounded text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">
              MCMC Steps
            </label>
            <input
              type="number"
              value={config.mcmcSteps}
              onChange={(e) =>
                updateConfig({ mcmcSteps: parseInt(e.target.value) || 100 })
              }
              min={10}
              max={500}
              className="w-full px-2 py-1 text-sm bg-slate-700 border border-slate-600 rounded text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">
              Diffusion Steps
            </label>
            <input
              type="number"
              value={config.diffusionSteps}
              onChange={(e) =>
                updateConfig({
                  diffusionSteps: parseInt(e.target.value) || 100,
                })
              }
              min={10}
              max={500}
              className="w-full px-2 py-1 text-sm bg-slate-700 border border-slate-600 rounded text-white"
            />
          </div>
        </div>
      </div>

      {/* Comparison Panels Placeholder */}
      <div className="p-4">
        <div className="grid grid-cols-2 gap-4">
          {/* MCMC Panel */}
          <div className="bg-slate-700/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-orange-400">
                MCMC (Metropolis-Hastings)
              </h3>
              {mcmcState.isLoading && (
                <span className="text-xs text-slate-400 animate-pulse">
                  Sampling...
                </span>
              )}
            </div>
            <div className="aspect-square bg-slate-800 rounded flex items-center justify-center">
              {mcmcState.frames.length > 0 ? (
                <span className="text-sm text-slate-400">
                  Frame {mcmcState.currentFrame + 1} / {mcmcState.frames.length}
                </span>
              ) : (
                <span className="text-sm text-slate-500">
                  Run comparison to see results
                </span>
              )}
            </div>
            {mcmcState.metadata[mcmcState.currentFrame] && (
              <div className="mt-2 text-xs text-slate-400">
                Energy: {mcmcState.metadata[mcmcState.currentFrame].energy.toFixed(3)}
              </div>
            )}
          </div>

          {/* Diffusion Panel */}
          <div className="bg-slate-700/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-purple-400">
                Diffusion (Score-Based)
              </h3>
              {diffusionState.isLoading && (
                <span className="text-xs text-slate-400 animate-pulse">
                  Sampling...
                </span>
              )}
            </div>
            <div className="aspect-square bg-slate-800 rounded flex items-center justify-center">
              {diffusionState.frames.length > 0 ? (
                <span className="text-sm text-slate-400">
                  Frame {diffusionState.currentFrame + 1} /{' '}
                  {diffusionState.frames.length}
                </span>
              ) : (
                <span className="text-sm text-slate-500">
                  Run comparison to see results
                </span>
              )}
            </div>
            {diffusionState.metadata[diffusionState.currentFrame] && (
              <div className="mt-2 text-xs text-slate-400">
                t: {(diffusionState.metadata[diffusionState.currentFrame].t ?? 0).toFixed(3)}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="p-4 border-t border-slate-700">
        <div className="flex items-center justify-center gap-4">
          <button
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Run Comparison
          </button>
          {hasData && (
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className={`px-4 py-2 rounded transition-colors ${
                isPlaying
                  ? 'bg-purple-600 text-white'
                  : 'bg-slate-600 text-white hover:bg-slate-500'
              }`}
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default MCMCvsDiffusionComparison;
