import React from 'react';
import { FrameMetadata } from '../store/simulationStore';

interface DiffusionProgressVisualizationProps {
  metadata: FrameMetadata | null;
  className?: string;
}

/** Visual phases of the diffusion denoising process */
const DIFFUSION_PHASES = [
  { range: [0.8, 1.0], label: 'Pure Noise', color: 'from-pink-500 to-purple-500' },
  { range: [0.5, 0.8], label: 'Emerging Structure', color: 'from-purple-500 to-blue-500' },
  { range: [0.2, 0.5], label: 'Refining', color: 'from-blue-500 to-cyan-500' },
  { range: [0.0, 0.2], label: 'Crystallized', color: 'from-cyan-500 to-green-500' },
] as const;

/**
 * Get the current phase based on diffusion time t.
 */
function getCurrentPhase(t: number): (typeof DIFFUSION_PHASES)[number] {
  for (const phase of DIFFUSION_PHASES) {
    if (t >= phase.range[0] && t <= phase.range[1]) {
      return phase;
    }
  }
  return DIFFUSION_PHASES[DIFFUSION_PHASES.length - 1];
}

/**
 * Component to visualize the diffusion denoising trajectory.
 * Shows a timeline with phases and current position indicator.
 */
export const DiffusionProgressVisualization: React.FC<
  DiffusionProgressVisualizationProps
> = ({ metadata, className = '' }) => {
  if (!metadata || metadata.sampler !== 'diffusion') {
    return null;
  }

  const t = metadata.t ?? 1.0;
  const currentPhase = getCurrentPhase(t);
  // Position from right (t=1) to left (t=0)
  const progressPercent = (1 - t) * 100;

  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-400">
          Denoising Trajectory
        </span>
        <span className="text-xs text-purple-400 font-medium">
          {currentPhase.label}
        </span>
      </div>

      {/* Phase timeline */}
      <div className="relative h-8 bg-slate-700 rounded-lg overflow-hidden">
        {/* Phase segments */}
        <div className="absolute inset-0 flex">
          {DIFFUSION_PHASES.slice()
            .reverse()
            .map((phase, index) => {
              const width = (phase.range[1] - phase.range[0]) * 100;
              return (
                <div
                  key={phase.label}
                  className={`h-full bg-gradient-to-r ${phase.color} opacity-30`}
                  style={{ width: `${width}%` }}
                  title={phase.label}
                />
              );
            })}
        </div>

        {/* Progress fill */}
        <div
          className={`absolute h-full bg-gradient-to-r ${currentPhase.color} transition-all duration-150`}
          style={{ width: `${progressPercent}%` }}
        />

        {/* Current position marker */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-white shadow-lg transition-all duration-150"
          style={{ left: `${progressPercent}%`, transform: 'translateX(-50%)' }}
        />

        {/* Phase labels */}
        <div className="absolute inset-0 flex items-center justify-between px-2 text-[10px] text-white/60 pointer-events-none">
          <span>t=0</span>
          <span>t=0.5</span>
          <span>t=1</span>
        </div>
      </div>

      {/* Detailed stats */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="bg-slate-700/50 rounded p-2">
          <div className="text-xs text-slate-500">Time</div>
          <div className="text-sm font-mono text-slate-300">
            {t.toFixed(3)}
          </div>
        </div>
        <div className="bg-slate-700/50 rounded p-2">
          <div className="text-xs text-slate-500">Sigma</div>
          <div className="text-sm font-mono text-purple-400">
            {(metadata.sigma ?? 0).toFixed(3)}
          </div>
        </div>
        <div className="bg-slate-700/50 rounded p-2">
          <div className="text-xs text-slate-500">Progress</div>
          <div className="text-sm font-mono text-green-400">
            {progressPercent.toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiffusionProgressVisualization;
