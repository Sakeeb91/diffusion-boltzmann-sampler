import React from 'react';

interface LoadingSpinnerProps {
  /** Size of the spinner */
  size?: 'sm' | 'md' | 'lg';
  /** Optional label text */
  label?: string;
  /** Center in container */
  centered?: boolean;
}

const sizeClasses = {
  sm: 'w-4 h-4 border-2',
  md: 'w-8 h-8 border-3',
  lg: 'w-12 h-12 border-4',
};

/**
 * Loading spinner component with optional label.
 */
export function LoadingSpinner({
  size = 'md',
  label,
  centered = false,
}: LoadingSpinnerProps): JSX.Element {
  const spinner = (
    <div className={centered ? 'flex flex-col items-center gap-3' : 'inline-flex flex-col items-center gap-2'}>
      <div
        className={`
          ${sizeClasses[size]}
          border-slate-600
          border-t-blue-500
          rounded-full
          animate-spin
        `}
        role="status"
        aria-label={label || 'Loading'}
      />
      {label && (
        <span className="text-sm text-slate-400">{label}</span>
      )}
    </div>
  );

  if (centered) {
    return (
      <div className="flex items-center justify-center min-h-[100px]">
        {spinner}
      </div>
    );
  }

  return spinner;
}

/**
 * Full-page loading overlay.
 */
export function LoadingOverlay({ label }: { label?: string }): JSX.Element {
  return (
    <div className="fixed inset-0 bg-slate-900/80 flex items-center justify-center z-50">
      <LoadingSpinner size="lg" label={label} />
    </div>
  );
}

export default LoadingSpinner;
