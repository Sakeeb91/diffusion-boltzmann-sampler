/**
 * Component tests for LoadingSpinner and LoadingOverlay.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LoadingSpinner, LoadingOverlay } from './LoadingSpinner';

describe('LoadingSpinner', () => {
  describe('rendering', () => {
    it('should render spinner element', () => {
      render(<LoadingSpinner />);
      expect(screen.getByRole('status')).toBeInTheDocument();
    });

    it('should have default aria-label', () => {
      render(<LoadingSpinner />);
      expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'Loading');
    });

    it('should use custom aria-label when label provided', () => {
      render(<LoadingSpinner label="Processing" />);
      expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'Processing');
    });
  });

  describe('label', () => {
    it('should not show label by default', () => {
      render(<LoadingSpinner />);
      expect(screen.queryByText(/Loading/)).not.toBeInTheDocument();
    });

    it('should show label when provided', () => {
      render(<LoadingSpinner label="Please wait..." />);
      expect(screen.getByText('Please wait...')).toBeInTheDocument();
    });

    it('should style label correctly', () => {
      render(<LoadingSpinner label="Loading data" />);
      const label = screen.getByText('Loading data');
      expect(label.className).toContain('text-sm');
      expect(label.className).toContain('text-slate-400');
    });
  });

  describe('sizes', () => {
    it('should use medium size by default', () => {
      render(<LoadingSpinner />);
      const spinner = screen.getByRole('status');
      expect(spinner.className).toContain('w-8');
      expect(spinner.className).toContain('h-8');
    });

    it('should render small size', () => {
      render(<LoadingSpinner size="sm" />);
      const spinner = screen.getByRole('status');
      expect(spinner.className).toContain('w-4');
      expect(spinner.className).toContain('h-4');
    });

    it('should render large size', () => {
      render(<LoadingSpinner size="lg" />);
      const spinner = screen.getByRole('status');
      expect(spinner.className).toContain('w-12');
      expect(spinner.className).toContain('h-12');
    });
  });

  describe('centered prop', () => {
    it('should not be centered by default', () => {
      const { container } = render(<LoadingSpinner />);
      expect(container.querySelector('.min-h-\\[100px\\]')).not.toBeInTheDocument();
    });

    it('should center when centered prop is true', () => {
      const { container } = render(<LoadingSpinner centered />);
      expect(container.querySelector('.justify-center')).toBeInTheDocument();
    });

    it('should have flex-col when centered', () => {
      const { container } = render(<LoadingSpinner centered label="Loading" />);
      expect(container.querySelector('.flex-col')).toBeInTheDocument();
    });
  });

  describe('animation', () => {
    it('should have spin animation class', () => {
      render(<LoadingSpinner />);
      const spinner = screen.getByRole('status');
      expect(spinner.className).toContain('animate-spin');
    });

    it('should have rounded-full for circular shape', () => {
      render(<LoadingSpinner />);
      const spinner = screen.getByRole('status');
      expect(spinner.className).toContain('rounded-full');
    });
  });

  describe('colors', () => {
    it('should have slate border color', () => {
      render(<LoadingSpinner />);
      const spinner = screen.getByRole('status');
      expect(spinner.className).toContain('border-slate-600');
    });

    it('should have blue top border for visible spinner', () => {
      render(<LoadingSpinner />);
      const spinner = screen.getByRole('status');
      expect(spinner.className).toContain('border-t-blue-500');
    });
  });
});

describe('LoadingOverlay', () => {
  it('should render with fixed positioning', () => {
    const { container } = render(<LoadingOverlay />);
    expect(container.querySelector('.fixed')).toBeInTheDocument();
    expect(container.querySelector('.inset-0')).toBeInTheDocument();
  });

  it('should have high z-index', () => {
    const { container } = render(<LoadingOverlay />);
    expect(container.querySelector('.z-50')).toBeInTheDocument();
  });

  it('should have semi-transparent background', () => {
    const { container } = render(<LoadingOverlay />);
    expect(container.querySelector('.bg-slate-900\\/80')).toBeInTheDocument();
  });

  it('should center content', () => {
    const { container } = render(<LoadingOverlay />);
    expect(container.querySelector('.flex')).toBeInTheDocument();
    expect(container.querySelector('.items-center')).toBeInTheDocument();
    expect(container.querySelector('.justify-center')).toBeInTheDocument();
  });

  it('should render large spinner', () => {
    render(<LoadingOverlay />);
    const spinner = screen.getByRole('status');
    expect(spinner.className).toContain('w-12');
    expect(spinner.className).toContain('h-12');
  });

  it('should show label when provided', () => {
    render(<LoadingOverlay label="Saving changes..." />);
    expect(screen.getByText('Saving changes...')).toBeInTheDocument();
  });

  it('should not show label when not provided', () => {
    render(<LoadingOverlay />);
    expect(screen.queryByText(/Loading/)).not.toBeInTheDocument();
  });
});
