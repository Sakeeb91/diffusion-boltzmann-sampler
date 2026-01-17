/**
 * Component tests for MCMCvsDiffusionComparison.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MCMCvsDiffusionComparison } from './MCMCvsDiffusionComparison';

describe('MCMCvsDiffusionComparison', () => {
  describe('initial render', () => {
    it('should render component title', () => {
      render(<MCMCvsDiffusionComparison />);
      expect(screen.getByText('MCMC vs Diffusion Comparison')).toBeInTheDocument();
    });

    it('should render synchronized checkbox', () => {
      render(<MCMCvsDiffusionComparison />);
      expect(screen.getByText('Synchronized')).toBeInTheDocument();
      expect(screen.getByRole('checkbox')).toBeChecked();
    });

    it('should render configuration panel', () => {
      render(<MCMCvsDiffusionComparison />);
      expect(screen.getByText('Lattice Size')).toBeInTheDocument();
      expect(screen.getByText('MCMC Temperature')).toBeInTheDocument();
      expect(screen.getByText('MCMC Steps')).toBeInTheDocument();
      expect(screen.getByText('Diffusion Steps')).toBeInTheDocument();
    });

    it('should render both sampler panels', () => {
      render(<MCMCvsDiffusionComparison />);
      expect(screen.getByText('MCMC (Metropolis-Hastings)')).toBeInTheDocument();
      expect(screen.getByText('Diffusion (Score-Based)')).toBeInTheDocument();
    });

    it('should render run comparison button', () => {
      render(<MCMCvsDiffusionComparison />);
      expect(screen.getByText('Run Comparison')).toBeInTheDocument();
    });

    it('should show placeholder text in panels', () => {
      render(<MCMCvsDiffusionComparison />);
      const placeholders = screen.getAllByText('Run comparison to see results');
      expect(placeholders).toHaveLength(2);
    });
  });

  describe('configuration inputs', () => {
    it('should have default lattice size of 32', () => {
      render(<MCMCvsDiffusionComparison />);
      const input = screen.getByLabelText(/Lattice Size/i).closest('div')?.querySelector('input');
      expect(input).toHaveValue(32);
    });

    it('should have default temperature of 2.27', () => {
      render(<MCMCvsDiffusionComparison />);
      const input = screen.getByLabelText(/MCMC Temperature/i).closest('div')?.querySelector('input');
      expect(input).toHaveValue(2.27);
    });

    it('should have default MCMC steps of 100', () => {
      render(<MCMCvsDiffusionComparison />);
      const input = screen.getByLabelText(/MCMC Steps/i).closest('div')?.querySelector('input');
      expect(input).toHaveValue(100);
    });

    it('should have default diffusion steps of 100', () => {
      render(<MCMCvsDiffusionComparison />);
      const input = screen.getByLabelText(/Diffusion Steps/i).closest('div')?.querySelector('input');
      expect(input).toHaveValue(100);
    });

    it('should update lattice size when changed', () => {
      render(<MCMCvsDiffusionComparison />);
      const input = screen.getByLabelText(/Lattice Size/i).closest('div')?.querySelector('input');

      fireEvent.change(input!, { target: { value: '64' } });

      expect(input).toHaveValue(64);
    });

    it('should update temperature when changed', () => {
      render(<MCMCvsDiffusionComparison />);
      const input = screen.getByLabelText(/MCMC Temperature/i).closest('div')?.querySelector('input');

      fireEvent.change(input!, { target: { value: '3.0' } });

      expect(input).toHaveValue(3.0);
    });
  });

  describe('synchronized toggle', () => {
    it('should toggle synchronized state when clicked', () => {
      render(<MCMCvsDiffusionComparison />);
      const checkbox = screen.getByRole('checkbox');

      expect(checkbox).toBeChecked();

      fireEvent.click(checkbox);

      expect(checkbox).not.toBeChecked();
    });
  });

  describe('play button', () => {
    it('should not show play button initially (no data)', () => {
      render(<MCMCvsDiffusionComparison />);
      expect(screen.queryByText('Play')).not.toBeInTheDocument();
    });
  });

  describe('custom className', () => {
    it('should apply custom className', () => {
      const { container } = render(
        <MCMCvsDiffusionComparison className="custom-class" />
      );
      expect(container.firstChild).toHaveClass('custom-class');
    });
  });
});
