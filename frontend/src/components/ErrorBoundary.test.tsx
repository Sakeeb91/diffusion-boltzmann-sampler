/**
 * Component tests for ErrorBoundary.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorBoundary } from './ErrorBoundary';

// Component that throws an error
const ThrowingComponent = ({ shouldThrow }: { shouldThrow: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error message');
  }
  return <div>Child content</div>;
};

// Suppress console.error for cleaner test output
beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

describe('ErrorBoundary', () => {
  describe('when no error occurs', () => {
    it('should render children normally', () => {
      render(
        <ErrorBoundary>
          <div>Test child content</div>
        </ErrorBoundary>
      );
      expect(screen.getByText('Test child content')).toBeInTheDocument();
    });

    it('should render multiple children', () => {
      render(
        <ErrorBoundary>
          <div>First child</div>
          <div>Second child</div>
        </ErrorBoundary>
      );
      expect(screen.getByText('First child')).toBeInTheDocument();
      expect(screen.getByText('Second child')).toBeInTheDocument();
    });
  });

  describe('when error occurs', () => {
    it('should catch error and show default fallback', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('should display error message', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );
      expect(screen.getByText('Test error message')).toBeInTheDocument();
    });

    it('should show Try Again button', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );
      expect(screen.getByText('Try Again')).toBeInTheDocument();
    });

    it('should log error to console', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );
      expect(console.error).toHaveBeenCalledWith(
        'Error caught by boundary:',
        expect.any(Error),
        expect.anything()
      );
    });
  });

  describe('custom fallback', () => {
    it('should render custom fallback when provided', () => {
      render(
        <ErrorBoundary fallback={<div>Custom error UI</div>}>
          <ThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );
      expect(screen.getByText('Custom error UI')).toBeInTheDocument();
      expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
    });
  });

  describe('retry functionality', () => {
    it('should reset error state when Try Again is clicked', () => {
      const { rerender } = render(
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();

      // Click retry
      fireEvent.click(screen.getByText('Try Again'));

      // Re-render with non-throwing component
      rerender(
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(screen.getByText('Child content')).toBeInTheDocument();
      expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
    });
  });

  describe('error without message', () => {
    it('should show default message when error has no message', () => {
      const ThrowsEmptyError = () => {
        throw new Error();
      };

      render(
        <ErrorBoundary>
          <ThrowsEmptyError />
        </ErrorBoundary>
      );

      // Should show default message when error message is empty
      expect(screen.getByText('An unexpected error occurred')).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('should have correct styling classes', () => {
      const { container } = render(
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(container.querySelector('.bg-slate-800')).toBeInTheDocument();
      expect(container.querySelector('.rounded-lg')).toBeInTheDocument();
    });

    it('should have styled Try Again button', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );

      const button = screen.getByText('Try Again');
      expect(button.className).toContain('bg-blue-600');
      expect(button.className).toContain('rounded-lg');
    });
  });
});
