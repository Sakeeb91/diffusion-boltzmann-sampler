/**
 * Component tests for CorrelationPlot and DistributionPlot.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CorrelationPlot, DistributionPlot } from './CorrelationPlot';

// Mock react-plotly.js
vi.mock('react-plotly.js', () => ({
  default: vi.fn(({ data, layout }: { data: unknown[]; layout: { title?: { text?: string } } }) => (
    <div data-testid="plotly-mock" data-title={layout.title?.text}>
      <div data-testid="plot-data">{JSON.stringify(data)}</div>
    </div>
  )),
}));

// Mock export utilities
const mockExportCorrelation = vi.fn();
const mockExportDistribution = vi.fn();
vi.mock('../utils/export', () => ({
  exportCorrelationData: (...args: unknown[]) => mockExportCorrelation(...args),
  exportDistributionData: (...args: unknown[]) => mockExportDistribution(...args),
}));

describe('CorrelationPlot', () => {
  const mockMcmcData = { r: [1, 2, 3], C_r: [1, 0.5, 0.25] };
  const mockDiffusionData = { r: [1, 2, 3], C_r: [1, 0.48, 0.23] };

  beforeEach(() => {
    mockExportCorrelation.mockClear();
  });

  describe('empty state', () => {
    it('should show empty message when no data', () => {
      render(<CorrelationPlot />);
      expect(screen.getByText('Run comparison to see correlation data')).toBeInTheDocument();
    });

    it('should show title in empty state', () => {
      render(<CorrelationPlot title="Custom Title" />);
      expect(screen.getByText('Custom Title')).toBeInTheDocument();
    });
  });

  describe('with data', () => {
    it('should render plot with MCMC data', () => {
      render(<CorrelationPlot mcmcData={mockMcmcData} />);
      expect(screen.getByTestId('plotly-mock')).toBeInTheDocument();
    });

    it('should render plot with diffusion data', () => {
      render(<CorrelationPlot diffusionData={mockDiffusionData} />);
      expect(screen.getByTestId('plotly-mock')).toBeInTheDocument();
    });

    it('should render plot with both data sources', () => {
      render(<CorrelationPlot mcmcData={mockMcmcData} diffusionData={mockDiffusionData} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');
      expect(data).toHaveLength(2);
    });

    it('should use correct title', () => {
      render(<CorrelationPlot mcmcData={mockMcmcData} title="Spin-Spin Correlation" />);
      expect(screen.getByTestId('plotly-mock')).toHaveAttribute('data-title', 'Spin-Spin Correlation');
    });
  });

  describe('plot data structure', () => {
    it('should include MCMC trace with correct properties', () => {
      render(<CorrelationPlot mcmcData={mockMcmcData} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].x).toEqual(mockMcmcData.r);
      expect(data[0].y).toEqual(mockMcmcData.C_r);
      expect(data[0].name).toBe('MCMC');
      expect(data[0].type).toBe('scatter');
    });

    it('should include diffusion trace with correct properties', () => {
      render(<CorrelationPlot diffusionData={mockDiffusionData} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].x).toEqual(mockDiffusionData.r);
      expect(data[0].y).toEqual(mockDiffusionData.C_r);
      expect(data[0].name).toBe('Diffusion');
    });
  });

  describe('export functionality', () => {
    it('should show export button when data present', () => {
      render(<CorrelationPlot mcmcData={mockMcmcData} />);
      expect(screen.getByTitle('Export data as CSV')).toBeInTheDocument();
    });

    it('should hide export button when showExport is false', () => {
      render(<CorrelationPlot mcmcData={mockMcmcData} showExport={false} />);
      expect(screen.queryByTitle('Export data as CSV')).not.toBeInTheDocument();
    });

    it('should call export function when button clicked', () => {
      render(<CorrelationPlot mcmcData={mockMcmcData} diffusionData={mockDiffusionData} />);
      fireEvent.click(screen.getByTitle('Export data as CSV'));
      expect(mockExportCorrelation).toHaveBeenCalledWith(
        mockMcmcData,
        mockDiffusionData,
        'correlation_data.csv'
      );
    });
  });
});

describe('DistributionPlot', () => {
  const mockMcmcData = { values: [0, 0.5, 1], probabilities: [0.3, 0.4, 0.3] };
  const mockDiffusionData = { values: [0, 0.5, 1], probabilities: [0.28, 0.42, 0.3] };

  beforeEach(() => {
    mockExportDistribution.mockClear();
  });

  describe('empty state', () => {
    it('should show empty message when no data', () => {
      render(<DistributionPlot />);
      expect(screen.getByText('Run comparison to see distribution')).toBeInTheDocument();
    });

    it('should show custom title in empty state', () => {
      render(<DistributionPlot title="Energy Distribution" />);
      expect(screen.getByText('Energy Distribution')).toBeInTheDocument();
    });
  });

  describe('with data', () => {
    it('should render plot with MCMC data', () => {
      render(<DistributionPlot mcmcData={mockMcmcData} />);
      expect(screen.getByTestId('plotly-mock')).toBeInTheDocument();
    });

    it('should render plot with diffusion data', () => {
      render(<DistributionPlot diffusionData={mockDiffusionData} />);
      expect(screen.getByTestId('plotly-mock')).toBeInTheDocument();
    });

    it('should render plot with both data sources', () => {
      render(<DistributionPlot mcmcData={mockMcmcData} diffusionData={mockDiffusionData} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');
      expect(data).toHaveLength(2);
    });
  });

  describe('plot data structure', () => {
    it('should use bar chart type', () => {
      render(<DistributionPlot mcmcData={mockMcmcData} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');
      expect(data[0].type).toBe('bar');
    });

    it('should include MCMC distribution data', () => {
      render(<DistributionPlot mcmcData={mockMcmcData} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].x).toEqual(mockMcmcData.values);
      expect(data[0].y).toEqual(mockMcmcData.probabilities);
      expect(data[0].name).toBe('MCMC');
    });

    it('should include diffusion distribution data', () => {
      render(<DistributionPlot diffusionData={mockDiffusionData} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].x).toEqual(mockDiffusionData.values);
      expect(data[0].y).toEqual(mockDiffusionData.probabilities);
      expect(data[0].name).toBe('Diffusion');
    });
  });

  describe('export functionality', () => {
    it('should show export button when data present', () => {
      render(<DistributionPlot mcmcData={mockMcmcData} />);
      expect(screen.getByTitle('Export data as CSV')).toBeInTheDocument();
    });

    it('should hide export button when showExport is false', () => {
      render(<DistributionPlot mcmcData={mockMcmcData} showExport={false} />);
      expect(screen.queryByTitle('Export data as CSV')).not.toBeInTheDocument();
    });

    it('should call export function with correct filename', () => {
      render(
        <DistributionPlot
          mcmcData={mockMcmcData}
          diffusionData={mockDiffusionData}
          title="Energy Distribution"
        />
      );
      fireEvent.click(screen.getByTitle('Export data as CSV'));
      expect(mockExportDistribution).toHaveBeenCalledWith(
        mockMcmcData,
        mockDiffusionData,
        'energy_distribution_data.csv'
      );
    });

    it('should use default title for filename', () => {
      render(<DistributionPlot mcmcData={mockMcmcData} />);
      fireEvent.click(screen.getByTitle('Export data as CSV'));
      expect(mockExportDistribution).toHaveBeenCalledWith(
        mockMcmcData,
        undefined,
        'distribution_data.csv'
      );
    });
  });

  describe('xlabel prop', () => {
    it('should accept custom xlabel', () => {
      render(<DistributionPlot mcmcData={mockMcmcData} xlabel="Magnetization" />);
      expect(screen.getByTestId('plotly-mock')).toBeInTheDocument();
    });
  });
});
