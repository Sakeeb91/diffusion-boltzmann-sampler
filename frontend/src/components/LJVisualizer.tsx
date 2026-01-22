import React, { useMemo, useRef, useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

type PlotSize = 'sm' | 'md' | 'lg' | 'auto';

interface LJVisualizerProps {
  /** Particle positions as array of [x, y] coordinates */
  positions: [number, number][] | null;
  /** Size of the periodic box */
  boxSize: number;
  /** Plot title */
  title?: string;
  /** Plot size: 'sm' (250px), 'md' (350px), 'lg' (450px), 'auto' (responsive) */
  plotSize?: PlotSize;
  /** Show position on hover */
  showHover?: boolean;
  /** Particle color */
  particleColor?: string;
  /** Particle size (radius in pixels) */
  particleSize?: number;
  /** Show periodic box boundary */
  showBox?: boolean;
}

const plotSizes: Record<Exclude<PlotSize, 'auto'>, number> = {
  sm: 250,
  md: 350,
  lg: 450,
};

export const LJVisualizer: React.FC<LJVisualizerProps> = ({
  positions,
  boxSize,
  title = 'Particle Configuration',
  plotSize = 'md',
  showHover = true,
  particleColor = '#3b82f6',
  particleSize = 12,
  showBox = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: plotSizes.md, height: plotSizes.md });

  // Handle responsive sizing
  useEffect(() => {
    if (plotSize !== 'auto') {
      const pixelSize = plotSizes[plotSize];
      setDimensions({ width: pixelSize, height: pixelSize });
      return;
    }

    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth - 32;
        const constrainedSize = Math.min(Math.max(width, 200), 600);
        setDimensions({ width: constrainedSize, height: constrainedSize });
      }
    };

    updateDimensions();
    const observer = new ResizeObserver(updateDimensions);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, [plotSize]);

  const plotData = useMemo(() => {
    if (!positions || positions.length === 0) {
      return [
        {
          x: [] as number[],
          y: [] as number[],
          mode: 'markers' as const,
          type: 'scatter' as const,
          marker: { size: particleSize, color: particleColor },
        },
      ];
    }

    return [
      {
        x: positions.map((p) => p[0]),
        y: positions.map((p) => p[1]),
        mode: 'markers' as const,
        type: 'scatter' as const,
        marker: {
          size: particleSize,
          color: particleColor,
          line: {
            color: '#1e3a5f',
            width: 1,
          },
        },
        hovertemplate: showHover
          ? 'x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
          : '',
        hoverinfo: showHover ? undefined : ('none' as const),
      },
    ];
  }, [positions, particleColor, particleSize, showHover]);

  const layout = useMemo(
    () => ({
      width: dimensions.width,
      height: dimensions.height,
      margin: { t: 30, b: 40, l: 40, r: 20 },
      title: {
        text: title,
        font: { size: 14, color: '#e2e8f0' },
      },
      xaxis: {
        range: [0, boxSize],
        title: 'x',
        titlefont: { size: 12, color: '#94a3b8' },
        tickfont: { size: 10, color: '#94a3b8' },
        gridcolor: '#334155',
        zerolinecolor: '#475569',
        showline: showBox,
        linecolor: '#64748b',
        mirror: showBox,
      },
      yaxis: {
        range: [0, boxSize],
        title: 'y',
        titlefont: { size: 12, color: '#94a3b8' },
        tickfont: { size: 10, color: '#94a3b8' },
        gridcolor: '#334155',
        zerolinecolor: '#475569',
        scaleanchor: 'x' as const,
        showline: showBox,
        linecolor: '#64748b',
        mirror: showBox,
      },
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#1e293b',
      showlegend: false,
    }),
    [title, dimensions, boxSize, showBox]
  );

  return (
    <div ref={containerRef} className="bg-slate-800 rounded-lg p-4 shadow-lg">
      <Plot
        data={plotData}
        layout={layout}
        config={{
          displayModeBar: false,
          responsive: plotSize === 'auto',
        }}
      />
      {positions && (
        <div className="mt-2 text-xs text-slate-400 text-center">
          {positions.length} particles | Box: {boxSize.toFixed(1)} × {boxSize.toFixed(1)}
        </div>
      )}
    </div>
  );
};

export default LJVisualizer;
