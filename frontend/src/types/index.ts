/**
 * Shared TypeScript type definitions for the frontend.
 */

// ============================================================================
// Spin Configuration Types
// ============================================================================

/** A single spin value (+1 or -1) */
export type SpinValue = 1 | -1;

/** A row of spin values in the lattice */
export type SpinRow = number[];

/** 2D spin configuration (lattice) */
export type SpinConfiguration = number[][];

// ============================================================================
// Physical Parameter Types
// ============================================================================

/** Temperature value (must be positive) */
export type Temperature = number;

/** Lattice size (must be positive integer) */
export type LatticeSize = number;

/** Energy value (can be negative) */
export type Energy = number;

/** Magnetization value (between -1 and 1) */
export type Magnetization = number;

// ============================================================================
// Sampler Types
// ============================================================================

/** Available sampler types */
export type SamplerType = 'mcmc' | 'diffusion';

/** Sampling parameters */
export interface SamplingParams {
  temperature: Temperature;
  latticeSize: LatticeSize;
  samplerType: SamplerType;
  numSteps: number;
}

// ============================================================================
// Animation Types
// ============================================================================

/** Single animation frame with metadata */
export interface AnimationFrame {
  spins: SpinConfiguration;
  energy: Energy;
  magnetization: Magnetization;
  step: number;
  t?: number; // Diffusion time
}

/** Animation playback state */
export interface AnimationState {
  frames: SpinConfiguration[];
  currentFrame: number;
  isPlaying: boolean;
  playbackSpeed: number;
}

// ============================================================================
// API Response Types
// ============================================================================

/** Health check response */
export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  version?: string;
  title?: string;
  features?: Record<string, boolean>;
}

/** Random configuration response */
export interface RandomConfigResponse {
  spins: SpinConfiguration;
  energy: Energy;
  magnetization: Magnetization;
}

/** Sample response from MCMC or diffusion */
export interface SampleResponse {
  samples: SpinConfiguration[];
  energies: Energy[];
  magnetizations: Magnetization[];
  temperature: Temperature;
  lattice_size: LatticeSize;
}

/** WebSocket frame data */
export interface WSFrameData {
  type: 'frame' | 'done' | 'error';
  spins?: SpinConfiguration;
  energy?: Energy;
  magnetization?: Magnetization;
  t?: number;
  message?: string;
}

// ============================================================================
// Analysis Types
// ============================================================================

/** Correlation data */
export interface CorrelationData {
  r: number[];
  C_r: number[];
}

/** Distribution data */
export interface DistributionData {
  values: number[];
  probabilities: number[];
}

/** Magnetization distribution from API */
export interface MagnetizationDistribution {
  M: number[];
  P_M: number[];
}

/** Energy distribution from API */
export interface EnergyDistribution {
  E: number[];
  P_E: number[];
}

/** Sampler statistics */
export interface SamplerStats {
  magnetization: MagnetizationDistribution;
  energy: EnergyDistribution;
  correlation: CorrelationData;
  mean_mag: number;
  var_mag: number;
  autocorrelation_time?: number;
}

/** Comparison analysis response */
export interface ComparisonResponse {
  mcmc: SamplerStats & { autocorrelation_time: number };
  diffusion: SamplerStats;
  comparison_metrics: Record<string, number>;
  temperature: Temperature;
  lattice_size: LatticeSize;
}

/** Phase diagram response */
export interface PhaseDiagramResponse {
  temperatures: number[];
  mean_magnetization: number[];
  std_magnetization: number[];
  T_critical: number;
  lattice_size: LatticeSize;
}

// ============================================================================
// UI State Types
// ============================================================================

/** Connection status */
export interface ConnectionStatus {
  isConnected: boolean;
  isChecking: boolean;
  error: string | null;
  lastChecked: Date | null;
}

/** Notification type */
export type NotificationType = 'info' | 'success' | 'warning' | 'error';

/** Toast notification */
export interface ToastNotification {
  id: string;
  type: NotificationType;
  message: string;
  duration?: number;
}

// ============================================================================
// Component Props Types
// ============================================================================

/** Base props for visualization components */
export interface VisualizationProps {
  width?: number;
  height?: number;
  title?: string;
}

/** Plot color scheme */
export interface ColorScheme {
  primary: string;
  secondary: string;
  background: string;
  text: string;
  grid: string;
}
