export type AppState = 'IDLE' | 'RECORDING' | 'ANALYZING' | 'RESULT';

export interface SpectrumBand {
  freq_hz: number;
  total_db: number;
  music_db: number;
  noise_db: number;
}

export interface AIDiagnostics {
  model_name: string;
  probabilities: {
    music: number;
    speech: number;
    ambient_noise: number;
  };
  dominant_scene: string;
  scene_description: string;
  confidence: number;
  neural_mask_applied: boolean;
  device: string;
}

export interface AnalysisResult {
  duration_seconds: number;
  metrics: {
    total_dba: number;
    harmonic_music_dba: number;
    noise_dba: number;
    snr_db: number;
    spectral_flatness: number;
    is_music_detected: boolean;
    band_dba?: Record<string, number>;
    lufs?: number;
    music_score?: number;
  };
  recommendation: {
    adjustment_db: number;
    status: string;
    confidence: number;
  };
  ai_diagnostics: AIDiagnostics;
  spectrum_analysis: SpectrumBand[];
  window_count?: number;
}
