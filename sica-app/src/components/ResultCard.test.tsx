import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ResultCard } from './ResultCard';
import type { AnalysisResult } from '../types';

const mockAnalysis: AnalysisResult = {
  duration_seconds: 4.5,
  metrics: {
    total_dba: 65.0,
    harmonic_music_dba: 60.0,
    noise_dba: 55.0,
    snr_db: 5.0,
    spectral_flatness: 0.1,
    is_music_detected: true,
  },
  recommendation: {
    adjustment_db: 0,
    status: 'Nível equilibrado',
    confidence: 0.8,
  },
  ai_diagnostics: {
    model_name: 'Test Model',
    probabilities: { music: 50.0, speech: 30.0, ambient_noise: 20.0 },
    dominant_scene: 'Música',
    scene_description: 'Música Ambiente Predominante',
    confidence: 0.8,
    neural_mask_applied: false,
    device: 'cpu',
  },
  spectrum_analysis: [],
};

describe('ResultCard', () => {
  it('renders recommendation card', () => {
    render(<ResultCard analysis={mockAnalysis} onReset={vi.fn()} />);
    expect(screen.getByText('Recomendação Inteligente SICA')).toBeInTheDocument();
    expect(screen.getByText('0')).toBeInTheDocument();
    expect(screen.getByText('dB')).toBeInTheDocument();
    expect(screen.getByText('Nível equilibrado')).toBeInTheDocument();
  });

  it('renders AI diagnostics card', () => {
    render(<ResultCard analysis={mockAnalysis} onReset={vi.fn()} />);
    expect(screen.getByText('Test Model')).toBeInTheDocument();
    expect(screen.getByText('Música Ambiente Predominante')).toBeInTheDocument();
    expect(screen.getByText('Música')).toBeInTheDocument();
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('30%')).toBeInTheDocument();
    expect(screen.getByText('20%')).toBeInTheDocument();
  });

  it('renders metrics grid', () => {
    render(<ResultCard analysis={mockAnalysis} onReset={vi.fn()} />);
    expect(screen.getByText('NÍVEL TOTAL dB(A)')).toBeInTheDocument();
    expect(screen.getByText('65 dBA')).toBeInTheDocument();
    expect(screen.getByText('RELAÇÃO SINAL/RUÍDO (SNR)')).toBeInTheDocument();
    expect(screen.getByText('5 dB')).toBeInTheDocument();
  });

  it('calls onReset when button clicked', () => {
    const mockOnReset = vi.fn();
    render(<ResultCard analysis={mockAnalysis} onReset={mockOnReset} />);
    const button = screen.getByRole('button', { name: /nova análise/i });
    fireEvent.click(button);
    expect(mockOnReset).toHaveBeenCalledTimes(1);
  });

  it('handles negative adjustment with red color', () => {
    const analysisWithNegative = {
      ...mockAnalysis,
      recommendation: { ...mockAnalysis.recommendation, adjustment_db: -3 },
    };
    render(<ResultCard analysis={analysisWithNegative} onReset={vi.fn()} />);
    const adjustmentText = screen.getByText('-3');
    expect(adjustmentText).toHaveClass('text-red-400');
  });

  it('handles positive adjustment with green color', () => {
    const analysisWithPositive = {
      ...mockAnalysis,
      recommendation: { ...mockAnalysis.recommendation, adjustment_db: 3 },
    };
    render(<ResultCard analysis={analysisWithPositive} onReset={vi.fn()} />);
    const adjustmentText = screen.getByText('+3');
    expect(adjustmentText).toHaveClass('text-emerald-400');
  });

  it('shows neural mask indicator when applied', () => {
    const analysisWithMask = {
      ...mockAnalysis,
      ai_diagnostics: { ...mockAnalysis.ai_diagnostics, neural_mask_applied: true },
    };
    render(<ResultCard analysis={analysisWithMask} onReset={vi.fn()} />);
    expect(screen.getByText(/máscara neural ativa/i)).toBeInTheDocument();
  });
});
