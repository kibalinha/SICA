import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AudioVisualizer } from './AudioVisualizer';

describe('AudioVisualizer', () => {
  const mockGetByteFrequencyData = vi.fn();
  const mockAnalyser = {
    frequencyBinCount: 128,
    getByteFrequencyData: mockGetByteFrequencyData,
  } as unknown as AnalyserNode;

  beforeEach(() => {
    mockGetByteFrequencyData.mockClear();

    // jsdom não tem canvas 2d real: getContext('2d') retorna null,
    // então o loop de desenho é pulado e o componente ainda deve renderizar.
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      return setTimeout(cb as () => void, 0) as unknown as number;
    });
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation((id) => {
      clearTimeout(id);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders canvas when analyser provided and recording', () => {
    render(<AudioVisualizer analyser={mockAnalyser} isRecording={true} />);
    const canvas = screen.getByTestId('audio-visualizer-canvas');
    expect(canvas).toBeInTheDocument();
  });

  it('does not render canvas when not recording', () => {
    render(<AudioVisualizer analyser={mockAnalyser} isRecording={false} />);
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
  });

  it('does not render canvas when analyser is null', () => {
    render(<AudioVisualizer analyser={null} isRecording={true} />);
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
  });

  it('renders FFT Spectrum label', () => {
    render(<AudioVisualizer analyser={mockAnalyser} isRecording={true} />);
    expect(screen.getByText('FFT SPECTRUM (FFT_SIZE=256)')).toBeInTheDocument();
  });

  it('renders LIVE CAPTURE indicator', () => {
    render(<AudioVisualizer analyser={mockAnalyser} isRecording={true} />);
    expect(screen.getByText('LIVE CAPTURE')).toBeInTheDocument();
  });
});