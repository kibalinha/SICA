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

    // Mock requestAnimationFrame
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      return setTimeout(cb, 0);
    });
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation((id) => {
      clearTimeout(id);
    });

    // Mock HTMLCanvasElement and its getContext method
    const mockCanvas = {
      width: 300,
      height: 150,
      style: {},
      getContext: vi.fn().mockReturnValue({
        clearRect: vi.fn(),
        fillRect: vi.fn(),
        fillStyle: '',
        setTransform: vi.fn()
      }),
      getBoundingClientRect: vi.fn().mockReturnValue({
        width: 300,
        height: 150,
        left: 0,
        top: 0,
        right: 300,
        bottom: 150
      })
    } as unknown as HTMLCanvasElement;
    
    // Mock document.createElement to return our canvas
    vi.spyOn(document, 'createElement').mockReturnValue(mockCanvas);
    
    // Mock document.createElement to return our canvas
    vi.spyOn(document, 'createElement').mockReturnValue(mockCanvas);
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
