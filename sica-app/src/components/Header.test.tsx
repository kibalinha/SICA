import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Header } from './Header';

describe('Header', () => {
  it('renders SICA title', () => {
    render(<Header />);
    expect(screen.getByText('SICA')).toBeInTheDocument();
  });

  it('renders subtitle with tech stack info', () => {
    render(<Header />);
    expect(screen.getByText('Deep Learning (PyTorch CNN) & Fourier (STFT)')).toBeInTheDocument();
  });

  it('has proper ARIA structure', () => {
    render(<Header />);
    const header = screen.getByRole('banner');
    expect(header).toBeInTheDocument();
  });

  it('applies correct styling classes', () => {
    const { container } = render(<Header />);
    const headerElement = container.querySelector('header');
    expect(headerElement).toHaveClass('bg-gradient-to-r');
    expect(headerElement).toHaveClass('from-blue-900');
    expect(headerElement).toHaveClass('to-indigo-900');
  });
});
