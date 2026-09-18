import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { IdleState } from './IdleState';

describe('IdleState', () => {
  it('renders start button', () => {
    const mockOnClick = vi.fn();
    render(<IdleState onStartRecording={mockOnClick} />);
    const button = screen.getByRole('button', { name: /iniciar gravação/i });
    expect(button).toBeInTheDocument();
  });

  it('calls onStartRecording when button clicked', () => {
    const mockOnClick = vi.fn();
    render(<IdleState onStartRecording={mockOnClick} />);
    const button = screen.getByRole('button', { name: /iniciar gravação/i });
    fireEvent.click(button);
    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });

  it('renders descriptive text', () => {
    render(<IdleState onStartRecording={vi.fn()} />);
    expect(screen.getByText('Captura Espectral Real')).toBeInTheDocument();
    expect(screen.getByText(/classificação neural/i)).toBeInTheDocument();
  });

  it('has proper accessibility attributes', () => {
    render(<IdleState onStartRecording={vi.fn()} />);
    const button = screen.getByRole('button', { name: /iniciar gravação/i });
    expect(button).toHaveAttribute('aria-label', 'Iniciar gravação de áudio e análise de IA');
  });
});
