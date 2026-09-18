import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ErrorMessage } from './ErrorMessage';

describe('ErrorMessage', () => {
  it('renders error message', () => {
    render(<ErrorMessage message="Test error message" />);
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });

  it('renders error icon and title', () => {
    render(<ErrorMessage message="Error" />);
    expect(screen.getByText('⚠️')).toBeInTheDocument();
    expect(screen.getByText('Erro de Processamento')).toBeInTheDocument();
  });

  it('applies correct styling classes', () => {
    const { container } = render(<ErrorMessage message="Error" />);
    const errorDiv = container.querySelector('div');
    expect(errorDiv).toHaveClass('bg-red-950/80');
    expect(errorDiv).toHaveClass('border-red-600');
  });
});
