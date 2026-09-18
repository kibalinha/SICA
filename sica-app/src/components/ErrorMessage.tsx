interface ErrorMessageProps {
  message: string;
}

export function ErrorMessage({ message }: ErrorMessageProps) {
  return (
    <div className="sica-fade-in flex items-start space-x-2 rounded-xl border border-red-600 bg-red-950/80 p-4 text-xs text-red-200">
      <span className="text-base">⚠️</span>
      <div>
        <strong className="mb-1 block font-bold">Erro de Processamento</strong>
        {message}
      </div>
    </div>
  );
}
