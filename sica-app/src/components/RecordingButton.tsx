interface RecordingButtonProps {
  onClick: () => void;
  disabled?: boolean;
}

export function RecordingButton({ onClick, disabled = false }: RecordingButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label="Iniciar gravação de áudio e análise de IA"
      className="flex w-full transform items-center justify-center space-x-2 rounded-2xl border border-blue-400/30 bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4 font-bold text-white shadow-xl shadow-blue-950/50 transition-all hover:from-blue-500 hover:to-indigo-500 focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-slate-900 focus:outline-none active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
    >
      <span className="h-3 w-3 animate-pulse rounded-full bg-red-500" aria-hidden="true" />
      <span>Iniciar Gravação &amp; Análise IA</span>
    </button>
  );
}
