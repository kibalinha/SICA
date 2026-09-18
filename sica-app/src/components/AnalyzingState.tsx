export function AnalyzingState() {
  return (
    <div className="sica-fade-in flex flex-1 flex-col items-center justify-center space-y-6 text-center">
      <div className="relative h-20 w-20">
        <div className="absolute inset-0 animate-spin rounded-full border-4 border-blue-500/20 border-t-blue-500" />
        <div
          className="absolute inset-2 animate-spin rounded-full border-4 border-indigo-500/20 border-t-indigo-400"
          style={{ animationDirection: 'reverse', animationDuration: '1.2s' }}
        />
      </div>
      <div className="space-y-2">
        <h2 className="text-xl font-bold text-white">Classificação Neural &amp; Fourier...</h2>
        <p className="text-xs text-slate-400">
          Processando cena acústica com PyTorch CNN e decomposição HPSS.
        </p>
      </div>
    </div>
  );
}
