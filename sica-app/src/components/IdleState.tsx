import { RecordingButton } from './RecordingButton';

interface IdleStateProps {
  onStartRecording: () => void;
}

export function IdleState({ onStartRecording }: IdleStateProps) {
  return (
    <div className="sica-fade-in flex flex-1 flex-col items-center justify-center space-y-6 text-center">
      <div className="group relative flex h-28 w-28 items-center justify-center rounded-full border border-blue-500/30 bg-blue-950/60 shadow-inner">
        <div className="absolute inset-0 rounded-full bg-blue-500/10 blur-xl transition-all group-hover:bg-blue-500/20" />
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="relative z-10 h-14 w-14 text-blue-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
          />
        </svg>
      </div>

      <div className="space-y-2">
        <h2 className="text-2xl font-bold tracking-tight text-white">Captura Espectral Real</h2>
        <p className="px-6 text-xs leading-relaxed text-slate-400">
          Classificação neural via <strong className="text-blue-300">PyTorch CNN</strong> e
          decomposição espectral via <strong className="text-indigo-300">STFT + HPSS</strong>.
        </p>
      </div>

      <RecordingButton onClick={onStartRecording} />
    </div>
  );
}
